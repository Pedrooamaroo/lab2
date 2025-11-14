"""
Training Pipeline para AlphaZero
Implementa self-play com MCTS guiado por rede neural
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import random
import math
from network import GomokuNet, board_to_tensor, save_checkpoint
from data_buffer import AugmentedBuffer
from game_gomoku import GomokuGame


class MCTSNodeNN:
    """Nó MCTS com suporte para rede neural (policy priors)"""
    
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_prob
    
    def value(self):
        """Valor médio do nó"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, c_puct=1.5):
        """UCB score com policy prior (fórmula AlphaZero)"""
        if self.visits == 0:
            return float('inf')
        
        # U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + u
    
    def select_child(self, c_puct=1.5):
        """Seleciona melhor filho por UCB"""
        return max(self.children.items(), key=lambda item: item[1].ucb_score(c_puct))
    
    def expand(self, policy_probs, legal_moves):
        """
        Expande nó com priors da rede neural
        
        Args:
            policy_probs: numpy array com probabilidades de cada ação
            legal_moves: lista de (row, col) válidas
        """
        board_size = int(math.sqrt(len(policy_probs)))
        
        # Normaliza priors apenas sobre movimentos legais
        legal_probs = []
        for row, col in legal_moves:
            action_idx = row * board_size + col
            legal_probs.append(policy_probs[action_idx])
        
        # Normaliza
        legal_probs = np.array(legal_probs)
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            # Fallback: uniform distribution
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Cria filhos com priors
        for (row, col), prior in zip(legal_moves, legal_probs):
            if (row, col) not in self.children:
                self.children[(row, col)] = MCTSNodeNN(parent=self, prior_prob=prior)
    
    def update(self, value):
        """Backpropagation"""
        self.visits += 1
        self.value_sum += value


class MCTSAlphaZero:
    """MCTS guiado por rede neural (AlphaZero style)"""
    
    def __init__(self, network, game=None, c_puct=1.5, num_simulations=800):
        self.network = network
        self.game = game if game is not None else GomokuGame(board_size=15)
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root = MCTSNodeNN()
    
    def search(self, board, player, temperature=1.0):
        """
        Executa MCTS search e retorna distribuição de policy
        
        Args:
            board: numpy array (board_size, board_size)
            player: int (1 ou 2)
            temperature: float - controla exploração (1.0 = full, 0.1 = greedy)
        
        Returns:
            policy: numpy array (board_size * board_size) com distribuição de visitas
            value: float - valor da posição (da perspectiva do player)
        """
        board_size = self.game.board_size
        
        for _ in range(self.num_simulations):
            # Copia board para simulação
            node = self.root
            temp_board = board.copy()
            temp_player = player
            search_path = [node]
            
            # 1. Selection: desce até folha
            while node.children:
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
                temp_board[move] = temp_player
                temp_player = 3 - temp_player
            
            # 2. Expansion & Evaluation
            winner = self.game.check_winner(temp_board)
            
            if winner == 0:  # Jogo não terminou
                # Expande com rede neural
                legal_moves = self.game.get_legal_moves(temp_board)
                
                if legal_moves:
                    # Avalia posição com rede neural
                    state = board_to_tensor(temp_board, temp_player, board_size=board_size)
                    policy_probs, value = self.network.predict(state)
                    
                    # Expande nó
                    node.expand(policy_probs, legal_moves)
                    
                    # Valor da perspectiva do jogador original
                    # Se temp_player != player, inverte sinal
                    if temp_player != player:
                        value = -value
                else:
                    value = 0.0  # Empate
            else:
                # Jogo terminou
                if winner == player:
                    value = 1.0
                elif winner == 3 - player:
                    value = -1.0
                else:
                    value = 0.0
            
            # 3. Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Alterna perspectiva
        
        # Extrai distribuição de policy baseada em visitas
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        
        if temperature < 0.01:  # Greedy
            # Escolhe movimento com mais visitas
            best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
            row, col = best_move
            policy[row * board_size + col] = 1.0
        else:
            # Distribuição proporcional a visitas^(1/temp)
            visits = np.array([
                child.visits for child in self.root.children.values()
            ])
            moves = list(self.root.children.keys())
            
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            
            visits_sum = visits.sum()
            if visits_sum > 0:
                visit_probs = visits / visits_sum
                
                for move, prob in zip(moves, visit_probs):
                    row, col = move
                    policy[row * board_size + col] = prob
        
        return policy, self.root.value()
    
    def get_action(self, board, player, temperature=1.0):
        """
        Retorna ação escolhida por MCTS
        
        Returns:
            (row, col): melhor movimento
            policy: distribuição completa
        """
        policy, _ = self.search(board, player, temperature)
        
        # Amostra ação da distribuição
        board_size = self.game.board_size
        action_idx = np.random.choice(len(policy), p=policy)
        row = action_idx // board_size
        col = action_idx % board_size
        
        return (row, col), policy
    
    def update_root(self, move):
        """Move raiz para filho correspondente ao movimento"""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = MCTSNodeNN()


def self_play_game(network, game=None, num_simulations=800, temperature_threshold=15):
    """
    Joga um jogo completo de self-play
    
    Args:
        network: GomokuNet
        game: GomokuGame ou GamePente
        num_simulations: número de simulações MCTS por jogada
        temperature_threshold: jogadas até usar temp=1.0, depois temp=0.1
    
    Returns:
        game_data: lista de (state, policy, player)
        winner: int (1, 2, ou 0)
    """
    if game is None:
        game = GomokuGame(board_size=15)
    board = np.zeros((game.board_size, game.board_size), dtype=np.int32)
    player = 1
    game_data = []
    last_move = None
    
    mcts = MCTSAlphaZero(network, game, num_simulations=num_simulations)
    
    for turn in range(game.board_size * game.board_size):
        # Temperature decay: exploração no início, greedy no final
        temperature = 1.0 if turn < temperature_threshold else 0.1
        
        # Estado atual
        state = board_to_tensor(board, player, last_move, game.board_size)
        
        # MCTS search
        policy, value = mcts.search(board, player, temperature)
        
        # Guarda experiência
        game_data.append((state, policy, player))
        
        # Escolhe ação
        action_idx = np.random.choice(len(policy), p=policy)
        row = action_idx // game.board_size
        col = action_idx % game.board_size
        
        # Executa movimento
        board[row, col] = player
        last_move = (row, col)
        
        # Atualiza MCTS root
        mcts.update_root((row, col))
        
        # Verifica vitória
        winner = game.check_winner(board)
        if winner != 0:
            return game_data, winner
        
        # Próximo jogador
        player = 3 - player
    
    # Empate
    return game_data, 0


def train_network(network, optimizer, buffer, batch_size=64, epochs=10):
    """
    Treina rede neural com experiências do buffer
    
    Args:
        network: GomokuNet
        optimizer: torch optimizer
        buffer: ExperienceBuffer
        batch_size: tamanho do batch
        epochs: número de épocas
    
    Returns:
        losses: dict com histórico de losses
    """
    network.train()
    device = next(network.parameters()).device
    
    losses = {'policy': [], 'value': [], 'total': []}
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        num_batches = 0
        
        # Itera sobre todo o buffer em batches
        buffer_size = len(buffer)
        num_steps = max(1, buffer_size // batch_size)
        
        for _ in range(num_steps):
            # Sample batch
            states, target_policies, target_values = buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(device)
            target_policies = torch.FloatTensor(target_policies).to(device)
            target_values = torch.FloatTensor(target_values).to(device)
            
            # Forward pass
            log_policies, values = network(states)
            
            # Loss de policy (cross-entropy)
            policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
            
            # Loss de value (MSE)
            value_loss = F.mse_loss(values, target_values)
            
            # Loss total
            total_loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
        
        # Média das losses
        if num_batches > 0:
            epoch_policy_loss /= num_batches
            epoch_value_loss /= num_batches
            epoch_total_loss /= num_batches
        
        losses['policy'].append(epoch_policy_loss)
        losses['value'].append(epoch_value_loss)
        losses['total'].append(epoch_total_loss)
        
        print(f"  Época {epoch+1}/{epochs}: "
              f"Policy Loss={epoch_policy_loss:.4f}, "
              f"Value Loss={epoch_value_loss:.4f}, "
              f"Total={epoch_total_loss:.4f}")
    
    return losses


if __name__ == "__main__":
    print("Training pipeline criado!")
    print("Use train.py para executar treino completo")
