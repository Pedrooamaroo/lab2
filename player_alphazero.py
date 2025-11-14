"""
Player AlphaZero - MCTS guiado por Rede Neural
Usa GomokuNet treinada para guiar busca MCTS
"""

import numpy as np
import time
import math
import os
import torch
from network import GomokuNet, board_to_tensor, load_checkpoint, create_network
from numba import njit


@njit(cache=True)
def check_win_fast(board, board_size):
    """Ultra-fast winner detection with Numba JIT"""
    for row in range(board_size):
        for col in range(board_size):
            player = board[row, col]
            if player == 0:
                continue
            
            directions = ((0, 1), (1, 0), (1, 1), (1, -1))
            for dr, dc in directions:
                count = 1
                r, c = row + dr, col + dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                
                if count >= 5:
                    return player
    
    return 0


@njit(cache=True)
def get_legal_moves_proximity(board, board_size, distance=2):
    """Retorna movimentos legais perto de pedras existentes"""
    moves = []
    
    if np.sum(board) == 0:
        center = board_size // 2
        return [(center, center)]
    
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                for dr in range(-distance, distance + 1):
                    for dc in range(-distance, distance + 1):
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < board_size and 0 <= nc < board_size and 
                            board[nr, nc] == 0):
                            moves.append((nr, nc))
    
    # Remove duplicates
    if len(moves) == 0:
        return moves
    
    unique_moves = []
    seen = set()
    for move in moves:
        if move not in seen:
            unique_moves.append(move)
            seen.add(move)
    
    return unique_moves


class MCTSNodeAlphaZero:
    """Nó MCTS com policy priors da rede neural"""
    
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_prob
    
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, c_puct=1.5):
        """UCB com policy prior (AlphaZero formula)"""
        if self.visits == 0:
            return float('inf')
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + u
    
    def select_child(self, c_puct=1.5):
        return max(self.children.items(), key=lambda item: item[1].ucb_score(c_puct))
    
    def expand(self, policy_probs, legal_moves, board_size):
        """Expande com priors da rede neural"""
        legal_probs = []
        for row, col in legal_moves:
            action_idx = row * board_size + col
            legal_probs.append(policy_probs[action_idx])
        
        legal_probs = np.array(legal_probs)
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        for (row, col), prior in zip(legal_moves, legal_probs):
            if (row, col) not in self.children:
                self.children[(row, col)] = MCTSNodeAlphaZero(parent=self, prior_prob=prior)
    
    def update(self, value):
        self.visits += 1
        self.value_sum += value


class Player:
    """Player AlphaZero para competição"""
    
    def __init__(self, rules='gomoku', board_size=15):
        self.rules = rules
        self.board_size = board_size
        self.player_id = None
        self.time_limit = 4.5
        
        # Carrega rede neural
        self.network = self._load_network()
        self.network.eval()
        
        # Configuração MCTS
        self.c_puct = 1.5
        self.root = MCTSNodeAlphaZero()
        
        print(f"AlphaZero Player inicializado (regras: {rules})")
    
    def _load_network(self):
        """Carrega rede neural treinada"""
        checkpoint_path = "model_checkpoint.pth"
        
        if os.path.exists(checkpoint_path):
            try:
                model, _, epoch = load_checkpoint(
                    checkpoint_path,
                    board_size=self.board_size,
                    num_filters=64,
                    num_blocks=4
                )
                print(f"Rede carregada: {checkpoint_path} (época {epoch})")
                return model
            except Exception as e:
                print(f"Erro ao carregar checkpoint: {e}")
        
        # Fallback: rede nova (sem treino)
        print("AVISO: Usando rede não treinada!")
        return create_network(self.board_size, num_filters=64, num_blocks=4)
    
    def play(self, board, turn_number, last_opponent_move):
        """
        Retorna próxima jogada
        
        Args:
            board: numpy array (board_size, board_size) com 0/1/2
            turn_number: int
            last_opponent_move: tuple (row, col) ou None
        
        Returns:
            (row, col): posição da jogada
        """
        start_time = time.time()
        
        # Determina player_id na primeira jogada
        if self.player_id is None:
            self.player_id = 1 if turn_number == 1 else 2
        
        # Primeira jogada: centro
        if turn_number == 1:
            center = self.board_size // 2
            return (center, center)
        
        # MCTS search com rede neural
        iterations = 0
        max_iterations = 50000
        
        while time.time() - start_time < self.time_limit and iterations < max_iterations:
            self._mcts_iteration(board.copy(), self.player_id)
            iterations += 1
        
        elapsed = time.time() - start_time
        
        # Escolhe melhor movimento (greedy: mais visitas)
        if not self.root.children:
            # Fallback: movimento legal aleatório
            legal = get_legal_moves_proximity(board, self.board_size, distance=3)
            if legal:
                move = legal[np.random.randint(len(legal))]
            else:
                move = (self.board_size // 2, self.board_size // 2)
            print(f"Turn {turn_number}: Fallback move {move}")
            return move
        
        best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
        visits = self.root.children[best_move].visits
        value = self.root.children[best_move].value()
        
        print(f"Turn {turn_number}: {best_move} "
              f"(iter={iterations}, time={elapsed:.2f}s, visits={visits}, value={value:.3f})")
        
        # Atualiza root
        if best_move in self.root.children:
            self.root = self.root.children[best_move]
            self.root.parent = None
        else:
            self.root = MCTSNodeAlphaZero()
        
        return best_move
    
    def _mcts_iteration(self, board, player):
        """Uma iteração MCTS"""
        node = self.root
        temp_board = board
        temp_player = player
        search_path = [node]
        
        # 1. Selection
        while node.children:
            move, node = node.select_child(self.c_puct)
            search_path.append(node)
            temp_board[move] = temp_player
            temp_player = 3 - temp_player
        
        # 2. Expansion & Evaluation
        winner = check_win_fast(temp_board, self.board_size)
        
        if winner == 0:
            legal_moves = get_legal_moves_proximity(temp_board, self.board_size, distance=2)
            
            if legal_moves:
                # Avalia com rede neural
                state = board_to_tensor(temp_board, temp_player, board_size=self.board_size)
                policy_probs, value = self.network.predict(state)
                
                # Expande
                node.expand(policy_probs, legal_moves, self.board_size)
                
                # Ajusta valor para perspectiva do jogador original
                if temp_player != player:
                    value = -value
            else:
                value = 0.0
        else:
            if winner == player:
                value = 1.0
            elif winner == 3 - player:
                value = -1.0
            else:
                value = 0.0
        
        # 3. Backpropagation
        for node in reversed(search_path):
            node.update(value)
            value = -value


if __name__ == "__main__":
    # Teste rápido
    print("Testando Player AlphaZero...")
    
    player = Player(rules='gomoku', board_size=15)
    board = np.zeros((15, 15), dtype=np.int32)
    
    # Primeira jogada
    move = player.play(board, 1, None)
    print(f"Primeira jogada: {move}")
    
    # Segunda jogada (simula oponente no centro)
    board[7, 7] = 1
    board[7, 8] = 2
    move = player.play(board, 3, (7, 8))
    print(f"Terceira jogada: {move}")
    
    print("\n✅ Player AlphaZero criado com sucesso!")
    print("\nNOTA: Para ter performance competitiva, execute train.py para treinar a rede!")
