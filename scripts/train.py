"""
Script de Treino AlphaZero - Self-Play + Training Loop
Executa ciclos de self-play e treino da rede neural
"""

import numpy as np
import torch
import torch.optim as optim
import time
import os
from datetime import datetime

from network import create_network, save_checkpoint, load_checkpoint
from data_buffer import AugmentedBuffer
from training import self_play_game, train_network, MCTSAlphaZero
from game_gomoku import GomokuGame


class AlphaZeroTrainer:
    """Gerencia processo completo de treino AlphaZero"""
    
    def __init__(self, 
                 board_size=15,
                 num_filters=64,
                 num_blocks=4,
                 learning_rate=0.001,
                 buffer_size=100000):
        """
        Args:
            board_size: tamanho do tabuleiro
            num_filters: filtros nas convoluções
            num_blocks: blocos residuais
            learning_rate: taxa de aprendizado
            buffer_size: tamanho máximo do buffer
        """
        self.board_size = board_size
        self.game = GomokuGame(board_size=board_size)
        
        # Rede neural
        self.network = create_network(board_size, num_filters, num_blocks)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Buffer de experiências com augmentation
        self.buffer = AugmentedBuffer(max_size=buffer_size)
        
        # Estatísticas
        self.iteration = 0
        self.total_games = 0
        
        print(f"AlphaZero Trainer inicializado:")
        print(f"  Board size: {board_size}x{board_size}")
        print(f"  Network: {num_blocks} blocks, {num_filters} filters")
        print(f"  Parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"  Buffer size: {buffer_size:,}")
    
    def load_checkpoint(self, filename):
        """Carrega checkpoint anterior"""
        if os.path.exists(filename):
            self.network, self.optimizer, self.iteration = load_checkpoint(
                filename,
                board_size=self.board_size
            )
            print(f"Checkpoint carregado: iteração {self.iteration}")
            return True
        return False
    
    def self_play(self, num_games=100, num_simulations=800, temperature_threshold=15):
        """
        Executa self-play games
        
        Args:
            num_games: número de jogos a jogar
            num_simulations: simulações MCTS por jogada
            temperature_threshold: jogadas até usar temp=1.0
        
        Returns:
            stats: dict com estatísticas dos jogos
        """
        print(f"\n{'='*60}")
        print(f"SELF-PLAY: {num_games} jogos ({num_simulations} simulações/jogada)")
        print(f"{'='*60}")
        
        stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'total_experiences': 0
        }
        
        start_time = time.time()
        
        for game_num in range(num_games):
            game_start = time.time()
            
            # Joga um jogo
            game_data, winner = self_play_game(
                self.network,
                self.game,
                num_simulations=num_simulations,
                temperature_threshold=temperature_threshold
            )
            
            # Processa e adiciona ao buffer (com augmentation = 8x mais dados!)
            self.buffer.process_game_with_augmentation(
                game_data, 
                winner, 
                board_size=self.board_size
            )
            
            # Atualiza estatísticas
            if winner == 1:
                stats['player1_wins'] += 1
            elif winner == 2:
                stats['player2_wins'] += 1
            else:
                stats['draws'] += 1
            
            stats['avg_game_length'] += len(game_data)
            stats['total_experiences'] += len(game_data) * 8  # 8x por causa do augmentation
            
            game_time = time.time() - game_start
            
            if (game_num + 1) % 10 == 0:
                print(f"  Jogo {game_num+1}/{num_games}: "
                      f"Winner={winner}, Length={len(game_data)}, "
                      f"Time={game_time:.1f}s, Buffer={len(self.buffer):,}")
        
        elapsed = time.time() - start_time
        stats['avg_game_length'] /= num_games
        self.total_games += num_games
        
        print(f"\nSelf-play completado em {elapsed/60:.1f} min:")
        print(f"  P1 wins: {stats['player1_wins']} ({100*stats['player1_wins']/num_games:.1f}%)")
        print(f"  P2 wins: {stats['player2_wins']} ({100*stats['player2_wins']/num_games:.1f}%)")
        print(f"  Draws: {stats['draws']} ({100*stats['draws']/num_games:.1f}%)")
        print(f"  Avg game length: {stats['avg_game_length']:.1f} jogadas")
        print(f"  Total experiences: {stats['total_experiences']:,}")
        print(f"  Buffer size: {len(self.buffer):,}")
        
        return stats
    
    def train(self, batch_size=64, epochs=10):
        """
        Treina rede neural com buffer atual
        
        Args:
            batch_size: tamanho do batch
            epochs: número de épocas
        
        Returns:
            losses: histórico de losses
        """
        print(f"\n{'='*60}")
        print(f"TRAINING: {epochs} épocas, batch_size={batch_size}")
        print(f"{'='*60}")
        
        if len(self.buffer) < batch_size:
            print(f"Buffer muito pequeno ({len(self.buffer)} < {batch_size}), pulando treino")
            return None
        
        start_time = time.time()
        
        losses = train_network(
            self.network,
            self.optimizer,
            self.buffer,
            batch_size=batch_size,
            epochs=epochs
        )
        
        elapsed = time.time() - start_time
        print(f"\nTreino completado em {elapsed:.1f}s")
        print(f"  Final losses: Policy={losses['policy'][-1]:.4f}, "
              f"Value={losses['value'][-1]:.4f}, Total={losses['total'][-1]:.4f}")
        
        return losses
    
    def save(self, filename="model_checkpoint.pth"):
        """Salva checkpoint"""
        save_checkpoint(self.network, self.optimizer, self.iteration, filename)
    
    def run_iteration(self, 
                     num_games=100,
                     num_simulations=800,
                     batch_size=64,
                     epochs=10):
        """
        Executa uma iteração completa: self-play + treino
        
        Args:
            num_games: jogos de self-play
            num_simulations: simulações MCTS por jogada
            batch_size: batch size para treino
            epochs: épocas de treino
        """
        self.iteration += 1
        
        print(f"\n{'#'*60}")
        print(f"ITERAÇÃO {self.iteration}")
        print(f"{'#'*60}")
        
        iteration_start = time.time()
        
        # 1. Self-play
        self.self_play(num_games, num_simulations)
        
        # 2. Treino
        self.train(batch_size, epochs)
        
        # 3. Save checkpoint
        self.save(f"model_checkpoint.pth")
        self.save(f"model_checkpoint_iter{self.iteration}.pth")
        
        # 4. Save buffer
        self.buffer.save(f"experience_buffer.pkl")
        
        elapsed = time.time() - iteration_start
        print(f"\nIteração {self.iteration} completada em {elapsed/60:.1f} min")
        print(f"Total de jogos até agora: {self.total_games}")
    
    def run_training(self, 
                    num_iterations=10,
                    games_per_iteration=100,
                    num_simulations=800,
                    batch_size=64,
                    epochs=10):
        """
        Executa loop completo de treino
        
        Args:
            num_iterations: número de iterações
            games_per_iteration: jogos por iteração
            num_simulations: simulações MCTS
            batch_size: batch size
            epochs: épocas por iteração
        """
        print(f"\n{'='*60}")
        print(f"INICIANDO TREINO ALPHAZERO")
        print(f"{'='*60}")
        print(f"Configuração:")
        print(f"  Iterações: {num_iterations}")
        print(f"  Jogos/iteração: {games_per_iteration}")
        print(f"  Simulações MCTS: {num_simulations}")
        print(f"  Batch size: {batch_size}")
        print(f"  Épocas: {epochs}")
        print(f"  Total estimado de jogos: {num_iterations * games_per_iteration}")
        
        total_start = time.time()
        
        for i in range(num_iterations):
            self.run_iteration(
                num_games=games_per_iteration,
                num_simulations=num_simulations,
                batch_size=batch_size,
                epochs=epochs
            )
        
        total_elapsed = time.time() - total_start
        
        print(f"\n{'='*60}")
        print(f"TREINO COMPLETADO!")
        print(f"{'='*60}")
        print(f"Tempo total: {total_elapsed/3600:.2f} horas")
        print(f"Total de jogos: {self.total_games}")
        print(f"Experiências no buffer: {len(self.buffer):,}")
        print(f"Checkpoint final: model_checkpoint.pth")


def main():
    """Função principal de treino"""
    
    # Configuração rápida (Mini-AlphaZero): ~5-10 horas CPU
    CONFIG_MINI = {
        'num_iterations': 5,
        'games_per_iteration': 50,
        'num_simulations': 400,
        'batch_size': 32,
        'epochs': 5,
        'num_filters': 32,
        'num_blocks': 3,
    }
    
    # Configuração média: ~20-30 horas CPU
    CONFIG_MEDIUM = {
        'num_iterations': 10,
        'games_per_iteration': 100,
        'num_simulations': 800,
        'batch_size': 64,
        'epochs': 10,
        'num_filters': 64,
        'num_blocks': 4,
    }
    
    # Configuração completa: ~50-100 horas CPU
    CONFIG_FULL = {
        'num_iterations': 20,
        'games_per_iteration': 200,
        'num_simulations': 1000,
        'batch_size': 128,
        'epochs': 15,
        'num_filters': 64,
        'num_blocks': 5,
    }
    
    # Escolha a configuração (mude aqui!)
    CONFIG = CONFIG_MINI  # <-- MUDE PARA CONFIG_MEDIUM ou CONFIG_FULL se tiver tempo
    
    print(f"Usando configuração: {'MINI' if CONFIG == CONFIG_MINI else 'MEDIUM' if CONFIG == CONFIG_MEDIUM else 'FULL'}")
    
    # Cria trainer
    trainer = AlphaZeroTrainer(
        board_size=15,
        num_filters=CONFIG['num_filters'],
        num_blocks=CONFIG['num_blocks'],
        learning_rate=0.001,
        buffer_size=100000
    )
    
    # Tenta carregar checkpoint anterior (para continuar treino)
    trainer.load_checkpoint("model_checkpoint.pth")
    
    # Executa treino
    trainer.run_training(
        num_iterations=CONFIG['num_iterations'],
        games_per_iteration=CONFIG['games_per_iteration'],
        num_simulations=CONFIG['num_simulations'],
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs']
    )


if __name__ == "__main__":
    # Verifica se CUDA está disponível
    if torch.cuda.is_available():
        print("⚡ GPU detectada! Usando CUDA para acelerar treino")
        device = torch.device('cuda')
    else:
        print("💻 Usando CPU (sem GPU)")
        device = torch.device('cpu')
    
    # Inicia treino
    main()
