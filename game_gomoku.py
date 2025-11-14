"""
Gomoku Game Implementation (15x15)
Five in a row wins!
"""
import numpy as np


class GomokuGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 = Black (starts), 2 = White
        self.winner = None
        self.move_history = []
        
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        
    def is_valid_move(self, row, col):
        """Check if a move is valid"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    @staticmethod
    def check_winner(board):
        """
        Static method para verificar vencedor em qualquer board
        Usado pelo MCTS para avaliar estados
        
        Args:
            board: numpy array (board_size, board_size)
        
        Returns:
            0 (no winner), 1 (player 1), 2 (player 2)
        """
        board_size = board.shape[0]
        
        for row in range(board_size):
            for col in range(board_size):
                player = board[row, col]
                if player == 0:
                    continue
                
                # Check 4 directions: horizontal, vertical, diag\, diag/
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                for dr, dc in directions:
                    count = 1
                    # Forward direction
                    r, c = row + dr, col + dc
                    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                        count += 1
                        r += dr
                        c += dc
                    
                    if count >= 5:
                        return player
        
        return 0
    
    @staticmethod
    def get_legal_moves(board):
        """
        Static method para obter movimentos legais
        
        Args:
            board: numpy array (board_size, board_size)
        
        Returns:
            list of (row, col) tuples
        """
        board_size = board.shape[0]
        moves = []
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == 0:
                    moves.append((row, col))
        return moves
    
    def get_valid_moves(self):
        """Return list of all valid moves as (row, col) tuples"""
        return [(r, c) for r in range(self.board_size) 
                for c in range(self.board_size) 
                if self.board[r, c] == 0]
    
    def make_move(self, row, col):
        """
        Make a move at (row, col) for current player
        Returns True if successful, False otherwise
        """
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # Check for win
        if self.check_win(row, col):
            self.winner = self.current_player
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        return True
    
    def check_win(self, row, col):
        """Check if the last move at (row, col) resulted in a win"""
        player = self.board[row, col]
        
        # Directions: horizontal, vertical, diagonal /, diagonal \
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal \
            [(1, -1), (-1, 1)]   # diagonal /
        ]
        
        for direction_pair in directions:
            count = 1  # count the placed stone
            
            # Check both directions
            for dr, dc in direction_pair:
                r, c = row + dr, col + dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == player:
                        count += 1
                        r += dr
                        c += dc
                    else:
                        break
            
            if count >= 5:
                return True
        
        return False
    
    def is_game_over(self):
        """Check if game is over (win or draw)"""
        if self.winner is not None:
            return True
        # Draw: board is full
        return len(self.get_valid_moves()) == 0
    
    def get_winner(self):
        """Return the winner (1 or 2) or None if no winner yet, 0 for draw"""
        if self.winner:
            return self.winner
        if self.is_game_over():
            return 0  # Draw
        return None
    
    def get_board_copy(self):
        """Return a copy of the current board"""
        return self.board.copy()
    
    def get_board_for_player(self, player):
        """
        Return board from player's perspective
        Player is always 1, opponent is always 2
        """
        if player == 1:
            return self.board.copy()
        else:
            # Swap 1s and 2s
            board = self.board.copy()
            board[board == 1] = 9  # temp
            board[board == 2] = 1
            board[board == 9] = 2
            return board
    
    def display(self):
        """Display the board in a human-readable format"""
        symbols = {0: '·', 1: '●', 2: '○'}
        
        # Column numbers
        print('   ', end='')
        for c in range(self.board_size):
            print(f'{c:2}', end=' ')
        print()
        
        for r in range(self.board_size):
            print(f'{r:2} ', end='')
            for c in range(self.board_size):
                print(f' {symbols[self.board[r, c]]} ', end='')
            print()
        print()


if __name__ == "__main__":
    # Test the game
    game = GomokuGame(15)
    print("Gomoku Game Test")
    print("=" * 50)
    game.display()
    
    # Make some test moves
    test_moves = [(7, 7), (7, 8), (8, 7), (8, 8), (6, 6), (6, 7)]
    for row, col in test_moves:
        player = game.current_player
        if game.make_move(row, col):
            print(f"Player {player} ({'●' if player == 1 else '○'}) played at ({row}, {col})")
            game.display()
            if game.is_game_over():
                winner = game.get_winner()
                if winner == 0:
                    print("Game Over: Draw!")
                else:
                    print(f"Game Over: Player {winner} wins!")
                break
