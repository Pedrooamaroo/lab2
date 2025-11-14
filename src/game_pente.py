"""
Pente Game Implementation (15x15)
Five in a row OR capture 10 stones (5 pairs) wins!
"""
import numpy as np


class PenteGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 = Black (starts), 2 = White
        self.winner = None
        self.move_history = []
        self.captures = {1: 0, 2: 0}  # stones captured by each player
        
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        self.captures = {1: 0, 2: 0}
        
    def is_valid_move(self, row, col):
        """Check if a move is valid"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    def get_valid_moves(self):
        """Return list of all valid moves as (row, col) tuples"""
        return [(r, c) for r in range(self.board_size) 
                for c in range(self.board_size) 
                if self.board[r, c] == 0]
    
    def check_capture(self, row, col):
        """
        Check for captures after placing stone at (row, col)
        Capture: flank exactly 2 opponent stones in a straight line
        Returns number of stones captured
        """
        player = self.board[row, col]
        opponent = 3 - player
        captured_count = 0
        
        # 8 directions to check
        directions = [
            (0, 1),   # right
            (0, -1),  # left
            (1, 0),   # down
            (-1, 0),  # up
            (1, 1),   # down-right
            (-1, -1), # up-left
            (1, -1),  # down-left
            (-1, 1)   # up-right
        ]
        
        for dr, dc in directions:
            # Check if we have: player - opponent - opponent - player
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2*dr, col + 2*dc
            r3, c3 = row + 3*dr, col + 3*dc
            
            # Check bounds
            if not (0 <= r3 < self.board_size and 0 <= c3 < self.board_size):
                continue
            
            # Check pattern: opponent, opponent, player
            if (self.board[r1, c1] == opponent and 
                self.board[r2, c2] == opponent and 
                self.board[r3, c3] == player):
                
                # Capture!
                self.board[r1, c1] = 0
                self.board[r2, c2] = 0
                captured_count += 2
        
        return captured_count
    
    def make_move(self, row, col):
        """
        Make a move at (row, col) for current player
        Returns True if successful, False otherwise
        """
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # Check for captures
        captured = self.check_capture(row, col)
        if captured > 0:
            self.captures[self.current_player] += captured
        
        # Check for win by capture (10 stones = 5 pairs)
        if self.captures[self.current_player] >= 10:
            self.winner = self.current_player
        # Check for win by five in a row
        elif self.check_win(row, col):
            self.winner = self.current_player
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        return True
    
    def check_win(self, row, col):
        """
        Check if the last move at (row, col) resulted in five in a row
        
        PENTE RULE: Black (player 1) cannot win with 6+ in a row (overline).
        White (player 2) can win with any 5+ in a row.
        """
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
            
            # Rule: Both players win with 5 or more in a row
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
    
    @staticmethod
    def check_winner(board, captures_p1=0, captures_p2=0):
        """
        Static method para verificar vencedor em qualquer board
        Usado pelo MCTS e interface
        
        Args:
            board: numpy array (board_size, board_size)
            captures_p1: número de capturas do jogador 1
            captures_p2: número de capturas do jogador 2
        
        Returns:
            0 (no winner), 1 (player 1), 2 (player 2)
        """
        # Check win by capture first
        if captures_p1 >= 10:
            return 1
        if captures_p2 >= 10:
            return 2
        
        # Check win by 5 in a row
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
                    
                    # Both players can win with 5 or more in a row
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
    
    def display(self):
        """Display the board in a human-readable format"""
        symbols = {0: '·', 1: '●', 2: '○'}
        
        print(f"Captures - Black (●): {self.captures[1]}, White (○): {self.captures[2]}")
        
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
    # Test the game with captures
    game = PenteGame(15)
    print("Pente Game Test")
    print("=" * 50)
    game.display()
    
    # Test capture scenario
    print("\nTesting capture scenario...")
    game.board[7, 6] = 1  # Black
    game.board[7, 7] = 2  # White
    game.board[7, 8] = 2  # White
    game.current_player = 1
    game.display()
    
    print("Black plays at (7, 9) - should capture 2 white stones!")
    game.make_move(7, 9)
    game.display()
    print(f"Black has captured {game.captures[1]} stones")
