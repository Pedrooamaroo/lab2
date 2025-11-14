"""
Heuristic Player - uses simple strategies
Better than random, but not optimal
"""
import random


class Player:
    def __init__(self, rules, board_size):
        """
        Initialize the player
        
        Args:
            rules: 'gomoku' or 'pente'
            board_size: size of the board (15 for this project)
        """
        self.rules = rules
        self.board_size = board_size
        self.name = "Heuristic Bot"
        
    def play(self, board, turn_number, last_opponent_move):
        """
        Make a move using simple heuristics
        
        Strategy:
        1. Win if possible (make 5 in a row)
        2. Block opponent from winning
        3. Play near existing stones
        4. Play center on first move
        
        Args:
            board: numpy array, 0 = empty, 1 = you, 2 = opponent
            turn_number: current turn number
            last_opponent_move: tuple (row, col) or None if first move
            
        Returns:
            tuple (row, col) for the move
        """
        # First move: play center
        if turn_number == 0:
            center = self.board_size // 2
            if board[center][center] == 0:
                return (center, center)
        
        # Try to win
        winning_move = self.find_winning_move(board, player=1)
        if winning_move:
            return winning_move
        
        # Block opponent from winning
        blocking_move = self.find_winning_move(board, player=2)
        if blocking_move:
            return blocking_move
        
        # Play near existing stones (good strategy for connection games)
        near_move = self.find_move_near_stones(board)
        if near_move:
            return near_move
        
        # Fallback: random move
        return self.random_move(board)
    
    def find_winning_move(self, board, player):
        """Find a move that creates 5 in a row for player"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == 0:
                    # Try this move
                    board[row][col] = player
                    if self.check_win(board, row, col, player):
                        board[row][col] = 0  # undo
                        return (row, col)
                    board[row][col] = 0  # undo
        return None
    
    def check_win(self, board, row, col, player):
        """Check if move at (row, col) creates 5 in a row"""
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal \
            [(1, -1), (-1, 1)]   # diagonal /
        ]
        
        for direction_pair in directions:
            count = 1
            for dr, dc in direction_pair:
                r, c = row + dr, col + dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if board[r][c] == player:
                        count += 1
                        r += dr
                        c += dc
                    else:
                        break
            if count >= 5:
                return True
        return False
    
    def find_move_near_stones(self, board):
        """Find an empty position near existing stones"""
        candidates = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == 0:
                    # Check if there's a stone nearby
                    if self.has_neighbor(board, row, col):
                        candidates.append((row, col))
        
        if candidates:
            return random.choice(candidates)
        return None
    
    def has_neighbor(self, board, row, col, distance=2):
        """Check if position has any stones within distance"""
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if board[r][c] != 0:
                        return True
        return False
    
    def random_move(self, board):
        """Random valid move as fallback"""
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == 0:
                    valid_moves.append((row, col))
        
        if valid_moves:
            return random.choice(valid_moves)
        return (0, 0)


if __name__ == "__main__":
    # Test the player
    import numpy as np
    
    player = Player('gomoku', 15)
    board = np.zeros((15, 15))
    
    print(f"Player: {player.name}")
    print("\nTest 1: First move (should be center)")
    move = player.play(board, 0, None)
    print(f"  Move: {move}")
    
    print("\nTest 2: Winning move detection")
    board = np.zeros((15, 15))
    board[7, 5:9] = 1  # 4 in a row for player
    print("  Board has 4 in a row at (7, 5-8)")
    move = player.play(board, 5, None)
    print(f"  Move to win: {move}")
