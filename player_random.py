"""
Random Player - makes random valid moves
This is the baseline bot to beat!
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
        self.name = "Random Bot"
        
    def play(self, board, turn_number, last_opponent_move):
        """
        Make a move
        
        Args:
            board: numpy array, 0 = empty, 1 = you, 2 = opponent
            turn_number: current turn number
            last_opponent_move: tuple (row, col) or None if first move
            
        Returns:
            tuple (row, col) for the move
        """
        # Find all valid moves (empty positions)
        valid_moves = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 0:
                    valid_moves.append((row, col))
        
        # Choose a random valid move
        if valid_moves:
            return random.choice(valid_moves)
        
        # Should never happen in a normal game
        return (0, 0)


if __name__ == "__main__":
    # Test the player
    import numpy as np
    
    player = Player('gomoku', 15)
    board = np.zeros((15, 15))
    
    print(f"Player: {player.name}")
    print("Making 5 random moves:")
    for i in range(5):
        move = player.play(board, i, None)
        print(f"  Move {i+1}: {move}")
        board[move[0], move[1]] = 1
