"""
MCTS Player - Monte Carlo Tree Search
A strong AI player using MCTS algorithm
"""
import random
import math
import time
from copy import deepcopy


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board, player, parent=None, move=None):
        self.board = board  # numpy array
        self.player = player  # 1 or 2
        self.parent = parent
        self.move = move  # (row, col) that led to this node
        
        self.children = []
        self.untried_moves = None  # Will be populated lazily
        
        self.visits = 0
        self.wins = 0  # from perspective of parent's player
        
    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        if self.untried_moves is None:
            self.untried_moves = self._get_legal_moves()
        return len(self.untried_moves) == 0 and len(self.children) > 0
    
    def is_terminal(self):
        """Check if this is a game-ending state"""
        return self._check_winner() is not None or len(self._get_legal_moves()) == 0
    
    def _get_legal_moves(self):
        """Get all legal moves from current position"""
        # Optimization: Only consider moves near existing stones
        # This drastically reduces search space
        
        # Find all stones on board
        stone_positions = []
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                if self.board[r][c] != 0:
                    stone_positions.append((r, c))
        
        if not stone_positions:
            # Empty board, return center
            center = len(self.board) // 2
            return [(center, center)]
        
        # Get all empty positions within distance 2 of any stone
        candidates = set()
        for sr, sc in stone_positions:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = sr + dr, sc + dc
                    if (0 <= r < len(self.board) and 
                        0 <= c < len(self.board[0]) and 
                        self.board[r][c] == 0):
                        candidates.add((r, c))
        
        return list(candidates)
    
    def _check_winner(self):
        """
        Check if there's a winner
        Returns: 1 (player 1 wins), 2 (player 2 wins), 0 (draw), None (game continues)
        """
        # Check for 5 in a row
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                if self.board[r][c] != 0:
                    player = self.board[r][c]
                    # Check 4 directions
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    for dr, dc in directions:
                        count = 1
                        # Check forward
                        nr, nc = r + dr, c + dc
                        while (0 <= nr < len(self.board) and 
                               0 <= nc < len(self.board[0]) and 
                               self.board[nr][nc] == player):
                            count += 1
                            nr += dr
                            nc += dc
                        # Check backward
                        nr, nc = r - dr, c - dc
                        while (0 <= nr < len(self.board) and 
                               0 <= nc < len(self.board[0]) and 
                               self.board[nr][nc] == player):
                            count += 1
                            nr -= dr
                            nc -= dc
                        
                        if count >= 5:
                            return player
        
        # Check for draw (board full)
        if len(self._get_legal_moves()) == 0:
            return 0
        
        return None
    
    def ucb1(self, exploration=1.41):
        """
        UCB1 formula for node selection
        exploration = sqrt(2) is theoretically optimal
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration_term
    
    def best_child(self, exploration=1.41):
        """Select best child using UCB1"""
        return max(self.children, key=lambda child: child.ucb1(exploration))
    
    def expand(self):
        """Expand tree by adding a new child node"""
        if self.untried_moves is None:
            self.untried_moves = self._get_legal_moves()
        
        if not self.untried_moves:
            return None
        
        # Prioritize good moves (winning/blocking)
        best_moves = []
        good_moves = []
        normal_moves = []
        
        for move in self.untried_moves:
            # Check if this move wins
            test_board = self.board.copy()
            test_board[move[0], move[1]] = self.player
            if self._quick_check_winner(test_board) == self.player:
                best_moves.append(move)
            # Check if this move blocks opponent win
            else:
                opponent = 3 - self.player
                test_board[move[0], move[1]] = opponent
                if self._quick_check_winner(test_board) == opponent:
                    good_moves.append(move)
                else:
                    normal_moves.append(move)
        
        # Pick from best available category
        if best_moves:
            move = best_moves[0]
            self.untried_moves = [m for m in self.untried_moves if m != move]
        elif good_moves:
            move = good_moves[random.randint(0, len(good_moves) - 1)]
            self.untried_moves = [m for m in self.untried_moves if m != move]
        else:
            move = self.untried_moves.pop(random.randint(0, len(self.untried_moves) - 1))
        
        # Create new board state
        new_board = self.board.copy()
        new_board[move[0], move[1]] = self.player
        
        # Create child node (next player's turn)
        next_player = 3 - self.player
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child)
        
        return child
    
    def simulate(self):
        """
        Simulate a random game from this position
        Returns: winner (1, 2, or 0 for draw)
        Uses lightweight simulation for speed
        """
        sim_board = self.board.copy()
        sim_player = self.player
        
        max_moves = 50  # Limit simulation depth
        moves = 0
        
        while moves < max_moves:
            # Quick terminal check
            winner = self._quick_check_winner(sim_board)
            if winner is not None:
                return winner
            
            # Get moves (limited area)
            legal_moves = self._get_legal_moves_fast(sim_board)
            if not legal_moves:
                return 0  # Draw
            
            # Pick random move
            move = random.choice(legal_moves)
            sim_board[move[0], move[1]] = sim_player
            
            # Switch player
            sim_player = 3 - sim_player
            moves += 1
        
        # Evaluate position if we hit move limit
        return self._evaluate_position(sim_board)
    
    def _quick_check_winner(self, board):
        """Ultra-fast winner check - only check last few rows/cols"""
        size = len(board)
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue
                player = board[r][c]
                # Only check horizontally and vertically for speed
                if c <= size - 5:
                    if all(board[r][c+i] == player for i in range(5)):
                        return player
                if r <= size - 5:
                    if all(board[r+i][c] == player for i in range(5)):
                        return player
                if r <= size - 5 and c <= size - 5:
                    if all(board[r+i][c+i] == player for i in range(5)):
                        return player
                if r <= size - 5 and c >= 4:
                    if all(board[r+i][c-i] == player for i in range(5)):
                        return player
        return None
    
    def _evaluate_position(self, board):
        """Quick position evaluation when simulation times out"""
        # Count stones for each player
        p1_count = 0
        p2_count = 0
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 1:
                    p1_count += 1
                elif board[r][c] == 2:
                    p2_count += 1
        
        # Return player with more stones (simple heuristic)
        if p1_count > p2_count:
            return 1
        elif p2_count > p1_count:
            return 2
        return 0
    
    def _get_legal_moves_fast(self, board):
        """Fast legal moves for simulation - very limited search"""
        moves = []
        # Find stones
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] != 0:
                    # Check immediate neighbors only (distance 1)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < len(board) and 0 <= nc < len(board[0]) and 
                                board[nr][nc] == 0 and (nr, nc) not in moves):
                                moves.append((nr, nc))
                    if len(moves) > 15:  # Limit for speed
                        return moves
        
        if not moves:
            # Empty board
            center = len(board) // 2
            return [(center, center)]
        
        return moves
    
    def backpropagate(self, result):
        """
        Backpropagate the result up the tree
        result: winner (1 or 2) or 0 for draw
        """
        self.visits += 1
        
        # Update wins from parent's perspective
        if self.parent:
            # If result is parent's player, it's a win for parent
            if result == self.parent.player:
                self.wins += 1
            elif result == 0:  # Draw
                self.wins += 0.5
            
            self.parent.backpropagate(result)


class Player:
    """MCTS Player"""
    
    def __init__(self, rules, board_size):
        self.rules = rules
        self.board_size = board_size
        self.name = "MCTS Bot"
        self.time_limit = 4.5  # seconds (留 0.5s buffer)
        
    def play(self, board, turn_number, last_opponent_move):
        """
        Make a move using MCTS
        
        Args:
            board: numpy array, 0 = empty, 1 = you, 2 = opponent
            turn_number: current turn number
            last_opponent_move: tuple (row, col) or None
            
        Returns:
            tuple (row, col)
        """
        # Quick first move
        if turn_number == 0:
            center = self.board_size // 2
            return (center, center)
        
        # Run MCTS
        root = MCTSNode(board.copy(), player=1, parent=None, move=None)
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < self.time_limit:
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal():
                node = node.expand()
                if node is None:
                    continue
            
            # Simulation
            result = node.simulate()
            
            # Backpropagation
            node.backpropagate(result)
            
            iterations += 1
        
        # Choose best move
        if not root.children:
            # Fallback to random
            legal_moves = root._get_legal_moves()
            return random.choice(legal_moves) if legal_moves else (0, 0)
        
        # Choose child with most visits (most reliable)
        best_child = max(root.children, key=lambda c: c.visits)
        
        print(f"  MCTS: {iterations} iterations, {best_child.visits} visits, "
              f"winrate: {best_child.wins/best_child.visits:.1%}")
        
        return best_child.move
    
    def _select(self, node):
        """
        Selection phase: traverse tree using UCB1 until leaf node
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child()
        return node


if __name__ == "__main__":
    # Test the MCTS player
    import numpy as np
    
    player = Player('gomoku', 15)
    board = np.zeros((15, 15))
    
    print(f"Player: {player.name}")
    print("\nTest 1: First move (should be center)")
    move = player.play(board, 0, None)
    print(f"  Move: {move}")
    
    print("\nTest 2: Second move")
    board[7, 7] = 2  # Opponent played center
    move = player.play(board, 1, (7, 7))
    print(f"  Move: {move}")
