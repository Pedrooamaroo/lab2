"""
Main game runner - play games between different agents
Usage: python play.py <player1.py> <player2.py> [--game gomoku|pente] [--display]
"""
import sys
import importlib.util
import time
from game_gomoku import GomokuGame
from game_pente import PenteGame


def load_player(filepath, player_number):
    """
    Dynamically load a player from a Python file
    
    Args:
        filepath: path to the player file (e.g., 'player_random.py')
        player_number: 1 or 2
        
    Returns:
        Player instance
    """
    try:
        spec = importlib.util.spec_from_file_location(f"player{player_number}", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Player
    except Exception as e:
        print(f"Error loading player from {filepath}: {e}")
        sys.exit(1)


def play_game(player1_file, player2_file, game_type='gomoku', display=True, timeout=5.0):
    """
    Play a game between two players
    
    Args:
        player1_file: path to player 1 file
        player2_file: path to player 2 file
        game_type: 'gomoku' or 'pente'
        display: whether to display the board
        timeout: time limit per move in seconds
        
    Returns:
        winner (1 or 2), 0 for draw, -1 for error
    """
    # Load players
    Player1Class = load_player(player1_file, 1)
    Player2Class = load_player(player2_file, 2)
    
    # Create game
    if game_type == 'gomoku':
        game = GomokuGame(15)
    elif game_type == 'pente':
        game = PenteGame(15)
    else:
        print(f"Unknown game type: {game_type}")
        return -1
    
    # Initialize players
    try:
        player1 = Player1Class(game_type, 15)
        player2 = Player2Class(game_type, 15)
    except Exception as e:
        print(f"Error initializing players: {e}")
        return -1
    
    player1_name = getattr(player1, 'name', 'Player 1')
    player2_name = getattr(player2, 'name', 'Player 2')
    
    print(f"\n{'='*60}")
    print(f"Game: {game_type.upper()}")
    print(f"Black (●): {player1_name}")
    print(f"White (○): {player2_name}")
    print(f"{'='*60}\n")
    
    if display:
        game.display()
    
    # Game loop
    turn_number = 0
    last_move = None
    
    while not game.is_game_over():
        current_player_obj = player1 if game.current_player == 1 else player2
        current_player_name = player1_name if game.current_player == 1 else player2_name
        
        # Get board from player's perspective
        board_for_player = game.get_board_for_player(game.current_player)
        
        # Get move with timeout
        try:
            start_time = time.time()
            move = current_player_obj.play(board_for_player, turn_number, last_move)
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                print(f"⚠️  {current_player_name} exceeded time limit ({elapsed:.2f}s > {timeout}s)")
                print(f"   Making random move instead...")
                # Random fallback
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    import random
                    move = random.choice(valid_moves)
                else:
                    print("No valid moves available!")
                    break
            
            row, col = move
            
        except Exception as e:
            print(f"❌ Error from {current_player_name}: {e}")
            print(f"   Making random move instead...")
            # Random fallback
            valid_moves = game.get_valid_moves()
            if valid_moves:
                import random
                row, col = random.choice(valid_moves)
            else:
                print("No valid moves available!")
                break
        
        # Make move
        if game.make_move(row, col):
            symbol = '●' if game.current_player == 2 else '○'  # switched because player already changed
            print(f"Turn {turn_number + 1}: {current_player_name} ({symbol}) → ({row}, {col})")
            
            if display:
                game.display()
            
            last_move = (row, col)
            turn_number += 1
        else:
            print(f"❌ Invalid move from {current_player_name}: ({row}, {col})")
            print(f"   Player {game.current_player} forfeits!")
            return 3 - game.current_player  # Other player wins
    
    # Game over
    winner = game.get_winner()
    
    print(f"\n{'='*60}")
    if winner == 0:
        print("🤝 Game Over: DRAW!")
    elif winner == 1:
        print(f"🏆 Game Over: {player1_name} (●) WINS!")
    elif winner == 2:
        print(f"🏆 Game Over: {player2_name} (○) WINS!")
    
    if game_type == 'pente':
        print(f"\nCaptures - Black: {game.captures[1]}, White: {game.captures[2]}")
    
    print(f"Total turns: {turn_number}")
    print(f"{'='*60}\n")
    
    return winner


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python play.py <player1.py> <player2.py> [--game gomoku|pente] [--display] [--nodisplay]")
        print("\nExamples:")
        print("  python play.py player_random.py player_heuristic.py")
        print("  python play.py player.py player_random.py --game pente")
        print("  python play.py player1.py player2.py --nodisplay")
        sys.exit(1)
    
    player1_file = sys.argv[1]
    player2_file = sys.argv[2]
    
    # Parse options
    game_type = 'gomoku'
    display = True
    
    for arg in sys.argv[3:]:
        if arg == '--game' and len(sys.argv) > sys.argv.index(arg) + 1:
            game_type = sys.argv[sys.argv.index(arg) + 1]
        elif arg == '--nodisplay':
            display = False
        elif arg == '--display':
            display = True
    
    # Play the game
    winner = play_game(player1_file, player2_file, game_type, display)
    
    # Return exit code (useful for automation)
    sys.exit(0 if winner in [0, 1, 2] else 1)


if __name__ == "__main__":
    main()
