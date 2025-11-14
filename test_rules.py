"""
Test script to verify game rules are correctly implemented
"""
from game_gomoku import GomokuGame
from game_pente import PenteGame
import numpy as np


def test_gomoku_win():
    """Test Gomoku win conditions"""
    print("=" * 60)
    print("TEST 1: Gomoku - 5 in a row wins")
    print("=" * 60)
    
    game = GomokuGame(15)
    
    # Create 5 in a row horizontally
    for i in range(5):
        game.board[7, 5 + i] = 1
    
    game.display()
    
    # Check if it's a win
    if game.check_win(7, 7):
        print("✅ PASS: 5 in a row detected as win")
    else:
        print("❌ FAIL: 5 in a row not detected")
    
    print()


def test_gomoku_six_wins():
    """Test that 6+ in a row also wins in Gomoku"""
    print("=" * 60)
    print("TEST 2: Gomoku - 6+ in a row also wins")
    print("=" * 60)
    
    game = GomokuGame(15)
    
    # Create 6 in a row
    for i in range(6):
        game.board[7, 5 + i] = 1
    
    game.display()
    
    if game.check_win(7, 7):
        print("✅ PASS: 6 in a row wins in Gomoku (correct)")
    else:
        print("❌ FAIL: 6 in a row should win in Gomoku")
    
    print()


def test_pente_five_wins():
    """Test that exactly 5 wins in Pente for Black"""
    print("=" * 60)
    print("TEST 3: Pente - Black wins with exactly 5")
    print("=" * 60)
    
    game = PenteGame(15)
    
    # Create 5 in a row for Black (player 1)
    for i in range(5):
        game.board[7, 5 + i] = 1
    
    game.display()
    
    if game.check_win(7, 7):
        print("✅ PASS: Black wins with exactly 5 in a row")
    else:
        print("❌ FAIL: Black should win with 5 in a row")
    
    print()


def test_pente_overline_black():
    """Test that 6+ does NOT win for Black in Pente (overline rule)"""
    print("=" * 60)
    print("TEST 4: Pente - Black CANNOT win with 6+ (overline)")
    print("=" * 60)
    
    game = PenteGame(15)
    
    # Create 6 in a row for Black (player 1)
    for i in range(6):
        game.board[7, 5 + i] = 1
    
    game.display()
    
    if not game.check_win(7, 7):
        print("✅ PASS: Black correctly blocked from winning with 6 (overline rule)")
    else:
        print("❌ FAIL: Black should NOT win with 6+ in a row (overline rule)")
    
    print()


def test_pente_overline_white():
    """Test that 6+ DOES win for White in Pente"""
    print("=" * 60)
    print("TEST 5: Pente - White CAN win with 6+")
    print("=" * 60)
    
    game = PenteGame(15)
    
    # Create 6 in a row for White (player 2)
    for i in range(6):
        game.board[7, 5 + i] = 2
    
    game.display()
    
    if game.check_win(7, 7):
        print("✅ PASS: White wins with 6 in a row (correct)")
    else:
        print("❌ FAIL: White should win with 6+ in a row")
    
    print()


def test_pente_capture():
    """Test Pente capture mechanism"""
    print("=" * 60)
    print("TEST 6: Pente - Capture mechanism")
    print("=" * 60)
    
    game = PenteGame(15)
    
    # Setup: Black - White - White - (empty)
    game.board[7, 6] = 1  # Black
    game.board[7, 7] = 2  # White
    game.board[7, 8] = 2  # White
    game.current_player = 1
    
    print("Before capture:")
    game.display()
    
    # Black plays at (7, 9) - should capture the two white stones
    game.make_move(7, 9)
    
    print("After Black plays at (7, 9):")
    game.display()
    
    if game.captures[1] == 2:
        print("✅ PASS: Captured 2 stones correctly")
    else:
        print(f"❌ FAIL: Expected 2 captures, got {game.captures[1]}")
    
    print()


def test_pente_capture_win():
    """Test winning by capturing 10 stones (5 pairs)"""
    print("=" * 60)
    print("TEST 7: Pente - Win by capturing 10 stones")
    print("=" * 60)
    
    game = PenteGame(15)
    game.captures[1] = 8  # Black already has 8 captures
    
    # Setup another capture opportunity
    game.board[7, 6] = 1
    game.board[7, 7] = 2
    game.board[7, 8] = 2
    game.current_player = 1
    
    print(f"Black has {game.captures[1]} captures")
    game.display()
    
    # Make the capturing move (should win by reaching 10 captures)
    game.make_move(7, 9)
    
    print(f"After capture, Black has {game.captures[1]} captures")
    
    if game.winner == 1 and game.captures[1] >= 10:
        print("✅ PASS: Black wins by capturing 10 stones")
    else:
        print(f"❌ FAIL: Black should win with 10 captures (has {game.captures[1]})")
    
    print()


def test_pente_no_capture_three():
    """Test that capturing 3 stones doesn't work (must be exactly 2)"""
    print("=" * 60)
    print("TEST 8: Pente - Cannot capture 3 stones")
    print("=" * 60)
    
    game = PenteGame(15)
    
    # Setup: Black - White - White - White - (empty)
    game.board[7, 6] = 1
    game.board[7, 7] = 2
    game.board[7, 8] = 2
    game.board[7, 9] = 2  # 3 whites
    game.current_player = 1
    
    print("Before move (3 white stones in a row):")
    game.display()
    
    # Black plays at (7, 10) - should NOT capture (needs exactly 2)
    game.make_move(7, 10)
    
    print("After Black plays at (7, 10):")
    game.display()
    
    if game.captures[1] == 0:
        print("✅ PASS: Cannot capture 3 stones (must be exactly 2)")
    else:
        print(f"❌ FAIL: Should not capture, but captured {game.captures[1]}")
    
    print()


if __name__ == "__main__":
    print("\n" + "🎮 TESTING GAME RULES 🎮".center(60))
    print("\n")
    
    # Gomoku tests
    test_gomoku_win()
    test_gomoku_six_wins()
    
    # Pente tests
    test_pente_five_wins()
    test_pente_overline_black()
    test_pente_overline_white()
    test_pente_capture()
    test_pente_capture_win()
    test_pente_no_capture_three()
    
    print("=" * 60)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 60)
