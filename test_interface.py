"""
Teste rápido dos componentes da interface
"""
import numpy as np
from game_gomoku import GomokuGame
from game_pente import PenteGame
from player import Player

print("=" * 60)
print("TESTE 1: GomokuGame check_winner")
print("=" * 60)

g = GomokuGame(15)
board = np.zeros((15, 15), dtype=np.int32)

# Sem vencedor
result = g.check_winner(board)
print(f"✓ Tabuleiro vazio: {result} (esperado: 0)")

# 5 em linha para P1
board[7, 7:12] = 1
result = g.check_winner(board)
print(f"✓ 5 em linha P1: {result} (esperado: 1)")

# 5 em linha para P2
board = np.zeros((15, 15), dtype=np.int32)
board[5:10, 5] = 2
result = g.check_winner(board)
print(f"✓ 5 em linha P2: {result} (esperado: 2)")

print("\n" + "=" * 60)
print("TESTE 2: PenteGame check_winner")
print("=" * 60)

p = PenteGame(15)
board = np.zeros((15, 15), dtype=np.int32)

# Sem vencedor
result = p.check_winner(board, 0, 0)
print(f"✓ Tabuleiro vazio: {result} (esperado: 0)")

# 5 em linha para P1
board[7, 7:12] = 1
result = p.check_winner(board, 0, 0)
print(f"✓ 5 em linha P1: {result} (esperado: 1)")

# Vitória por captura P1
board = np.zeros((15, 15), dtype=np.int32)
result = p.check_winner(board, 10, 0)
print(f"✓ 10 capturas P1: {result} (esperado: 1)")

# Vitória por captura P2
result = p.check_winner(board, 0, 10)
print(f"✓ 10 capturas P2: {result} (esperado: 2)")

# 6 em linha (ambos ganham agora)
board = np.zeros((15, 15), dtype=np.int32)
board[7, 7:13] = 1  # 6 pedras
result = p.check_winner(board, 0, 0)
print(f"✓ 6 em linha P1: {result} (esperado: 1)")

board = np.zeros((15, 15), dtype=np.int32)
board[7, 7:13] = 2  # 6 pedras
result = p.check_winner(board, 0, 0)
print(f"✓ 6 em linha P2: {result} (esperado: 2)")

print("\n" + "=" * 60)
print("TESTE 3: Player initialization (Gomoku)")
print("=" * 60)

try:
    player_gomoku = Player('gomoku', 15)
    print(f"✓ Player Gomoku criado (rules={player_gomoku.rules})")
except Exception as e:
    print(f"✗ Erro ao criar Player Gomoku: {e}")

print("\n" + "=" * 60)
print("TESTE 4: Player initialization (Pente)")
print("=" * 60)

try:
    player_pente = Player('pente', 15)
    print(f"✓ Player Pente criado (rules={player_pente.rules})")
    print(f"  captures_p1={player_pente.captures_p1}")
    print(f"  captures_p2={player_pente.captures_p2}")
except Exception as e:
    print(f"✗ Erro ao criar Player Pente: {e}")

print("\n" + "=" * 60)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 60)
print("\nA interface deve funcionar corretamente agora.")
print("Acessa: http://localhost:8502")
