"""
Motor Pente otimizado com Numba
Funções JIT-compiled para detecção de capturas e vitória
"""

import numpy as np
from numba import njit


@njit(cache=True)
def check_win_pente(board, board_size):
    """
    Verifica vencedor no Pente (5 ou mais em linha)
    Returns: 0 (no winner), 1 (player 1), 2 (player 2)
    """
    for row in range(board_size):
        for col in range(board_size):
            player = board[row, col]
            if player == 0:
                continue
            
            # Check 4 directions: horizontal, vertical, diag\, diag/
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
def detect_capture_pente(board, row, col, board_size):
    """
    Detecta capturas após colocar pedra em (row, col)
    Captura: flanquear exatamente 2 pedras adversárias numa linha
    
    Returns: número de pedras capturadas (0, 2, 4, 6, ou 8)
    """
    player = board[row, col]
    if player == 0:
        return 0
    
    opponent = 3 - player
    captured_count = 0
    
    # 8 direções para verificar
    directions = (
        (0, 1),   # direita
        (0, -1),  # esquerda
        (1, 0),   # baixo
        (-1, 0),  # cima
        (1, 1),   # diagonal baixo-direita
        (-1, -1), # diagonal cima-esquerda
        (1, -1),  # diagonal baixo-esquerda
        (-1, 1)   # diagonal cima-direita
    )
    
    for dr, dc in directions:
        # Verifica padrão: jogador - oponente - oponente - jogador
        r1, c1 = row + dr, col + dc
        r2, c2 = row + 2*dr, col + 2*dc
        r3, c3 = row + 3*dr, col + 3*dc
        
        # Verifica bounds
        if not (0 <= r3 < board_size and 0 <= c3 < board_size):
            continue
        
        # Verifica padrão: oponente, oponente, jogador
        if (board[r1, c1] == opponent and 
            board[r2, c2] == opponent and 
            board[r3, c3] == player):
            
            captured_count += 2
    
    return captured_count


@njit(cache=True)
def apply_capture_pente(board, row, col, board_size):
    """
    Aplica capturas no tabuleiro após colocar pedra em (row, col)
    Modifica board in-place e retorna número de pedras capturadas
    """
    player = board[row, col]
    if player == 0:
        return 0
    
    opponent = 3 - player
    captured_count = 0
    
    directions = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    )
    
    for dr, dc in directions:
        r1, c1 = row + dr, col + dc
        r2, c2 = row + 2*dr, col + 2*dc
        r3, c3 = row + 3*dr, col + 3*dc
        
        if not (0 <= r3 < board_size and 0 <= c3 < board_size):
            continue
        
        if (board[r1, c1] == opponent and 
            board[r2, c2] == opponent and 
            board[r3, c3] == player):
            
            # Captura!
            board[r1, c1] = 0
            board[r2, c2] = 0
            captured_count += 2
    
    return captured_count


@njit(cache=True)
def check_win_by_capture(captures_p1, captures_p2):
    """
    Verifica se algum jogador ganhou por captura (10+ pedras)
    Returns: 0 (no winner), 1 (player 1), 2 (player 2)
    """
    if captures_p1 >= 10:
        return 1
    if captures_p2 >= 10:
        return 2
    return 0


@njit(cache=True)
def get_legal_moves_proximity_pente(board, board_size, distance=2):
    """
    Retorna movimentos legais perto de pedras existentes (Pente)
    Similar ao Gomoku mas considera espaços criados por capturas
    """
    moves = []
    
    # Primeira jogada: centro
    if np.sum(board) == 0:
        center = board_size // 2
        return [(center, center)]
    
    # Procura movimentos perto de pedras
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                for dr in range(-distance, distance + 1):
                    for dc in range(-distance, distance + 1):
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < board_size and 0 <= nc < board_size and 
                            board[nr, nc] == 0):
                            moves.append((nr, nc))
    
    # Remove duplicados
    if len(moves) == 0:
        return moves
    
    unique_moves = []
    seen = set()
    for move in moves:
        if move not in seen:
            unique_moves.append(move)
            seen.add(move)
    
    return unique_moves


@njit(cache=True)
def simulate_game_pente(board, current_player, board_size, captures_p1, captures_p2, max_moves=50):
    """
    Simula jogo de Pente até fim ou max_moves
    Considera capturas e vitória por captura
    
    Returns: winner (0=draw, 1=player1, 2=player2)
    """
    temp_board = board.copy()
    temp_captures_p1 = captures_p1
    temp_captures_p2 = captures_p2
    player = current_player
    
    for _ in range(max_moves):
        # Verifica vitória por captura
        winner = check_win_by_capture(temp_captures_p1, temp_captures_p2)
        if winner != 0:
            return winner
        
        # Verifica vitória por 5 em linha
        winner = check_win_pente(temp_board, board_size)
        if winner != 0:
            return winner
        
        # Pega movimentos legais
        legal_moves = get_legal_moves_proximity_pente(temp_board, board_size, distance=2)
        if len(legal_moves) == 0:
            return 0  # Empate
        
        # Escolhe movimento aleatório
        move_idx = np.random.randint(len(legal_moves))
        row, col = legal_moves[move_idx]
        
        # Faz jogada
        temp_board[row, col] = player
        
        # Aplica capturas
        captured = apply_capture_pente(temp_board, row, col, board_size)
        if player == 1:
            temp_captures_p1 += captured
        else:
            temp_captures_p2 += captured
        
        # Próximo jogador
        player = 3 - player
    
    # Sem vencedor após max_moves
    return 0


if __name__ == "__main__":
    # Teste das funções
    print("Testando motor Pente...")
    
    board_size = 15
    board = np.zeros((board_size, board_size), dtype=np.int32)
    
    # Teste 1: Vitória por 5 em linha
    board[7, 5:10] = 1
    winner = check_win_pente(board, board_size)
    print(f"Teste vitória 5 em linha: {winner} (esperado: 1)")
    
    # Teste 2: Captura
    board = np.zeros((board_size, board_size), dtype=np.int32)
    board[7, 7] = 1  # Jogador 1
    board[7, 8] = 2  # Adversário
    board[7, 9] = 2  # Adversário
    board[7, 10] = 1 # Jogador 1 (flanqueia)
    
    captured = detect_capture_pente(board, 7, 10, board_size)
    print(f"Teste detecção captura: {captured} pedras (esperado: 2)")
    
    # Teste 3: Aplicar captura
    board_before = board.copy()
    captured = apply_capture_pente(board, 7, 10, board_size)
    print(f"Teste aplicar captura: {captured} pedras capturadas")
    print(f"Posição (7,8) antes: {board_before[7,8]}, depois: {board[7,8]} (esperado: 2 -> 0)")
    
    # Teste 4: Vitória por captura
    winner = check_win_by_capture(10, 8)
    print(f"Teste vitória por captura P1: {winner} (esperado: 1)")
    
    winner = check_win_by_capture(8, 12)
    print(f"Teste vitória por captura P2: {winner} (esperado: 2)")
    
    print("\n✅ Motor Pente criado com sucesso!")
