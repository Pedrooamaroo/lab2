"""
Interface Streamlit para Gomoku AlphaZero
GUI moderna e interativa para jogar contra o bot
"""

import streamlit as st
import numpy as np
import time
from player_mcts_turbo import Player as MCTSTurboPlayer
from player_alphazero import Player as AlphaZeroPlayer
from player_random import Player as RandomPlayer
from game_gomoku import GomokuGame
from game_pente import PenteGame

# Configuração da página
st.set_page_config(
    page_title="Gomoku & Pente AlphaZero",
    page_icon="⚫",
    layout="wide"
)

# Inicializa estado da sessão
if 'game' not in st.session_state:
    st.session_state.game_type = 'gomoku'
    st.session_state.game = GomokuGame(board_size=15)
    st.session_state.board = np.zeros((15, 15), dtype=np.int32)
    st.session_state.human_player = 1
    st.session_state.bot_player = 2
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.move_history = []
    st.session_state.last_move = None
    st.session_state.bot_stats = {}
    st.session_state.bot = None
    st.session_state.bot_type = "MCTS Turbo"
    st.session_state.captures_p1 = 0
    st.session_state.captures_p2 = 0

def reset_game():
    """Reinicia o jogo"""
    if st.session_state.game_type == 'gomoku':
        st.session_state.game = GomokuGame(board_size=15)
    else:
        st.session_state.game = PenteGame(board_size=15)
    st.session_state.board = np.zeros((15, 15), dtype=np.int32)
    st.session_state.current_player = 1
    st.session_state.captures_p1 = 0
    st.session_state.captures_p2 = 0
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.move_history = []
    st.session_state.last_move = None
    st.session_state.bot_stats = {}

def create_bot(bot_type):
    """Cria bot selecionado"""
    rules = st.session_state.game_type
    if bot_type == "MCTS Turbo":
        bot = MCTSTurboPlayer(rules=rules, board_size=15)
        bot.time_limit = 3.0
    elif bot_type == "AlphaZero":
        bot = AlphaZeroPlayer(rules=rules, board_size=15)
        bot.time_limit = 3.0
    else:  # Random
        bot = RandomPlayer(rules=rules, board_size=15)
    return bot

def make_move(row, col):
    """Executa jogada no tabuleiro"""
    if st.session_state.game_over:
        return False
    
    if st.session_state.board[row, col] != 0:
        return False
    
    # Faz jogada
    st.session_state.board[row, col] = st.session_state.current_player
    st.session_state.move_history.append((row, col, st.session_state.current_player))
    st.session_state.last_move = (row, col)
    
    # Se for Pente, atualizar capturas
    if st.session_state.game_type == 'pente':
        from pente_engine import detect_capture_pente, apply_capture_pente
        captured = detect_capture_pente(st.session_state.board, row, col, 15)
        if captured > 0:
            apply_capture_pente(st.session_state.board, row, col, 15)
            if st.session_state.current_player == 1:
                st.session_state.captures_p1 += captured
            else:
                st.session_state.captures_p2 += captured
    
    # Verifica vitória
    if st.session_state.game_type == 'pente':
        winner = st.session_state.game.check_winner(
            st.session_state.board, 
            st.session_state.captures_p1, 
            st.session_state.captures_p2
        )
    else:
        winner = st.session_state.game.check_winner(st.session_state.board)
    
    if winner != 0:
        st.session_state.game_over = True
        st.session_state.winner = winner
        return True
    
    # Verifica empate
    if len(st.session_state.move_history) >= 15 * 15:
        st.session_state.game_over = True
        st.session_state.winner = 0
        return True
    
    # Alterna jogador
    st.session_state.current_player = 3 - st.session_state.current_player
    return True

def bot_move():
    """Bot faz jogada"""
    if st.session_state.game_over:
        return
    
    if st.session_state.bot is None:
        st.session_state.bot = create_bot(st.session_state.bot_type)
    
    # Bot joga
    turn_number = len(st.session_state.move_history) + 1
    
    start_time = time.time()
    move = st.session_state.bot.play(
        st.session_state.board.copy(),
        turn_number,
        st.session_state.last_move
    )
    elapsed = time.time() - start_time
    
    # Guarda estatísticas
    st.session_state.bot_stats = {
        'move': move,
        'time': elapsed,
        'turn': turn_number
    }
    
    # Executa jogada
    make_move(move[0], move[1])

def render_board():
    """Renderiza o tabuleiro"""
    board = st.session_state.board
    last_move = st.session_state.last_move
    
    # Gera HTML do tabuleiro
    html = "<div style='display: inline-block; background: #DEB887; padding: 10px; border-radius: 5px;'>"
    html += "<table style='border-collapse: collapse;'>"
    
    # Header com números das colunas
    html += "<tr><td style='width: 30px;'></td>"
    for col in range(15):
        html += f"<td style='width: 30px; text-align: center; font-size: 10px;'>{col}</td>"
    html += "</tr>"
    
    for row in range(15):
        html += f"<tr><td style='height: 30px; text-align: center; font-size: 10px;'>{row}</td>"
        for col in range(15):
            cell_value = board[row, col]
            
            # Determina cor da célula
            is_last = last_move and last_move == (row, col)
            
            if cell_value == 0:
                bg_color = "#DEB887"
                content = "+"
                color = "#8B4513"
            elif cell_value == 1:
                bg_color = "#FF6B6B" if is_last else "#000000"
                content = "●"
                color = "#FFFFFF" if cell_value == 1 else "#000000"
            else:  # 2
                bg_color = "#4ECDC4" if is_last else "#FFFFFF"
                content = "●"
                color = "#000000"
            
            html += f"<td style='width: 30px; height: 30px; background: {bg_color}; "
            html += f"border: 1px solid #8B4513; text-align: center; vertical-align: middle; "
            html += f"font-size: 18px; color: {color}; cursor: pointer;'>{content}</td>"
        html += "</tr>"
    
    html += "</table></div>"
    return html

# Interface principal
game_title = "⚫⚪ Gomoku AlphaZero" if st.session_state.game_type == 'gomoku' else "⚫⚪ Pente AlphaZero"
st.title(game_title)
st.markdown("---")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Escolher jogo
    game_options = {"Gomoku": "gomoku", "Pente": "pente"}
    selected_game = st.selectbox(
        "🎮 Escolher Jogo",
        list(game_options.keys()),
        index=list(game_options.values()).index(st.session_state.game_type)
    )
    
    if game_options[selected_game] != st.session_state.game_type:
        st.session_state.game_type = game_options[selected_game]
        reset_game()
        st.session_state.bot = None  # Reset bot com novas regras
        st.rerun()
    
    st.markdown("---")
    
    # Escolher bot
    bot_options = ["MCTS Turbo", "AlphaZero", "Random"]
    selected_bot = st.selectbox(
        "🤖 Escolher Bot",
        bot_options,
        index=bot_options.index(st.session_state.bot_type)
    )
    
    if selected_bot != st.session_state.bot_type:
        st.session_state.bot_type = selected_bot
        st.session_state.bot = None  # Reset bot
    
    # Escolher quem começa
    st.session_state.human_player = st.radio(
        "Jogas com",
        [1, 2],
        format_func=lambda x: "⚫ Preto (1º)" if x == 1 else "⚪ Branco (2º)",
        disabled=len(st.session_state.move_history) > 0
    )
    st.session_state.bot_player = 3 - st.session_state.human_player
    
    st.markdown("---")
    
    # Informação do jogo
    st.subheader("📊 Estado do Jogo")
    st.write(f"**Jogadas:** {len(st.session_state.move_history)}")
    
    # Mostrar capturas se for Pente
    if st.session_state.game_type == 'pente':
        st.write(f"**Capturas ⚫:** {st.session_state.captures_p1}")
        st.write(f"**Capturas ⚪:** {st.session_state.captures_p2}")
    
    if st.session_state.current_player == st.session_state.human_player:
        st.write("**Turno:** 👤 Humano")
    else:
        st.write("**Turno:** 🤖 Bot")
    
    if st.session_state.game_over:
        if st.session_state.winner == 0:
            st.success("🤝 Empate!")
        elif st.session_state.winner == st.session_state.human_player:
            st.success("🎉 Ganhaste!")
        else:
            st.error("😢 Bot ganhou!")
    
    st.markdown("---")
    
    # Estatísticas do bot
    if st.session_state.bot_stats:
        st.subheader("🤖 Última Jogada do Bot")
        stats = st.session_state.bot_stats
        st.write(f"**Movimento:** {stats['move']}")
        st.write(f"**Tempo:** {stats['time']:.2f}s")
        st.write(f"**Turno:** {stats['turn']}")
    
    st.markdown("---")
    
    # Botões de controlo
    if st.button("🔄 Novo Jogo", use_container_width=True):
        reset_game()
        st.rerun()
    
    if st.button("↩️ Desfazer", use_container_width=True, disabled=len(st.session_state.move_history) < 2):
        # Desfaz última jogada do humano e do bot
        if len(st.session_state.move_history) >= 2:
            for _ in range(2):
                if st.session_state.move_history:
                    row, col, player = st.session_state.move_history.pop()
                    st.session_state.board[row, col] = 0
            st.session_state.current_player = st.session_state.human_player
            st.session_state.game_over = False
            st.session_state.winner = None
            if st.session_state.move_history:
                st.session_state.last_move = st.session_state.move_history[-1][:2]
            else:
                st.session_state.last_move = None
            st.rerun()

# Layout principal - duas colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎮 Tabuleiro")
    
    # Renderiza tabuleiro
    board_html = render_board()
    st.markdown(board_html, unsafe_allow_html=True)
    
    # Input de jogada manual
    if not st.session_state.game_over and st.session_state.current_player == st.session_state.human_player:
        st.write("**Tua vez! Insere coordenadas (0-14):**")
        col_input1, col_input2, col_input3 = st.columns([1, 1, 2])
        with col_input1:
            row_input = st.number_input("Linha", 0, 14, 7, key="row_input")
        with col_input2:
            col_input = st.number_input("Coluna", 0, 14, 7, key="col_input")
        with col_input3:
            if st.button("✓ Jogar", use_container_width=True):
                if make_move(row_input, col_input):
                    st.rerun()
                else:
                    st.error("Jogada inválida!")

with col2:
    st.subheader("📜 Histórico")
    
    if st.session_state.move_history:
        history_text = ""
        for i, (row, col, player) in enumerate(st.session_state.move_history):
            symbol = "⚫" if player == 1 else "⚪"
            player_name = "Tu" if player == st.session_state.human_player else "Bot"
            history_text += f"{i+1}. {symbol} {player_name}: ({row},{col})\n"
        st.text_area("Jogadas", history_text, height=400)
    else:
        st.info("Nenhuma jogada ainda")

# Auto-play do bot
if not st.session_state.game_over and st.session_state.current_player == st.session_state.bot_player:
    with st.spinner(f"🤖 {st.session_state.bot_type} a pensar..."):
        bot_move()
    st.rerun()

# Informações no rodapé
st.markdown("---")
st.caption("🎯 Gomoku & Pente AlphaZero | Projeto LAB 2 | Rui Almeida")
