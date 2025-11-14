# Projeto 2 - Lab IA & DC
## Narabe Games Player (Gomoku & Pente) - MCTS Turbo 🚀

### 🎮 Jogos Implementados
- **Gomoku 15×15**: 5 em linha para ganhar
- **Pente 15×15**: 5 em linha OU capturar 10 pedras (5 pares)

### 🚀 **NOVO: MCTS Turbo com Numba JIT**
**Performance**: 10.000-14.000 iterações/turno (30-40x mais rápido que MCTS original!)

### 📁 Estrutura dos Ficheiros

```
├── game_gomoku.py          # Lógica do jogo Gomoku
├── game_pente.py           # Lógica do jogo Pente (com capturas)
├── play.py                 # Sistema principal para jogar
├── player_random.py        # Bot aleatório (baseline)
├── player_heuristic.py     # Bot com heurísticas simples
├── player_mcts.py          # MCTS original (150-350 iter/turno)
├── player_mcts_turbo.py    # 🚀 MCTS otimizado com Numba (10k+ iter/turno)
├── player.py               # ⭐ O TEU BOT (MCTS Turbo - para competição)
├── requirements.txt        # Dependências Python
├── PERFORMANCE_REPORT.md   # 📊 Relatório detalhado de performance
└── README.md               # Este ficheiro
```

### 🔥 Como Usar o MCTS Turbo

#### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

**Dependências necessárias:**
- `numpy` - Arrays e cálculos
- `numba` - JIT compilation (torna o código 30-40x mais rápido!)

#### 2. Testar o bot otimizado
```bash
# MCTS Turbo vs Random
python play.py player.py player_random.py --nodisplay

# MCTS Turbo vs Heuristic (teste difícil)
python play.py player.py player_heuristic.py --nodisplay

# Ver o tabuleiro durante o jogo
python play.py player.py player_random.py
```

#### 3. Comparar performance (MCTS original vs Turbo)
```bash
# MCTS original (lento)
python play.py player_mcts.py player_random.py --nodisplay

# MCTS Turbo (RÁPIDO! 🚀)
python play.py player_mcts_turbo.py player_random.py --nodisplay
```

### 📊 Performance Comparison

| Bot | Iterações/Turno | Velocidade | Força |
|-----|-----------------|------------|-------|
| Random | N/A | Instant | ⭐ |
| Heuristic | N/A | Instant | ⭐⭐ |
| MCTS Original | 150-350 | 80 iter/s | ⭐⭐⭐ |
| **MCTS Turbo** | **10k-14k** | **2k-3k iter/s** | **⭐⭐⭐⭐⭐** |

### 🏆 Resultados de Testes

**MCTS Turbo vs Random:**
- ✅ Vitória em 9 turnos
- Pico: 12.322 iterações (2.735 iter/s)

**MCTS Turbo vs Heuristic:**
- ✅ Vitória em 11 turnos  
- Pico: 14.193 iterações (3.154 iter/s)
- Winrate final: 100%

### 🔧 Como Funciona o MCTS Turbo

**Otimizações aplicadas:**

1. **Numba JIT Compilation** - Funções críticas compiladas para código nativo:
   - `check_win_fast()` - Detecção de vencedor ultra-rápida
   - `get_legal_moves_proximity()` - Geração inteligente de movimentos
   - `simulate_game()` - Simulações rápidas (rollouts)

2. **Algoritmo MCTS Completo:**
   - **Selection**: UCB1 para balancear exploration/exploitation
   - **Expansion**: Prioriza jogadas vencedoras e bloqueios
   - **Simulation**: Rollouts rápidos (limite 50 movimentos)
   - **Backpropagation**: Atualiza estatísticas da árvore

3. **Heurísticas Inteligentes:**
   - Busca apenas movimentos próximos de pedras (raio 2)
   - Prioriza jogadas que ganham imediatamente
   - Bloqueia vitórias do oponente

### 📝 Como Criar o Teu Bot

Se quiseres modificar `player.py`:

```python
class Player:
    def __init__(self, rules, board_size):
        self.rules = rules  # 'gomoku' ou 'pente'
        self.board_size = board_size  # 15
        self.name = "Meu Bot"
        
    def play(self, board, turn_number, last_opponent_move):
        # board: matriz numpy
        #   0 = vazio
        #   1 = tu
        #   2 = oponente
        # 
        # Retorna: (row, col)
        
        # A TUA LÓGICA AQUI!
        row, col = ... 
        return (row, col)
```

**Importante:**
- O tabuleiro é sempre da tua perspectiva (tu = 1, oponente = 2)
- **Timeout: 5 segundos por jogada** (MCTS Turbo usa 4.5s)
- Se excederes o tempo → jogada aleatória

### 🏆 Competição

1. O teu bot de competição está em `player.py` (já é o MCTS Turbo!)
2. Faz push para o GitLab (branch main)
3. Resultados publicados diariamente às 01:00 no Moodle

### 📦 Packages Permitidos
- numpy ✅
- numba ✅
- scikit-learn
- torch
- torchvision  
- stable-baselines3

### 🎯 Status do Projeto

**Concluído:** ✅
- [x] Jogos implementados e testados
- [x] Sistema de competição funcional
- [x] Bots baseline (Random, Heuristic)
- [x] MCTS completo e otimizado
- [x] Performance 30-40x melhor que original
- [x] Testes vs oponentes (100% winrate)
- [x] Bot de competição pronto (`player.py`)

**Opcional (melhorias futuras):**
- [ ] Rede neural simples (AlphaZero lite)
- [ ] Multiprocessing (paralelização CPU)
- [ ] RAVE (Rapid Action Value Estimation)
- [ ] Tuning de hiperparâmetros

### 🐛 Debugging

Se houver erros:
```bash
# Verificar sintaxe
python -m py_compile player.py

# Instalar dependências
pip install numpy numba

# Testar contra random
python play.py player.py player_random.py

# Testar ambos os jogos
python play.py player.py player_random.py --game gomoku
python play.py player.py player_random.py --game pente
```

### 📚 Recursos
- **Relatório de Performance**: Ver `PERFORMANCE_REPORT.md`
- AlphaZero: https://en.wikipedia.org/wiki/AlphaZero
- Gomoku: https://en.wikipedia.org/wiki/Gomoku
- Pente: https://en.wikipedia.org/wiki/Pente
- Numba JIT: https://numba.pydata.org/

---
**Bot atual**: MCTS Turbo 🚀  
**Performance**: 10.000-14.000 iterações/turno  
**Status**: ✅ PRONTO PARA COMPETIÇÃO  
**Força**: ⭐⭐⭐⭐⭐
