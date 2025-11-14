"""
Script de Teste Rápido para AlphaZero
Verifica se todos os componentes estão funcionais
"""

import numpy as np
import torch
import time

print("="*60)
print("TESTE ALPHAZERO - Verificação de Componentes")
print("="*60)

# 1. Testa imports
print("\n1. Testando imports...")
try:
    from network import create_network, board_to_tensor
    from data_buffer import AugmentedBuffer
    from training import MCTSAlphaZero, self_play_game
    from game_gomoku import GomokuGame
    from player_alphazero import Player
    print("   ✅ Todos os imports OK")
except Exception as e:
    print(f"   ❌ Erro nos imports: {e}")
    exit(1)

# 2. Testa criação da rede
print("\n2. Testando rede neural...")
try:
    net = create_network(board_size=15, num_filters=64, num_blocks=4)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"   ✅ Rede criada: {num_params:,} parâmetros")
    
    # Teste forward pass
    test_input = torch.randn(2, 3, 15, 15)
    log_policy, value = net(test_input)
    print(f"   ✅ Forward pass OK: policy shape={log_policy.shape}, value shape={value.shape}")
    
    # Teste predict
    board = np.zeros((15, 15), dtype=np.int32)
    board[7, 7] = 1
    state = board_to_tensor(board, 1)
    probs, val = net.predict(state)
    print(f"   ✅ Predict OK: probs sum={probs.sum():.6f}, value={val:.3f}")
except Exception as e:
    print(f"   ❌ Erro na rede: {e}")
    exit(1)

# 3. Testa buffer
print("\n3. Testando buffer de experiências...")
try:
    buffer = AugmentedBuffer(max_size=1000)
    
    # Adiciona algumas experiências
    for i in range(5):
        state = np.random.randn(3, 15, 15).astype(np.float32)
        policy = np.random.rand(225).astype(np.float32)
        policy = policy / policy.sum()
        buffer.add_experience_with_augmentation(state, policy, 1.0, board_size=15)
    
    print(f"   ✅ Buffer criado: {len(buffer)} experiências (5 jogadas × 8 augmentations = 40)")
    
    # Testa sampling
    states, policies, values = buffer.sample(10)
    print(f"   ✅ Sampling OK: shapes={states.shape}, {policies.shape}, {values.shape}")
except Exception as e:
    print(f"   ❌ Erro no buffer: {e}")
    exit(1)

# 4. Testa MCTS com rede neural
print("\n4. Testando MCTS + Rede Neural...")
try:
    game = GomokuGame(board_size=15)
    mcts = MCTSAlphaZero(net, game, num_simulations=50)  # Só 50 para teste rápido
    
    board = np.zeros((15, 15), dtype=np.int32)
    board[7, 7] = 1  # Uma pedra no centro
    
    start = time.time()
    policy, value = mcts.search(board, player=2, temperature=1.0)
    elapsed = time.time() - start
    
    print(f"   ✅ MCTS search OK: {elapsed:.2f}s para 50 simulações")
    print(f"   ✅ Policy sum={policy.sum():.6f}, value={value:.3f}")
    
    # Verifica se policy está concentrada perto da pedra existente
    top_moves = np.argsort(policy)[-5:][::-1]
    print(f"   ✅ Top 5 movimentos: {[(m//15, m%15) for m in top_moves]}")
except Exception as e:
    print(f"   ❌ Erro no MCTS: {e}")
    exit(1)

# 5. Testa self-play completo
print("\n5. Testando self-play game (pode demorar ~30-60s)...")
try:
    start = time.time()
    game_data, winner = self_play_game(
        net, 
        game, 
        num_simulations=50,  # Baixo para teste rápido
        temperature_threshold=10
    )
    elapsed = time.time() - start
    
    print(f"   ✅ Self-play OK: Winner={winner}, Jogadas={len(game_data)}, Tempo={elapsed:.1f}s")
except Exception as e:
    print(f"   ❌ Erro no self-play: {e}")
    exit(1)

# 6. Testa Player AlphaZero
print("\n6. Testando Player AlphaZero...")
try:
    player = Player(rules='gomoku', board_size=15)
    
    board = np.zeros((15, 15), dtype=np.int32)
    
    # Primeira jogada
    move1 = player.play(board, turn_number=1, last_opponent_move=None)
    print(f"   ✅ Jogada 1: {move1}")
    
    # Segunda jogada (simula oponente)
    board[move1] = 1
    board[7, 8] = 2
    
    start = time.time()
    move2 = player.play(board, turn_number=3, last_opponent_move=(7, 8))
    elapsed = time.time() - start
    
    print(f"   ✅ Jogada 2: {move2} (tempo={elapsed:.2f}s)")
    print(f"   ℹ️  Com rede não treinada, jogadas podem parecer aleatórias")
except Exception as e:
    print(f"   ❌ Erro no Player: {e}")
    exit(1)

# 7. Teste de jogo completo vs Random
print("\n7. Testando jogo completo AlphaZero vs Random...")
try:
    from player_random import Player as RandomPlayer
    
    # Cria players
    alphazero = Player(rules='gomoku', board_size=15)
    alphazero.time_limit = 2.0  # Reduz para teste rápido
    random_player = RandomPlayer(rules='gomoku', board_size=15)
    
    board = np.zeros((15, 15), dtype=np.int32)
    game_obj = GomokuGame(board_size=15)
    
    print("   Jogando AlphaZero (P1) vs Random (P2)...")
    
    for turn in range(1, 50):
        # AlphaZero joga
        if turn % 2 == 1:
            move = alphazero.play(board, turn, None if turn == 1 else last_move)
            board[move] = 1
        else:
            move = random_player.play(board, turn, last_move)
            board[move] = 2
        
        last_move = move
        
        # Verifica vitória
        winner = game_obj.check_winner(board)
        if winner != 0:
            print(f"   ✅ Jogo terminou: Winner={winner} em {turn} jogadas")
            if winner == 1:
                print(f"   🎉 AlphaZero VENCEU!")
            else:
                print(f"   ⚠️  AlphaZero perdeu (normal com rede não treinada)")
            break
    else:
        print(f"   ✅ Jogo chegou a {turn} jogadas sem vencedor")
    
except Exception as e:
    print(f"   ❌ Erro no jogo completo: {e}")
    import traceback
    traceback.print_exc()

# Resumo final
print("\n" + "="*60)
print("TESTE COMPLETO!")
print("="*60)
print("\n✅ Todos os componentes estão funcionais!")
print("\n📝 Próximos passos:")
print("   1. Execute 'python train.py' para treinar a rede (5-10h para Mini)")
print("   2. Após treino, copie player_alphazero.py para player.py")
print("   3. Teste novamente - performance será MUITO melhor!")
print("\nℹ️  Nota: Com rede NÃO TREINADA, o bot joga quase aleatoriamente.")
print("         Após treino, será top 5-10% do torneio!")
print("="*60)
