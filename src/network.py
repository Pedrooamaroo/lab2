"""
Rede Neural AlphaZero para Gomoku/Pente
Arquitetura: ResNet com Policy e Value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Bloco residual com 2 convoluções e skip connection"""
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class GomokuNet(nn.Module):
    """
    Rede Neural AlphaZero para Gomoku/Pente
    
    Entrada: tensor (batch, 3, board_size, board_size)
        - Canal 0: posições do jogador atual (1 onde há peça, 0 caso contrário)
        - Canal 1: posições do adversário (1 onde há peça, 0 caso contrário)
        - Canal 2: última jogada (1 na posição da última jogada, 0 resto)
    
    Saída:
        - policy: probabilidades para cada ação (batch, board_size * board_size)
        - value: avaliação da posição (batch, 1) em [-1, 1]
    """
    
    def __init__(self, board_size=15, num_filters=64, num_blocks=4):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size
        
        # Camada de entrada: 3 canais -> num_filters
        self.conv_input = nn.Conv2d(3, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Blocos residuais
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])
        
        # Policy head: prediz probabilidades de cada ação
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)
        
        # Value head: prediz valor da posição
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: tensor (batch, 3, board_size, board_size)
        
        Returns:
            policy: tensor (batch, board_size * board_size) com log-probabilidades
            value: tensor (batch, 1) em [-1, 1]
        """
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # Log-prob para estabilidade numérica
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # [-1, 1]
        
        return policy, value
    
    def predict(self, board_state):
        """
        Predição para um único estado (usado durante o jogo)
        
        Args:
            board_state: numpy array (3, board_size, board_size)
        
        Returns:
            policy_probs: numpy array (board_size * board_size) com probabilidades
            value: float em [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # Converte para tensor
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0)
            
            # Forward pass
            log_policy, value = self.forward(state_tensor)
            
            # Converte de log-prob para prob
            policy_probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value = value.item()
        
        return policy_probs, value


def board_to_tensor(board, current_player, last_move=None, board_size=15):
    """
    Converte tabuleiro numpy para tensor de entrada da rede
    
    Args:
        board: numpy array (board_size, board_size) com 0/1/2
        current_player: int (1 ou 2) - jogador que vai jogar
        last_move: tuple (row, col) ou None - última jogada do adversário
        board_size: int - tamanho do tabuleiro
    
    Returns:
        tensor: numpy array (3, board_size, board_size)
    """
    tensor = np.zeros((3, board_size, board_size), dtype=np.float32)
    
    # Canal 0: posições do jogador atual
    tensor[0] = (board == current_player).astype(np.float32)
    
    # Canal 1: posições do adversário
    opponent = 3 - current_player  # 1->2, 2->1
    tensor[1] = (board == opponent).astype(np.float32)
    
    # Canal 2: última jogada
    if last_move is not None:
        row, col = last_move
        if 0 <= row < board_size and 0 <= col < board_size:
            tensor[2, row, col] = 1.0
    
    return tensor


def create_network(board_size=15, num_filters=64, num_blocks=4):
    """
    Factory function para criar rede neural
    
    Args:
        board_size: tamanho do tabuleiro
        num_filters: número de filtros nas convoluções
        num_blocks: número de blocos residuais
    
    Returns:
        GomokuNet: rede neural
    """
    return GomokuNet(board_size, num_filters, num_blocks)


def save_checkpoint(model, optimizer, epoch, filename):
    """Salva checkpoint do modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'board_size': model.board_size,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvo: {filename}")


def load_checkpoint(filename, board_size=15, num_filters=64, num_blocks=4):
    """
    Carrega checkpoint do modelo
    
    Returns:
        model, optimizer, epoch
    """
    checkpoint = torch.load(filename)
    
    model = create_network(
        board_size=checkpoint.get('board_size', board_size),
        num_filters=num_filters,
        num_blocks=num_blocks
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    
    print(f"Checkpoint carregado: {filename} (época {epoch})")
    return model, optimizer, epoch


if __name__ == "__main__":
    # Teste da rede
    print("Testando GomokuNet...")
    
    net = create_network(board_size=15, num_filters=64, num_blocks=4)
    print(f"Parâmetros da rede: {sum(p.numel() for p in net.parameters()):,}")
    
    # Teste com batch aleatório
    batch_size = 4
    board_size = 15
    x = torch.randn(batch_size, 3, board_size, board_size)
    
    log_policy, value = net(x)
    print(f"Policy shape: {log_policy.shape}")  # (4, 225)
    print(f"Value shape: {value.shape}")  # (4, 1)
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    # Teste de predição única
    board = np.zeros((15, 15), dtype=np.int32)
    board[7, 7] = 1  # Jogador 1 no centro
    board[7, 8] = 2  # Jogador 2 ao lado
    
    state = board_to_tensor(board, current_player=1, last_move=(7, 8))
    probs, val = net.predict(state)
    
    print(f"\nPredição para estado de teste:")
    print(f"Policy probs sum: {probs.sum():.6f} (deve ser ~1.0)")
    print(f"Value: {val:.3f}")
    print(f"Top 5 ações: {np.argsort(probs)[-5:][::-1]}")
    
    print("\n✅ Rede neural criada com sucesso!")
