import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepExpert(nn.Module):
    """Deep Expert with multiple hidden layers and residual connections."""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_prob: float = 0.1):
        super(DeepExpert, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

        # Projection for residual connections (if needed)
        self.residual_projections = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            if prev_size != hidden_size:
                self.residual_projections.append(nn.Linear(prev_size, hidden_size))
            else:
                self.residual_projections.append(nn.Identity())
            prev_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        for i, (layer, bn, proj) in enumerate(zip(self.layers, self.bn_layers, self.residual_projections)):
            identity = proj(residual)
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + identity  # Residual connection
            residual = x

        x = self.output_layer(x)
        return x


class AttentionExpert(nn.Module):
    """Expert with self-attention mechanism for feature importance."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_heads: int = 4, dropout_prob: float = 0.1):
        super(AttentionExpert, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Multi-head attention
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Initial projection
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Self-attention
        residual = x
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.view(batch_size, -1)
        attention_output = self.attention_output(attention_output)

        x = self.ln1(residual + self.dropout(attention_output))

        # Feed-forward network
        residual = x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ResidualExpert(nn.Module):
    """Expert with residual blocks similar to ResNet."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_blocks: int = 2, dropout_prob: float = 0.1):
        super(ResidualExpert, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.bn_input = nn.BatchNorm1d(hidden_size)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_prob) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.bn_input(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    """A residual block with batch normalization."""

    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out


class ConvolutionalExpert(nn.Module):
    """Convolutional Expert for image/spatial data."""

    def __init__(self, in_channels: int, hidden_channels: int, output_size: int,
                 kernel_size: int = 3, dropout_prob: float = 0.1):
        super(ConvolutionalExpert, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels * 2, output_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
