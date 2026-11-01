"""The three networks LHD is tested on.

Each model lists its watched layers with lhd_layers(), in the order
data flows through them, so TorchAdapter can wrap it directly.
Every ReLU is its own module, so its output can be captured alone.
"""

import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    """Small convolutional network for 32x32 color images.

    Layout: conv1, conv2, conv3 (8, 16, and 32 filters, each 3x3,
    ReLU, 2x2 max pool), then fc1 (64 units, ReLU) and fc2 (class
    scores). Kept small so hundreds of runs fit on one laptop CPU.
    """

    def __init__(self, classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.act3 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = self.pool(self.act3(self.conv3(x)))
        x = self.act4(self.fc1(x.flatten(1)))
        return self.fc2(x)

    def lhd_layers(self) -> dict:
        """Watched layers for TorchAdapter."""
        return {
            "conv1": (self.conv1, self.act1, 1),
            "conv2": (self.conv2, self.act2, 1),
            "conv3": (self.conv3, self.act3, 1),
            "fc1": (self.fc1, self.act4, -1),
            "fc2": (self.fc2, self.fc2, -1),
        }


class MLP(nn.Module):
    """Fully connected network: 784 -> 256 -> 128 -> 64 -> classes, ReLU between."""

    def __init__(self, inputs: int, classes: int):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.fc4(x)

    def lhd_layers(self) -> dict:
        """Watched layers for TorchAdapter."""
        return {
            "fc1": (self.fc1, self.act1, -1),
            "fc2": (self.fc2, self.act2, -1),
            "fc3": (self.fc3, self.act3, -1),
            "fc4": (self.fc4, self.fc4, -1),
        }


class Block(nn.Module):
    """One pre-norm transformer block with causal self-attention and a ReLU feed-forward."""

    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, hidden)
        self.act = nn.ReLU()
        self.ff2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(width, dim=2)
        shape = (batch, steps, self.heads, width // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, steps, width))
        return x + self.ff2(self.act(self.ff1(self.ln2(x))))


class TinyTransformer(nn.Module):
    """Character-level language model: token and position embeddings,
    two transformer blocks, and a linear head.

    The feed-forward layers use ReLU, as in the original transformer,
    so dead units can be counted the same way as in the other networks.
    """

    def __init__(self, vocab: int, context: int, width: int = 64,
                 heads: int = 4, depth: int = 2, hidden: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads, hidden) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok(idx) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def lhd_layers(self) -> dict:
        """Watched layers for TorchAdapter."""
        layers = {
            "tok": (self.tok, self.tok, -1),
            "pos": (self.pos, self.pos, -1),
        }
        for i, block in enumerate(self.blocks):
            layers[f"b{i}.qkv"] = (block.qkv, block.qkv, -1)
            layers[f"b{i}.proj"] = (block.proj, block.proj, -1)
            layers[f"b{i}.ff1"] = (block.ff1, block.act, -1)
            layers[f"b{i}.ff2"] = (block.ff2, block.ff2, -1)
        layers["head"] = (self.head, self.head, -1)
        return layers


def relu_layers(model: nn.Module) -> list[str]:
    """Watched layers whose output passes through a ReLU, where units can die."""
    return [name for name, (_, out, _) in model.lhd_layers().items()
            if isinstance(out, nn.ReLU)]
