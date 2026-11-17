"""Text and image encoders."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(token_ids).mean(dim=1))


class ImageEncoder(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
