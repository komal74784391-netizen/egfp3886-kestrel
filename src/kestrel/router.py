"""Cross-modal sparse routing layer."""

import torch
from torch import nn


class SparseRouter(nn.Module):
    def __init__(self, dim: int = 512, experts: int = 8) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(experts)])

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(queries), dim=-1)
        outputs = torch.stack([expert(keys) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)
