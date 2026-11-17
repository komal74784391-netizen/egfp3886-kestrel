import torch
from kestrel.router import SparseRouter


def test_router_shapes():
    router = SparseRouter(dim=16, experts=4)
    queries = torch.randn(2, 16)
    keys = torch.randn(5, 16)
    assert router(queries, keys).shape == (2, 16)
