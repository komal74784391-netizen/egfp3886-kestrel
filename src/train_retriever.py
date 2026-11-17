"""Train the Kestrel retriever."""

import argparse

import torch

from kestrel.encoders import ImageEncoder, TextEncoder
from kestrel.router import SparseRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acl26.yaml")
    args = parser.parse_args()
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    router = SparseRouter()
    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()) + list(image_encoder.parameters()) + list(router.parameters()),
        lr=1e-3,
    )
    print("Kestrel retriever ready:", args.config, optimizer.defaults["lr"])


if __name__ == "__main__":
    main()
