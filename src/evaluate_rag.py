"""End-to-end retrieval-augmented generation evaluation."""

import argparse


def evaluate(config: str) -> None:
    print(f"Running RAG evaluation with {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acl26.yaml")
    args = parser.parse_args()
    evaluate(args.config)
