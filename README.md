# Kestrel: Cross-Modal Sparse Routing for Retrieval-Augmented Generation

Official implementation for the paper "Kestrel: Cross-Modal Sparse Routing for
Retrieval-Augmented Generation".

## Contents

- `src/kestrel/router.py` - cross-modal sparse router
- `src/kestrel/encoders.py` - text and image encoders
- `src/train_retriever.py` - retriever training script
- `src/evaluate_rag.py` - end-to-end RAG evaluation
- `scripts/prepare_corpus.py` - corpus preparation pipeline
- `configs/acl26.yaml` - configuration used in the paper
- `tests/test_router.py` - unit tests
- `checkpoints/README.md` - checkpoint download notes

## Status

Released implementation with training and evaluation code matching the
camera-ready paper.

## Citation

S. Park, A. Flores, and D. Nwosu. "Kestrel: Cross-Modal Sparse Routing for
Retrieval-Augmented Generation." ACL 2026.
