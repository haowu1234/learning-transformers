# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

**learning-transformers** — A hands-on project for learning the Transformer architecture by implementing it from scratch, and fine-tuning on classic downstream tasks (no pretraining).

## Architecture

```
learning-transformers/
├── src/                    # Source code
│   ├── models/             # Model implementations
│   │   ├── attention.py    # Multi-Head Attention
│   │   ├── embedding.py    # Token embedding & positional encoding
│   │   ├── transformer.py  # Transformer encoder/decoder
│   │   └── bert.py         # BERT model
│   ├── tasks/              # Downstream task heads
│   │   ├── classification.py  # Text classification
│   │   ├── ner.py             # Named Entity Recognition
│   │   ├── qa.py              # Extractive QA
│   │   └── sentence_pair.py   # NLI / Semantic similarity
│   └── utils/              # Utilities
│       ├── data.py         # Data loading & preprocessing
│       ├── training.py     # Training loop & scheduling
│       └── metrics.py      # Evaluation metrics
├── configs/                # YAML config files
├── scripts/                # Training & evaluation entry points
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for exploration
└── requirements.txt        # Python dependencies
```

## Tech Stack

- Python 3.10+
- PyTorch (core framework)
- HuggingFace `tokenizers` & `datasets`
- YAML-based configuration

## Conventions

- All model code in `src/models/`, task heads in `src/tasks/`, utilities in `src/utils/`
- Use pretrained weights (e.g. from HuggingFace) for fine-tuning; no pretraining in this project
- Config-driven: hyperparameters live in `configs/` YAML files
- Entry points are in `scripts/`
