# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

**learning-transformers** — A hands-on project for learning the Transformer architecture by implementing it from scratch, and fine-tuning on classic downstream tasks (no pretraining).

## Architecture

```
src/
├── models/                     # Pure model components (task-agnostic)
│   ├── attention.py            # Multi-Head Attention
│   ├── embedding.py            # BertEmbedding (token + position + segment)
│   ├── encoder.py              # TransformerEncoderLayer + TransformerEncoder
│   └── bert.py                 # BertModel + BertForTask + weight loading
├── heads/                      # Pluggable task heads (Registry pattern)
│   ├── base.py                 # BaseHead abstract class
│   ├── classification.py       # ClassificationHead (sentence-level)
│   ├── token_classification.py # TokenClassificationHead (NER)
│   └── qa.py                   # QAHead (span extraction)
├── data/                       # Pluggable data modules (Registry pattern)
│   ├── base.py                 # BaseDataModule abstract class
│   ├── collator.py             # Generic DataCollator
│   └── sst2.py                 # SST-2 sentiment dataset
├── training/                   # Task-agnostic training engine
│   ├── trainer.py              # Unified Trainer
│   └── callbacks.py            # EarlyStopping, ModelCheckpoint
├── evaluation/                 # Pluggable metrics (Registry pattern)
│   ├── base.py                 # BaseMetric abstract class
│   └── metrics.py              # Accuracy, F1
└── utils/
    └── registry.py             # Generic Registry mechanism

configs/                        # YAML config per experiment
scripts/                        # Entry points (train.py, evaluate.py)
tests/                          # Unit tests
docs/                           # Design documents
```

## Tech Stack

- Python 3.10+
- PyTorch (core framework)
- HuggingFace `transformers`, `tokenizers`, `datasets`
- YAML-based configuration

## Design Patterns

- **Registry Pattern**: HEAD_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY — new tasks require zero changes to existing code
- **Abstract Base Classes**: BaseHead, BaseDataModule, BaseMetric define contracts
- **Composition**: BertForTask = BertModel (backbone) + any BaseHead (task head)
- **Config-driven**: each experiment has its own YAML file in `configs/`

## Conventions

- Model components in `src/models/`, task heads in `src/heads/`, data in `src/data/`
- Use pretrained weights (from HuggingFace) for fine-tuning; no pretraining
- Entry points in `scripts/`, run via `python scripts/train.py --config configs/xxx.yaml`
- To add a new task: add a head + data module + config file (open-closed principle)
