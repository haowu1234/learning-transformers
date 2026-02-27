---
name: learning-transformers-dev
description: "This skill provides development guidance for the learning-transformers project, a hands-on Transformer implementation with BERT fine-tuning on downstream tasks. This skill should be used when adding new tasks (heads, data modules, metrics), modifying existing components, writing configs, creating design docs, or debugging training issues. It encodes the project Registry pattern, abstract base class contracts, composition architecture, and config-driven workflow."
---

# Learning Transformers Development Skill

## Purpose

Provide standardized workflows and domain knowledge for developing within the learning-transformers project — a from-scratch Transformer implementation with pluggable BERT fine-tuning on downstream NLP tasks.

## When to Use

- Adding a new downstream task (head + data module + metric + config)
- Modifying or debugging existing model components, training, or evaluation
- Writing YAML experiment configs
- Creating design documents for new tasks
- Understanding the project's architecture and conventions

## Project Architecture

```
src/
├── models/          # Task-agnostic backbone (BertModel, BertForTask, BertBiEncoder)
├── heads/           # Pluggable task heads (Registry pattern)
├── data/            # Pluggable data modules (Registry pattern)
├── evaluation/      # Pluggable metrics (Registry pattern)
├── training/        # Unified Trainer + Callbacks
└── utils/           # Registry mechanism
configs/             # YAML config per experiment
scripts/             # Entry points (train.py, evaluate.py)
```

## Core Design Patterns

### Registry Pattern

Three Registry instances govern extensibility:

- `HEAD_REGISTRY` — defined in `src/heads/classification.py`, re-exported from `src/heads/__init__.py`
- `DATASET_REGISTRY` — defined in `src/data/sst2.py`, re-exported from `src/data/__init__.py`
- `METRIC_REGISTRY` — defined in `src/evaluation/metrics.py`, re-exported from `src/evaluation/__init__.py`

Register new components via `@REGISTRY.register("name")` decorator. **Critical**: after creating a new module file, add `from . import module_name  # noqa: F401` to the corresponding `__init__.py` to trigger decorator execution.

### Model Composition

- **Single-Encoder tasks** (classification, token_classification, qa, regression): `BertForTask(backbone, head)`
- **Dual-Encoder tasks** (similarity): `BertBiEncoder(backbone, head)` — shared backbone, two inputs

### Config-Driven Workflow

Every experiment is defined by a YAML file in `configs/`. Run via:
```bash
python scripts/train.py --config configs/xxx.yaml
```

---

## Workflow: Adding a New Task

This is the primary development workflow. Follow these steps in order.

### Step 1: Write a Design Document

Before coding, create `docs/<task>_design.md` covering:
1. Task description and motivation
2. Dataset source and format
3. Label schema and preprocessing strategy
4. Head architecture choice (which backbone output to use)
5. Loss function selection
6. Evaluation metric(s)
7. Any special considerations (subword alignment, dual-encoder, etc.)
8. Change checklist (files to create/modify)

Refer to `references/architecture.md` for base class contracts and existing patterns.

### Step 2: Implement the Task Head

Create `src/heads/<task_name>.py`:

```python
"""<Task description>."""

import torch.nn as nn
from torch import Tensor

from .base import BaseHead
from .classification import HEAD_REGISTRY  # Import the singleton Registry


@HEAD_REGISTRY.register("<task_name>")
class <TaskName>Head(BaseHead):

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        # Define layers

    def forward(self, backbone_outputs: dict[str, Tensor],
                attention_mask: Tensor | None = None, **kwargs) -> Tensor:
        # Choose backbone_outputs["pooled_output"] for sentence-level
        # or backbone_outputs["sequence_output"] for token-level
        ...

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        ...
```

Key decisions:
- **Sentence-level** tasks → use `pooled_output` (B, H)
- **Token-level** tasks → use `sequence_output` (B, L, H)
- **Regression** tasks → output shape (B,), use `MSELoss`
- **Classification** tasks → output shape (B, num_labels), use `CrossEntropyLoss`
- **Token classification** → use `CrossEntropyLoss(ignore_index=-100)`

Register in `src/heads/__init__.py`:
```python
from . import <task_name>  # noqa: F401
```

### Step 3: Implement the Data Module

Create `src/data/<dataset_name>.py`:

```python
"""<Dataset> data module."""

from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .collator import DataCollator
from .sst2 import DATASET_REGISTRY  # Import the singleton Registry


class <Name>Dataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@DATASET_REGISTRY.register("<dataset_name>")
class <Name>DataModule(BaseDataModule):

    LABELS = [...]  # or dynamically collect

    def __init__(self, pretrained_model_name: str = "bert-base-uncased",
                 max_seq_length: int = 128, batch_size: int = 32, num_workers: int = 0):
        ...
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def setup(self) -> None:
        from datasets import load_dataset
        # Load and tokenize

    def train_dataloader(self) -> DataLoader:
        return DataLoader(..., collate_fn=DataCollator(pad_token_id=self.tokenizer.pad_token_id))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(..., shuffle=False, collate_fn=DataCollator(...))

    def get_labels(self) -> list[str]:
        return self.LABELS
```

**Tokenization patterns by task type**:
- **Sentence classification**: `tokenizer(text, max_length=..., truncation=True, padding=False)`
- **Token classification (word-level)**: `tokenizer(words, is_split_into_words=True)` + `word_ids()` for subword-label alignment; first subword gets label, rest get `-100`
- **Token classification (pre-tokenized)**: `convert_tokens_to_ids()` directly
- **Sentence pair (cross-encoder)**: `tokenizer(sent1, sent2, ...)` — single input with `[SEP]`
- **Sentence pair (bi-encoder)**: Tokenize A and B separately; use `PairDataCollator`

**Collator selection**:
- Standard tasks → `DataCollator(pad_token_id=..., label_dtype="long"|"float")`
- Bi-encoder → `PairDataCollator` (defined in `src/data/stsb.py`)
- Sequence labels (list[int]) → auto-padded with `-100` by `DataCollator`

Register in `src/data/__init__.py`:
```python
from . import <dataset_name>  # noqa: F401
```

### Step 4: Implement Metric (if needed)

Create `src/evaluation/<metric_name>.py` if no existing metric fits:

```python
"""<Metric description>."""

from torch import Tensor

from .base import BaseMetric
from .metrics import METRIC_REGISTRY  # Import the singleton Registry


@METRIC_REGISTRY.register("<metric_name>")
class <MetricName>(BaseMetric):

    def __init__(self, **kwargs):
        self.reset()

    def update(self, preds: Tensor, labels: Tensor) -> None:
        # Accumulate batch results

    def compute(self) -> dict[str, float]:
        # Return {"metric_name": value}

    def reset(self) -> None:
        # Clear state
```

If the metric constructor requires special arguments (e.g., `num_labels`, `label_list`), add an `elif` branch in `scripts/train.py` and `scripts/evaluate.py` metric instantiation section.

Register in `src/evaluation/__init__.py`:
```python
from . import <metric_name>  # noqa: F401
```

### Step 5: Create YAML Config

Create `configs/<task_name>.yaml`:

```yaml
# <Task Description>

task:
  head: "<head_registry_name>"
  dataset: "<dataset_registry_name>"
  metrics: ["<metric1>", "<metric2>"]

model:
  pretrained: "bert-base-uncased"  # or bert-base-cased, bert-base-multilingual-cased
  vocab_size: 30522               # must match pretrained model
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 128
  device: "auto"
  log_interval: 50

checkpoint:
  save_dir: "checkpoints"
  mode: "max"              # "max" for accuracy/f1, "min" for loss

early_stopping:
  patience: 3
  mode: "max"              # must match checkpoint mode
```

**Pretrained model ↔ vocab_size mapping**:
- `bert-base-uncased` → 30522
- `bert-base-cased` → 28996
- `bert-base-multilingual-cased` → 119547

### Step 6: Handle Special Cases in train.py

If the new task requires special handling in `scripts/train.py`:

- **New model architecture** (not BertForTask or BertBiEncoder): Add an `elif` in the model construction section
- **Head with non-standard constructor**: Add an `elif` in the head instantiation section
- **Metric with special args**: Add an `elif` in the metric instantiation loop

### Step 7: Verify

```bash
# Train
python scripts/train.py --config configs/<task_name>.yaml

# Evaluate (full validation set)
python scripts/evaluate.py --config configs/<task_name>.yaml --checkpoint checkpoints/best_model.pt

# Evaluate (single text inference)
python scripts/evaluate.py --config configs/<task_name>.yaml --checkpoint checkpoints/best_model.pt --text "example input"
```

---

## Existing Task Reference

| Task | Head | Dataset | Metrics | Pretrained Model |
|------|------|---------|---------|-----------------|
| Sentiment Classification | `classification` | `sst2` | `accuracy`, `f1` | bert-base-uncased |
| Named Entity Recognition | `token_classification` | `conll2003` | `seqeval` | bert-base-cased |
| PII Detection | `token_classification` | `pii` | `seqeval` | bert-base-multilingual-cased |
| Semantic Similarity (Bi-Enc) | `similarity` | `stsb` | `spearman` | bert-base-uncased |
| Semantic Similarity (Cross-Enc) | `regression` | `stsb_cross_encoder` | `spearman` | bert-base-uncased |

---

## Important Conventions

1. **Lazy imports**: Use lazy imports for heavy dependencies (`transformers`, `datasets`) inside methods, not at module top level
2. **Tokenizer as property**: Define `_tokenizer = None` in `__init__`, load on first access via `@property`
3. **Labels contract**: `get_labels()` returns `list[str]`; `num_labels` is derived as `len(get_labels())`; for regression tasks, return `[]` (num_labels = 0)
4. **DataCollator**: Always use `padding=False` in tokenizer, let `DataCollator` handle dynamic padding per batch
5. **Subword alignment**: For word-level token classification, assign label to first subword only, use `-100` for continuation subwords
6. **Trainer monitors**: Priority order for best-metric selection: accuracy → f1 → spearman → -loss

## Debugging Tips

- **Weight loading issues**: Run `python scripts/diagnose_weights.py --config configs/xxx.yaml` to check parameter mapping
- **Metric not found**: Ensure `__init__.py` imports the module containing the `@register` decorator
- **Shape mismatch in head**: Verify `num_labels` matches between data module `get_labels()` and head constructor
- **NaN loss**: Check label alignment — ensure non-predicted positions use `-100`, not `0`
