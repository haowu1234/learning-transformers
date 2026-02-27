# Architecture Reference

This document provides the complete contracts and signatures for all base classes, registries, and key components in the learning-transformers project.

---

## Registry Pattern

**File**: `src/utils/registry.py`

```python
class Registry:
    def __init__(self, name: str)
    def register(self, name: str) -> Callable    # Decorator: @REGISTRY.register("name")
    def get(self, name: str) -> Type              # Retrieve class by name
    def list(self) -> list[str]                   # List all registered names
```

**Three singleton instances**:
- `HEAD_REGISTRY` — defined in `src/heads/classification.py`
- `DATASET_REGISTRY` — defined in `src/data/sst2.py`
- `METRIC_REGISTRY` — defined in `src/evaluation/metrics.py`

---

## Base Classes

### BaseHead (`src/heads/base.py`)

```python
class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, backbone_outputs: dict[str, Tensor],
                attention_mask: Tensor | None = None, **kwargs) -> Tensor:
        """
        backbone_outputs keys:
          - "sequence_output": (B, L, H) — all token representations
          - "pooled_output":   (B, H)    — [CLS] representation after pooling

        Returns: logits tensor (task-dependent shape)
        """

    @abstractmethod
    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        """Returns: scalar loss tensor"""
```

### BaseDataModule (`src/data/base.py`)

```python
class BaseDataModule(ABC):
    @abstractmethod def setup(self) -> None
    @abstractmethod def train_dataloader(self) -> DataLoader
    @abstractmethod def val_dataloader(self) -> DataLoader
    @abstractmethod def get_labels(self) -> list[str]

    @property
    def num_labels(self) -> int:
        return len(self.get_labels())
```

**Constructor convention** — all data modules accept:
```python
def __init__(self, pretrained_model_name: str, max_seq_length: int, batch_size: int, num_workers: int = 0)
```

### BaseMetric (`src/evaluation/base.py`)

```python
class BaseMetric(ABC):
    @abstractmethod def update(self, preds: Tensor, labels: Tensor) -> None
    @abstractmethod def compute(self) -> dict[str, float]
    @abstractmethod def reset(self) -> None
```

---

## Model Components

### BertModel (`src/models/bert.py`)

```python
class BertModel(nn.Module):
    def __init__(self, config: BertConfig)
    def forward(self, input_ids, attention_mask, token_type_ids) -> dict:
        # Returns: {"sequence_output": (B, L, H), "pooled_output": (B, H)}
```

### BertForTask — Single Encoder (`src/models/bert.py`)

```python
class BertForTask(nn.Module):
    def __init__(self, backbone: BertModel, head: nn.Module)
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, **kwargs):
        # Returns: {"loss": Tensor|None, "logits": Tensor}
```

### BertBiEncoder — Dual Encoder (`src/models/bi_encoder.py`)

```python
class BertBiEncoder(nn.Module):
    def __init__(self, backbone: BertModel, head: nn.Module)
    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a,
                input_ids_b, attention_mask_b, token_type_ids_b, labels=None):
        # Encodes A and B through shared backbone
        # Returns: {"loss": Tensor|None, "logits": Tensor}
```

### BertConfig

```python
@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "BertConfig"
```

---

## DataCollator (`src/data/collator.py`)

```python
@dataclass
class DataCollator:
    pad_token_id: int = 0
    pad_to_max_length: bool = False
    max_length: int = 128
    label_dtype: str = "long"   # "long" for classification, "float" for regression

    def __call__(self, batch: list[dict]) -> dict[str, Tensor]:
        # Handles: input_ids, attention_mask, token_type_ids (optional)
        # Labels: scalar → tensor, list[int] → padded with -100
```

## PairDataCollator (`src/data/stsb.py`)

For bi-encoder tasks. Pads A-side and B-side independently:
```python
class PairDataCollator:
    def __call__(self, batch) -> dict:
        # Returns: input_ids_a, attention_mask_a, token_type_ids_a,
        #          input_ids_b, attention_mask_b, token_type_ids_b, labels
```

---

## Training Components

### TrainingConfig (`src/training/trainer.py`)

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "auto"
    log_interval: int = 50

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig"
```

### Trainer

```python
class Trainer:
    def __init__(self, model, config: TrainingConfig,
                 early_stopping=None, checkpoint=None)
    def fit(self, train_dataloader, val_dataloader, metrics) -> dict
    def evaluate(self, dataloader, metrics) -> dict
    def predict(self, dataloader) -> list[Tensor]
```

### Callbacks

```python
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max")
    def step(self, value: float) -> bool  # Returns True if should stop

class ModelCheckpoint:
    def __init__(self, save_dir: str = "checkpoints", mode: str = "max")
    def step(self, value: float, model: nn.Module) -> None
```

---

## Existing Implementations Quick Reference

### Heads (5)
| Registry Name | Class | Input | Output | Loss |
|--------------|-------|-------|--------|------|
| `classification` | ClassificationHead | pooled_output (B,H) | (B, num_labels) | CrossEntropyLoss |
| `token_classification` | TokenClassificationHead | sequence_output (B,L,H) | (B, L, num_labels) | CrossEntropyLoss(ignore=-100) |
| `qa` | QAHead | sequence_output (B,L,H) | (B, L, 2) | CrossEntropyLoss(ignore=-100) |
| `similarity` | SimilarityHead | two backbone outputs | (B,) cosine_sim | MSELoss |
| `regression` | RegressionHead | pooled_output (B,H) | (B,) scalar | MSELoss |

### Data Modules (5)
| Registry Name | Pretrained | Task Type |
|--------------|-----------|-----------|
| `sst2` | bert-base-uncased | Sentence classification |
| `conll2003` | bert-base-cased | Token classification (NER) |
| `pii` | bert-base-multilingual-cased | Token classification (PII) |
| `stsb` | bert-base-uncased | Bi-encoder similarity |
| `stsb_cross_encoder` | bert-base-uncased | Cross-encoder regression |

### Metrics (4)
| Registry Name | Class | Output Keys |
|--------------|-------|-------------|
| `accuracy` | Accuracy | accuracy |
| `f1` | F1Score(num_labels) | f1, precision, recall |
| `seqeval` | SeqevalF1(label_list) | f1, precision, recall |
| `spearman` | SpearmanCorrelation | spearman, pearson |

---

## __init__.py Pattern

Each package's `__init__.py` must:
1. Import the singleton Registry from the module that defines it
2. Import every submodule to trigger `@register` decorators

```python
# src/heads/__init__.py
from .classification import HEAD_REGISTRY
from . import classification  # noqa: F401
from . import token_classification  # noqa: F401
from . import new_task  # noqa: F401  ← ADD THIS for new tasks
```
