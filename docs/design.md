# 文本分类下游任务 — 设计方案

## 1. 任务定义

**情感分析（Sentiment Analysis）** — 对电影评论进行二分类（正面/负面），基于 SST-2 数据集，使用 BERT-base 微调。

---

## 2. 架构总览

```
src/
├── models/                     # 纯模型组件层（与任务无关）
│   ├── __init__.py
│   ├── attention.py            # Multi-Head Attention
│   ├── embedding.py            # BertEmbedding (token + position + segment)
│   ├── encoder.py              # TransformerEncoderLayer + TransformerEncoder
│   └── bert.py                 # BertModel + BertForTask（backbone + 任务头组合）
│
├── heads/                      # 任务头层（可插拔，注册表管理）
│   ├── __init__.py             # HEAD_REGISTRY
│   ├── base.py                 # BaseHead 抽象基类
│   ├── classification.py       # ClassificationHead
│   ├── token_classification.py # TokenClassificationHead (NER)
│   ├── qa.py                   # QAHead (span extraction)
│   └── sentence_pair.py        # SentencePairHead
│
├── data/                       # 数据处理层（可插拔，注册表管理）
│   ├── __init__.py             # DATASET_REGISTRY
│   ├── base.py                 # BaseDataModule 抽象基类
│   ├── sst2.py                 # SST2DataModule
│   └── collator.py             # 通用 DataCollator
│
├── training/                   # 训练引擎（任务无关）
│   ├── __init__.py
│   ├── trainer.py              # Trainer（统一训练循环）
│   └── callbacks.py            # EarlyStopping, ModelCheckpoint
│
├── evaluation/                 # 评估层（可插拔，注册表管理）
│   ├── __init__.py             # METRIC_REGISTRY
│   ├── base.py                 # BaseMetric 抽象基类
│   └── metrics.py              # Accuracy, F1, SpanF1
│
└── utils/                      # 通用工具
    ├── __init__.py
    └── registry.py             # Registry 注册表机制
```

---

## 3. 关键设计模式

### 3.1 注册表模式（Registry Pattern）

所有可扩展组件通过注册表管理，新增任务 **零修改** 已有代码：

```python
# src/utils/registry.py
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"[{self.name}] '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name]
```

使用方式：

```python
# heads/__init__.py
HEAD_REGISTRY = Registry("heads")

# heads/classification.py
@HEAD_REGISTRY.register("classification")
class ClassificationHead(BaseHead): ...
```

### 3.2 抽象基类定义契约

```python
# heads/base.py
class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """接收 backbone 输出，返回 logits"""

    @abstractmethod
    def compute_loss(self, logits, labels, **kwargs):
        """计算任务特定损失"""
```

```python
# data/base.py
class BaseDataModule(ABC):
    @abstractmethod
    def setup(self):
        """下载/加载数据集"""

    @abstractmethod
    def train_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def get_labels(self) -> list[str]:
        """返回标签列表，用于自动推断 num_labels"""
```

```python
# evaluation/base.py
class BaseMetric(ABC):
    @abstractmethod
    def update(self, preds, labels): ...

    @abstractmethod
    def compute(self) -> dict[str, float]: ...

    @abstractmethod
    def reset(self): ...
```

### 3.3 TaskModel 组合 Backbone + Head

```python
# src/models/bert.py
class BertForTask(nn.Module):
    """通用组合：Backbone + 任意 Head"""
    def __init__(self, backbone: BertModel, head: BaseHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.backbone(input_ids, attention_mask)
        logits = self.head(hidden_states, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            loss = self.head.compute_loss(logits, labels, **kwargs)
        return {"loss": loss, "logits": logits}
```

### 3.4 统一 Trainer

```python
# src/training/trainer.py
class Trainer:
    def __init__(self, model, config, callbacks=None):
        self.model = model
        self.config = config
        self.callbacks = callbacks or []

    def fit(self, train_dataloader, val_dataloader, metrics): ...
    def evaluate(self, dataloader, metrics) -> dict: ...
    def predict(self, dataloader) -> list: ...
```

### 3.5 YAML 配置驱动

```yaml
# configs/sst2_classification.yaml
task:
  head: "classification"       # → HEAD_REGISTRY.get("classification")
  dataset: "sst2"              # → DATASET_REGISTRY.get("sst2")
  metrics: ["accuracy", "f1"]  # → METRIC_REGISTRY.get(...)

model:
  pretrained: "bert-base-uncased"
  hidden_size: 768
  num_labels: 2

training:
  batch_size: 32
  learning_rate: 2.0e-5
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 128
  max_grad_norm: 1.0
```

---

## 4. 模型结构

```
Input: "This movie is absolutely wonderful!"

         ┌─────────────────────────────┐
         │       Tokenizer             │
         │  [CLS] this movie is ... [SEP] │
         └──────────────┬──────────────┘
                        ▼
         ┌─────────────────────────────┐
         │     Token Embeddings        │
         │   + Position Embeddings     │
         │   + Segment Embeddings      │
         │   + LayerNorm + Dropout     │
         └──────────────┬──────────────┘
                        ▼
         ┌─────────────────────────────┐
         │    Transformer Encoder      │
         │    × 12 layers              │
         │  ┌───────────────────────┐  │
         │  │ Multi-Head Attention  │  │
         │  │ + Residual + LayerNorm│  │
         │  ├───────────────────────┤  │
         │  │ Feed Forward Network  │  │
         │  │ + Residual + LayerNorm│  │
         │  └───────────────────────┘  │
         └──────────────┬──────────────┘
                        ▼
         ┌─────────────────────────────┐
         │  [CLS] hidden state (768d)  │
         └──────────────┬──────────────┘
                        ▼
         ┌─────────────────────────────┐
         │  ClassificationHead         │
         │  Dropout(0.1) → Linear(768→2) │
         └──────────────┬──────────────┘
                        ▼
                   Logits [2]
                   → CrossEntropyLoss (训练)
                   → Softmax → argmax (推理)
```

### 模型选择

| 项目 | 选择 | 理由 |
|------|------|------|
| 预训练模型 | `bert-base-uncased` | 12层/768维/12头，110M参数，适合学习 |
| 权重来源 | HuggingFace Hub | 加载预训练权重，不做预训练 |
| 微调方式 | Full fine-tuning | 全参数训练，论文原始做法 |

---

## 5. 数据集

| 项目 | 说明 |
|------|------|
| 数据集 | **SST-2**（Stanford Sentiment Treebank） |
| 来源 | `datasets` 库：`load_dataset("glue", "sst2")` |
| 规模 | 训练集 67k，验证集 872，测试集 1.8k |
| 标签 | 0 = 负面，1 = 正面 |
| 输入长度 | 大部分 < 64 tokens，`max_length=128` |

**数据预处理流程：**

```
原始文本
  → Tokenizer 分词 (bert-base-uncased)
  → 生成 input_ids + attention_mask + token_type_ids
  → 截断/填充到 max_length=128
  → DataCollator 组装 batch
  → DataLoader (batch_size=32, shuffle=True)
```

---

## 6. 训练方案

| 超参数 | 值 | 说明 |
|--------|-----|------|
| Optimizer | AdamW | BERT 标准优化器 |
| Learning Rate | 2e-5 | BERT 微调经典值 |
| Batch Size | 32 | |
| Epochs | 3 | BERT 微调通常 2-4 轮 |
| Warmup | 前 10% steps | 线性 warmup |
| LR Scheduler | Linear decay | warmup 后线性衰减到 0 |
| Weight Decay | 0.01 | 应用于非 bias/LayerNorm 参数 |
| Max Seq Length | 128 | |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |

**训练伪代码：**

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # callbacks: logging, progress bar

    # epoch 结束评估
    results = trainer.evaluate(val_dataloader, metrics)
    # callbacks: early stopping, checkpoint
```

---

## 7. 评测方案

| 指标 | 说明 |
|------|------|
| **Accuracy** | 主指标，SST-2 官方评测指标 |
| **F1 Score** | 正负类的加权 F1 |

**预期效果：** BERT-base 在 SST-2 上准确率约 **92-93%**

**评测伪代码：**

```python
model.eval()
metric.reset()
with torch.no_grad():
    for batch in val_dataloader:
        outputs = model(input_ids, attention_mask)
        preds = outputs["logits"].argmax(dim=-1)
        metric.update(preds, labels)
results = metric.compute()  # {"accuracy": 0.93, "f1": 0.93}
```

---

## 8. 推理方案

```python
def predict(text: str) -> dict:
    tokens = tokenizer(text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])["logits"]
    probs = torch.softmax(logits, dim=-1)
    label_id = probs.argmax(dim=-1).item()
    return {
        "label": "positive" if label_id == 1 else "negative",
        "confidence": probs[0][label_id].item()
    }
```

支持单条推理和批量推理。

---

## 9. 实现顺序

| Step | 文件 | 内容 |
|------|------|------|
| 1 | `src/utils/registry.py` | Registry 注册表 |
| 2 | `src/models/attention.py` | Scaled Dot-Product + Multi-Head Attention |
| 3 | `src/models/embedding.py` | BertEmbedding |
| 4 | `src/models/encoder.py` | TransformerEncoderLayer + TransformerEncoder |
| 5 | `src/models/bert.py` | BertModel + BertForTask + 权重加载 |
| 6 | `src/heads/base.py` | BaseHead 抽象基类 |
| 7 | `src/heads/classification.py` | ClassificationHead |
| 8 | `src/evaluation/base.py` | BaseMetric 抽象基类 |
| 9 | `src/evaluation/metrics.py` | Accuracy, F1 |
| 10 | `src/data/base.py` | BaseDataModule 抽象基类 |
| 11 | `src/data/collator.py` | DataCollator |
| 12 | `src/data/sst2.py` | SST2DataModule |
| 13 | `src/training/trainer.py` | 统一 Trainer |
| 14 | `src/training/callbacks.py` | EarlyStopping, ModelCheckpoint |
| 15 | `configs/sst2_classification.yaml` | SST-2 任务配置 |
| 16 | `scripts/train.py` | 训练入口 |
| 17 | `scripts/evaluate.py` | 评估/推理入口 |

---

## 10. 扩展性

新增一个下游任务只需 **3 步**，不修改任何已有代码：

1. 在 `src/heads/` 新增任务头，用 `@HEAD_REGISTRY.register(...)` 注册
2. 在 `src/data/` 新增数据模块，用 `@DATASET_REGISTRY.register(...)` 注册
3. 在 `configs/` 新增 YAML 配置文件

运行：`python scripts/train.py --config configs/<new_task>.yaml`
