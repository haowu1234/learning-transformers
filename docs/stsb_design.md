# STS-B 语义相似度任务设计方案（Bi-Encoder）

## 1. 任务概述

| 维度 | 说明 |
|------|------|
| **任务类型** | 句子对语义相似度回归 |
| **数据集** | STS-B（Semantic Textual Similarity Benchmark，GLUE 子任务） |
| **架构** | Bi-Encoder：两句分别编码 → pooling → 余弦相似度 |
| **标签** | 0~5 连续相似度分数（归一化到 0~1） |
| **评估指标** | Spearman 相关系数 / Pearson 相关系数 |
| **预期指标** | Spearman ~85-87% |

## 2. Bi-Encoder 架构详解

### 2.1 与已有任务的根本区别

已有 3 个任务都是 **单输入、单前向**：

```
Text → BERT → head → logits
```

Bi-Encoder 是 **双输入、双前向、共享权重**：

```
Sentence A → BERT(shared) → pooling → emb_a ─┐
                                                ├→ cosine_similarity → score
Sentence B → BERT(shared) → pooling → emb_b ─┘
```

### 2.2 为什么用 Bi-Encoder 而不是 Cross-Encoder？

| 方案 | 输入方式 | 编码次数 | 检索效率 | 精度 |
|------|---------|---------|---------|------|
| **Cross-Encoder** | `[CLS] A [SEP] B [SEP]` 拼接 | 1 次 | O(n²)，无法预计算 | 更高 |
| **Bi-Encoder** | A 和 B 分别编码 | 2 次 | O(n)，可预计算向量 | 稍低 |

Cross-Encoder 其实就是之前 SST-2 的分类方式（把句子对拼成一个输入）——我们已经会了。

Bi-Encoder 的学习价值在于：
- **独立编码** → 学习 sentence embedding
- **可用于检索** → 向量预计算 + ANN 近似搜索
- **对比学习思想** → 为后续 SimCSE、RAG 打基础

### 2.3 Pooling 策略

BERT 输出 `(batch, seq_len, hidden_size)`，需要压缩成一个句向量 `(batch, hidden_size)`。

常见策略：

| 策略 | 做法 | 效果 |
|------|------|------|
| CLS pooling | 取 `[CLS]` token 的隐状态 | 简单但效果一般 |
| **Mean pooling** | 对所有有效 token 取平均（考虑 attention_mask） | **最常用，效果最好** |
| Max pooling | 对每个维度取 max | 一般 |

本方案使用 **Mean pooling**：

```python
def mean_pooling(sequence_output, attention_mask):
    # sequence_output: (B, L, H), attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
    summed = (sequence_output * mask).sum(dim=1)       # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
    return summed / lengths                             # (B, H)
```

### 2.4 损失函数

STS-B 标签是 0~5 的连续值，归一化到 [0, 1] 后，使用 **Cosine Embedding Loss**：

```python
cos_sim = F.cosine_similarity(emb_a, emb_b)  # (B,) 范围 [-1, 1]
# 方式一：MSE loss（简单直接）
loss = MSE(cos_sim, labels)  # labels 归一化到 [0, 1] 或 [-1, 1]
```

也可以用 `CosineEmbeddingLoss`，但 MSE 更灵活，适合连续相似度分数。

本方案将标签 0~5 **线性映射到 [-1, 1]**（`label * 2/5 - 1`），与余弦相似度范围一致，使用 MSE loss。

## 3. 数据集分析

### 3.1 STS-B

```python
from datasets import load_dataset
raw = load_dataset("glue", "stsb")
```

| Split | 样本数 |
|-------|-------|
| Train | 5,749 |
| Validation | 1,500 |
| Test | 1,379（无标签，GLUE leaderboard 用） |

**字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `sentence1` | str | 第一个句子 |
| `sentence2` | str | 第二个句子 |
| `label` | float | 相似度分数，0.0 ~ 5.0 |

**样本示例：**

```
sentence1: "A plane is taking off."
sentence2: "An air plane is taking off."
label: 5.00

sentence1: "A man is playing a flute."
sentence2: "A man is surfing."
label: 0.60
```

## 4. 架构改动分析

### 4.1 核心挑战

当前 `BertForTask` 的 forward 只支持单次前向：

```python
def forward(self, input_ids, attention_mask, token_type_ids, labels):
    backbone_outputs = self.backbone(input_ids, attention_mask, token_type_ids)
    logits = self.head(backbone_outputs, attention_mask=attention_mask)
    ...
```

Bi-Encoder 需要对两个句子分别过 backbone，然后在 head 中计算相似度。

### 4.2 方案选择

**方案 A：修改 BertForTask 支持双输入** — 侵入性大，破坏已有任务  
**方案 B：新建 BertBiEncoder 模型** — 独立类，零改动已有代码 ✅

```python
class BertBiEncoder(nn.Module):
    """Bi-Encoder: shared backbone encodes two sentences independently."""

    def __init__(self, backbone: BertModel, head: nn.Module):
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a,
                      input_ids_b, attention_mask_b, token_type_ids_b,
                      labels=None):
        out_a = self.backbone(input_ids_a, attention_mask_a, token_type_ids_a)
        out_b = self.backbone(input_ids_b, attention_mask_b, token_type_ids_b)
        logits = self.head(out_a, out_b, attention_mask_a, attention_mask_b)
        ...
```

### 4.3 Trainer 兼容性

当前 Trainer 的 train/evaluate 循环中，batch 是一个 dict，直接传给 `self.model(...)`：

```python
outputs = self.model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch.get("token_type_ids"),
    labels=batch["labels"],
)
```

Bi-Encoder 的 batch 会包含 `input_ids_a`, `input_ids_b` 等字段。如果让 Trainer 硬编码这些字段名就太死板了。

**解决方案：让 Trainer 直接把整个 batch dict 解包传给 model**：

```python
# 修改前（硬编码字段名）：
outputs = self.model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    ...
)

# 修改后（通用解包）：
outputs = self.model(**batch)
```

这对已有任务完全兼容（batch 里本来就是 `input_ids`, `attention_mask`, `labels` 等），同时自然支持 Bi-Encoder 的 `input_ids_a`, `input_ids_b` 等新字段。

### 4.4 Evaluate 中的 argmax 问题

当前 Trainer.evaluate() 用 `logits.argmax(dim=-1)` 获取预测值，这对分类任务正确，但对回归任务（相似度分数）不适用——回归任务的 logits 就是预测值本身。

**解决方案：** 当 logits 是 1D（无分类维度）时，直接用 logits 作为 preds：

```python
# 修改前：
preds = outputs["logits"].argmax(dim=-1)

# 修改后：
logits = outputs["logits"]
if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
    preds = logits.squeeze(-1)   # 回归任务
else:
    preds = logits.argmax(dim=-1)  # 分类 / 序列标注
```

## 5. 变更清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/models/bi_encoder.py` | `BertBiEncoder` 模型（shared backbone + 双前向） |
| `src/heads/similarity.py` | `SimilarityHead`（mean pooling + 余弦相似度 + MSE loss） |
| `src/data/stsb.py` | STS-B 数据模块（句子对 tokenize + 分数归一化） |
| `src/evaluation/correlation.py` | Spearman / Pearson 相关系数指标 |
| `configs/stsb_similarity.yaml` | 实验配置 |

### 修改文件

| 文件 | 改动 | 影响 |
|------|------|------|
| `src/heads/__init__.py` | `+ from . import similarity` | 仅加 import |
| `src/data/__init__.py` | `+ from . import stsb` | 仅加 import |
| `src/evaluation/__init__.py` | `+ from . import correlation` | 仅加 import |
| `src/training/trainer.py` | ① batch 解包方式改为 `**batch` ② evaluate 中 argmax 改为回归兼容 | 对已有任务无影响 |
| `scripts/train.py` | 增加 Bi-Encoder 模型构建分支 + correlation metric 实例化 | 新增分支 |
| `scripts/evaluate.py` | 增加 Bi-Encoder 构建 + 句子对推理 | 新增分支 |

### 无需修改

| 文件 | 原因 |
|------|------|
| `src/models/bert.py` | BertModel backbone 直接复用 |
| `src/data/collator.py` | 需要新的 collator（句子对有 6 个字段），新建在 stsb.py 中 |
| `src/training/callbacks.py` | EarlyStopping / ModelCheckpoint 通用 |

## 6. 核心设计细节

### 6.1 SimilarityHead

```python
@HEAD_REGISTRY.register("similarity")
class SimilarityHead(BaseHead):
    """Bi-Encoder head: mean pooling → cosine similarity → MSE loss."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def mean_pooling(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (sequence_output * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    def forward(self, backbone_outputs_a, backbone_outputs_b,
                attention_mask_a, attention_mask_b, **kwargs):
        emb_a = self.mean_pooling(backbone_outputs_a["sequence_output"], attention_mask_a)
        emb_b = self.mean_pooling(backbone_outputs_b["sequence_output"], attention_mask_b)
        cos_sim = F.cosine_similarity(emb_a, emb_b)  # (B,)
        return cos_sim

    def compute_loss(self, logits, labels, **kwargs):
        return self.loss_fn(logits, labels)  # MSE(cos_sim, normalized_score)
```

注意：`SimilarityHead.forward()` 的签名与 `BaseHead` 不同（接收两组 backbone_outputs），这是 Bi-Encoder 架构的本质差异。`BertBiEncoder` 会正确调用它。

### 6.2 BertBiEncoder

```python
class BertBiEncoder(nn.Module):
    """Shared-backbone Bi-Encoder for sentence similarity."""

    def __init__(self, backbone: BertModel, head: nn.Module):
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a,
                      input_ids_b, attention_mask_b, token_type_ids_b,
                      labels=None, **kwargs):
        out_a = self.backbone(input_ids_a, attention_mask_a, token_type_ids_a)
        out_b = self.backbone(input_ids_b, attention_mask_b, token_type_ids_b)

        cos_sim = self.head(out_a, out_b, attention_mask_a, attention_mask_b)

        loss = None
        if labels is not None:
            loss = self.head.compute_loss(cos_sim, labels)

        return {"loss": loss, "logits": cos_sim}
```

返回格式与 `BertForTask` 一致（`{"loss": ..., "logits": ...}`），保证 Trainer 兼容。

### 6.3 STS-B DataModule

```python
@DATASET_REGISTRY.register("stsb")
class STSBDataModule(BaseDataModule):

    LABELS = []  # 回归任务无离散标签

    def _tokenize(self, split):
        examples = []
        for item in split:
            enc_a = self.tokenizer(item["sentence1"], ...)
            enc_b = self.tokenizer(item["sentence2"], ...)
            # 标签归一化：0~5 → -1~1（与余弦相似度范围对齐）
            score = item["label"] * 2.0 / 5.0 - 1.0
            examples.append({
                "input_ids_a": enc_a["input_ids"],
                "attention_mask_a": enc_a["attention_mask"],
                "token_type_ids_a": enc_a.get("token_type_ids", ...),
                "input_ids_b": enc_b["input_ids"],
                "attention_mask_b": enc_b["attention_mask"],
                "token_type_ids_b": enc_b.get("token_type_ids", ...),
                "labels": score,
            })
        return examples
```

**Collator**：句子对需要分别 padding A 和 B，新建 `PairDataCollator`：

```python
@dataclass
class PairDataCollator:
    """Pads sentence-pair inputs (A and B) independently."""
    pad_token_id: int = 0

    def __call__(self, batch):
        # Pad A fields to max_len_a, B fields to max_len_b
        # labels: float scalar → torch.float tensor
        ...
```

### 6.4 Correlation Metric

```python
@METRIC_REGISTRY.register("spearman")
class SpearmanCorrelation(BaseMetric):
    """Spearman rank correlation coefficient."""

    def update(self, preds, labels):
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())

    def compute(self):
        preds = torch.cat(self.all_preds)
        labels = torch.cat(self.all_labels)
        # Spearman = Pearson on ranks
        pred_ranks = self._rank(preds)
        label_ranks = self._rank(labels)
        return {"spearman": self._pearson(pred_ranks, label_ranks)}
```

### 6.5 train.py 修改

```python
# 模型构建分支
if config["task"]["head"] == "similarity":
    from src.models.bi_encoder import BertBiEncoder
    head = head_cls(hidden_size=bert_config.hidden_size)
    model = BertBiEncoder(backbone, head)
else:
    head = head_cls(hidden_size=..., num_labels=..., dropout_prob=...)
    model = BertForTask(backbone, head)

# Metric 分支
elif metric_name == "spearman":
    metrics.append(metric_cls())
```

### 6.6 配置文件

```yaml
task:
  head: "similarity"
  dataset: "stsb"
  metrics: ["spearman"]

model:
  pretrained: "bert-base-uncased"
  vocab_size: 30522
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1

training:
  batch_size: 32
  learning_rate: 2.0e-5
  num_epochs: 5
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 128
  device: "auto"
  log_interval: 50

checkpoint:
  save_dir: "checkpoints/stsb"
  mode: "max"

early_stopping:
  patience: 3
  mode: "max"
```

## 7. 数据流全链路

```
STS-B 原始数据
  sentence1: "A plane is taking off."
  sentence2: "An air plane is taking off."
  label: 5.0

    ↓ tokenizer(sentence1) + tokenizer(sentence2) 分别编码

  input_ids_a:      [101, 1037, 4946, 2003, 2635, 2125, 1012, 102]
  attention_mask_a:  [1,   1,    1,    1,    1,    1,    1,    1  ]
  input_ids_b:      [101, 2019, 2250, 4946, 2003, 2635, 2125, 1012, 102]
  attention_mask_b:  [1,   1,    1,    1,    1,    1,    1,    1,    1  ]
  labels: 1.0        # 5.0 * 2/5 - 1 = 1.0（满分相似）

    ↓ PairDataCollator（分别 padding A 和 B）

    ↓ BertBiEncoder:
      BERT(shared) → out_a (sequence_output_a)
      BERT(shared) → out_b (sequence_output_b)

    ↓ SimilarityHead:
      mean_pooling(out_a, mask_a) → emb_a (B, 768)
      mean_pooling(out_b, mask_b) → emb_b (B, 768)
      cosine_similarity(emb_a, emb_b) → cos_sim (B,)

    ↓ MSELoss(cos_sim, labels) → loss
    ↓ SpearmanCorrelation.update(cos_sim, labels)
    ↓ SpearmanCorrelation.compute() → {"spearman": 0.86}
```

## 8. 验证清单

- [ ] `BertBiEncoder` 返回格式与 `BertForTask` 一致 (`{"loss": ..., "logits": ...}`)
- [ ] Trainer 的 `**batch` 解包对已有 3 个任务无影响
- [ ] Trainer.evaluate() 的 argmax 逻辑对回归任务（1D logits）正确跳过
- [ ] Mean pooling 正确使用 attention_mask 过滤 padding token
- [ ] STS-B 标签归一化到 [-1, 1] 与余弦相似度范围匹配
- [ ] `PairDataCollator` 对 A 和 B 分别 padding
- [ ] Spearman 相关系数自行实现（rank → Pearson on ranks）
- [ ] 单文本推理：输入两个句子，输出相似度分数
