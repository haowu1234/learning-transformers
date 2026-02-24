# PII 检测任务设计方案

## 1. 任务概述

| 维度 | 说明 |
|------|------|
| **任务类型** | Token-level 序列标注（PII 实体检测） |
| **数据集** | [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k)（约 400K 样本） |
| **模型** | `bert-base-multilingual-cased`（多语言 mBERT） |
| **标签体系** | BIO 格式，35 类标签：`O` + 17 种 PII 类型 × (B/I) |
| **评估方式** | Entity-level F1（复用 `seqeval` 指标） |
| **应用场景** | 自动检测文本中的个人隐私信息（姓名、邮箱、电话、信用卡号等） |

### 什么是 PII？

PII（Personally Identifiable Information，个人可识别信息）是指能直接或间接识别自然人身份的信息。常见类别包括：

| 类别 | 英文标签 | 示例 |
|------|---------|------|
| 用户名 | USERNAME | john_doe |
| 名 | GIVENNAME | John |
| 姓 | SURNAME | Smith |
| 邮箱 | EMAIL | john@example.com |
| 电话 | PHONE | +1-555-0123 |
| 信用卡号 | CREDITCARDNUMBER | 4111-1111-1111-1111 |
| 身份证号 | IDCARD | 110101199001011234 |
| 街道地址 | STREET | 123 Main Street |
| 城市 | CITY | San Francisco |
| 邮编 | ZIPCODE | 94102 |
| 州/省 | STATE | California |
| 出生日期 | DATE | 1990-01-01 |
| 时间 | TIME | 14:30:00 |
| IP 地址 | IPADDRESS | 192.168.1.1 |
| 金额 | AMOUNT | $1,234.56 |
| 公司名 | COMPANYNAME | Google Inc. |
| 职位 | JOBTITLE | Software Engineer |

## 2. 数据集分析

### 2.1 ai4privacy/pii-masking-400k

该数据集由 AI4Privacy 组织发布，包含约 400K 条多语言 PII 标注样本。

**数据集规模：**

| Split | 样本数 |
|-------|-------|
| Train | ~326K |
| Validation | ~81K |
| Test | 无（需从 validation 中切分，或直接用 validation 评估） |

**关键字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_text` | str | 原始文本 |
| `mbert_tokens` | list[str] | mBERT tokenizer 已切分的 subword token 序列 |
| `mbert_token_classes` | list[str] | 对应的 BIO 标签序列（与 `mbert_tokens` 一一对应） |
| `language` | str | 文本语言（en, fr, de, ...） |

### 2.2 数据策略：直接使用预对齐的 token

该数据集已经提供了 **`mbert_tokens`** 和 **`mbert_token_classes`**，即已经用 `bert-base-multilingual-cased` tokenizer 进行了 subword 分词，并标注了每个 subword 的 BIO 标签。

**因此我们不需要自己做 subword-label 对齐**，可直接使用：

```
mbert_tokens:        ["John", "lives", "in", "San", "Fran", "##cis", "##co"]
mbert_token_classes: ["B-GIVENNAME", "O", "O", "B-CITY", "I-CITY", "I-CITY", "I-CITY"]
```

只需要：
1. 用 `tokenizer.convert_tokens_to_ids()` 将 token 转为 `input_ids`
2. 标签字符串转为 label id（通过标签映射表）
3. 手动添加 `[CLS]`/`[SEP]` 并给对应位置标 `-100`

**与 CoNLL-2003 的关键差异：**

| 对比项 | CoNLL-2003 | PII (ai4privacy) |
|--------|-----------|-------------------|
| 原始标注粒度 | word-level，需手动对齐 | subword-level，已预对齐 |
| 对齐方式 | `tokenizer(is_split_into_words=True)` + `word_ids()` | 直接 `convert_tokens_to_ids()` |
| Tokenizer | bert-base-cased (英文) | bert-base-multilingual-cased (多语言) |
| 语言覆盖 | 仅英文 | 多语言（en, fr, de, ...） |

### 2.3 标签体系

35 个 BIO 标签（`O` + 17 种 PII 类型 × B/I）：

```python
LABELS = [
    "O",
    "B-USERNAME",    "I-USERNAME",
    "B-EMAIL",       "I-EMAIL",
    "B-PHONE",       "I-PHONE",
    "B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER",
    "B-GIVENNAME",   "I-GIVENNAME",
    "B-SURNAME",     "I-SURNAME",
    "B-IDCARD",      "I-IDCARD",
    "B-STREET",      "I-STREET",
    "B-CITY",        "I-CITY",
    "B-ZIPCODE",     "I-ZIPCODE",
    "B-STATE",       "I-STATE",
    "B-DATE",        "I-DATE",
    "B-TIME",        "I-TIME",
    "B-IPADDRESS",   "I-IPADDRESS",
    "B-AMOUNT",      "I-AMOUNT",
    "B-COMPANYNAME", "I-COMPANYNAME",
    "B-JOBTITLE",    "I-JOBTITLE",
]
```

> **注意**：实际标签列表需要在实现时根据数据集中 `mbert_token_classes` 的实际取值进行确认和调整，上面是根据数据集文档列出的预期标签。

## 3. 变更清单

遵循开闭原则，不修改已有核心代码。

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/data/pii.py` | PII 数据模块，注册到 `DATASET_REGISTRY` |
| `configs/pii_detection.yaml` | PII 检测实验配置 |

### 修改文件（仅添加 import 触发注册）

| 文件 | 改动 |
|------|------|
| `src/data/__init__.py` | `+ from . import pii` |

### 无需修改（直接复用）

| 文件 | 原因 |
|------|------|
| `src/heads/token_classification.py` | 已有 `"token_classification"` head，`CrossEntropyLoss(ignore_index=-100)` |
| `src/evaluation/seqeval.py` | 已有 `SeqevalF1`，BIO 解码逻辑通用 |
| `src/data/collator.py` | 已支持 `list[int]` label padding（`-100`） |
| `src/models/bert.py` | `BertForTask` 通用组合，mBERT 同架构 |
| `src/training/trainer.py` | `argmax(dim=-1)` 对 3D logits 同样适用 |
| `scripts/train.py` | `seqeval` 分支已有 |
| `scripts/evaluate.py` | token_classification 推理逻辑已有 |

## 4. 核心设计细节

### 4.1 数据模块 — PII DataModule

由于数据集已提供 mBERT subword 级标注，处理流程比 CoNLL-2003 更简单：

```python
@DATASET_REGISTRY.register("pii")
class PIIDataModule(BaseDataModule):

    LABELS = ["O", "B-USERNAME", "I-USERNAME", ...]  # 35 labels

    def setup(self):
        raw = load_dataset("ai4privacy/pii-masking-400k")
        self.train_dataset = PIIDataset(self._tokenize(raw["train"]))
        self.val_dataset = PIIDataset(self._tokenize(raw["validation"]))
        self.test_dataset = self.val_dataset  # 无 test split，复用 validation
```

**核心处理逻辑 `_tokenize`：**

```python
def _tokenize(self, split) -> list[dict]:
    label2id = {label: i for i, label in enumerate(self.LABELS)}
    examples = []
    for item in split:
        tokens = item["mbert_tokens"]           # 已是 subword 序列
        tag_strs = item["mbert_token_classes"]   # 已是 BIO 标签字符串

        # Truncate to max_seq_length - 2 (leave room for [CLS] and [SEP])
        max_tokens = self.max_seq_length - 2
        tokens = tokens[:max_tokens]
        tag_strs = tag_strs[:max_tokens]

        # Convert tokens to ids
        input_ids = [self.tokenizer.cls_token_id]
        input_ids += self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [self.tokenizer.sep_token_id]

        # Convert label strings to ids, with -100 for [CLS] and [SEP]
        labels = [-100]
        labels += [label2id.get(tag, 0) for tag in tag_strs]
        labels += [-100]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        })
    return examples
```

**关键点：**

1. **无需 subword 对齐**：`mbert_tokens` 已经是 subword 粒度，与 `mbert_token_classes` 一一对应
2. **手动添加特殊 token**：`[CLS]` 和 `[SEP]` 需手动拼接，标签为 `-100`
3. **标签映射容错**：遇到未知标签时 fallback 到 `O`（id=0）
4. **截断处理**：在添加 `[CLS]`/`[SEP]` 之前截断到 `max_seq_length - 2`

### 4.2 与 CoNLL-2003 数据处理的对比

```
=== CoNLL-2003 ===
原始: ["John", "lives", "in", "New", "York"]   ← word-level
  ↓ tokenizer(is_split_into_words=True)         ← tokenizer 自动拆分
  ↓ word_ids() 获取对齐关系
  ↓ _align_labels() 手动对齐
结果: input_ids + aligned_labels

=== PII (ai4privacy) ===
原始: ["John", "lives", "in", "San", "Fran", "##cis", "##co"]  ← 已是 subword
  ↓ convert_tokens_to_ids()                     ← 直接转换
  ↓ 手动加 [CLS]/[SEP] 和对应 -100
结果: input_ids + labels
```

### 4.3 配置文件

```yaml
# PII Detection with mBERT
task:
  head: "token_classification"
  dataset: "pii"
  metrics: ["seqeval"]

model:
  pretrained: "bert-base-multilingual-cased"
  vocab_size: 119547
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1

training:
  batch_size: 16
  learning_rate: 3.0e-5
  num_epochs: 3          # 数据量大，3 epoch 即可
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 128
  device: "auto"
  log_interval: 100      # 数据量大，减少日志频率

checkpoint:
  save_dir: "checkpoints/pii"
  mode: "max"

early_stopping:
  patience: 2
  mode: "max"
```

**参数选择说明：**

- **模型**: `bert-base-multilingual-cased`，因数据集多语言，且标注基于此 tokenizer
- **num_epochs: 3**: 训练集约 326K 条，远大于 CoNLL-2003（14K），无需太多 epoch
- **batch_size: 16**: 与 NER 一致，token-level 任务序列较长
- **max_seq_length: 128**: mBERT subword 序列平均长度适中，128 足以覆盖大部分样本
- **early_stopping patience: 2**: 数据量足够，收敛较快，patience 可减小

## 5. 数据流全链路

```
ai4privacy/pii-masking-400k 原始数据
  mbert_tokens:        ["Dear", "John", "Smith", ",", "your", "email", "is", "john", "@", "example", ".", "com"]
  mbert_token_classes: ["O", "B-GIVENNAME", "B-SURNAME", "O", "O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]

    ↓ convert_tokens_to_ids() + 手动加 [CLS]/[SEP]

  input_ids:      [101, 39385, 2198, 3489, 117, ..., 102]
  attention_mask: [1,   1,     1,    1,    1,   ..., 1  ]
  labels:         [-100, 0,    9,    11,   0,   ..., -100]
                   CLS   O  B-GIVEN B-SUR  O         SEP

    ↓ DataCollator（padding to batch max_len, label pad = -100）

    ↓ BertModel → sequence_output (B, L, 768)
    ↓ TokenClassificationHead → logits (B, L, 35)
    ↓ CrossEntropyLoss(ignore_index=-100) → loss
    ↓ argmax(dim=-1) → preds (B, L)

    ↓ SeqevalF1.update(preds, labels)   # 过滤 -100，解码 BIO
    ↓ SeqevalF1.compute() → {"f1": ..., "precision": ..., "recall": ...}
```

## 6. 潜在风险与注意事项

1. **标签覆盖确认**：实现时需遍历数据集确认 `mbert_token_classes` 的所有唯一值，确保 `LABELS` 列表完整
2. **未知标签处理**：`label2id.get(tag, 0)` fallback 到 `O`，避免 KeyError
3. **无 test split**：数据集只有 train/validation，可选择：
   - 直接用 validation 做评估（简单方案）
   - 从 train 中切分一部分做 test（更严谨）
4. **数据量大**：326K 训练样本，首次 setup 下载和 tokenize 会比较慢
5. **多语言 tokenizer**：必须使用 `bert-base-multilingual-cased`，因为数据集的 `mbert_tokens` 就是用它分的

## 7. 验证清单

- [ ] `LABELS` 列表与数据集中实际标签一致
- [ ] `convert_tokens_to_ids()` 能正确处理 `mbert_tokens` 中的所有 token
- [ ] `[CLS]`/`[SEP]` 正确添加，对应标签为 `-100`
- [ ] 截断逻辑正确：先截断再加特殊 token
- [ ] `DataCollator` 对 PII 的 `list[int]` labels padding 正常工作
- [ ] `SeqevalF1` 能正确解码 35 类 BIO 标签
- [ ] `evaluate.py` 的单文本推理对 PII 任务正常工作
- [ ] 训练跑通，loss 正常下降
