# CoNLL-2003 NER 任务设计方案

## 1. 任务概述

| 维度 | 说明 |
|------|------|
| **任务类型** | Token-level 序列标注（命名实体识别） |
| **数据集** | CoNLL-2003（NER 领域最经典的 benchmark） |
| **标签体系** | BIO 格式，9 类标签：`O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC` |
| **评估方式** | Entity-level F1（实体级别精确匹配，而非 token 级别） |
| **预期指标** | BERT-base 在 CoNLL-2003 上约 **91-92 F1** |

## 2. 与 SST-2 任务的核心差异

| 对比项 | SST-2（句子分类） | CoNLL-2003（NER） |
|--------|------------------|-------------------|
| 分类粒度 | 整个句子 → 1 个标签 | 每个 token → 1 个标签 |
| Head 输入 | `pooled_output` (B, H) | `sequence_output` (B, L, H) |
| Label 形状 | 标量 `int` | 序列 `list[int]`，长度 = seq_len |
| Padding 标签 | 不涉及 | `-100`（subword 对齐 + padding） |
| 评估指标 | Token-level accuracy/F1 | **Entity-level** F1（需解码 BIO 序列） |

## 3. 变更清单

遵循开闭原则，不修改已有核心代码。

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/data/conll2003.py` | CoNLL-2003 数据模块，注册到 `DATASET_REGISTRY` |
| `src/evaluation/seqeval.py` | Entity-level F1 指标，注册到 `METRIC_REGISTRY` |
| `configs/conll2003_ner.yaml` | NER 实验配置 |

### 修改文件（仅添加 import 触发注册）

| 文件 | 改动 |
|------|------|
| `src/data/__init__.py` | `+ from . import conll2003` |
| `src/evaluation/__init__.py` | `+ from . import seqeval` |
| `scripts/train.py` | metric 实例化增加 `seqeval` 分支 |

### 无需修改

| 文件 | 原因 |
|------|------|
| `src/heads/token_classification.py` | 已有，直接复用 `"token_classification"` |
| `src/data/collator.py` | 已支持 `list[int]` label padding（`-100`） |
| `src/models/bert.py` | `BertForTask` 通用组合模型 |
| `src/training/trainer.py` | `argmax(dim=-1)` 对 3D logits 同样适用 |

## 4. 核心设计细节

### 4.1 数据模块 — Subword-Label 对齐

CoNLL-2003 标注在 **word 级别**，而 BERT tokenizer 会把 word 拆成多个 subword，这是 NER 数据处理中最关键的问题。

**示例（无拆分）：**

```
原始:     ["John",  "lives", "in", "New",  "York"]
标签:     [B-PER,   O,       O,    B-LOC,  I-LOC]
Tokenize: ["john",  "lives", "in", "new",  "york"]
对齐标签: [B-PER,   O,       O,    B-LOC,  I-LOC]   ← 一一对应
```

**示例（有拆分）：**

```
原始:     ["Washington"]
标签:     [B-PER]
Tokenize: ["wash", "##ing", "##ton"]
对齐标签: [B-PER,  -100,    -100]   ← 只有首个 subword 保留标签
```

**对齐策略：**

- 使用 `tokenizer(words, is_split_into_words=True)` 保留 word 边界信息
- 通过 `word_ids()` 获取每个 subword 对应的原始 word 索引
- 规则：每个 word 的**第一个** subword 取该 word 的标签，其余 subword 设为 `-100`
- `[CLS]` 和 `[SEP]` 特殊 token 也设为 `-100`

**对齐核心代码：**

```python
def _align_labels(self, word_ids, word_labels):
    """将 word-level 标签对齐到 subword-level"""
    aligned = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:           # [CLS], [SEP], padding
            aligned.append(-100)
        elif word_id != prev_word_id:  # word 的首个 subword
            aligned.append(word_labels[word_id])
        else:                          # word 的后续 subword
            aligned.append(-100)
        prev_word_id = word_id
    return aligned
```

**数据加载：**

- 使用 HuggingFace `load_dataset("conll2003")`
- 每条样本包含 `tokens: list[str]` 和 `ner_tags: list[int]`
- 标签映射：`{0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}`

### 4.2 评估指标 — Entity-level F1

NER 的标准评估方式是 **entity-level F1**，不是 token-level。

**区别示例：**

```
预测: [O, B-PER, I-PER, O, B-LOC, O    ]
真实: [O, B-PER, I-PER, O, B-LOC, I-LOC]

Token-level: 5/6 = 83.3% accuracy
Entity-level: PER 命中 1/1, LOC 未命中 0/1 → F1 = 66.7%
```

**实现要点：**

- 注册为 `@METRIC_REGISTRY.register("seqeval")`
- 构造时传入 `label_list: list[str]`（标签名称列表，用于 id → 标签名转换）
- `update(preds, labels)`：收集 batch 级预测和真实标签，过滤 `-100` 位置
- `compute()`：
  1. 将 id 序列转回 BIO 标签字符串
  2. 解码 BIO 序列提取实体 span `(type, start, end)`
  3. 计算 entity-level precision / recall / F1
- **自行实现** BIO 解码逻辑，保持项目的学习价值

**BIO 解码规则：**

- 遇到 `B-XXX` → 开始一个新的 XXX 类型实体
- 遇到 `I-XXX` 且前一个标签是 `B-XXX` 或 `I-XXX` → 延续当前实体
- 遇到 `I-XXX` 但前一个不是同类型 → 视为新实体的开始（容错处理）
- 遇到 `O` 或不同类型 → 结束当前实体

### 4.3 train.py 修改

仅需在 metric 实例化处增加一个分支：

```python
# 现有逻辑
if metric_name == "f1":
    metric = metric_cls(num_labels=data_module.num_labels)
# 新增
elif metric_name == "seqeval":
    metric = metric_cls(label_list=data_module.get_labels())
else:
    metric = metric_cls()
```

### 4.4 配置文件

```yaml
task:
  head: "token_classification"    # 复用已有 head
  dataset: "conll2003"
  metrics: ["seqeval"]            # entity-level F1

model:
  pretrained: "bert-base-cased"   # NER 用 cased（大小写对实体识别重要）

training:
  batch_size: 16                  # NER 序列较长，batch 适当减小
  learning_rate: 3.0e-5           # NER 常用 3e-5 或 5e-5
  num_epochs: 5                   # NER 通常需要更多 epoch
  max_seq_length: 128
```

> **注意**：NER 任务通常使用 `bert-base-cased`（区分大小写），因为大小写对实体识别很重要（如 "Apple" 公司 vs "apple" 水果）。

## 5. 数据流全链路

```
CoNLL-2003 原始数据
  tokens:   ["John", "lives", "in", "New", "York"]
  ner_tags: [1, 0, 0, 5, 6]   # B-PER, O, O, B-LOC, I-LOC

    ↓ tokenizer(is_split_into_words=True) + 标签对齐

  input_ids:      [101, 2198, 3268, 1999, 2047, 2259, 102]
  attention_mask:  [1,   1,    1,    1,    1,    1,    1  ]
  labels:         [-100, 1,    0,    0,    5,    6,   -100]
                   CLS                                SEP

    ↓ DataCollator（padding to batch max_len）

  input_ids:      [101, 2198, 3268, 1999, 2047, 2259, 102, 0, 0]
  attention_mask:  [1,   1,    1,    1,    1,    1,    1,  0, 0]
  labels:         [-100, 1,    0,    0,    5,    6,   -100,-100,-100]

    ↓ BertModel → sequence_output (B, L, 768)
    ↓ TokenClassificationHead → logits (B, L, 9)
    ↓ CrossEntropyLoss(ignore_index=-100) → loss
    ↓ argmax(dim=-1) → preds (B, L)

    ↓ SeqevalF1.update(preds, labels)   # 过滤 -100，解码 BIO
    ↓ SeqevalF1.compute() → {"f1": 0.91, "precision": ..., "recall": ...}
```

## 6. 验证清单

- [ ] `TokenClassificationHead` 直接复用，无需改动
- [ ] `DataCollator` 已支持 `list[int]` label padding（`-100`），无需改动
- [ ] `Trainer.evaluate()` 中 `argmax(dim=-1)` 对 3D logits 正确工作，无需改动
- [ ] 新增文件遵循 Registry 模式，零修改已有核心代码
- [ ] Subword 对齐逻辑正确处理 `[CLS]`/`[SEP]`/后续 subword → `-100`
- [ ] Entity-level F1 而非 token-level，符合 NER 标准评估
