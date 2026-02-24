# learning-transformers

从零手写 Transformer，并基于 BERT 进行经典下游任务微调（不做预训练）。

## 已支持任务

| 任务 | 数据集 | 模型 | 指标 | 结果 |
|------|--------|------|------|------|
| 情感分类 | SST-2 | bert-base-uncased | Accuracy | **92.55%** |
| 命名实体识别 | CoNLL-2003 | bert-base-cased | Entity F1 | **94.88%** |
| PII 检测 | ai4privacy/pii-masking-400k | bert-base-multilingual-cased | Entity F1 | **93.09%** |
| 语义相似度 | STS-B (GLUE) | bert-base-uncased | Spearman | — |

## 特性

- 从零实现 Multi-Head Attention、Transformer Encoder、BERT
- 加载 HuggingFace 预训练权重进行微调
- Registry 模式，新增任务零改动已有代码
- 统一 Trainer + EarlyStopping + ModelCheckpoint
- config 驱动，一份 YAML 跑一个实验

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 选择任务，开始训练
python scripts/train.py --config configs/sst2_classification.yaml
python scripts/train.py --config configs/conll2003_ner.yaml
python scripts/train.py --config configs/pii_detection.yaml
python scripts/train.py --config configs/stsb_similarity.yaml

# 3. 评估 (默认 validation，加 --split test 切换到 test)
python scripts/evaluate.py --config configs/sst2_classification.yaml \
    --checkpoint checkpoints/best_model.pt

python scripts/evaluate.py --config configs/conll2003_ner.yaml \
    --checkpoint checkpoints/ner/best_model.pt --split test

# 4. 单条推理
python scripts/evaluate.py --config configs/sst2_classification.yaml \
    --checkpoint checkpoints/best_model.pt \
    --text "This movie is great!"

python scripts/evaluate.py --config configs/conll2003_ner.yaml \
    --checkpoint checkpoints/ner/best_model.pt \
    --text "John lives in New York"

python scripts/evaluate.py --config configs/pii_detection.yaml \
    --checkpoint checkpoints/pii/best_model.pt \
    --text "Contact John Smith at john@example.com"

python scripts/evaluate.py --config configs/stsb_similarity.yaml \
    --checkpoint checkpoints/stsb/best_model.pt \
    --text "A plane is taking off. ||| An air plane is taking off."
```

## 项目结构

```
src/
├── models/          # Multi-Head Attention → Embedding → Encoder → BERT / Bi-Encoder
├── heads/           # 可插拔任务头: classification, token_classification, similarity, qa
├── data/            # 可插拔数据模块: SST-2, CoNLL-2003, PII, STS-B
├── training/        # Trainer + EarlyStopping + ModelCheckpoint
├── evaluation/      # 可插拔指标: Accuracy, F1, SeqevalF1, Spearman
└── utils/           # Registry 注册表
configs/             # 每个实验一份 YAML
scripts/             # train.py / evaluate.py 入口
docs/                # 任务设计文档
```

## 扩展新任务

只需 3 步，不修改已有代码（开闭原则）：

1. `src/heads/` 新增任务头，用 `@HEAD_REGISTRY.register(...)` 注册
2. `src/data/` 新增数据模块，用 `@DATASET_REGISTRY.register(...)` 注册
3. `configs/` 新增 YAML 配置文件

```yaml
# configs/your_task.yaml
task:
  head: "token_classification"   # 复用已有 head 或新建
  dataset: "your_dataset"        # 对应注册名
  metrics: ["seqeval"]           # accuracy / f1 / seqeval / spearman
model:
  pretrained: "bert-base-uncased"
training:
  batch_size: 16
  learning_rate: 3.0e-5
  num_epochs: 5
```

## 技术栈

- Python 3.10+ / PyTorch 2.0+
- HuggingFace transformers / tokenizers / datasets
- PyYAML
