# learning-transformers

从零手写 Transformer，并基于 BERT 进行经典下游任务微调。

## 特性

- 从零实现 Multi-Head Attention、Transformer Encoder、BERT 模型
- 加载 HuggingFace 预训练权重进行微调（不做预训练）
- Registry 注册表模式，新增任务零改动已有代码
- 统一 Trainer，config 驱动，一份 YAML 跑一个实验

## 项目结构

```
src/
├── models/          # Attention → Embedding → Encoder → BERT (纯 backbone)
├── heads/           # 可插拔任务头: classification, token_classification, qa
├── data/            # 可插拔数据模块: SST-2, ...
├── training/        # 统一 Trainer + Callbacks
├── evaluation/      # 可插拔指标: Accuracy, F1
└── utils/           # Registry 注册表
configs/             # 每个实验一份 YAML
scripts/             # train.py / evaluate.py 入口
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练 SST-2 情感分类
PYTHONPATH=. python scripts/train.py --config configs/sst2_classification.yaml

# 评估
PYTHONPATH=. python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt

# 单条推理
PYTHONPATH=. python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt --text "This movie is great!"
```

## 扩展新任务

只需 3 步，不修改已有代码：

1. `src/heads/` 新增任务头，用 `@HEAD_REGISTRY.register(...)` 注册
2. `src/data/` 新增数据模块，用 `@DATASET_REGISTRY.register(...)` 注册
3. `configs/` 新增 YAML 配置文件

## 技术栈

- Python 3.10+
- PyTorch
- HuggingFace transformers / tokenizers / datasets
- PyYAML
