"""Training entry point — config-driven.

Usage:
    python scripts/train.py --config configs/sst2_classification.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.bert import BertConfig, BertModel, BertForTask, load_pretrained_weights
from src.heads import HEAD_REGISTRY
from src.data import DATASET_REGISTRY
from src.evaluation import METRIC_REGISTRY
from src.training import Trainer, TrainingConfig, EarlyStopping, ModelCheckpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a BERT downstream task")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {args.config}")

    # --- Data ---
    dataset_cls = DATASET_REGISTRY.get(config["task"]["dataset"])
    data_module = dataset_cls(
        pretrained_model_name=config["model"]["pretrained"],
        max_seq_length=config["training"]["max_seq_length"],
        batch_size=config["training"]["batch_size"],
    )
    data_module.setup()
    logger.info(f"Dataset: {config['task']['dataset']} — {data_module.num_labels} labels")

    # --- Model ---
    bert_config = BertConfig.from_dict(config["model"])
    backbone = BertModel(bert_config)
    backbone = load_pretrained_weights(backbone, config["model"]["pretrained"])

    head_cls = HEAD_REGISTRY.get(config["task"]["head"])
    head = head_cls(
        hidden_size=bert_config.hidden_size,
        num_labels=data_module.num_labels,
        dropout_prob=bert_config.hidden_dropout_prob,
    )

    model = BertForTask(backbone, head)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # --- Metrics ---
    metrics = []
    for metric_name in config["task"]["metrics"]:
        metric_cls = METRIC_REGISTRY.get(metric_name)
        if metric_name == "f1":
            metrics.append(metric_cls(num_labels=data_module.num_labels))
        elif metric_name == "seqeval":
            metrics.append(metric_cls(label_list=data_module.get_labels()))
        else:
            metrics.append(metric_cls())

    # --- Callbacks ---
    early_stopping = None
    if "early_stopping" in config:
        early_stopping = EarlyStopping(**config["early_stopping"])

    checkpoint = None
    if "checkpoint" in config:
        checkpoint = ModelCheckpoint(**config["checkpoint"])

    # --- Trainer ---
    training_config = TrainingConfig.from_dict(config["training"])
    trainer = Trainer(
        model=model,
        config=training_config,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
    )

    # --- Train ---
    results = trainer.fit(
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        metrics=metrics,
    )

    logger.info(f"Training complete. Final results: {results}")


if __name__ == "__main__":
    main()
