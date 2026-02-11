"""Evaluation and inference entry point.

Usage:
    # Evaluate on validation set
    python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt

    # Single text inference
    python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt --text "This movie is great!"
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.bert import BertConfig, BertModel, BertForTask
from src.heads import HEAD_REGISTRY
from src.data import DATASET_REGISTRY
from src.evaluation import METRIC_REGISTRY
from src.training import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def predict_single(model, tokenizer, text: str, device: torch.device, labels: list[str]) -> dict:
    """Run inference on a single text string."""
    enc = tokenizer(text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )

    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=-1)
    label_id = probs.argmax(dim=-1).item()

    return {
        "text": text,
        "label": labels[label_id],
        "confidence": probs[0][label_id].item(),
        "probabilities": {labels[i]: probs[0][i].item() for i in range(len(labels))},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate or run inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Text for single inference")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Data module (for labels and tokenizer) ---
    dataset_cls = DATASET_REGISTRY.get(config["task"]["dataset"])
    data_module = dataset_cls(
        pretrained_model_name=config["model"]["pretrained"],
        max_seq_length=config["training"]["max_seq_length"],
        batch_size=config["training"]["batch_size"],
    )

    # --- Build model ---
    bert_config = BertConfig.from_dict(config["model"])
    backbone = BertModel(bert_config)
    head_cls = HEAD_REGISTRY.get(config["task"]["head"])
    head = head_cls(
        hidden_size=bert_config.hidden_size,
        num_labels=data_module.num_labels,
        dropout_prob=bert_config.hidden_dropout_prob,
    )
    model = BertForTask(backbone, head)

    # Load checkpoint
    training_config = TrainingConfig.from_dict(config["training"])
    device = training_config.resolve_device()
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # --- Single text inference ---
    if args.text:
        result = predict_single(model, data_module.tokenizer, args.text, device, data_module.get_labels())
        print(f"\nPrediction:")
        print(f"  Text:       {result['text']}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probs:      {result['probabilities']}")
        return

    # --- Full evaluation ---
    data_module.setup()
    metrics = []
    for metric_name in config["task"]["metrics"]:
        metric_cls = METRIC_REGISTRY.get(metric_name)
        if metric_name == "f1":
            metrics.append(metric_cls(num_labels=data_module.num_labels))
        else:
            metrics.append(metric_cls())

    trainer = Trainer(model=model, config=training_config)
    results = trainer.evaluate(data_module.val_dataloader(), metrics)

    print(f"\nEvaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
