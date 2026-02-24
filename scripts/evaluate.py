"""Evaluation and inference entry point.

Usage:
    # Evaluate on validation set
    python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt

    # Evaluate on test set
    python scripts/evaluate.py --config configs/conll2003_ner.yaml --checkpoint checkpoints/ner/best_model.pt --split test

    # Single text inference (classification)
    python scripts/evaluate.py --config configs/sst2_classification.yaml --checkpoint checkpoints/best_model.pt --text "This movie is great!"

    # Single text inference (NER)
    python scripts/evaluate.py --config configs/conll2003_ner.yaml --checkpoint checkpoints/ner/best_model.pt --text "John lives in New York"
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


def predict_single(model, tokenizer, text: str, device: torch.device, labels: list[str], task_head: str) -> dict:
    """Run inference on a single text string."""
    is_token_task = task_head == "token_classification"

    if is_token_task:
        words = text.split()
        enc = tokenizer(words, is_split_into_words=True, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    else:
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

    if is_token_task:
        # Token-level: decode each token
        pred_ids = logits[0].argmax(dim=-1).tolist()
        word_ids = enc.get("word_ids", None)
        # Reconstruct word_ids from tokenizer
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        result_tokens = []
        prev_word_id = None
        for i, (token, pred_id) in enumerate(zip(tokens, pred_ids)):
            if token in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                continue
            if token.startswith("##"):
                continue
            tag = labels[pred_id] if 0 <= pred_id < len(labels) else "O"
            result_tokens.append((token, tag))
        return {"text": text, "tokens": result_tokens}
    else:
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
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on (validation/test)")
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
        result = predict_single(model, data_module.tokenizer, args.text, device,
                                data_module.get_labels(), config["task"]["head"])
        print(f"\nPrediction:")
        print(f"  Text: {result['text']}")
        if "tokens" in result:
            # Token-level task (NER)
            print(f"  Entities:")
            for token, tag in result["tokens"]:
                if tag != "O":
                    print(f"    {token:15s} → {tag}")
            print(f"  All tokens:")
            for token, tag in result["tokens"]:
                print(f"    {token:15s} → {tag}")
        else:
            # Sentence-level task
            print(f"  Label:      {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Probs:      {result['probabilities']}")
        return

    # --- Full evaluation ---
    data_module.setup()

    # Select split
    if args.split == "test":
        eval_dataloader = data_module.test_dataloader()
    else:
        eval_dataloader = data_module.val_dataloader()

    metrics = []
    for metric_name in config["task"]["metrics"]:
        metric_cls = METRIC_REGISTRY.get(metric_name)
        if metric_name == "f1":
            metrics.append(metric_cls(num_labels=data_module.num_labels))
        elif metric_name == "seqeval":
            metrics.append(metric_cls(label_list=data_module.get_labels()))
        else:
            metrics.append(metric_cls())

    trainer = Trainer(model=model, config=training_config)
    results = trainer.evaluate(eval_dataloader, metrics)

    print(f"\nEvaluation Results ({args.split}):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
