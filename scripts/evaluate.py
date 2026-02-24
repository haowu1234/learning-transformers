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

    # Single text inference (PII)
    python scripts/evaluate.py --config configs/pii_detection.yaml --checkpoint checkpoints/pii/best_model.pt --text "Contact John Smith at john@example.com"

    # Sentence pair inference (similarity)
    python scripts/evaluate.py --config configs/stsb_similarity.yaml --checkpoint checkpoints/stsb/best_model.pt --text "A plane is taking off. ||| An air plane is taking off."
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


def predict_single(model, tokenizer, text: str, device: torch.device,
                   labels: list[str], task_head: str, dataset_name: str) -> dict:
    """Run inference on a single text string."""
    is_token_task = task_head == "token_classification"

    # NER datasets (e.g. CoNLL-2003) trained with is_split_into_words=True;
    # PII dataset trained with direct tokenizer(text) since mbert_tokens are pre-tokenized.
    use_word_split = is_token_task and dataset_name not in ("pii",)

    if use_word_split:
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
        pred_ids = logits[0].argmax(dim=-1).tolist()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        # Merge subword tokens (##xxx) into their preceding token for display
        result_tokens = []
        for token, pred_id in zip(tokens, pred_ids):
            if token in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                continue
            tag = labels[pred_id] if 0 <= pred_id < len(labels) else "O"
            if token.startswith("##") and result_tokens:
                # Append subword to previous token's display text
                prev_text, prev_tag = result_tokens[-1]
                result_tokens[-1] = (prev_text + token[2:], prev_tag)
            else:
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


def predict_pair(model, tokenizer, text: str, device: torch.device) -> dict:
    """Run inference on a sentence pair (separated by [SEP] or |||)."""
    # Split input into two sentences
    if "|||" in text:
        sent_a, sent_b = [s.strip() for s in text.split("|||", 1)]
    else:
        # Fallback: split by [SEP]
        parts = text.split("[SEP]")
        if len(parts) >= 2:
            sent_a, sent_b = parts[0].strip(), parts[1].strip()
        else:
            return {"error": "Please separate two sentences with ||| (e.g. 'sentence A ||| sentence B')"}

    enc_a = tokenizer(sent_a, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    enc_b = tokenizer(sent_b, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    batch = {
        "input_ids_a": enc_a["input_ids"].to(device),
        "attention_mask_a": enc_a["attention_mask"].to(device),
        "token_type_ids_a": enc_a.get("token_type_ids", torch.zeros_like(enc_a["input_ids"])).to(device),
        "input_ids_b": enc_b["input_ids"].to(device),
        "attention_mask_b": enc_b["attention_mask"].to(device),
        "token_type_ids_b": enc_b.get("token_type_ids", torch.zeros_like(enc_b["input_ids"])).to(device),
    }

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)

    cos_sim = outputs["logits"].item()
    # Map [-1, 1] back to [0, 5] for human-readable score
    score_0_5 = (cos_sim + 1.0) * 5.0 / 2.0

    return {
        "sentence1": sent_a,
        "sentence2": sent_b,
        "cosine_similarity": cos_sim,
        "score_0_5": score_0_5,
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
    data_module.setup()

    # --- Build model ---
    bert_config = BertConfig.from_dict(config["model"])
    backbone = BertModel(bert_config)
    head_cls = HEAD_REGISTRY.get(config["task"]["head"])

    if config["task"]["head"] == "similarity":
        from src.models.bi_encoder import BertBiEncoder
        head = head_cls(hidden_size=bert_config.hidden_size)
        model = BertBiEncoder(backbone, head)
    else:
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
        if config["task"]["head"] == "similarity":
            result = predict_pair(model, data_module.tokenizer, args.text, device)
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nPrediction:")
                print(f"  Sentence 1: {result['sentence1']}")
                print(f"  Sentence 2: {result['sentence2']}")
                print(f"  Cosine Sim: {result['cosine_similarity']:.4f}")
                print(f"  Score (0-5): {result['score_0_5']:.2f}")
        else:
            result = predict_single(model, data_module.tokenizer, args.text, device,
                                    data_module.get_labels(), config["task"]["head"],
                                    config["task"]["dataset"])
            print(f"\nPrediction:")
            print(f"  Text: {result['text']}")
            if "tokens" in result:
                print(f"  Entities:")
                for token, tag in result["tokens"]:
                    if tag != "O":
                        print(f"    {token:15s} → {tag}")
                print(f"  All tokens:")
                for token, tag in result["tokens"]:
                    print(f"    {token:15s} → {tag}")
            else:
                print(f"  Label:      {result['label']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Probs:      {result['probabilities']}")
        return

    # --- Full evaluation ---
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
