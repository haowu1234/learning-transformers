"""Entity-level F1 metric for NER (seqeval-style)."""

import torch
from torch import Tensor

from .base import BaseMetric
from .metrics import METRIC_REGISTRY


@METRIC_REGISTRY.register("seqeval")
class SeqevalF1(BaseMetric):
    """Entity-level precision / recall / F1 for BIO-tagged sequences.

    Decodes BIO tag sequences, extracts entity spans, and computes
    strict entity-level matching (type + boundaries must both match).
    """

    def __init__(self, label_list: list[str]):
        self.label_list = label_list
        self.reset()

    def update(self, preds: Tensor, labels: Tensor) -> None:
        # preds, labels: (batch, seq_len)
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())

    def compute(self) -> dict[str, float]:
        preds = torch.cat(self.all_preds, dim=0)   # (N, L)
        labels = torch.cat(self.all_labels, dim=0)  # (N, L)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i in range(preds.size(0)):
            pred_ids = preds[i].tolist()
            label_ids = labels[i].tolist()

            # Filter out -100 positions and convert to tag strings
            pred_tags = []
            gold_tags = []
            for p, g in zip(pred_ids, label_ids):
                if g == -100:
                    continue
                pred_tags.append(self.label_list[p] if 0 <= p < len(self.label_list) else "O")
                gold_tags.append(self.label_list[g])

            pred_entities = self._extract_entities(pred_tags)
            gold_entities = self._extract_entities(gold_tags)

            # Strict matching: (type, start, end) must all match
            pred_set = set(pred_entities)
            gold_set = set(gold_entities)

            total_tp += len(pred_set & gold_set)
            total_fp += len(pred_set - gold_set)
            total_fn += len(gold_set - pred_set)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []

    @staticmethod
    def _extract_entities(tags: list[str]) -> list[tuple[str, int, int]]:
        """Extract entity spans from a BIO tag sequence.

        Returns list of (entity_type, start_idx, end_idx) tuples.
        end_idx is inclusive.
        """
        entities = []
        current_type = None
        current_start = None

        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                # Close previous entity if any
                if current_type is not None:
                    entities.append((current_type, current_start, i - 1))
                # Start new entity
                current_type = tag[2:]
                current_start = i
            elif tag.startswith("I-"):
                tag_type = tag[2:]
                if current_type == tag_type:
                    # Continue current entity
                    pass
                else:
                    # Type mismatch: close previous, treat as new B-
                    if current_type is not None:
                        entities.append((current_type, current_start, i - 1))
                    current_type = tag_type
                    current_start = i
            else:
                # O tag: close previous entity if any
                if current_type is not None:
                    entities.append((current_type, current_start, i - 1))
                    current_type = None
                    current_start = None

        # Close last entity if sequence ends mid-entity
        if current_type is not None:
            entities.append((current_type, current_start, len(tags) - 1))

        return entities
