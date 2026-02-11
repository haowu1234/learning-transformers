"""Concrete metric implementations."""

import torch
from torch import Tensor

from .base import BaseMetric
from ..utils.registry import Registry

METRIC_REGISTRY = Registry("metrics")


@METRIC_REGISTRY.register("accuracy")
class Accuracy(BaseMetric):
    """Simple accuracy: correct / total."""

    def __init__(self):
        self.reset()

    def update(self, preds: Tensor, labels: Tensor) -> None:
        self.correct += (preds == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> dict[str, float]:
        acc = self.correct / self.total if self.total > 0 else 0.0
        return {"accuracy": acc}

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


@METRIC_REGISTRY.register("f1")
class F1Score(BaseMetric):
    """Binary / macro F1 score."""

    def __init__(self, num_labels: int = 2, average: str = "macro"):
        self.num_labels = num_labels
        self.average = average
        self.reset()

    def update(self, preds: Tensor, labels: Tensor) -> None:
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())

    def compute(self) -> dict[str, float]:
        preds = torch.cat(self.all_preds)
        labels = torch.cat(self.all_labels)

        f1_per_class = []
        precision_per_class = []
        recall_per_class = []

        for c in range(self.num_labels):
            tp = ((preds == c) & (labels == c)).sum().float()
            fp = ((preds == c) & (labels != c)).sum().float()
            fn = ((preds != c) & (labels == c)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precision_per_class.append(precision.item())
            recall_per_class.append(recall.item())
            f1_per_class.append(f1.item())

        if self.average == "macro":
            return {
                "f1": sum(f1_per_class) / len(f1_per_class),
                "precision": sum(precision_per_class) / len(precision_per_class),
                "recall": sum(recall_per_class) / len(recall_per_class),
            }
        else:
            # Return per-class values
            return {
                f"f1_class_{c}": v for c, v in enumerate(f1_per_class)
            }

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []
