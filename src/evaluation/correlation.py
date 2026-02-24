"""Spearman and Pearson correlation metrics for regression tasks."""

import torch
from torch import Tensor

from .base import BaseMetric
from .metrics import METRIC_REGISTRY


@METRIC_REGISTRY.register("spearman")
class SpearmanCorrelation(BaseMetric):
    """Spearman rank correlation coefficient.

    Spearman = Pearson correlation on rank-transformed values.
    """

    def __init__(self):
        self.reset()

    def update(self, preds: Tensor, labels: Tensor) -> None:
        self.all_preds.append(preds.cpu().float())
        self.all_labels.append(labels.cpu().float())

    def compute(self) -> dict[str, float]:
        preds = torch.cat(self.all_preds)
        labels = torch.cat(self.all_labels)

        pred_ranks = self._rank(preds)
        label_ranks = self._rank(labels)

        spearman = self._pearson(pred_ranks, label_ranks)
        pearson = self._pearson(preds, labels)

        return {
            "spearman": spearman,
            "pearson": pearson,
        }

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []

    @staticmethod
    def _rank(x: Tensor) -> Tensor:
        """Compute ranks (1-based) with average tie-breaking."""
        sorted_indices = x.argsort()
        ranks = torch.empty_like(x)
        ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype)

        # Average tie-breaking
        unique_vals = x.unique()
        for val in unique_vals:
            mask = x == val
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    @staticmethod
    def _pearson(x: Tensor, y: Tensor) -> float:
        """Pearson correlation coefficient."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        num = (x_centered * y_centered).sum()
        den = (x_centered.norm() * y_centered.norm()).clamp(min=1e-8)
        return (num / den).item()
