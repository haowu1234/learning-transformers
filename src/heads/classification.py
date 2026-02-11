"""Classification head for sentence-level tasks (e.g. sentiment analysis)."""

import torch.nn as nn
from torch import Tensor

from .base import BaseHead
from ..utils.registry import Registry

HEAD_REGISTRY = Registry("heads")


@HEAD_REGISTRY.register("classification")
class ClassificationHead(BaseHead):
    """[CLS] pooled output → Dropout → Linear → logits.

    Suitable for single-sentence and sentence-pair classification tasks.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        backbone_outputs: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        pooled_output = backbone_outputs["pooled_output"]  # (batch, hidden_size)
        return self.classifier(self.dropout(pooled_output))  # (batch, num_labels)

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        return self.loss_fn(logits, labels)
