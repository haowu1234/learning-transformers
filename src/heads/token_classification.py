"""Token classification head for NER and similar tasks."""

import torch.nn as nn
from torch import Tensor

from .base import BaseHead
from .classification import HEAD_REGISTRY


@HEAD_REGISTRY.register("token_classification")
class TokenClassificationHead(BaseHead):
    """Apply a classifier on each token's hidden state.

    Suitable for NER, POS tagging, etc.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        backbone_outputs: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        sequence_output = backbone_outputs["sequence_output"]  # (batch, seq_len, hidden)
        return self.classifier(self.dropout(sequence_output))  # (batch, seq_len, num_labels)

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        # logits: (batch, seq_len, num_labels) → flatten
        # labels: (batch, seq_len) → flatten
        return self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
