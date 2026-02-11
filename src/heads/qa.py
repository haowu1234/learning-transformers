"""Extractive QA head (SQuAD-style span extraction)."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseHead
from .classification import HEAD_REGISTRY


@HEAD_REGISTRY.register("qa")
class QAHead(BaseHead):
    """Predict start and end positions of the answer span.

    Each token gets a start score and an end score.
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        backbone_outputs: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        sequence_output = backbone_outputs["sequence_output"]  # (batch, seq_len, hidden)
        start_logits = self.start_classifier(sequence_output).squeeze(-1)  # (batch, seq_len)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        return torch.stack([start_logits, end_logits], dim=-1)  # (batch, seq_len, 2)

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        # logits: (batch, seq_len, 2)
        # labels: (batch, 2) — [start_position, end_position]
        start_logits = logits[:, :, 0]  # (batch, seq_len)
        end_logits = logits[:, :, 1]
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]

        start_loss = self.loss_fn(start_logits, start_positions)
        end_loss = self.loss_fn(end_logits, end_positions)
        return (start_loss + end_loss) / 2
