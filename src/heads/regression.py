"""Regression head for Cross-Encoder similarity: [CLS] → Linear(1) → MSE loss."""

import torch.nn as nn
from torch import Tensor

from .base import BaseHead
from .classification import HEAD_REGISTRY


@HEAD_REGISTRY.register("regression")
class RegressionHead(BaseHead):
    """[CLS] pooled output → Dropout → Linear(hidden_size, 1) → squeeze → MSE loss.

    Designed for Cross-Encoder sentence similarity / regression tasks.
    The model sees both sentences concatenated as a single input:
        [CLS] sent_a [SEP] sent_b [SEP]
    """

    def __init__(self, hidden_size: int, num_labels: int = 1,
                 dropout_prob: float = 0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        backbone_outputs: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        pooled_output = backbone_outputs["pooled_output"]  # (B, H)
        logits = self.regressor(self.dropout(pooled_output))  # (B, 1)
        return logits.squeeze(-1)  # (B,)

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        return self.loss_fn(logits, labels)
