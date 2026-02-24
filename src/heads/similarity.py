"""Similarity head for Bi-Encoder: mean pooling + cosine similarity + MSE loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseHead
from .classification import HEAD_REGISTRY


@HEAD_REGISTRY.register("similarity")
class SimilarityHead(BaseHead):
    """Mean pooling → cosine similarity → MSE loss.

    Designed for Bi-Encoder sentence similarity tasks.
    The forward() signature differs from BaseHead: it receives two sets of
    backbone outputs and attention masks (one per sentence).
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def mean_pooling(sequence_output: Tensor, attention_mask: Tensor) -> Tensor:
        """Average pool over non-padding tokens.

        Args:
            sequence_output: (B, L, H)
            attention_mask:  (B, L)

        Returns:
            (B, H) sentence embedding
        """
        mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
        summed = (sequence_output * mask).sum(dim=1)      # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        return summed / lengths

    def forward(
        self,
        backbone_outputs_a: dict[str, Tensor],
        backbone_outputs_b: dict[str, Tensor],
        attention_mask_a: Tensor,
        attention_mask_b: Tensor,
        **kwargs,
    ) -> Tensor:
        emb_a = self.mean_pooling(backbone_outputs_a["sequence_output"], attention_mask_a)
        emb_b = self.mean_pooling(backbone_outputs_b["sequence_output"], attention_mask_b)
        return F.cosine_similarity(emb_a, emb_b)  # (B,)

    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        return self.loss_fn(logits, labels)
