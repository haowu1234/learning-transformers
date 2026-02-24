"""Bi-Encoder: shared BERT backbone encodes two sentences independently."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .bert import BertModel


class BertBiEncoder(nn.Module):
    """Shared-backbone Bi-Encoder for sentence-pair tasks (e.g. similarity).

    Two sentences are encoded independently by the same BERT backbone,
    then the head computes a score (e.g. cosine similarity).
    """

    def __init__(self, backbone: BertModel, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids_a: Tensor,
        attention_mask_a: Tensor,
        token_type_ids_a: Tensor | None = None,
        input_ids_b: Tensor | None = None,
        attention_mask_b: Tensor | None = None,
        token_type_ids_b: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor | None]:
        out_a = self.backbone(input_ids_a, attention_mask_a, token_type_ids_a)
        out_b = self.backbone(input_ids_b, attention_mask_b, token_type_ids_b)

        cos_sim = self.head(out_a, out_b, attention_mask_a, attention_mask_b)

        loss = None
        if labels is not None:
            loss = self.head.compute_loss(cos_sim, labels)

        return {"loss": loss, "logits": cos_sim}
