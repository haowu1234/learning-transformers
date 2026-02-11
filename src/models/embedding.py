"""BERT Embedding: Token + Position + Segment embeddings."""

import torch
import torch.nn as nn
from torch import Tensor


class BertEmbedding(nn.Module):
    """Construct embeddings from token, position, and segment embeddings.

    This matches the original BERT implementation:
        Embedding = TokenEmbed + PositionEmbed + SegmentEmbed
        → LayerNorm → Dropout
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Register position_ids as buffer (not a parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).unsqueeze(0),  # (1, max_pos)
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids:     (batch, seq_len)
            token_type_ids: (batch, seq_len) — 0 for sentence A, 1 for sentence B

        Returns:
            (batch, seq_len, hidden_size)
        """
        seq_len = input_ids.size(1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.position_ids[:, :seq_len]

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
