"""Scaled Dot-Product Attention and Multi-Head Attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute Scaled Dot-Product Attention.

    Args:
        query: (batch, heads, seq_len, d_k)
        key:   (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_k)
        mask:  (batch, 1, 1, seq_len) — 1 for valid, 0 for pad
        dropout: optional dropout on attention weights

    Returns:
        output:  (batch, heads, seq_len, d_k)
        weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    Splits hidden_size into num_heads parallel attention heads,
    applies scaled dot-product attention, then concatenates and projects.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            hidden_states:  (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) — 1 for valid, 0 for pad

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Linear projections → (batch, seq_len, hidden_size)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape → (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand mask: (batch, seq_len) → (batch, 1, 1, seq_len)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(q, k, v, attention_mask, self.dropout)

        # Concat heads → (batch, seq_len, hidden_size)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )

        # Final linear projection
        return self.output(attn_output)
