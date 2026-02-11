"""Transformer Encoder Layer and Encoder stack."""

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network: Linear → GELU → Linear → Dropout."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.dense2(self.activation(self.dense1(x))))


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer.

    Structure:
        x → MultiHeadAttention → Residual + LayerNorm
          → FeedForward        → Residual + LayerNorm
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)

        self.ffn = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Self-attention + residual + norm
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.attention_dropout(attn_output))

        # FFN + residual + norm
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_output)

        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of N Transformer encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
