"""Tests for attention mechanism."""

import torch
from src.models.attention import scaled_dot_product_attention, MultiHeadAttention


def test_scaled_dot_product_attention_shape():
    batch, heads, seq_len, d_k = 2, 4, 8, 64
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch, heads, seq_len, d_k)
    assert weights.shape == (batch, heads, seq_len, seq_len)


def test_scaled_dot_product_attention_with_mask():
    batch, heads, seq_len, d_k = 2, 4, 8, 64
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)
    mask = torch.ones(batch, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0  # mask last 2 positions

    output, weights = scaled_dot_product_attention(q, k, v, mask)

    # Attention weights for masked positions should be ~0
    assert weights[:, :, :, -2:].max().item() < 1e-6


def test_multi_head_attention_shape():
    batch, seq_len, hidden_size, num_heads = 2, 10, 768, 12
    mha = MultiHeadAttention(hidden_size, num_heads, dropout_prob=0.0)

    x = torch.randn(batch, seq_len, hidden_size)
    mask = torch.ones(batch, seq_len)

    output = mha(x, mask)
    assert output.shape == (batch, seq_len, hidden_size)


def test_multi_head_attention_no_mask():
    batch, seq_len, hidden_size, num_heads = 2, 10, 768, 12
    mha = MultiHeadAttention(hidden_size, num_heads, dropout_prob=0.0)

    x = torch.randn(batch, seq_len, hidden_size)
    output = mha(x)
    assert output.shape == (batch, seq_len, hidden_size)


if __name__ == "__main__":
    test_scaled_dot_product_attention_shape()
    test_scaled_dot_product_attention_with_mask()
    test_multi_head_attention_shape()
    test_multi_head_attention_no_mask()
    print("All attention tests passed!")
