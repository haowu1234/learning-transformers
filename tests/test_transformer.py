"""Tests for transformer encoder and BERT model."""

import torch
from src.models.embedding import BertEmbedding
from src.models.encoder import TransformerEncoderLayer, TransformerEncoder
from src.models.bert import BertConfig, BertModel, BertForTask
from src.heads.classification import ClassificationHead


def test_bert_embedding_shape():
    emb = BertEmbedding(vocab_size=1000, hidden_size=128, max_position_embeddings=64)
    input_ids = torch.randint(0, 1000, (2, 16))
    output = emb(input_ids)
    assert output.shape == (2, 16, 128)


def test_encoder_layer_shape():
    layer = TransformerEncoderLayer(
        hidden_size=128, num_attention_heads=4, intermediate_size=512,
    )
    x = torch.randn(2, 16, 128)
    mask = torch.ones(2, 16)
    output = layer(x, mask)
    assert output.shape == (2, 16, 128)


def test_transformer_encoder_shape():
    encoder = TransformerEncoder(
        num_layers=2, hidden_size=128, num_attention_heads=4, intermediate_size=512,
    )
    x = torch.randn(2, 16, 128)
    output = encoder(x)
    assert output.shape == (2, 16, 128)


def test_bert_model_outputs():
    config = BertConfig(
        vocab_size=1000, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512, max_position_embeddings=64,
    )
    model = BertModel(config)
    input_ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    outputs = model(input_ids, mask)
    assert outputs["sequence_output"].shape == (2, 16, 128)
    assert outputs["pooled_output"].shape == (2, 128)


def test_bert_for_task_classification():
    config = BertConfig(
        vocab_size=1000, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512, max_position_embeddings=64,
    )
    backbone = BertModel(config)
    head = ClassificationHead(hidden_size=128, num_labels=2)
    model = BertForTask(backbone, head)

    input_ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.tensor([0, 1])

    outputs = model(input_ids, mask, labels=labels)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0  # scalar


if __name__ == "__main__":
    test_bert_embedding_shape()
    test_encoder_layer_shape()
    test_transformer_encoder_shape()
    test_bert_model_outputs()
    test_bert_for_task_classification()
    print("All transformer tests passed!")
