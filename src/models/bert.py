"""BERT model: backbone + generic task wrapper + pretrained weight loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .embedding import BertEmbedding
from .encoder import TransformerEncoder

logger = logging.getLogger(__name__)


@dataclass
class BertConfig:
    """Configuration for BertModel, mirrors bert-base-uncased defaults."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    @classmethod
    def from_dict(cls, d: dict) -> "BertConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BertPooler(nn.Module):
    """Take the [CLS] token hidden state → dense + tanh."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        cls_hidden = hidden_states[:, 0]  # (batch, hidden_size)
        return self.activation(self.dense(cls_hidden))


class BertModel(nn.Module):
    """Pure BERT backbone — embeddings + encoder + pooler.

    Returns both sequence_output and pooled_output.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.pooler = BertPooler(config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Returns:
            {
                "sequence_output": (batch, seq_len, hidden_size),
                "pooled_output":   (batch, hidden_size)
            }
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(sequence_output)

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }


class BertForTask(nn.Module):
    """Generic wrapper: BertModel backbone + any pluggable task head.

    The head must implement:
        forward(backbone_outputs: dict, attention_mask, **kwargs) -> Tensor
        compute_loss(logits, labels, **kwargs) -> Tensor
    """

    def __init__(self, backbone: BertModel, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor | None]:
        backbone_outputs = self.backbone(input_ids, attention_mask, token_type_ids)
        logits = self.head(backbone_outputs, attention_mask=attention_mask, **kwargs)

        loss = None
        if labels is not None:
            loss = self.head.compute_loss(logits, labels, **kwargs)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

# Mapping from our parameter names → HuggingFace bert-base-uncased names
_PARAM_MAPPING = {
    # Embeddings
    "embeddings.token_embeddings.weight": "bert.embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
    "embeddings.segment_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
    "embeddings.layer_norm.weight": "bert.embeddings.LayerNorm.weight",
    "embeddings.layer_norm.bias": "bert.embeddings.LayerNorm.bias",
    # Pooler
    "pooler.dense.weight": "bert.pooler.dense.weight",
    "pooler.dense.bias": "bert.pooler.dense.bias",
}


def _build_encoder_mapping(num_layers: int) -> dict[str, str]:
    """Build parameter name mapping for encoder layers."""
    mapping = {}
    for i in range(num_layers):
        ours = f"encoder.layers.{i}"
        hf = f"bert.encoder.layer.{i}"

        # Self-attention Q/K/V/O
        for proj in ("query", "key", "value"):
            mapping[f"{ours}.attention.{proj}.weight"] = f"{hf}.attention.self.{proj}.weight"
            mapping[f"{ours}.attention.{proj}.bias"] = f"{hf}.attention.self.{proj}.bias"
        mapping[f"{ours}.attention.output.weight"] = f"{hf}.attention.output.dense.weight"
        mapping[f"{ours}.attention.output.bias"] = f"{hf}.attention.output.dense.bias"

        # Attention LayerNorm
        mapping[f"{ours}.attention_norm.weight"] = f"{hf}.attention.output.LayerNorm.weight"
        mapping[f"{ours}.attention_norm.bias"] = f"{hf}.attention.output.LayerNorm.bias"

        # FFN
        mapping[f"{ours}.ffn.dense1.weight"] = f"{hf}.intermediate.dense.weight"
        mapping[f"{ours}.ffn.dense1.bias"] = f"{hf}.intermediate.dense.bias"
        mapping[f"{ours}.ffn.dense2.weight"] = f"{hf}.output.dense.weight"
        mapping[f"{ours}.ffn.dense2.bias"] = f"{hf}.output.dense.bias"

        # FFN LayerNorm
        mapping[f"{ours}.ffn_norm.weight"] = f"{hf}.output.LayerNorm.weight"
        mapping[f"{ours}.ffn_norm.bias"] = f"{hf}.output.LayerNorm.bias"

    return mapping


def load_pretrained_weights(model: BertModel, pretrained_model_name: str = "bert-base-uncased"):
    """Load weights from a HuggingFace pretrained BERT model into our BertModel.

    Requires `transformers` library to be installed (used only for downloading weights).
    """
    try:
        from transformers import BertModel as HFBertModel
    except ImportError:
        raise ImportError(
            "transformers library is required to load pretrained weights. "
            "Install it with: pip install transformers"
        )

    logger.info(f"Loading pretrained weights from '{pretrained_model_name}'...")
    hf_model = HFBertModel.from_pretrained(pretrained_model_name)
    hf_state = hf_model.state_dict()

    # Build full mapping
    mapping = {**_PARAM_MAPPING}
    mapping.update(_build_encoder_mapping(model.config.num_hidden_layers))

    # Reverse: our_name → hf_name  →  load hf weights into our model
    our_state = model.state_dict()
    new_state = {}
    missing, unexpected = [], []

    for our_name, param in our_state.items():
        hf_name = mapping.get(our_name)
        if hf_name is None or hf_name not in hf_state:
            missing.append(our_name)
            new_state[our_name] = param  # keep init value
            continue

        hf_param = hf_state[hf_name]
        if param.shape != hf_param.shape:
            logger.warning(
                f"Shape mismatch for {our_name}: ours={param.shape}, hf={hf_param.shape}. Skipping."
            )
            new_state[our_name] = param
            continue

        new_state[our_name] = hf_param

    model.load_state_dict(new_state, strict=False)

    if missing:
        logger.info(f"Parameters not found in pretrained: {missing}")

    # Handle buffers like position_ids
    logger.info("Pretrained weights loaded successfully.")
    return model
