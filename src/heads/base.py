"""Abstract base class for all task heads."""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseHead(nn.Module, ABC):
    """Base class that every task head must inherit from.

    A head receives the backbone outputs dict and produces task-specific logits.
    It also defines how to compute the loss for that task.
    """

    @abstractmethod
    def forward(
        self,
        backbone_outputs: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Produce logits from backbone outputs.

        Args:
            backbone_outputs: dict with "sequence_output" and "pooled_output"
            attention_mask: (batch, seq_len)

        Returns:
            logits tensor (shape depends on task)
        """

    @abstractmethod
    def compute_loss(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        """Compute task-specific loss.

        Args:
            logits: output of forward()
            labels: ground-truth labels

        Returns:
            scalar loss tensor
        """
