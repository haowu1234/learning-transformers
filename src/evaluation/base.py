"""Abstract base class for evaluation metrics."""

from abc import ABC, abstractmethod

from torch import Tensor


class BaseMetric(ABC):
    """Base class that every metric must inherit from.

    Metrics accumulate predictions over batches, then compute final values.
    """

    @abstractmethod
    def update(self, preds: Tensor, labels: Tensor) -> None:
        """Accumulate a batch of predictions and labels."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric values from all accumulated data.

        Returns:
            dict mapping metric names to float values.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new evaluation run."""
