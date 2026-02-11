"""Training callbacks: EarlyStopping, ModelCheckpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of evaluations with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: "min" for loss, "max" for accuracy.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"EarlyStopping triggered after {self.patience} evaluations without improvement.")

        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoint when a monitored metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        mode: "min" for loss, "max" for accuracy.
    """

    def __init__(self, save_dir: str = "checkpoints", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def __call__(self, value: float, model: torch.nn.Module, epoch: int) -> bool:
        improved = (
            value > self.best_value
            if self.mode == "max"
            else value < self.best_value
        )

        if improved:
            self.best_value = value
            path = self.save_dir / "best_model.pt"
            torch.save(model.state_dict(), path)
            logger.info(f"Epoch {epoch}: Saved best model ({value:.4f}) → {path}")
            return True
        return False
