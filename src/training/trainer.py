"""Unified Trainer: train loop, evaluation, prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.base import BaseMetric
from .callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    log_interval: int = 50

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def _get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay scheduler."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Task-agnostic trainer.

    Handles:
      - AdamW optimizer with weight decay (skip bias / LayerNorm)
      - Linear warmup + decay scheduler
      - Gradient clipping
      - Evaluation after each epoch
      - Callbacks (EarlyStopping, ModelCheckpoint)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        early_stopping: EarlyStopping | None = None,
        checkpoint: ModelCheckpoint | None = None,
    ):
        self.config = config
        self.device = config.resolve_device()
        self.model = model.to(self.device)
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint

        logger.info(f"Using device: {self.device}")

    def _build_optimizer(self):
        """Build AdamW with weight-decay exclusion for bias and LayerNorm."""
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"}
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.config.learning_rate)

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics: list[BaseMetric],
    ) -> dict:
        """Full training loop.

        Returns:
            dict with final evaluation results.
        """
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        optimizer = self._build_optimizer()
        scheduler = _get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        best_results = {}

        for epoch in range(1, self.config.num_epochs + 1):
            # --- Train ---
            self.model.train()
            total_loss = 0.0
            num_steps = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]")
            for step, batch in enumerate(pbar, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels=batch["labels"],
                )
                loss = outputs["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                num_steps += 1

                if step % self.config.log_interval == 0:
                    avg_loss = total_loss / num_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            avg_loss = total_loss / num_steps
            logger.info(f"Epoch {epoch} — avg train loss: {avg_loss:.4f}")

            # --- Evaluate ---
            results = self.evaluate(val_dataloader, metrics)
            results_str = ", ".join(f"{k}: {v:.4f}" for k, v in results.items())
            logger.info(f"Epoch {epoch} — eval: {results_str}")

            # --- Callbacks ---
            monitor_value = results.get("accuracy", -results.get("loss", 0))

            if self.checkpoint:
                self.checkpoint(monitor_value, self.model, epoch)

            if self.early_stopping:
                if self.early_stopping(monitor_value):
                    logger.info("Early stopping triggered. Stopping training.")
                    break

            best_results = results

        return best_results

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, metrics: list[BaseMetric]) -> dict:
        """Run evaluation on a dataloader.

        Returns:
            dict of metric name → value.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for m in metrics:
            m.reset()

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

            preds = outputs["logits"].argmax(dim=-1)
            labels = batch["labels"]

            for m in metrics:
                m.update(preds.cpu(), labels.cpu())

        results = {"loss": total_loss / max(num_batches, 1)}
        for m in metrics:
            results.update(m.compute())

        self.model.train()
        return results

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor]:
        """Run prediction, return list of logit tensors."""
        self.model.eval()
        all_logits = []

        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            all_logits.append(outputs["logits"].cpu())

        return all_logits
