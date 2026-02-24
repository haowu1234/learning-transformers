"""STS-B dataset module for sentence-pair semantic similarity (Bi-Encoder)."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .sst2 import DATASET_REGISTRY


class STSBDataset(Dataset):
    """Thin wrapper around tokenized STS-B examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class PairDataCollator:
    """Pads sentence-pair inputs (A and B fields) independently.

    Each sample has:
        input_ids_a, attention_mask_a, token_type_ids_a,
        input_ids_b, attention_mask_b, token_type_ids_b,
        labels (float scalar)
    """

    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict[str, Tensor]:
        max_len_a = max(len(item["input_ids_a"]) for item in batch)
        max_len_b = max(len(item["input_ids_b"]) for item in batch)

        result = {
            "input_ids_a": [], "attention_mask_a": [], "token_type_ids_a": [],
            "input_ids_b": [], "attention_mask_b": [], "token_type_ids_b": [],
            "labels": [],
        }

        for item in batch:
            # Pad A
            pad_a = max_len_a - len(item["input_ids_a"])
            result["input_ids_a"].append(item["input_ids_a"] + [self.pad_token_id] * pad_a)
            result["attention_mask_a"].append(item["attention_mask_a"] + [0] * pad_a)
            result["token_type_ids_a"].append(item["token_type_ids_a"] + [0] * pad_a)

            # Pad B
            pad_b = max_len_b - len(item["input_ids_b"])
            result["input_ids_b"].append(item["input_ids_b"] + [self.pad_token_id] * pad_b)
            result["attention_mask_b"].append(item["attention_mask_b"] + [0] * pad_b)
            result["token_type_ids_b"].append(item["token_type_ids_b"] + [0] * pad_b)

            result["labels"].append(item["labels"])

        return {
            "input_ids_a": torch.tensor(result["input_ids_a"], dtype=torch.long),
            "attention_mask_a": torch.tensor(result["attention_mask_a"], dtype=torch.long),
            "token_type_ids_a": torch.tensor(result["token_type_ids_a"], dtype=torch.long),
            "input_ids_b": torch.tensor(result["input_ids_b"], dtype=torch.long),
            "attention_mask_b": torch.tensor(result["attention_mask_b"], dtype=torch.long),
            "token_type_ids_b": torch.tensor(result["token_type_ids_b"], dtype=torch.long),
            "labels": torch.tensor(result["labels"], dtype=torch.float),
        }


@DATASET_REGISTRY.register("stsb")
class STSBDataModule(BaseDataModule):
    """STS-B semantic textual similarity (GLUE subtask).

    Label range 0~5 is linearly mapped to [-1, 1] to align with cosine similarity.
    """

    LABELS = []  # regression task, no discrete labels

    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        max_seq_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def setup(self) -> None:
        from datasets import load_dataset

        raw = load_dataset("glue", "stsb")
        self.train_dataset = STSBDataset(self._tokenize(raw["train"]))
        self.val_dataset = STSBDataset(self._tokenize(raw["validation"]))
        if "test" in raw:
            self.test_dataset = STSBDataset(self._tokenize(raw["test"]))

    def _tokenize(self, split) -> list[dict]:
        examples = []
        for item in split:
            enc_a = self.tokenizer(
                item["sentence1"],
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )
            enc_b = self.tokenizer(
                item["sentence2"],
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )
            # Map 0~5 → [-1, 1] to align with cosine similarity range
            score = item["label"] * 2.0 / 5.0 - 1.0

            examples.append({
                "input_ids_a": enc_a["input_ids"],
                "attention_mask_a": enc_a["attention_mask"],
                "token_type_ids_a": enc_a.get("token_type_ids", [0] * len(enc_a["input_ids"])),
                "input_ids_b": enc_b["input_ids"],
                "attention_mask_b": enc_b["attention_mask"],
                "token_type_ids_b": enc_b.get("token_type_ids", [0] * len(enc_b["input_ids"])),
                "labels": score,
            })
        return examples

    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PairDataCollator(pad_token_id=self.tokenizer.pad_token_id),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("STS-B test split has no labels; use validation for evaluation.")
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def get_labels(self) -> list[str]:
        return self.LABELS
