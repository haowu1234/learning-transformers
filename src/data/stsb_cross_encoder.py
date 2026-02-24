"""STS-B dataset module for Cross-Encoder: sentence pair as single input."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .sst2 import DATASET_REGISTRY
from .collator import DataCollator


class STSBCrossEncoderDataset(Dataset):
    """Thin wrapper around tokenized STS-B cross-encoder examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@DATASET_REGISTRY.register("stsb_cross_encoder")
class STSBCrossEncoderDataModule(BaseDataModule):
    """STS-B for Cross-Encoder: concatenate sentence pair into single input.

    Input format: [CLS] sentence1 [SEP] sentence2 [SEP]
    Label range 0~5 is linearly mapped to [0, 1].
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
        self.train_dataset = STSBCrossEncoderDataset(self._tokenize(raw["train"]))
        self.val_dataset = STSBCrossEncoderDataset(self._tokenize(raw["validation"]))
        if "test" in raw:
            self.test_dataset = STSBCrossEncoderDataset(self._tokenize(raw["test"]))

    def _tokenize(self, split) -> list[dict]:
        examples = []
        for item in split:
            enc = self.tokenizer(
                item["sentence1"],
                item["sentence2"],
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )
            # Map 0~5 → [0, 1]
            score = item["label"] / 5.0

            examples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "labels": score,
            })
        return examples

    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                label_dtype="float",
            ),
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
