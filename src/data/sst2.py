"""SST-2 dataset module for sentiment classification."""

from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .collator import DataCollator
from ..utils.registry import Registry

DATASET_REGISTRY = Registry("datasets")


class SST2Dataset(Dataset):
    """Thin wrapper around tokenized SST-2 examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@DATASET_REGISTRY.register("sst2")
class SST2DataModule(BaseDataModule):
    """SST-2 binary sentiment classification.

    Downloads via HuggingFace `datasets` and tokenizes with `tokenizers`.
    """

    LABELS = ["negative", "positive"]

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
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def setup(self) -> None:
        from datasets import load_dataset

        raw = load_dataset("glue", "sst2")
        self.train_dataset = SST2Dataset(self._tokenize(raw["train"]))
        self.val_dataset = SST2Dataset(self._tokenize(raw["validation"]))

    def _tokenize(self, split) -> list[dict]:
        examples = []
        for item in split:
            enc = self.tokenizer(
                item["sentence"],
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,  # collator handles padding
            )
            examples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "labels": item["label"],
            })
        return examples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def get_labels(self) -> list[str]:
        return self.LABELS
