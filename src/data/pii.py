"""PII detection dataset module using ai4privacy/pii-masking-400k."""

import logging

from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .collator import DataCollator
from .sst2 import DATASET_REGISTRY

logger = logging.getLogger(__name__)


class PIIDataset(Dataset):
    """Thin wrapper around tokenized PII examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@DATASET_REGISTRY.register("pii")
class PIIDataModule(BaseDataModule):
    """PII detection on ai4privacy/pii-masking-400k.

    Uses the pre-tokenized `mbert_tokens` and `mbert_token_classes` fields,
    which are already aligned at the mBERT subword level (no manual alignment needed).
    """

    def __init__(
        self,
        pretrained_model_name: str = "bert-base-multilingual-cased",
        max_seq_length: int = 128,
        batch_size: int = 16,
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
        self._labels: list[str] | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def setup(self) -> None:
        from datasets import load_dataset

        raw = load_dataset("ai4privacy/pii-masking-400k")

        # Dynamically collect all unique BIO labels from the dataset
        self._labels = self._collect_labels(raw)
        logger.info(f"PII labels ({len(self._labels)}): {self._labels}")

        label2id = {label: i for i, label in enumerate(self._labels)}

        self.train_dataset = PIIDataset(self._tokenize(raw["train"], label2id))
        self.val_dataset = PIIDataset(self._tokenize(raw["validation"], label2id))
        # No test split — reuse validation
        self.test_dataset = self.val_dataset

    @staticmethod
    def _collect_labels(raw) -> list[str]:
        """Scan all splits to collect the complete set of BIO labels.

        Returns a sorted list with "O" always at index 0.
        """
        tag_set: set[str] = set()
        for split_name in raw:
            for item in raw[split_name]:
                tag_set.update(item["mbert_token_classes"])

        tag_set.discard("O")
        # Sort so B- comes before I- for each entity type
        sorted_tags = sorted(tag_set)
        return ["O"] + sorted_tags

    def _tokenize(self, split, label2id: dict[str, int]) -> list[dict]:
        examples = []
        max_tokens = self.max_seq_length - 2  # room for [CLS] and [SEP]
        skipped = 0

        for item in split:
            tokens = item["mbert_tokens"]
            tag_strs = item["mbert_token_classes"]

            if not tokens:
                skipped += 1
                continue

            # Truncate
            tokens = tokens[:max_tokens]
            tag_strs = tag_strs[:max_tokens]

            # Convert tokens → input_ids (with [CLS] and [SEP])
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]

            # Convert label strings → label ids (with -100 for special tokens)
            labels = [-100] + [label2id.get(tag, 0) for tag in tag_strs] + [-100]

            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            })

        if skipped:
            logger.warning(f"Skipped {skipped} empty samples")
        logger.info(f"Tokenized {len(examples)} examples")
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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def get_labels(self) -> list[str]:
        if self._labels is None:
            raise RuntimeError("Call setup() before get_labels()")
        return self._labels
