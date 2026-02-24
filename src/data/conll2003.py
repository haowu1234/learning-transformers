"""CoNLL-2003 dataset module for Named Entity Recognition."""

from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .collator import DataCollator
from .sst2 import DATASET_REGISTRY


class CoNLL2003Dataset(Dataset):
    """Thin wrapper around tokenized CoNLL-2003 examples."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@DATASET_REGISTRY.register("conll2003")
class CoNLL2003DataModule(BaseDataModule):
    """CoNLL-2003 Named Entity Recognition.

    Downloads via HuggingFace `datasets` and tokenizes with subword-label alignment.
    """

    LABELS = [
        "O",
        "B-ORG", "B-MISC", "B-PER", "I-PER",
        "B-LOC", "I-ORG", "I-MISC", "I-LOC",
    ]

    def __init__(
        self,
        pretrained_model_name: str = "bert-base-cased",
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

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def setup(self) -> None:
        from datasets import load_dataset

        raw = load_dataset("tner/conll2003")
        self.train_dataset = CoNLL2003Dataset(self._tokenize(raw["train"]))
        self.val_dataset = CoNLL2003Dataset(self._tokenize(raw["validation"]))
        self.test_dataset = CoNLL2003Dataset(self._tokenize(raw["test"]))

    def _align_labels(self, word_ids: list[int | None], word_labels: list[int]) -> list[int]:
        """Align word-level NER labels to subword tokens.

        Strategy:
        - [CLS], [SEP], padding (word_id=None) → -100
        - First subword of each word → original label
        - Subsequent subwords of same word → -100
        """
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(word_labels[word_id])
            else:
                aligned.append(-100)
            prev_word_id = word_id
        return aligned

    def _tokenize(self, split) -> list[dict]:
        examples = []
        for item in split:
            tokens = item["tokens"]
            ner_tags = item["tags"]

            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )

            aligned_labels = self._align_labels(enc.word_ids(), ner_tags)

            examples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "labels": aligned_labels,
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
        return self.LABELS
