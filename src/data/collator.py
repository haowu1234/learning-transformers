"""Generic data collator for padding and batching."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DataCollator:
    """Pads input_ids, attention_mask, and labels to the longest sequence in a batch.

    Expects each sample to be a dict with at least:
        - input_ids: list[int]
        - attention_mask: list[int]
        - labels: int or list[int]

    Optional:
        - token_type_ids: list[int]
    """

    pad_token_id: int = 0
    pad_to_max_length: bool = False
    max_length: int = 128

    def __call__(self, batch: list[dict]) -> dict[str, Tensor]:
        if self.pad_to_max_length:
            max_len = self.max_length
        else:
            max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        has_token_type_ids = "token_type_ids" in batch[0]

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)

            if has_token_type_ids:
                token_type_ids.append(item["token_type_ids"] + [0] * pad_len)

            labels.append(item["labels"])

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if has_token_type_ids:
            result["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)

        # labels: scalar (classification) or sequence (token classification)
        if isinstance(labels[0], list):
            # Pad token-level labels with -100
            padded_labels = []
            for lbl in labels:
                pad_len = max_len - len(lbl)
                padded_labels.append(lbl + [-100] * pad_len)
            result["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        else:
            result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result
