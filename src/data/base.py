"""Abstract base class for data modules."""

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataModule(ABC):
    """Base class for all dataset modules.

    Encapsulates downloading, preprocessing, tokenization, and DataLoader creation.
    """

    @abstractmethod
    def setup(self) -> None:
        """Download / load raw data and run preprocessing."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""

    @abstractmethod
    def get_labels(self) -> list[str]:
        """Return the ordered list of label names.

        len(get_labels()) is used to infer num_labels for the task head.
        """

    @property
    def num_labels(self) -> int:
        return len(self.get_labels())
