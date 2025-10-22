import numpy as np
from typing import List
from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from .dataset import BaseDataset


class RandomSplitter(BaseDatasetSplitter):
    """
    Randomly splits dataset across clients with optional validation sets.

    For 'local' validation: Each client gets a random train set from the original train set and a random val set from the original val set
    For 'global' validation: The global validation set is taken from the original val set
    For 'none': No validation sets are created
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset | None,
        n_clients: int,
        heterogeneity: float = 0.0,
        valset_type: str = "global",
        seed: int | None = None,
    ):

        super().__init__(train_dataset, val_dataset, n_clients, heterogeneity, valset_type, seed)

    def _compute_client_splits(self) -> DatasetSplits:
        """
        Perform random splitting and return the complete split structure.

        Returns:
            DatasetSplits dictionary with client splits and optional global validation
        """

        # Seed already set in base class init

        splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

        train_indices = np.arange(len(self.train_dataset))
        np.random.shuffle(train_indices)
        client_train_indices = np.array_split(train_indices, self.n_clients)

        if self.valset_type == "global":
            return self._split_with_global_val(client_train_indices, splits)
        elif self.valset_type == "local":
            return self._split_with_local_val(client_train_indices, splits)
        else:  # valset_type == "none"
            return self._split_without_val(client_train_indices, splits)

    def _split_with_global_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:

        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for global validation splitting.")

        splits["global_val"] = self._load_dataset_samples(self.val_dataset)

        for i, indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, indices.tolist())

            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])

        return splits

    def _split_with_local_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for local validation splitting.")

        val_indices = np.arange(len(self.val_dataset))
        np.random.shuffle(val_indices)
        client_val_indices = np.array_split(val_indices, self.n_clients)

        for i, (train_indices, val_indices) in enumerate(zip(client_train_indices, client_val_indices)):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            client_val_data = self._load_dataset_samples(self.val_dataset, val_indices.tolist())

            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=client_val_data)

        return splits

    def _split_without_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        for i, train_indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])

        return splits
