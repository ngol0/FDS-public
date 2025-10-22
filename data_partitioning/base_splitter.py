from dataset import BaseDataset
from typing import Dict, List, Optional, TypedDict, Any
import numpy as np
import torch
from abc import ABC, abstractmethod
import json
import pickle


class ClientSplit(TypedDict):
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]


class DatasetSplits(TypedDict):
    clients: Dict[str, ClientSplit]
    global_val: Optional[List[Dict[str, Any]]]


class BaseDatasetSplitter(ABC):
    """
    Abstract base class for splitting datasets across multiple clients.

    Subclasses should implement specific splitting strategies while maintaining
    a consistent interface for client data distribution.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset],
        n_clients: int,
        heterogeneity: float,
        valset_type: str,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset: The BaseDataset instance to split
            n_clients: Number of clients to split data across
            heterogeneity: Degree of data heterogeneity (0=homogeneous, 1=heterogeneous)
            valset_type: Type of validation set - "local", "global", or "none"
            seed: Random seed for reproducibility
        """

        valset_type_valid_options = ["local", "global", "none"]
        if valset_type not in valset_type_valid_options:
            raise ValueError(f"Invalid valset_type '{valset_type}'. Valid options: {valset_type_valid_options}")

        if valset_type in ["local", "global"] and val_dataset is None:
            raise ValueError(f"val_dataset must be provided when valset_type is '{valset_type}'")

        if valset_type == "none" and val_dataset is not None:
            print("Warning: val_dataset provided but valset_type is 'none'. The val_dataset will be ignored.")

        if not (0.0 <= heterogeneity <= 1.0):
            raise ValueError(f"Heterogeneity must be between 0.0 and 1.0, got {heterogeneity}")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_clients = n_clients
        self.heterogeneity = heterogeneity
        self.valset_type = valset_type
        self.seed = seed
        self.splits: Optional[DatasetSplits] = None

        self._set_random_seed(seed)

    def _set_random_seed(self, seed: Optional[int]) -> None:
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    @abstractmethod
    def _compute_client_splits(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_client_splits(self) -> DatasetSplits:
        if self.splits is None:
            self._compute_client_splits()
        return self.splits  # type: ignore

    def store_client_splits(self, filename: str, format: str = "json") -> None:
        """
        Store the client data splits to a file.

        Args:
            filename: The name of the file to store the splits
            format: The format to store the splits in ("json" or "csv")
        """
        splits = self.get_client_splits()

        if format == "json":
            with open(filename, "w") as f:
                serializable_splits = self._make_serializable(splits)
                json.dump(serializable_splits, f, indent=2)
        elif format == "pkl":
            with open(filename, "wb") as f:
                pickle.dump(splits, f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'.")

    def _make_serializable(self, splits: DatasetSplits) -> DatasetSplits:
        """Convert any non-serializable objects in splits to serializable forms."""
        serializable_splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

        if splits["global_val"] is not None:
            serializable_splits["global_val"] = [
                {"image": str(sample["image"]), "label": sample["label"]} for sample in splits["global_val"]
            ]

        for client_id, client_split in splits["clients"].items():
            serializable_train = [{"image": str(sample["image"]), "label": sample["label"]} for sample in client_split["train"]]
            serializable_val = [{"image": str(sample["image"]), "label": sample["label"]} for sample in client_split["val"]]
            serializable_splits["clients"][client_id] = ClientSplit(
                train=serializable_train,
                val=serializable_val,
            )

        return serializable_splits

    def _load_dataset_samples(self, dataset: BaseDataset, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Load samples from the dataset. If indices are provided, load only those samples."""
        if indices is not None:
            return [{"image": str(dataset[idx][0]), "label": dataset[idx][1]} for idx in indices]
        else:
            return [{"image": str(dataset[idx][0]), "label": dataset[idx][1]} for idx in range(len(dataset))]

    def load_client_splits(self, filename: str, format: str = "json") -> None:
        if format == "json":
            with open(filename, "r") as f:
                self.splits = json.load(f)
        elif format == "pkl":
            with open(filename, "rb") as f:
                self.splits = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'.")

    def set_heterogeneity(self, heterogeneity: float) -> None:
        if not (0.0 <= heterogeneity <= 1.0):
            raise ValueError(f"Heterogeneity must be between 0.0 and 1.0, got {heterogeneity}")
        self.heterogeneity = heterogeneity
        self.splits = None  # Invalidate existing splits

    def set_seed(self, seed: int) -> None:
        self._set_random_seed(seed)
        self.splits = None  # Invalidate existing splits

    def get_split_summary(self) -> Dict[str, Any]:
        splits = self.get_client_splits()
        client_sizes = {}
        total_train = 0
        total_val = 0

        for client_id, split in splits["clients"].items():
            n_train = len(split["train"])
            n_val = len(split["val"])
            client_sizes[client_id] = {"train_samples": n_train, "val_samples": n_val}
            total_train += n_train
            total_val += n_val

        summary = {
            "n_clients": self.n_clients,
            "heterogeneity": self.heterogeneity,
            "valset_type": self.valset_type,
            "total_train_samples": total_train,
            "total_val_samples": total_val,
            "client_sizes": client_sizes,
            "seed": self.seed,
        }

        if self.valset_type == "global" and splits["global_val"] is not None:
            summary["global_val_samples"] = len(splits["global_val"])

        return summary

    def validate_splits(self) -> bool:
        """Validate that the splits cover the entire dataset without overlap."""
        splits = self.get_client_splits()
        if len(splits["clients"]) != self.n_clients:
            raise ValueError(f"Number of clients in splits ({len(splits['clients'])}) does not match n_clients ({self.n_clients})")
        
        all_train_paths = set()
        for client_id, split in splits["clients"].items():
            client_train_paths = {sample["image"] for sample in split["train"]}
            if all_train_paths.intersection(client_train_paths):
                raise ValueError(f"Overlap detected in training data for client {client_id}")
            all_train_paths.update(client_train_paths)
        expected_train_paths = {str(self.train_dataset[idx][0]) for idx in range(len(self.train_dataset))}
        if all_train_paths != expected_train_paths:
            missing = expected_train_paths - all_train_paths
            extra = all_train_paths - expected_train_paths
            raise ValueError(f"Training data splits do not cover the entire dataset. Missing: {missing}, Extra: {extra}")
        
        
        all_val_paths = set()
        if self.valset_type == "global" and splits["global_val"] is not None and self.val_dataset is not None:
            all_val_paths = {sample["image"] for sample in splits["global_val"]}
            expected_val_paths = {str(self.val_dataset[idx][0]) for idx in range(len(self.val_dataset))}
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Global validation data does not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        elif self.valset_type == "local" and self.val_dataset is not None:
            for client_id, split in splits["clients"].items():
                client_val_paths = {sample["image"] for sample in split["val"]}
                if all_val_paths.intersection(client_val_paths):
                    raise ValueError(f"Overlap detected in validation data for client {client_id}")
                all_val_paths.update(client_val_paths)
            expected_val_paths = {str(self.val_dataset[idx][0]) for idx in range(len(self.val_dataset))}
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Local validation data splits do not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        
        print("All splits are valid.")
        return True
