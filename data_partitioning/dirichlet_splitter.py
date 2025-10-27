import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
from scipy.stats import entropy

from base_splitter import DatasetSplits, ClientSplit
from dataset import BaseDataset
from random_splitter import RandomSplitter


class DirichletSplitter(RandomSplitter):
    """
    Splits dataset across clients using a Dirichlet distribution to control heterogeneity.

    For 'local' validation: Each client gets a Dirichlet-sampled train set from the original train set and a Dirichlet-sampled val set from the original val set
    For 'global' validation: The global validation set is taken from the original val set
    For 'none': No validation sets are created
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset | None,
        n_clients: int,
        heterogeneity: float = 0.5,
        valset_type: str = "global",
        seed: int | None = None,
        min_samples_per_client: int = 10,
        max_retries: int = 1000,
    ):
        self.min_samples_per_client = min_samples_per_client
        self.max_retries = max_retries
        super().__init__(train_dataset, val_dataset, n_clients, heterogeneity, valset_type, seed)

    def _compute_client_splits(self) -> DatasetSplits:
        splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

        train_labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        unique_labels = sorted(set(train_labels))

        client_train_indices = self._partition_with_retry(train_labels, unique_labels, len(self.train_dataset), "train")

        client_train_arrays = [np.array(indices) for indices in client_train_indices]

        if self.valset_type == "global":
            return self._split_with_global_val(client_train_arrays, splits)
        elif self.valset_type == "local":
            return self._split_with_local_val(client_train_arrays, splits)
        else:  # valset_type == "none"
            return self._split_without_val(client_train_arrays, splits)

    def _partition_with_retry(
        self, all_labels: List, unique_labels: List, total_samples: int, dataset_type: str = "train"
    ) -> List[List[int]]:
        """
        Partition data using Dirichlet distribution with retry logic to ensure minimum samples per client.

        Args:
            all_labels: List of labels for all samples in the dataset
            unique_labels: Sorted list of unique class labels
            total_samples: Total number of samples in the dataset
            dataset_type: "train" or "val" for error messages

        Returns:
            List of client indices lists
        """

        for attempt in range(self.max_retries):
            client_indices = [[] for _ in range(self.n_clients)]
            min_size = 0

            for class_label in unique_labels:

                class_indices = np.where(np.array(all_labels) == class_label)[0]
                np.random.shuffle(class_indices)
                proportions = np.random.dirichlet([self.heterogeneity] * self.n_clients)

                # Apply capacity constraint
                proportions = np.array(
                    [p * (len(client_idx) < total_samples / self.n_clients) for p, client_idx in zip(proportions, client_indices)]
                )
                if proportions.sum() == 0:
                    continue
                proportions = proportions / proportions.sum()

                # Convert proportions to split points and split
                split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
                class_splits = np.split(class_indices, split_points) if len(split_points) > 0 else [class_indices]
                for client_id in range(self.n_clients):
                    client_indices[client_id].extend(class_splits[client_id].tolist())

                min_size = min(len(idx) for idx in client_indices)

            if min_size >= self.min_samples_per_client:
                for client_id in range(self.n_clients):
                    np.random.shuffle(client_indices[client_id])
                return client_indices

        raise ValueError(
            f"Failed to create valid {dataset_type} partition after {self.max_retries} attempts. "
            f"Minimum samples per client: {self.min_samples_per_client}, "
            f"achieved: {min_size}. Try increasing alpha or decreasing n_clients."
        )

    def _split_with_local_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for local validation splitting.")

        # Get validation data labels and indices
        val_labels = [self.val_dataset[i][1] for i in range(len(self.val_dataset))]
        unique_labels = sorted(set(val_labels))

        # Partition validation data with retry logic
        client_val_indices = self._partition_with_retry(val_labels, unique_labels, len(self.val_dataset), "val")

        # Convert to numpy arrays
        client_val_arrays = [np.array(indices) for indices in client_val_indices]

        # Create client splits using Dirichlet-distributed train and validation sets
        for i, (train_indices, val_indices) in enumerate(zip(client_train_indices, client_val_arrays)):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            client_val_data = self._load_dataset_samples(self.val_dataset, val_indices.tolist())
            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=client_val_data)

        return splits

    def _split_with_global_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for global validation splitting.")

        splits["global_val"] = self._load_dataset_samples(self.val_dataset)

        for i, indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, indices.tolist())
            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])

        return splits

    def _split_without_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        for i, train_indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])

        return splits

    def visualize_class_distribution(self, max_classes_to_show: int = 20, figsize: tuple = (12, 8)) -> None:
        """
        Create a horizontal bar chart showing class distribution for each client.

        Args:
            max_classes_to_show: Maximum number of classes to show individually (top N by frequency)
            figsize: Figure size (width, height)
        """
        splits = self.get_client_splits()

        all_labels = set()
        client_label_counts = {}

        for client_id, split in splits["clients"].items():
            client_labels = [sample["label"] for sample in split["train"]]
            unique, counts = np.unique(client_labels, return_counts=True)
            client_label_counts[client_id] = dict(zip(unique, counts))
            all_labels.update(unique)

        total_class_counts = {}
        for label in all_labels:
            total_class_counts[label] = sum(client_counts.get(label, 0) for client_counts in client_label_counts.values())

        sorted_classes = sorted(total_class_counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_classes) > max_classes_to_show:
            top_classes = [cls for cls, count in sorted_classes[:max_classes_to_show]]
            other_classes = [cls for cls, count in sorted_classes[max_classes_to_show:]]
        else:
            top_classes = [cls for cls, count in sorted_classes]
            other_classes = []

        clients = sorted(splits["clients"].keys())
        n_clients = len(clients)

        if len(top_classes) <= 10:
            cmap = get_cmap("tab10")
        elif len(top_classes) <= 20:
            cmap = get_cmap("tab20")
        else:
            cmap = get_cmap("viridis")

        colors = [cmap(i % cmap.N) for i in range(len(top_classes))]
        if other_classes:
            colors.append("lightgray")

        fig, ax = plt.subplots(figsize=figsize)
        bar_height = 0.8
        client_positions = np.arange(n_clients)
        bottom = np.zeros(n_clients)

        for i, class_label in enumerate(top_classes + (["Other"] if other_classes else [])):
            if class_label == "Other":
                values = []
                for client_id in clients:
                    other_count = sum(count for cls, count in client_label_counts[client_id].items() if cls in other_classes)
                    values.append(other_count)
            else:
                values = [client_label_counts[client_id].get(class_label, 0) for client_id in clients]

            total_samples = [sum(client_label_counts[client_id].values()) for client_id in clients]
            percentages = [100 * value / total if total > 0 else 0 for value, total in zip(values, total_samples)]

            bars = ax.barh(
                client_positions, percentages, bar_height, left=bottom, color=colors[i], edgecolor="white", linewidth=0.5, label=class_label
            )
            bottom += percentages

        ax.set_xlabel("Class Distribution (%)", fontsize=12)
        ax.set_ylabel("Clients", fontsize=12)
        ax.set_title(f"Class Distribution per Client (Dirichlet Split, α={self.heterogeneity})", fontsize=14, fontweight="bold")
        ax.set_yticks(client_positions)
        ax.set_yticklabels(clients)
        ax.set_xlim(0, 100)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

        n_legend_items = len(top_classes) + (1 if other_classes else 0)
        if n_legend_items > 15:
            legend_classes = top_classes[:10] + (["Other"] if other_classes else [])
            legend_colors = colors[:10] + ([colors[-1]] if other_classes else [])
            legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(legend_classes, legend_colors)]
            if len(top_classes) > 10:
                legend_handles.append(mpatches.Patch(color="white", label=f"+ {len(top_classes) - 10} more classes"))
        else:
            legend_handles = [
                mpatches.Patch(color=color, label=label) for label, color in zip(top_classes + (["Other"] if other_classes else []), colors)
            ]

        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, title="Classes", title_fontsize=11)

        # Add value annotations for significant portions (>5%)
        for i, client_id in enumerate(clients):
            total = sum(client_label_counts[client_id].values())
            current_bottom = 0
            for j, class_label in enumerate(top_classes + (["Other"] if other_classes else [])):
                if class_label == "Other":
                    value = sum(client_label_counts[client_id].get(cls, 0) for cls in other_classes)
                else:
                    value = client_label_counts[client_id].get(class_label, 0)

                percentage = 100 * value / total if total > 0 else 0
                if percentage > 5:  # Only annotate if >5%
                    x_pos = current_bottom + percentage / 2
                    ax.text(
                        x_pos,
                        i,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if percentage > 20 else "black",
                    )
                current_bottom += percentage

        plt.tight_layout()
        plt.show()

        print("\nDataset Statistics:")
        print(f"Total clients: {n_clients}")
        print(f"Total classes: {len(all_labels)}")
        print(f"Classes shown individually: {len(top_classes)}")
        if other_classes:
            print(f"Classes grouped as 'Other': {len(other_classes)}")

        self._print_heterogeneity_stats(client_label_counts, all_labels)

    def _print_heterogeneity_stats(self, client_label_counts: dict, all_labels: set) -> None:
        """Print heterogeneity statistics."""
        # Calculate label distribution skew across clients

        labels_list = sorted(all_labels)
        n_clients = len(client_label_counts)
        n_labels = len(labels_list)

        # Create distribution matrix: clients x labels
        dist_matrix = np.zeros((n_clients, n_labels))
        for i, (client_id, counts) in enumerate(client_label_counts.items()):
            for j, label in enumerate(labels_list):
                dist_matrix[i, j] = counts.get(label, 0)

        # Normalize to probabilities
        row_sums = dist_matrix.sum(axis=1, keepdims=True)
        dist_matrix = dist_matrix / np.where(row_sums > 0, row_sums, 1)

        # Calculate average KL divergence between client distributions
        avg_kl = 0
        count = 0
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                # Avoid division by zero
                p = dist_matrix[i] + 1e-10
                q = dist_matrix[j] + 1e-10
                kl = entropy(p, q)
                avg_kl += kl
                count += 1

        if count > 0:
            avg_kl /= count
            print(f"Average KL divergence between clients: {avg_kl:.3f}")
            print(f"Heterogeneity level (α={self.heterogeneity}): {'High' if avg_kl > 2.0 else 'Medium' if avg_kl > 1.0 else 'Low'}")
