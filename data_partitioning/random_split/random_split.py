import numpy as np
import json
from ..base_splitter import BaseDatasetSplitter

class RandomSplitter(BaseDatasetSplitter):
    def __init__(self, dataset, n_clients, heterogeneity=0.0, seed=None):
        super().__init__(dataset, n_clients, heterogeneity, seed)
        if seed is not None:
            np.random.seed(seed)

    def split(self):
        n_samples = len(self.dataset)
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)

        samples_per_client = n_samples // self.n_clients
        for client_id in range(self.n_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client if client_id != self.n_clients - 1 else n_samples
            self.client_data_indices[client_id] = all_indices[start_idx:end_idx].tolist()

        with open(f'random_split_{self.seed}.json', 'w') as f:
            json.dump(self.client_data_indices, f)


