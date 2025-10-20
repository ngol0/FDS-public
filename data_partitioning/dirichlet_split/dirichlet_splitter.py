import numpy as np
import json

from ..base_splitter import BaseDatasetSplitter

class DirichletSplitter(BaseDatasetSplitter):
    def __init__(self, dataset, n_clients, heterogeneity=0.5, seed=None):
        super().__init__(dataset, n_clients, heterogeneity, seed)
        if seed is not None:
            np.random.seed(seed)

    def split(self):
        n_samples = len(self.dataset)
        n_classes = len(set([label for _, label in self.dataset]))
        class_indices = {i: [] for i in range(n_classes)}

        for idx, (_, label) in enumerate(self.dataset):
            class_indices[label].append(idx)

        for c in range(n_classes):
            np.random.shuffle(class_indices[c])

        for c in range(n_classes):
            class_size = len(class_indices[c])
            proportions = np.random.dirichlet(np.repeat(self.heterogeneity, self.n_clients))
            proportions = (proportions * class_size).astype(int)

            current_idx = 0
            for client_id in range(self.n_clients):
                num_samples = proportions[client_id]
                if client_id == self.n_clients - 1:
                    num_samples = class_size - current_idx
                self.client_data_indices[client_id].extend(class_indices[c][current_idx:current_idx + num_samples])
                current_idx += num_samples

        for client_id in range(self.n_clients):
            np.random.shuffle(self.client_data_indices[client_id])

        with open(f'dirichlet_split_{self.seed}.json', 'w') as f:
            json.dump(self.client_data_indices, f)