class BaseDatasetSplitter:
    def __init__(self, dataset, n_clients, heterogeneity, seed):
        self.dataset = dataset
        self.n_clients = n_clients
        self.heterogeneity = heterogeneity
        self.seed = seed
        self.client_data_indices = {i: [] for i in range(n_clients)}
    def split(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    def get_client_data_indices(self):
        return self.client_data_indices
    def get_client_datasets(self):
        client_datasets = {}
        for client_id, indices in self.client_data_indices.items():
            client_datasets[client_id] = [self.dataset[i] for i in indices]
        return client_datasets
    def set_heterogeneity(self, heterogeneity):
        self.heterogeneity = heterogeneity