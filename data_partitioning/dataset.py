# We need a python script defining a pytorch Dataset class. 
# This class ingests images from a disk location. 
# Each different dataset we will use needs a separate constructor function 
# inside the class that maps from its specific pathing to this common class. 
# This is so “data->train->image->048234.jpg/label->048234.jpg” and 
# “data->image->train->048234.jpg” (random example) can be processed using the same code. 
# This class will not load any images into memory, it’s there to pre-process the paths 
# mostly (and maybe other stuff I can’t think of right now).

import os
import json
from torch.utils.data import Dataset
from typing import List, Tuple
from pathlib import Path

_loaders = {}

class BaseDataset(Dataset):
    def __init__(self, dataset_name: str, base_path: str, split='train'):
        """
        Args:
            dataset_name: Name of the dataset (must be registered)
            base_path: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
        """
        self.dataset_name = dataset_name
        self.base_path = Path(base_path)
        self.split = split
        self.data: List[str] = []
        self.labels: List[str] = []

        self._validate_inputs()
        self._load_dataset()

    def _validate_inputs(self):
        """Validate constructor parameters."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")
        
        if self.dataset_name not in self._loaders:
            raise ValueError(
                f"Dataset '{self.dataset_name}' not supported. "
                f"Available: {list(self._loaders.keys())}"
            )

    def _load_dataset(self):
        loader_func = self._loaders[self.dataset_name]
        loader_func(self)
    
    @classmethod
    def register_dataset(cls, name: str):
        """Decorator to register dataset loader functions."""
        def decorator(func):
            cls._loaders[name] = func
            return func
        return decorator

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx:int) -> Tuple[str, str]:
        return self.data[idx], self.labels[idx]
    
    def get_summary(self) -> dict:
        return {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'total_samples': len(self),
            'unique_labels': len(set(self.labels)),
            'base_path': str(self.base_path)
        }


@BaseDataset.register_dataset('tiny_imagenet')
def _load_tiny_imagenet(self) -> None:
    """Load Tiny ImageNet dataset."""
    valid_splits = ['train', 'val', 'test']
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")
    split_path = self.base_path / 'tiny_imagenet-200' / self.split

    if not split_path.exists():
        raise FileNotFoundError(f"Split path {split_path} does not exist")
    
    for class_dir in sorted(datapath.iterdir()):
        if class_dir.is_dir():
            label = class_dir.name
            img_path = class_dir / 'images'
            for img_file in sorted(img_path.glob('*.JPEG')):
                self.data.append(str(img_file))
                self.labels.append(label)

@BaseDataset.register_dataset('food101')
def _load_food101(self):
    """Load Food-101 dataset."""
    valid_splits = ['train', 'test']
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")
    
    data_path: Path = self.base_path / 'food101'

    split_file = data_path / 'meta' / f'{self.split}.json'
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} does not exist")
    
    # this json contains the label as a dict key and list of image paths as values
    with open(split_file, 'r') as f:
        split_data = json.load(f)
        
        for label, img_list in split_data.items():
            for img_rel_path in img_list:
                img_path = os.path.join(data_path.name, 'images', img_rel_path)
                self.data.append(img_path)
                self.labels.append(label)

@BaseDataset.register_dataset('ade20k')
def _load_ade20k(self):
    """Load ADE20K dataset."""
    valid_splits = ['train', 'val']
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")
    
    # convert to ADE20K naming
    self.split = 'training' if self.split == 'train' else 'validation'

    images_dir = self.base_path / 'ADE20K_2021_17_01' / 'images' / 'ADE' / self.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")
    
    for img_file in sorted(images_dir.glob('*.jpg')):
        self.data.append(str(img_file))
        # Placeholder until we set up our task and model
        self.labels.append('unknown')

if __name__ == "__main__":
    # Usage remains the same
    dataset = BaseDataset('tiny_imagenet', '/path/to/data', 'train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Summary: {dataset.get_summary()}")
    
    # First sample
    img_path, label = dataset[0]
    print(f"First sample - Path: {img_path}, Label: {label}")