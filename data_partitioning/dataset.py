import os
import json
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image


class ClassificationDataset(Dataset):
    def __init__(self, split_data: List[Dict[str, Any]], transform=None):
        self.samples = split_data
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


class SegmentationDataset(Dataset):
    def __init__(self, split_data: List[Dict[str, Any]], transform=None):
        self.samples = split_data  # [{"image": img_path, "label": mask_path}, ...]
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"])

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


class BaseDataset(Dataset):

    _loaders = {}

    def __init__(self, dataset_name: str, base_path: str, split="train"):
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
            raise ValueError(f"Dataset '{self.dataset_name}' not supported. " f"Available: {list(self._loaders.keys())}")

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

    def __getitem__(self, idx: int) -> Tuple[str, str]:

        return self.data[idx], self.labels[idx]

    def get_summary(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "total_samples": len(self),
            "unique_labels": len(set(self.labels)),
            "base_path": str(self.base_path),
            # "label_distribution": {label: self.labels.count(label) for label in set(self.labels)},
        }


@BaseDataset.register_dataset("tiny_imagenet")
def _load_tiny_imagenet(self) -> None:
    """Load Tiny ImageNet dataset."""
    valid_splits = ["train", "val"]
    # there's a test split but no labels provided
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")
    split_path = self.base_path / "tiny-imagenet-200" / self.split

    if not split_path.exists():
        raise FileNotFoundError(f"Split path {split_path} does not exist")

    # need to handle train and val differently
    if self.split == "train":
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                label = class_dir.name
                img_path = class_dir / "images"
                for img_file in sorted(img_path.glob("*.JPEG")):
                    self.data.append(str(img_file))
                    self.labels.append(label)
    else:
        # val annotations are in a text file
        val_annotations_file = split_path / "val_annotations.txt"
        if not val_annotations_file.exists():
            raise FileNotFoundError(f"Validation annotations file {val_annotations_file} does not exist")

        img_to_label = {}
        with open(val_annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_to_label[parts[0]] = parts[1]

        img_path = split_path / "images"
        for img_file in sorted(img_path.glob("*.JPEG")):
            img_name = img_file.name
            if img_name in img_to_label:
                self.data.append(str(img_file))
                self.labels.append(img_to_label[img_name])


@BaseDataset.register_dataset("food101")
def _load_food101(self):
    """Load Food-101 dataset."""
    valid_splits = ["train", "test"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    data_path: Path = self.base_path / "food101"

    split_file = data_path / "meta" / f"{self.split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} does not exist")

    images_dir = data_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")

    # this json contains the label as a dict key and list of image paths as values
    with open(split_file, "r") as f:
        split_data = json.load(f)

    for label, img_list in split_data.items():
        for img_rel_path in img_list:
            img_path = images_dir / f"{img_rel_path}.jpg"
            if img_path.exists():
                self.data.append(img_path)
                self.labels.append(label)
            else:
                raise Warning(f"Image path {img_path} does not exist")


@BaseDataset.register_dataset("ade20k")
def _load_ade20k(self):
    """Load ADE20K dataset."""
    valid_splits = ["train", "val"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    # convert to ADE20K naming
    self.split = "training" if self.split == "train" else "validation"

    images_dir = self.base_path / "ADE20K" / "ADE20K_2021_17_01" / "images" / "ADE" / self.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_file = Path(root) / file
                self.data.append(str(img_file))
                mask_file = img_file.with_name(img_file.stem + "_seg.png")
                if mask_file.exists():
                    self.labels.append(str(mask_file))
                else:
                    raise Warning(f"Mask file {mask_file} does not exist")


if __name__ == "__main__":

    dataset = BaseDataset("ade20k", "/users/adcw447/gtai/FDS", "train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Summary: {dataset.get_summary()}")

    img_path, label = dataset[0]
    print(f"First sample - Path: {img_path}, Label: {label}")
