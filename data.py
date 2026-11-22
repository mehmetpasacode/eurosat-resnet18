"""EuroSAT image transforms and stratified holdout splits."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision import datasets, transforms


class GaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def make_tf(augment=True):
    eval_transform = transforms.ToTensor()
    train_transform = transforms.ToTensor()
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.02),
            transforms.ToTensor(),
            GaussianNoise(0.01),
        ])
    return train_transform, eval_transform


def split_80_10_10(dataset, seed=42):
    targets = np.asarray(dataset.targets)
    holdout_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, holdout_idx = next(holdout_split.split(targets, targets))
    holdout_targets = targets[holdout_idx]
    val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(val_test_split.split(holdout_targets, holdout_targets))
    return train_idx.tolist(), holdout_idx[val_idx].tolist(), holdout_idx[test_idx].tolist()


def training_splits(data_root, seed, augment=True):
    train_transform, eval_transform = make_tf(augment)
    train_dataset = datasets.ImageFolder(data_root, transform=train_transform)
    eval_dataset = datasets.ImageFolder(data_root, transform=eval_transform)
    train_idx, val_idx, test_idx = split_80_10_10(train_dataset, seed)
    return (
        Subset(train_dataset, train_idx),
        Subset(eval_dataset, val_idx),
        (train_idx, val_idx, test_idx),
        train_dataset.classes,
    )


def load_test_split(data_root, out_dir, classes):
    dataset = datasets.ImageFolder(data_root, transform=transforms.ToTensor())
    if dataset.classes != classes:
        raise ValueError("Dataset classes do not match the checkpoint.")
    with np.load(Path(out_dir) / "splits_80_10_10.npz", allow_pickle=False) as splits:
        indices = splits["test"]
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty array of integer test indices.")
    if indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError("Test indices are out of range for this dataset.")
    return Subset(dataset, indices.tolist())
