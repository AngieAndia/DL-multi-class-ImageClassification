"""
pipeline_data.py
=================
Everything about getting CIFAR-10 into batches: device detection,
reproducibility, the three augmentation strategies, the stratified
train/val split (cached to disk so it's identical for both group members
and every run), and the DataLoader builder.

Importing this module does not download data or run anything -- that only
happens the first time load_raw_datasets() / get_stratified_split() is
actually called.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pipeline_config import (
    CIFAR_MEAN, CIFAR_STD, DATA_ROOT, SPLIT_SEED, VAL_FRACTION, SPLIT_CACHE_PATH,
)


# ---------------------------------------------------------------------------
# Device detection -- Colab (CUDA) first, MPS (Apple Silicon) as a local
# dev/debug fallback, CPU last.
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

if torch.cuda.is_available():
    # Lets cuDNN pick the fastest convolution algorithm for our fixed 32x32
    # input size. Speed-only knob -- has no effect on what gets computed.
    torch.backends.cudnn.benchmark = True


def dataloader_kwargs(device: torch.device) -> dict:
    """num_workers/pin_memory tuned per backend. Throughput only -- does not
    affect any reported metric."""
    if device.type == "cuda":
        return {"num_workers": 2, "pin_memory": True, "persistent_workers": True}
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seeds weight init, shuffling, and augmentation draws. Does NOT touch
    the validation split, which is fixed independently via SPLIT_SEED."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Transforms -- Factor 2 (data augmentation), applied to TRAINING data only.
# ---------------------------------------------------------------------------

# Deterministic transform used for validation AND test, always. Never augmented.
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

AUGMENTATIONS = {
    "none": eval_transform,

    "crop_flip": T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),

    "strong": T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        # AutoAugment's CIFAR10 policy (Cubuk et al., 2019) was found by search
        # directly on this dataset, unlike RandAugment's generic, un-searched
        # policy -- since we aren't redoing that search ourselves, the
        # dataset-specific found policy is the better-justified citation here.
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25),  # Zhong et al., 2020 -- unchanged, last step
    ]),
}


# ---------------------------------------------------------------------------
# Dataset loading and the one fixed stratified split
# ---------------------------------------------------------------------------

def load_raw_datasets(root: str = DATA_ROOT):
    """Downloads CIFAR-10 if needed. full_train has no transform attached yet
    (each augmentation is applied per-config by TransformedSubset). test_set
    always uses the fixed eval_transform."""
    full_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=eval_transform
    )
    return full_train, test_set


def get_stratified_split(
    targets,
    val_fraction: float = VAL_FRACTION,
    split_seed: int = SPLIT_SEED,
    cache_path: str = SPLIT_CACHE_PATH,
):
    """Returns (train_idx, val_idx). Computed once and cached to disk so that
    every notebook, every run, and both group members' Colab sessions reuse
    the exact same split -- required by the brief ("create the validation
    split once and reuse exactly the same split in every configuration")."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["train_idx"], cached["val_idx"]

    targets = np.asarray(targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_fraction,
        stratify=targets,
        random_state=split_seed,
    )
    np.savez(cache_path, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


class TransformedSubset(Dataset):
    """Wraps a base CIFAR10 dataset + a fixed list of indices + one transform."""

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        img = self.transform(img)
        return img, label


def make_loaders(
    aug_name: str,
    full_train,
    test_set,
    train_idx,
    val_idx,
    batch_size: int,
    eval_batch_size: int | None = None,
    subset_size: int | None = None,
    seed_for_shuffle: int | None = None,
    device: torch.device = DEVICE,
):
    """Builds train/val/test DataLoaders for one augmentation strategy.

    subset_size: if set, use only this many TRAINING images (smoke test).
    eval_batch_size: batch size for val/test loaders only -- a throughput
    knob (no gradients there), free to differ from the training batch size.
    """
    eval_batch_size = eval_batch_size or batch_size
    train_transform = AUGMENTATIONS[aug_name]

    t_idx = train_idx if subset_size is None else train_idx[:subset_size]
    train_ds = TransformedSubset(full_train, t_idx, train_transform)
    val_ds = TransformedSubset(full_train, val_idx, eval_transform)

    generator = None
    if seed_for_shuffle is not None:
        generator = torch.Generator().manual_seed(seed_for_shuffle)

    kwargs = dataloader_kwargs(device)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, **kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
