"""
shared_pipeline.py
===================
Single source of truth for the PGR207 CIFAR-10 OFAT experiment (architecture x
augmentation x optimizer). Both group members import from this module so that
the data split, transforms, model definitions, training/evaluation loop, and
metric definitions are byte-for-byte identical across every notebook and every
one of the 21 training runs.

Design decisions and where they come from:
- Architectures, augmentations, optimizers, and the 7-configuration OFAT table
  (Section 3.5 of the assignment brief) are fixed below in CONFIGS.
- Train/val split: stratified 90/10, created once, cached to disk, reused
  everywhere (Section 4 of the brief).
- Batch size, early-stopping patience, and per-optimizer LR/weight-decay were
  explicit choices made with the course assignment in mind — see the comments
  next to each constant for the justification to reuse in the Methods section.

Importing this module does not train, download data, or execute anything by
itself — it only defines constants, classes, and functions. Data download and
the validation-split cache are created the first time a notebook explicitly
calls load_raw_datasets() / get_stratified_split().
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Device detection — Colab (CUDA) first, MPS (Apple Silicon) as a local
#    dev/debug fallback, CPU last.
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def dataloader_kwargs(device: torch.device) -> dict:
    """num_workers/pin_memory tuned per backend. Purely a throughput knob —
    does not affect what is computed, so it is not part of the controlled
    experimental design and can differ between environments."""
    if device.type == "cuda":
        return {"num_workers": 2, "pin_memory": True, "persistent_workers": True}
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seeds weight init, shuffling, and augmentation draws. Does NOT touch
    the validation split, which is fixed independently (see SPLIT_SEED)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# cudnn.benchmark speeds up training substantially on a fixed 32x32 input size
# by letting cuDNN pick the fastest convolution algorithm for that shape. It
# has no effect on MPS/CPU. This trades exact bit-for-bit determinism for
# speed; the experiment's reproducibility claim rests on the fixed seeds
# controlling init/shuffling/augmentation, not on bit-identical reruns.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# 3. Data: CIFAR-10, transforms, stratified split (created once, cached)
# ---------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
# Matches torchvision.datasets.CIFAR10(...).classes exactly (index = label id).
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DATA_ROOT = "./data"
SPLIT_SEED = 42          # fixed once, independent of the 3 experiment seeds
VAL_FRACTION = 0.10      # 5,000 of the 50,000 training images
SPLIT_CACHE_PATH = "val_split_indices.npz"

# Deterministic transform used for validation AND test, always. Never augmented.
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Factor 2 (data augmentation), applied to TRAINING data only.
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
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25),
    ]),
}


def load_raw_datasets(root: str = DATA_ROOT):
    """Downloads CIFAR-10 if needed. full_train has no transform attached yet
    (transforms are applied per-augmentation by TransformedSubset). test_set
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
    the exact same split — required by the brief ("create the validation
    split once and reuse exactly the same split in every configuration").
    """
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
    eval_batch_size: batch size for val/test loaders only — purely a
    throughput knob (no gradients there), does not affect any reported
    metric, so it is free to differ from the training batch size.
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


# ---------------------------------------------------------------------------
# 4. Models — Factor 1 (architecture). All trained from scratch (weights=None).
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Hand-written baseline CNN: 3 conv blocks + 2 FC layers."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18, weights=None, adapted for 32x32 input: the default 7x7
    stride-2 first conv + maxpool shrink a 224x224 ImageNet image by 4x
    before the first residual block — applied to a 32x32 image that would
    collapse it to 8x8 almost immediately. We replace conv1 with a 3x3
    stride-1 convolution and remove the maxpool (nn.Identity), which is the
    standard CIFAR adaptation for ResNet (He et al., 2016, used a similar
    3x3-stem variant for their original CIFAR-10 experiments)."""
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_densenet121_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """DenseNet-121, weights=None, with the same stem adaptation logic as
    ResNet-18 above, applied to DenseNet's stem (features.conv0/pool0)."""
    m = models.densenet121(weights=None, num_classes=num_classes)
    m.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.features.pool0 = nn.Identity()
    return m


ARCHITECTURES = {
    "simple_cnn": SimpleCNN,
    "resnet18": build_resnet18_cifar,
    "densenet121": build_densenet121_cifar,
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_param_counts() -> dict:
    """Parameter count per architecture, for the Methods table."""
    return {name: count_params(builder()) for name, builder in ARCHITECTURES.items()}


# ---------------------------------------------------------------------------
# 5. Optimizers — Factor 3. Per-optimizer default LR/weight-decay (not a
#    same-LR-for-all-optimizers design — see Methods justification below).
# ---------------------------------------------------------------------------

# SGD tolerates and needs a substantially higher LR than the Adam family
# (Kingma & Ba, 2015, default lr=1e-3; He et al., 2016, use lr=0.1 for SGD on
# CIFAR-scale ResNets) — using one shared LR for all three optimizers would
# unfairly cripple SGD (too small) or destabilize Adam/AdamW (too large).
#
# Weight decay: Adam's L2 penalty is added directly to the gradient before
# adaptive per-parameter scaling, so it is NOT decoupled from the adaptive
# learning rate — the same nominal weight_decay value therefore acts as a
# stronger effective penalty under Adam than under SGD or AdamW's decoupled
# decay (Loshchilov & Hutter, 2019). We use a smaller weight_decay for Adam
# (1e-4) than for SGD/AdamW (5e-4) to keep the effective regularization
# strength comparable across optimizers, and state this explicitly as a
# controlled design choice in the Methods section.
OPTIMIZER_DEFAULTS = {
    "sgd": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
    "adam": {"lr": 0.001, "weight_decay": 1e-4},
    "adamw": {"lr": 0.001, "weight_decay": 5e-4},
}


def build_optimizer(name: str, params) -> optim.Optimizer:
    cfg = dict(OPTIMIZER_DEFAULTS[name])
    if name == "sgd":
        return optim.SGD(params, **cfg)
    if name == "adam":
        return optim.Adam(params, **cfg)
    if name == "adamw":
        return optim.AdamW(params, **cfg)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# 6. Training / evaluation loop, early stopping
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_n += images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / total_n, total_correct / total_n, all_preds, all_labels


class EarlyStopper:
    """Early stopping on validation loss. Remembers the best-so-far weights
    and restores them after training stops, so the evaluated model is always
    the best-on-validation checkpoint, never just the last epoch."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.best_epoch = -1

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, class_names=CLASS_NAMES) -> dict:
    """Metrics for ONE run. Aggregate across seeds afterwards by averaging
    these per-run values (mean +/- std) — never pool predictions across seeds
    before computing a metric.

    Averaging convention used throughout: macro averaging for F1/precision/
    recall (every class weighted equally, regardless of support) — stated
    once here and kept consistent across all experiments.
    """
    accuracy = float((preds == labels).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    macro_f1 = float(f1.mean())
    cm = sk_confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_per_class": {c: float(p) for c, p in zip(class_names, precision)},
        "recall_per_class": {c: float(r) for c, r in zip(class_names, recall)},
        "f1_per_class": {c: float(f) for c, f in zip(class_names, f1)},
        "confusion_matrix": cm.tolist(),
    }


def mean_confusion_matrix(list_of_cms) -> np.ndarray:
    """Element-wise mean of the 3 per-seed confusion matrices for one config
    (used for the best/worst-config figures in the results-aggregation
    notebook) — an explicit, documented choice allowed by the brief as an
    alternative to picking one representative seed."""
    arr = np.stack([np.asarray(cm, dtype=float) for cm in list_of_cms], axis=0)
    return arr.mean(axis=0)


# ---------------------------------------------------------------------------
# 8. The 7-configuration OFAT design (Section 3.5 of the brief) + seeds
# ---------------------------------------------------------------------------

BASELINE = {"architecture": "resnet18", "augmentation": "crop_flip", "optimizer": "sgd"}

CONFIGS = {
    # config_id: (dict of factor levels, which studies it belongs to)
    "C1": {**BASELINE},                                    # baseline (all 3 studies)
    "C2": {**BASELINE, "architecture": "simple_cnn"},       # architecture study
    "C3": {**BASELINE, "architecture": "densenet121"},      # architecture study
    "C4": {**BASELINE, "augmentation": "none"},             # augmentation study
    "C5": {**BASELINE, "augmentation": "strong"},           # augmentation study
    "C6": {**BASELINE, "optimizer": "adam"},                # optimizer study
    "C7": {**BASELINE, "optimizer": "adamw"},               # optimizer study
}

# Which configs belong to which factor study, with C1 (baseline) common to all
# three — used by the aggregation notebook to build the three per-factor
# comparison tables required by the brief.
FACTOR_STUDIES = {
    "architecture": ["C2", "C1", "C3"],   # ordered: simple_cnn, resnet18*, densenet121
    "augmentation": ["C4", "C1", "C5"],   # ordered: none, crop_flip*, strong
    "optimizer": ["C1", "C6", "C7"],      # ordered: sgd*, adam, adamw
}

SEEDS = [0, 1, 2]

# Fixed training settings shared by every run unless noted otherwise.
BATCH_SIZE = 128
# Epoch budget for the FULL experiment (Section 3.1: "use a fixed epoch
# budget you can afford to repeat many times"). 25 epochs was chosen as a
# fixed ceiling: cosine annealing needs enough steps for a meaningful LR
# decay tail, DenseNet-121 (the slowest of the three architectures) still
# completes 25 epochs x 21 runs in a affordable amount of Colab GPU time, and
# early stopping (patience=7 on validation loss) lets configurations that
# converge sooner stop without spending the full budget. This value and the
# patience below apply identically to every one of the 7 configurations, per
# the OFAT control requirement.
FULL_RUN_EPOCHS = 25
FULL_RUN_PATIENCE = 7

# Smoke-test settings (01_smoke_test.ipynb): small subset, few epochs, low
# patience — purpose is only to catch bugs cheaply, not to produce meaningful
# numbers.
SMOKE_SUBSET_SIZE = 1000
SMOKE_EPOCHS = 2
SMOKE_PATIENCE = 2


# ---------------------------------------------------------------------------
# 9. End-to-end single-run wrapper
# ---------------------------------------------------------------------------

def run_training(
    config_id: str,
    config: dict,
    seed: int,
    full_train,
    test_set,
    train_idx,
    val_idx,
    epochs: int,
    patience: int,
    batch_size: int = BATCH_SIZE,
    subset_size: int | None = None,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> dict:
    """Runs one full training run for one (configuration, seed) pair and
    returns a flat dict ready to be appended as one row of all_results.csv.
    Trains -> early-stops on val loss -> restores best weights -> evaluates
    once on the held-out test set.
    """
    set_seed(seed)

    train_loader, val_loader, test_loader = make_loaders(
        aug_name=config["augmentation"],
        full_train=full_train,
        test_set=test_set,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=batch_size,
        subset_size=subset_size,
        seed_for_shuffle=seed,
        device=device,
    )

    model = ARCHITECTURES[config["architecture"]]().to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config["optimizer"], model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStopper(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    epochs_run = 0
    early_stopped = False

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epochs_run = epoch + 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"  [{config_id} seed={seed}] epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

        if stopper.step(val_loss, model, epoch):
            early_stopped = True
            if verbose:
                print(f"  [{config_id} seed={seed}] early stopping at epoch {epoch + 1}")
            break

    stopper.restore_best(model)
    train_time_seconds = time.time() - t0

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(preds, labels)

    opt_cfg = OPTIMIZER_DEFAULTS[config["optimizer"]]
    result = {
        "config_id": config_id,
        "architecture": config["architecture"],
        "augmentation": config["augmentation"],
        "optimizer": config["optimizer"],
        "seed": seed,
        "lr": opt_cfg["lr"],
        "weight_decay": opt_cfg["weight_decay"],
        "batch_size": batch_size,
        "epoch_budget": epochs,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "best_epoch": stopper.best_epoch + 1,
        "best_val_loss": stopper.best_loss,
        "train_time_seconds": train_time_seconds,
        "num_params": n_params,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": metrics["macro_f1"],
        "precision_per_class": json.dumps(metrics["precision_per_class"]),
        "recall_per_class": json.dumps(metrics["recall_per_class"]),
        "f1_per_class": json.dumps(metrics["f1_per_class"]),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        "history": json.dumps(history),
        "device": device.type,
    }
    return result


# ---------------------------------------------------------------------------
# 10. Results file I/O — incremental save, resume support
# ---------------------------------------------------------------------------

RESULTS_CSV_DEFAULT = "all_results.csv"


def append_result(result: dict, csv_path: str = RESULTS_CSV_DEFAULT) -> None:
    """Appends one run's result as a new row, writing the header only if the
    file does not exist yet. Called after every single run (not just at the
    end) so a Colab disconnect never loses completed runs."""
    row_df = pd.DataFrame([result])
    file_exists = os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def load_results(csv_path: str = RESULTS_CSV_DEFAULT) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def get_completed_runs(csv_path: str = RESULTS_CSV_DEFAULT) -> set:
    """Set of (config_id, seed) pairs already present in the results file —
    used to resume a full-experiment notebook after an interruption without
    re-running (and duplicating) completed work."""
    df = load_results(csv_path)
    if df.empty:
        return set()
    return set(zip(df["config_id"], df["seed"]))


# ---------------------------------------------------------------------------
# 11. Aggregation helpers (used by 03_results_aggregation.ipynb)
# ---------------------------------------------------------------------------

def decode_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses the JSON-string columns written by run_training back into
    Python objects (dicts / nested lists) for downstream analysis."""
    df = df.copy()
    for col in ["precision_per_class", "recall_per_class", "f1_per_class", "confusion_matrix", "history"]:
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean +/- std over the 3 seeds for accuracy and macro-F1.
    Computed from the per-run values already in df (one row per run) — never
    from pooled predictions, per the brief's explicit warning."""
    summary = df.groupby("config_id").agg(
        architecture=("architecture", "first"),
        augmentation=("augmentation", "first"),
        optimizer=("optimizer", "first"),
        num_params=("num_params", "first"),
        n_seeds=("seed", "count"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        train_time_mean=("train_time_seconds", "mean"),
    ).reset_index()
    return summary


def build_factor_table(summary_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """One factor-study table (architecture / augmentation / optimizer),
    ordered per FACTOR_STUDIES, ready to render as the paper's per-study
    table (config, level, params, accuracy mean+-std, macro-F1 mean+-std)."""
    config_order = FACTOR_STUDIES[factor]
    table = summary_df[summary_df["config_id"].isin(config_order)].copy()
    table["config_id"] = pd.Categorical(table["config_id"], categories=config_order, ordered=True)
    return table.sort_values("config_id").reset_index(drop=True)


def best_worst_config_ids(summary_df: pd.DataFrame) -> tuple:
    """(best_config_id, worst_config_id) by mean test accuracy across seeds."""
    ranked = summary_df.sort_values("accuracy_mean", ascending=False)
    return ranked.iloc[0]["config_id"], ranked.iloc[-1]["config_id"]
