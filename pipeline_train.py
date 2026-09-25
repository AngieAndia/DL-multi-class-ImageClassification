"""
pipeline_train.py
====================
The mechanical core: one epoch of training, one evaluation pass, early
stopping, and run_training() -- the single function every one of the 21
runs calls. Only the config dict and seed change between calls; everything
else (loss function, model-selection rule, evaluation code) is identical
across every run, which is what makes the OFAT design actually controlled.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from pipeline_config import BATCH_SIZE, OPTIMIZER_DEFAULTS
from pipeline_data import DEVICE, make_loaders, set_seed
from pipeline_models import ARCHITECTURES, build_optimizer, count_params
from pipeline_metrics import compute_metrics

USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = (torch.bfloat16 if USE_AMP and torch.cuda.is_bf16_supported()
             else torch.float16)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # to adapt for the faster gpu from colab A100
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()  # again because of A100

        # needed for the gradient clipper (first trial without)
        #scaler.unscale_(optimizer)

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
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
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)

    # Warmup
    warmup_epochs = 3
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    stopper = EarlyStopper(patience=patience)



    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    epochs_run = 0
    early_stopped = False

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
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
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
    }
    return result
