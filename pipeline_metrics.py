"""
pipeline_metrics.py
=====================
Turning predictions into numbers (Section 5 of the brief), saving/loading
the results file, and aggregating across the 3 seeds. No PyTorch here at
all -- everything below just operates on numpy arrays / a results CSV.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from pipeline_config import CLASS_NAMES, FACTOR_STUDIES, RESULTS_CSV_DEFAULT


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels, class_names=CLASS_NAMES) -> dict:
    """Metrics for ONE run. Aggregate across seeds afterwards by averaging
    these per-run values (mean +/- std) -- never pool predictions across
    seeds before computing a metric.

    Averaging convention used throughout: macro averaging for F1/precision/
    recall (every class weighted equally, regardless of support) -- stated
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
    -- an explicit, documented choice allowed by the brief as an alternative
    to picking one representative seed."""
    arr = np.stack([np.asarray(cm, dtype=float) for cm in list_of_cms], axis=0)
    return arr.mean(axis=0)


# ---------------------------------------------------------------------------
# Results file I/O -- incremental save, resume support
# ---------------------------------------------------------------------------

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
    """Set of (config_id, seed) pairs already present in the results file --
    used to resume a full-experiment notebook after an interruption without
    re-running (and duplicating) completed work."""
    df = load_results(csv_path)
    if df.empty:
        return set()
    return set(zip(df["config_id"], df["seed"]))


# ---------------------------------------------------------------------------
# Aggregation across seeds
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
    Computed from the per-run values already in df (one row per run) --
    never from pooled predictions, per the brief's explicit warning."""
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
