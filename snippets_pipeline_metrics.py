"""
snippets_pipeline_metrics.py
=============================
NOT a module to import. Copy each block into pipeline_metrics.py at the place
its header says. Step numbers match the outline in the chat.
"""

# ==========================================================================
# STEP 1 — REPLACE the import line from pipeline_config with this one
# ==========================================================================
from pipeline_config import (
    BASELINE_ID, CLASS_NAMES, CONFIGS, FACTOR_STUDIES, PRETTY_NAMES,
    RESULTS_CSV_DEFAULT, SEEDS,
)


# ==========================================================================
# STEP 3 — ADD directly below load_results()
# ==========================================================================

def check_results_complete(df: pd.DataFrame) -> None:
    """Fails loudly unless the results file holds exactly one row per
    (config, seed) pair -- no missing runs and no duplicates. A duplicate
    would silently turn a 3-seed mean into a 4-seed mean."""
    expected = {(c, s) for c in CONFIGS for s in SEEDS}
    found = list(zip(df["config_id"], df["seed"]))
    duplicates = sorted({p for p in found if found.count(p) > 1})
    missing = sorted(expected - set(found))
    if duplicates or missing:
        raise ValueError(f"Results file is not clean. Duplicates: {duplicates}  Missing: {missing}")

# ==========================================================================
# STEP 4 — REPLACE the existing aggregate_over_seeds()
# ==========================================================================

def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean +/- std over the 3 seeds. Computed from the per-run
    values (one row per run) -- never from pooled predictions.
    Also adds the difference to the baseline (C1) in accuracy points."""
    df = df.copy()
    df["time_per_epoch"] = df["train_time_seconds"] / df["epochs_run"]
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
        test_loss_mean=("test_loss", "mean"),
        train_time_mean=("train_time_seconds", "mean"),
        time_per_epoch_mean=("time_per_epoch", "mean"),
        best_epoch_min=("best_epoch", "min"),
        best_epoch_max=("best_epoch", "max"),
    ).reset_index()
    base_acc = summary.loc[summary["config_id"] == BASELINE_ID, "accuracy_mean"].item()
    summary["delta_vs_C1_pp"] = (summary["accuracy_mean"] - base_acc) * 100
    return summary

# ==========================================================================
# STEP 5 — ADD below aggregate_over_seeds()
# ==========================================================================

def train_val_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Final-epoch train accuracy minus val accuracy, mean over seeds, in
    percentage points. Answers the brief's question on how the train/val gap
    behaves as augmentation gets stronger (Section 3.7)."""
    rows = []
    for cid, g in df.groupby("config_id"):
        gaps = [(h["train_acc"][-1] - h["val_acc"][-1]) * 100 for h in g["history"]]
        rows.append({"config_id": cid, "train_val_gap_pp": float(np.mean(gaps))})
    return pd.DataFrame(rows)

# ==========================================================================
# STEP 5 — ADD below train_val_gap()
# ==========================================================================

def convergence_epoch(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """First epoch (1-based) at which validation accuracy reaches `threshold`,
    per run. Used for the optimiser study's 'convergence speed' claim.
    A run that never reaches the threshold gets NaN."""
    rows = []
    for _, r in df.iterrows():
        va = np.asarray(r["history"]["val_acc"])
        hit = np.flatnonzero(va >= threshold)
        rows.append({
            "config_id": r["config_id"], "seed": r["seed"],
            f"epoch_to_{int(threshold * 100)}pct": int(hit[0]) + 1 if hit.size else np.nan,
        })
    return pd.DataFrame(rows)

# ==========================================================================
# STEP 6 — REPLACE the existing build_factor_table()
# ==========================================================================

def build_factor_table(summary_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """One factor-study table (architecture / augmentation / optimizer),
    ordered per FACTOR_STUDIES, formatted for the paper: accuracy in % with
    2 decimals, macro-F1 with 3 decimals, both as mean ± std over seeds, the
    difference to C1 in percentage points, parameters and time per epoch.
    Rounding happens only here, at display time."""
    config_order = FACTOR_STUDIES[factor]
    table = summary_df[summary_df["config_id"].isin(config_order)].copy()
    table["config_id"] = pd.Categorical(table["config_id"], categories=config_order, ordered=True)
    table = table.sort_values("config_id").reset_index(drop=True)

    table[factor] = table[factor].map(lambda x: PRETTY_NAMES.get(x, x))
    table["Params"] = (table["num_params"] / 1e6).map(lambda x: f"{x:.2f} M")
    table["Accuracy (%)"] = [
        f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(table["accuracy_mean"], table["accuracy_std"])
    ]
    table["Macro-F1"] = [
        f"{m:.3f} ± {s:.3f}" for m, s in zip(table["macro_f1_mean"], table["macro_f1_std"])
    ]
    table["Δ vs C1 (pp)"] = [
        "—" if cid == BASELINE_ID else f"{d:+.2f}"
        for cid, d in zip(table["config_id"], table["delta_vs_C1_pp"])
    ]
    table["Time/epoch (s)"] = table["time_per_epoch_mean"].map(lambda x: f"{x:.1f}")
    return table[["config_id", factor, "Params", "Accuracy (%)", "Macro-F1",
                  "Δ vs C1 (pp)", "Time/epoch (s)"]]

# ==========================================================================
# STEP 7 — ADD below best_worst_config_ids()
# ==========================================================================

def per_class_table(df: pd.DataFrame, config_id: str) -> pd.DataFrame:
    """Per-class precision / recall / F1 for one config, mean and std over
    seeds (each seed's value computed on its own predictions first)."""
    rows = df.loc[df["config_id"] == config_id]
    precision = pd.DataFrame(rows["precision_per_class"].tolist())
    recall = pd.DataFrame(rows["recall_per_class"].tolist())
    f1 = pd.DataFrame(rows["f1_per_class"].tolist())
    out = pd.DataFrame({
        "precision_mean": precision.mean(), "precision_std": precision.std(),
        "recall_mean": recall.mean(), "recall_std": recall.std(),
        "f1_mean": f1.mean(), "f1_std": f1.std(),
    })
    out.index.name = "class"
    return out.round(3)

# ==========================================================================
# STEP 7 — ADD below per_class_table()
# ==========================================================================

def recall_by_class_all_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per-class recall for every config (rows = classes, cols = configs).
    Shows whether the hardest class is the same everywhere (Section 3.7)."""
    out = {}
    for cid, g in df.groupby("config_id"):
        out[cid] = pd.DataFrame(g["recall_per_class"].tolist()).mean()
    return pd.DataFrame(out).round(3)

# ==========================================================================
# STEP 7 — ADD below recall_by_class_all_configs()
# ==========================================================================

def top_confusions(df: pd.DataFrame, config_id: str, k: int = 4) -> pd.DataFrame:
    """The k class pairs confused most often (both directions summed),
    from the element-wise mean confusion matrix over seeds."""
    cm = mean_confusion_matrix(df.loc[df["config_id"] == config_id, "confusion_matrix"].tolist())
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j] + cm[j, i]))
    out = pd.DataFrame(pairs, columns=["class_a", "class_b", "mean_errors"])
    return out.sort_values("mean_errors", ascending=False).head(k).round(1).reset_index(drop=True)

# ==========================================================================
# STEP 8 — REPLACE the existing factor_table_to_latex() (and delete the PRETTY_NAMES dict above it)
# ==========================================================================

def factor_table_to_latex(table: pd.DataFrame, factor: str, path: str) -> None:
    """Writes one formatted factor table (output of build_factor_table)
    to a .tex file, ready to \\input{} into the paper."""
    t = table.copy()
    t["config_id"] = t["config_id"].astype(str)
    for col in ["Accuracy (%)", "Macro-F1"]:
        t[col] = t[col].str.replace("±", r"$\pm$", regex=False)
    t["Δ vs C1 (pp)"] = t["Δ vs C1 (pp)"].str.replace("—", "--", regex=False)
    t = t.rename(columns={
        "config_id": "ID",
        factor: factor.capitalize(),
        "Accuracy (%)": r"Acc. (\%)",
        "Δ vs C1 (pp)": r"$\Delta$ (pp)",
        "Time/epoch (s)": "s/epoch",
    })
    t.to_latex(
        path,
        index=False,
        column_format="llrcccr",
        caption=f"{factor.capitalize()} study: mean $\\pm$ std over 3 seeds. "
                r"$\Delta$ is the difference to the baseline C1 in accuracy points.",
        label=f"tab:{factor}",
        position="t",
    )
