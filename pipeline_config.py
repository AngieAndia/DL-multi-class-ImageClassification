"""
pipeline_config.py
===================
Every constant, name, and configuration table in one place, with NO logic.

"""

# ---------------------------------------------------------------------------
# Dataset constants
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

# ---------------------------------------------------------------------------
# Optimizer defaults (Factor 3) - literature-based, per-optimizer LR/decay.
# ---------------------------------------------------------------------------

OPTIMIZER_DEFAULTS = {
    "sgd": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
    "adam": {"lr": 0.001, "weight_decay": 1e-4},
    "adamw": {"lr": 0.001, "weight_decay": 5e-4},
}

# ---------------------------------------------------------------------------
# The 7-configuration OFAT design 
# ---------------------------------------------------------------------------

BASELINE = {"architecture": "resnet18", "augmentation": "crop_flip", "optimizer": "sgd"}
BASELINE_ID = "C1"

CONFIGS = {
    "C1": {**BASELINE},                                     # baseline (all 3 studies)
    "C2": {**BASELINE, "architecture": "simple_cnn"},       # architecture study
    "C3": {**BASELINE, "architecture": "densenet121"},      # architecture study
    "C4": {**BASELINE, "augmentation": "none"},             # augmentation study
    "C5": {**BASELINE, "augmentation": "strong"},           # augmentation study
    "C6": {**BASELINE, "optimizer": "adam"},                # optimizer study
    "C7": {**BASELINE, "optimizer": "adamw"},               # optimizer study
}

# Used for graphs, tables, and results
PRETTY_NAMES = {
    "simple_cnn": "Simple CNN", "resnet18": "ResNet-18", "densenet121": "DenseNet-121",
    "none": "None", "crop_flip": "Crop + flip",
    "strong": "Crop + flip + AutoAugment + erasing",   # matches AUGMENTATIONS["strong"]
    "sgd": "SGD (Nesterov)", "adam": "Adam", "adamw": "AdamW",
}

# Which configs belong to which factor study, C1 (baseline) common to all three 
FACTOR_STUDIES = {
    "architecture": ["C2", "C1", "C3"],   # ordered: simple_cnn, resnet18*, densenet121
    "augmentation": ["C4", "C1", "C5"],   # ordered: none, crop_flip*, strong
    "optimizer": ["C1", "C6", "C7"],      # ordered: sgd*, adam, adamw
}

SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Fixed training settings shared by every run 
# ---------------------------------------------------------------------------

BATCH_SIZE = 128

# Epoch budget for the full experiment. 25 was chosen as a fixed ceiling.
FULL_RUN_EPOCHS = 25

RESULTS_CSV_DEFAULT = "all_results.csv"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
