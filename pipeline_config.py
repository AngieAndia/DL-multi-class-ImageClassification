"""
pipeline_config.py
===================
Every constant, knob, and configuration table in one place, with NO logic.
Read this file top to bottom and you can see, at a glance, every choice made
in this project and why -- this is the file both of you should be able to
walk through out loud, since it's the closest thing to a checklist of your
Methods section.

Nothing here imports torch or downloads anything -- it's pure data.
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
# Optimizer defaults (Factor 3) -- literature-based, per-optimizer LR/decay.
#
# SGD tolerates and needs a substantially higher LR than the Adam family
# (Kingma & Ba, 2015, default lr=1e-3; He et al., 2016, use lr=0.1 for SGD on
# CIFAR-scale ResNets) -- one shared LR for all three would unfairly cripple
# SGD (too small) or destabilize Adam/AdamW (too large).
#
# Weight decay: Adam's L2 penalty is added directly to the gradient before
# adaptive per-parameter scaling, so it is NOT decoupled from the adaptive
# learning rate -- the same nominal weight_decay acts as a stronger effective
# penalty under Adam than under SGD or AdamW's decoupled decay (Loshchilov &
# Hutter, 2019). A smaller weight_decay is used for Adam (1e-4) than for
# SGD/AdamW (5e-4) to keep effective regularization comparable -- state this
# explicitly as a controlled design choice in the Methods section.
#
# SGD uses Nesterov-accelerated momentum (Sutskever et al., 2013), matching
# the original training protocols of the architectures we cite (He et al.,
# 2016; Huang et al., 2017), which also use Nesterov SGD.
# ---------------------------------------------------------------------------

OPTIMIZER_DEFAULTS = {
    "sgd": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
    "adam": {"lr": 0.001, "weight_decay": 1e-4},
    "adamw": {"lr": 0.001, "weight_decay": 5e-4},
}

# ---------------------------------------------------------------------------
# The 7-configuration OFAT design (Section 3.5 of the brief)
# ---------------------------------------------------------------------------

BASELINE = {"architecture": "resnet18", "augmentation": "crop_flip", "optimizer": "sgd"}

CONFIGS = {
    "C1": {**BASELINE},                                    # baseline (all 3 studies)
    "C2": {**BASELINE, "architecture": "simple_cnn"},       # architecture study
    "C3": {**BASELINE, "architecture": "densenet121"},      # architecture study
    "C4": {**BASELINE, "augmentation": "none"},             # augmentation study
    "C5": {**BASELINE, "augmentation": "strong"},           # augmentation study
    "C6": {**BASELINE, "optimizer": "adam"},                # optimizer study
    "C7": {**BASELINE, "optimizer": "adamw"},               # optimizer study
}

# Which configs belong to which factor study, C1 (baseline) common to all
# three -- used to build the three per-factor comparison tables the brief
# requires.
FACTOR_STUDIES = {
    "architecture": ["C2", "C1", "C3"],   # ordered: simple_cnn, resnet18*, densenet121
    "augmentation": ["C4", "C1", "C5"],   # ordered: none, crop_flip*, strong
    "optimizer": ["C1", "C6", "C7"],      # ordered: sgd*, adam, adamw
}

SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Fixed training settings shared by every run (Section 3.1's control list)
# ---------------------------------------------------------------------------

BATCH_SIZE = 128

# Epoch budget for the FULL experiment. 25 was chosen as a fixed ceiling:
# cosine annealing needs enough steps for a meaningful LR decay tail,
# DenseNet-121 (the slowest architecture) still completes 25 epochs x 21 runs
# in an affordable amount of Colab GPU time, and early stopping (patience=7
# on validation loss) lets configs that converge sooner stop early. This
# value and the patience below apply identically to every one of the 7
# configurations, per the OFAT control requirement.
FULL_RUN_EPOCHS = 25
FULL_RUN_PATIENCE = 7

# Smoke-test settings: small subset, few epochs, low patience -- purpose is
# only to catch bugs cheaply, not to produce meaningful numbers.
SMOKE_SUBSET_SIZE = 1000
SMOKE_EPOCHS = 2
SMOKE_PATIENCE = 2

RESULTS_CSV_DEFAULT = "all_results.csv"
