"""
pipeline_models.py
====================
Factor 1 (architecture) and Factor 3 (optimizer construction). All models
trained from scratch (weights=None), adapted for 32x32 CIFAR-10 input.
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from pipeline_config import NUM_CLASSES, OPTIMIZER_DEFAULTS


# ---------------------------------------------------------------------------
# Architectures
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
    stride-2 first conv + maxpool shrink a 224x224 ImageNet image 4x before
    the first residual block -- on a 32x32 image that would collapse it to
    8x8 almost immediately. Replace conv1 with a 3x3 stride-1 convolution
    and remove the maxpool (nn.Identity) -- the standard CIFAR adaptation
    (He et al., 2016, used a similar 3x3-stem variant for CIFAR-10)."""
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_densenet121_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """DenseNet-121, weights=None, same stem-adaptation logic as ResNet-18
    above, applied to DenseNet's stem (features.conv0/pool0)."""
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
# Optimizers -- values/justification live in pipeline_config.OPTIMIZER_DEFAULTS
# ---------------------------------------------------------------------------

def build_optimizer(name: str, params) -> optim.Optimizer:
    cfg = dict(OPTIMIZER_DEFAULTS[name])
    if name == "sgd":
        return optim.SGD(params, **cfg)
    if name == "adam":
        return optim.Adam(params, **cfg)
    if name == "adamw":
        return optim.AdamW(params, **cfg)
    raise ValueError(f"Unknown optimizer: {name}")
