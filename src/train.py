# src/train.py
#
# [COPY-PASTE] Cutout + build_dataloaders — dataset/augmentation boilerplate.
# Per the roadmap §3.2: read where Cutout is applied before copying — it must
# run AFTER ToTensor()/Normalize(), otherwise the zeroed patch gets shifted away
# from zero by the per-channel normalization mean.
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Official per-channel CIFAR-100 statistics (not ImageNet's) — computed over the
# actual CIFAR-100 training set.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class Cutout:
    """Zero out one random size x size square patch of an image tensor."""

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1, y2 = max(y - self.size // 2, 0), min(y + self.size // 2, h)
        x1, x2 = max(x - self.size // 2, 0), min(x + self.size // 2, w)

        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            Cutout(size=16),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=cfg.data.root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=cfg.data.root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    return train_loader, test_loader


# --- Everything below is [WRITE FROM SCRATCH] per the roadmap §3.2 ---
#
#   class Trainer:
#       def __init__(self, model, cfg, device, output_dir, steps_per_epoch): ...
#       def _train_one_epoch(self, loader, epoch) -> float: ...
#       def _evaluate(self, loader, epoch) -> tuple[float, float]:  # top-1, top-5
#       def _save_checkpoint(self, path) -> None: ...
#       def fit(self, train_loader, test_loader) -> TrainerState: ...
#
# Write these yourself — SGD+Nesterov, label smoothing, OneCycleLR stepped per
# batch (not per epoch), gradient clipping, top-5 eval. See the roadmap for the
# reasoning behind each hyperparameter choice and the OneCycleLR placement trap.
