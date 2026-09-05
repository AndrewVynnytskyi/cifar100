# src/model.py
#
# [COPY-PASTE] FINE_TO_COARSE — the CIFAR-100 fine→superclass taxonomy.
# Derived directly from cifar-100-python/train's fine_labels/coarse_labels pickle
# fields (100 factual entries, not a code pattern) — verified against the actual
# dataset, not retyped from memory. index = fine label id, value = superclass id.
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

import torch.nn as nn


class VGG16CIFAR100(nn.Module):
    """Self-written VGG16, ported from notebooks/_reference_custom_cnn.ipynb and
    fixed for CIFAR-100. Two bugs in the original conv_l5 block are fixed here:

    1. conv_l5 had TWO MaxPool2d(2, 2) calls instead of one. conv_l1..conv_l4
       each downsample once (stride 2), so after 4 blocks a 32x32 input is
       already at 2x2. A second maxpool in conv_l5 would need the input to
       still be at least 2x2 going in and collapses it to 1x1 correctly only
       by accident — but the *real* bug is the extra pool made this stage
       inconsistent with every other stage (exactly one pool each) and left a
       stray nn.ReLU() sitting between the two MaxPool2d calls instead of
       after the BatchNorm/conv it belongs to.
    2. The 1x1 conv in stages 3-5 (conv_l3, conv_l4, conv_l5) used
       padding=1. For a 1x1 kernel that pads the input by 1 on each side and
       *grows* the spatial dims by 2 (out = in + 2*padding - kernel + 1),
       instead of preserving them. Fixed to padding=0.

    [WRITE FROM SCRATCH] forward() — assemble conv_l1..conv_l5, avgpool,
    flatten, classifier yourself. That assembly (and verifying the shape after
    each stage) is the actual exercise — see roadmap §3.1 for the spatial-
    dimension reasoning that motivated the avgpool fix in the first place.
    """

    def __init__(self, num_classes: int = 100, dropout_p: float = 0.5) -> None:
        super().__init__()

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_l3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_l4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_l5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 5 stride-2 maxpools: 32 -> 16 -> 8 -> 4 -> 2 -> 1. AdaptiveAvgPool2d
        # is defensive (keeps the model correct if input size ever changes)
        # rather than strictly required for a fixed 32x32 input.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # CIFAR-sized head (512 -> 512 -> num_classes), not the original
        # ImageNet-scale 512 -> 4096 -> 4096 -> num_classes: with 500
        # images/class, 4096-unit FC layers are wildly over-parameterised for
        # this dataset and mostly just memorise faster.
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    # [WRITE FROM SCRATCH] — write forward() yourself:
    #   feed x through conv_l1..conv_l5, self.avgpool, flatten to (B, 512),
    #   then self.classifier. Verify the shape after each stage first.


# --- Everything below is [WRITE FROM SCRATCH] per the roadmap §3.1 ---
#
#   def build_projection_matrix() -> torch.Tensor: ...
#   class VGG16DualHead(nn.Module): ...
#
# Write these yourself — see the roadmap for the projection-matrix shape
# reasoning and the dual-head forward/consumer pattern.
