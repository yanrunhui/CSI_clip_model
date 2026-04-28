from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr_scale: float = 1e-5 / 3e-4,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), min_lr_scale)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
