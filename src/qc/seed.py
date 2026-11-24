"""可复现性：固定随机种子。"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 random / numpy / torch / Python hash 种子，并尽量开启确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
