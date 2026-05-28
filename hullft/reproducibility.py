"""Process-wide RNG and backend determinism controls."""

import os
import random

import numpy as np
import torch

_VALID_CUBLAS_WORKSPACE_CONFIGS = {":16:8", ":4096:8"}
_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":16:8"


def set_global_determinism(seed: int) -> None:
    """Seed every RNG used by the pipeline and enable strict torch determinism."""
    seed = int(seed)

    current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current_workspace not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DEFAULT_CUBLAS_WORKSPACE_CONFIG

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)
