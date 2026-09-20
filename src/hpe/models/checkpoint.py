from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hpe.datasets.common import sha256_file


_WRAPPER_KEYS = ("model_state_dict", "state_dict", "model")


def _unwrap_state_dict(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("The checkpoint is not a state-dict mapping.")
    for key in _WRAPPER_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            value = candidate
            break
    if not value or not all(isinstance(key, str) for key in value):
        raise ValueError("No model state dict was found in the checkpoint.")
    if all(key.startswith("module.") for key in value):
        value = {key.removeprefix("module."): tensor for key, tensor in value.items()}
    return value


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "parameter_tensors": len(state_dict),
    }
