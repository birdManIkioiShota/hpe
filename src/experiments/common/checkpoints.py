from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import torch

from hpe.datasets.common import sha256_file


_WRAPPER_KEYS = ("model_state_dict", "state_dict", "model")


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    value: Any = payload
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    for key in _WRAPPER_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            value = candidate
            break
    if not value or not all(isinstance(key, str) for key in value):
        raise ValueError("No model state dict was found")
    if all(key.startswith("module.") for key in value):
        value = {key.removeprefix("module."): tensor for key, tensor in value.items()}
    if not all(torch.is_tensor(tensor) for tensor in value.values()):
        raise TypeError("State dict contains non-tensor values")
    state = {key: tensor.detach().cpu() for key, tensor in value.items()}
    return state, {
        "path": str(path),
        "sha256": sha256_file(path),
        "parameter_tensors": len(state),
    }


def interpolate_state_dict(
    base: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(base) != set(candidate):
        missing = sorted(set(base) - set(candidate))
        extra = sorted(set(candidate) - set(base))
        raise ValueError(f"Checkpoint keys differ: missing={missing[:5]}, extra={extra[:5]}")

    result: dict[str, torch.Tensor] = {}
    for key in base:
        left, right = base[key], candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(
                f"Tensor metadata differs for {key}: "
                f"{tuple(left.shape)}/{left.dtype} vs {tuple(right.shape)}/{right.dtype}"
            )
        if left.is_floating_point() or left.is_complex():
            result[key] = torch.lerp(left, right, alpha)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"Non-floating tensor differs for {key}")
            result[key] = left.clone()
    return result


def atomic_save_state_dict(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": dict(state_dict), "experiment": metadata}, temporary)
    os.replace(temporary, path)
