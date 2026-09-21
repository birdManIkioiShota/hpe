from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from hpe.geometry.rotations import geodesic_error_degrees


def configure_update_scope(model: nn.Module, scope: str) -> None:
    if scope not in {"head", "layer4", "all"}:
        raise ValueError("scope must be one of: head, layer4, all")
    for name, parameter in model.named_parameters():
        if scope == "all":
            trainable = True
        elif scope == "layer4":
            trainable = name.startswith("layer4.") or name.startswith("linear_reg.")
        else:
            trainable = name.startswith("linear_reg.")
        parameter.requires_grad_(trainable)


def training_mode(model: nn.Module, scope: str) -> None:
    model.train()
    configure_update_scope(model, scope)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(
    model: nn.Module,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head = [value for name, value in model.named_parameters()
            if value.requires_grad and name.startswith("linear_reg.")]
    backbone = [value for name, value in model.named_parameters()
                if value.requires_grad and not name.startswith("linear_reg.")]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": head_lr, "name": "head"})
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(
        groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
):
    if total_updates <= 0 or warmup_updates < 0:
        raise ValueError("Invalid scheduler update counts")

    def factor(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def rotation_loss_rad(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return torch.deg2rad(
            geodesic_error_degrees(
                prediction.float(),
                target.float(),
                stable=True,
            )
        )


def grouped_metrics(errors: torch.Tensor, metadata: dict[str, Any]) -> dict[str, dict[str, float]]:
    errors = errors.detach().cpu().double()
    rear = torch.as_tensor(metadata["is_rear"], dtype=torch.bool)
    azimuth = torch.as_tensor(metadata["azimuth_deg"], dtype=torch.float64)
    bands = list(metadata["pose_band"])
    datasets = list(metadata["dataset"])
    front = torch.tensor([value == "front_lt60" for value in bands])
    side = torch.tensor([value == "side_60_to_lt120" for value in bands])
    rear_near = torch.tensor([value == "rear_120_to_lt150" for value in bands])
    rear_deep = torch.tensor([value == "rear_150_to_180" for value in bands])

    masks: dict[str, torch.Tensor] = {
        "overall": torch.ones(len(errors), dtype=torch.bool),
        "front": front,
        "side": side,
        "rear": rear,
        "retention": ~rear,
        "rear_negative": rear & (azimuth < 0),
        "rear_positive": rear & (azimuth > 0),
        "rear_120_to_lt150": rear_near,
        "rear_150_to_180": rear_deep,
        "rear_negative_120_to_lt150": rear_near & (azimuth < 0),
        "rear_negative_150_to_180": rear_deep & (azimuth < 0),
        "rear_positive_120_to_lt150": rear_near & (azimuth > 0),
        "rear_positive_150_to_180": rear_deep & (azimuth > 0),
    }
    for dataset in sorted(set(datasets)):
        dataset_mask = torch.tensor([value == dataset for value in datasets])
        masks[f"dataset:{dataset}"] = dataset_mask
        masks[f"dataset:{dataset}:front"] = dataset_mask & front
        masks[f"dataset:{dataset}:side"] = dataset_mask & side
        masks[f"dataset:{dataset}:rear"] = dataset_mask & rear

    result: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        values = errors[mask]
        if not len(values):
            continue
        result[name] = {
            "count": int(len(values)),
            "mean_deg": float(values.mean()),
            "median_deg": float(values.quantile(0.5)),
            "p90_deg": float(values.quantile(0.9)),
            "over90_percent": float((values > 90).double().mean() * 100),
        }
    return result


@torch.inference_mode()
def evaluate_pose_model(model: nn.Module, loader, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    errors: list[torch.Tensor] = []
    metadata_rows: dict[str, list[Any]] = {
        "is_rear": [], "azimuth_deg": [], "pose_band": [], "dataset": [],
    }
    running_sum = 0.0
    count = 0
    bar = tqdm(loader, desc="Internal dev", unit="batch")
    for images, target, metadata in bar:
        prediction = model(images.to(device, non_blocking=True))
        values = geodesic_error_degrees(
            prediction.double(),
            target.to(device, non_blocking=True).double(),
            stable=True,
        )
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite dev metric")
        errors.append(values.cpu())
        for key in metadata_rows:
            value = metadata[key]
            if torch.is_tensor(value):
                metadata_rows[key].extend(value.cpu().tolist())
            else:
                metadata_rows[key].extend(list(value))
        running_sum += float(values.sum())
        count += len(values)
        bar.set_postfix(geo_deg=f"{running_sum / count:.3f}")
    if not errors:
        raise ValueError("Empty dev set")
    return grouped_metrics(torch.cat(errors), metadata_rows)


REAR_SELECTION_GROUPS = (
    "rear_negative_120_to_lt150",
    "rear_negative_150_to_180",
    "rear_positive_120_to_lt150",
    "rear_positive_150_to_180",
)


def selection_score(
    metrics: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    *,
    retention_tolerance_deg: float,
) -> tuple[float, ...]:
    rear = metrics["rear"]
    front_mean_degradation = (
        metrics["front"]["mean_deg"] - baseline["front"]["mean_deg"]
    )
    side_mean_degradation = (
        metrics["side"]["mean_deg"] - baseline["side"]["mean_deg"]
    )
    front_p90_degradation = (
        metrics["front"]["p90_deg"] - baseline["front"]["p90_deg"]
    )
    side_p90_degradation = (
        metrics["side"]["p90_deg"] - baseline["side"]["p90_deg"]
    )
    retention_degradations = (
        front_mean_degradation,
        side_mean_degradation,
        front_p90_degradation,
        side_p90_degradation,
    )
    max_retention_degradation = max(retention_degradations)
    feasible = all(
        degradation <= retention_tolerance_deg
        for degradation in retention_degradations
    )
    worst_rear_p90 = max(metrics[name]["p90_deg"] for name in REAR_SELECTION_GROUPS)
    worst_rear_over90 = max(
        metrics[name]["over90_percent"] for name in REAR_SELECTION_GROUPS
    )
    if feasible:
        return (
            0.0,
            worst_rear_p90,
            rear["p90_deg"],
            worst_rear_over90,
            rear["mean_deg"],
            max_retention_degradation,
            metrics["front"]["mean_deg"],
            metrics["side"]["mean_deg"],
            metrics["front"]["p90_deg"],
            metrics["side"]["p90_deg"],
        )
    return (
        1.0,
        max_retention_degradation,
        worst_rear_p90,
        rear["p90_deg"],
        worst_rear_over90,
        rear["mean_deg"],
    )


def restore_horizontal_flip_rotation(rotation: torch.Tensor) -> torch.Tensor:
    mirror = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=rotation.dtype, device=rotation.device)
    )
    return mirror @ rotation @ mirror


def flip_consistency_loss_rad(
    original_prediction: torch.Tensor,
    flipped_prediction: torch.Tensor,
) -> torch.Tensor:
    if original_prediction.shape != flipped_prediction.shape:
        raise ValueError("Flip-consistency predictions must have identical shapes")
    restored = restore_horizontal_flip_rotation(flipped_prediction)
    return rotation_loss_rad(original_prediction, restored)


def autocast_context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError("precision must be bf16 or fp32")
