from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from hpe.data.dataset import evaluation_transform, read_rgb_image


TEACHER_IDS = ("sixdrepnet360_base", "semiuhpe_effnetv2s", "whenet")
ADOPTION_RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)


def signed_yaw_degrees(value: float) -> float:
    """Convert YawPose [0, 360) yaw to the fixed [-180, 180) convention."""
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(result) < 1e-12 else result


def circular_difference_degrees(a: float, b: float) -> float:
    """Return signed shortest angular difference a-b in [-180, 180)."""
    return signed_yaw_degrees(float(a) - float(b))


def circular_distance_degrees(a: float, b: float) -> float:
    return abs(circular_difference_degrees(a, b))


def rear_bucket(yaw_deg: float) -> str:
    yaw = signed_yaw_degrees(yaw_deg)
    absolute = abs(yaw)
    if absolute < 120.0:
        raise ValueError("rear bucket requires |yaw| >= 120 degrees")
    side = "negative" if yaw < 0.0 else "positive"
    band = "120_to_lt150" if absolute < 150.0 else "150_to_180"
    return f"{side}:rear_{band}"


def yaw_from_rotation_matrix_rad(rotation: torch.Tensor) -> torch.Tensor:
    """Return head-forward azimuth atan2(R[0,2], R[2,2]) in radians."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in a 3x3 matrix")
    return torch.atan2(rotation[..., 0, 2], rotation[..., 2, 2])


def yaw_loss_rad(prediction: torch.Tensor, target_yaw_rad: torch.Tensor) -> torch.Tensor:
    """Per-sample circular absolute yaw error in radians."""
    predicted = yaw_from_rotation_matrix_rad(prediction.float())
    target = target_yaw_rad.to(device=predicted.device, dtype=predicted.dtype)
    delta = torch.atan2(torch.sin(predicted - target), torch.cos(predicted - target))
    return delta.abs()


def _average_percentile_ranks(values: Iterable[float]) -> list[float]:
    """Return deterministic [0,1] percentile ranks with average rank for ties."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("percentile ranking requires a non-empty 1D array")
    if not np.isfinite(array).all():
        raise ValueError("percentile ranking values must be finite")
    if len(array) == 1:
        return [0.0]
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        value = array[order[start]]
        while end < len(order) and array[order[end]] == value:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return (ranks / (len(array) - 1)).tolist()


def reliability_records(
    candidates: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Compute fixed three-teacher reliability metrics and ranking."""
    if set(predictions) != set(TEACHER_IDS):
        raise ValueError(f"teacher IDs must be exactly {TEACHER_IDS}")
    if not candidates:
        raise ValueError("no YawPose rear candidates")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        instance_id = str(candidate["instance_id"])
        yaw = float(candidate["canonical_yaw_deg"])
        teacher_yaws: dict[str, float] = {}
        for teacher_id in TEACHER_IDS:
            try:
                value = float(predictions[teacher_id][instance_id])
            except KeyError as exc:
                raise ValueError(
                    f"missing {teacher_id} prediction for {instance_id}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite {teacher_id} yaw for {instance_id}")
            teacher_yaws[teacher_id] = signed_yaw_degrees(value)
        gt_errors = [circular_distance_degrees(value, yaw) for value in teacher_yaws.values()]
        pairwise = []
        teacher_values = list(teacher_yaws.values())
        for left in range(len(teacher_values)):
            for right in range(left + 1, len(teacher_values)):
                pairwise.append(circular_distance_degrees(teacher_values[left], teacher_values[right]))
        row = {
            **candidate,
            "teacher_yaw_deg": teacher_yaws,
            "teacher_gt_error_deg": {
                teacher_id: circular_distance_degrees(value, yaw)
                for teacher_id, value in teacher_yaws.items()
            },
            "gt_median_error_deg": float(np.median(gt_errors)),
            "gt_max_error_deg": float(np.max(gt_errors)),
            "teacher_dispersion_deg": float(np.median(pairwise)),
            "agree_10": sum(value <= 10.0 for value in gt_errors) / len(gt_errors),
            "agree_20": sum(value <= 20.0 for value in gt_errors) / len(gt_errors),
            "agree_30": sum(value <= 30.0 for value in gt_errors) / len(gt_errors),
        }
        rows.append(row)

    r_median = _average_percentile_ranks(row["gt_median_error_deg"] for row in rows)
    r_dispersion = _average_percentile_ranks(row["teacher_dispersion_deg"] for row in rows)
    r_max = _average_percentile_ranks(row["gt_max_error_deg"] for row in rows)
    for row, median_rank, dispersion_rank, max_rank in zip(
        rows, r_median, r_dispersion, r_max
    ):
        row["r_median"] = median_rank
        row["r_dispersion"] = dispersion_rank
        row["r_max"] = max_rank
        row["reliability_score"] = (median_rank + dispersion_rank + max_rank) / 3.0

    rows.sort(
        key=lambda row: (
            row["reliability_score"],
            row["gt_median_error_deg"],
            row["teacher_dispersion_deg"],
            row["gt_max_error_deg"],
            str(row["instance_id"]),
        )
    )
    denominator = max(1, len(rows) - 1)
    for index, row in enumerate(rows):
        row["reliability_rank"] = index + 1
        row["reliability_percentile"] = index / denominator
    return rows


def subset_records(rows: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    if ratio not in ADOPTION_RATIOS:
        raise ValueError(f"unsupported adoption ratio: {ratio}")
    count = len(rows) if ratio == 1.0 else max(1, math.ceil(len(rows) * ratio))
    return rows[:count]


@dataclass(frozen=True)
class YawPoseSamplePlan:
    order: list[tuple[int, int]]
    unique_samples: int
    repeated_draws: int
    total_draws: int
    source_draws: dict[str, int]
    rear_bucket_draws: dict[str, int]


def build_yawpose_epoch_plan(
    records: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
    epoch: int,
) -> YawPoseSamplePlan:
    if not records or num_samples <= 0:
        raise ValueError("YawPose sampling requires records and positive num_samples")
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003 + 97_531)
    source = torch.arange(len(records), dtype=torch.int64)
    indices: list[int] = []
    while len(indices) < num_samples:
        indices.extend(source[torch.randperm(len(source), generator=generator)].tolist())
    indices = indices[:num_samples]
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    indices = [indices[index] for index in permutation]
    source_draws: dict[str, int] = {}
    bucket_draws: dict[str, int] = {}
    for index in indices:
        record = records[index]
        source_name = str(record.get("source", "unknown"))
        source_draws[source_name] = source_draws.get(source_name, 0) + 1
        bucket = str(record["rear_bucket"])
        bucket_draws[bucket] = bucket_draws.get(bucket, 0) + 1
    return YawPoseSamplePlan(
        order=[(index, epoch) for index in indices],
        unique_samples=len(set(indices)),
        repeated_draws=len(indices) - len(set(indices)),
        total_draws=len(indices),
        source_draws=source_draws,
        rear_bucket_draws=bucket_draws,
    )


class YawPoseDataset(Dataset):
    """Yaw-only YawPose dataset; horizontal flip is intentionally absent."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        *,
        augment: bool,
        seed: int,
        read_error_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.augment = augment
        self.seed = seed
        self.read_error_dir = read_error_dir
        self.transform = evaluation_transform()
        with self.manifest_path.open(encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError("empty YawPose manifest")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        record = self.records[index]
        image = read_rgb_image(
            self.project_root / record["image_path"],
            expected_sha256=record.get("image_sha256"),
            error_dir=self.read_error_dir,
        )
        if self.augment:
            rng = np.random.default_rng(
                np.uint64((self.seed * 1_000_003 + epoch * 97_409 + index) % 2**63)
            )
            for operation in (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation):
                image = operation(image, float(rng.uniform(0.8, 1.2)))
        tensor = self.transform(image)
        yaw_rad = math.radians(float(record["canonical_yaw_deg"]))
        metadata = {
            "instance_id": str(record["instance_id"]),
            "source": str(record.get("source", "unknown")),
            "rear_bucket": str(record["rear_bucket"]),
        }
        return tensor, torch.tensor(yaw_rad, dtype=torch.float32), metadata


def _yaw_metric_values(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().double()
    return {
        "count": int(len(values)),
        "mean_deg": float(values.mean()),
        "median_deg": float(values.quantile(0.5)),
        "p90_deg": float(values.quantile(0.9)),
        "over30_percent": float((values > 30.0).double().mean() * 100.0),
        "over60_percent": float((values > 60.0).double().mean() * 100.0),
        "over90_percent": float((values > 90.0).double().mean() * 100.0),
    }


def grouped_forward_yaw_metrics(
    errors: torch.Tensor,
    metadata: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    errors = errors.detach().cpu().double()
    azimuth = torch.as_tensor(metadata["azimuth_deg"], dtype=torch.float64)
    bands = list(metadata["pose_band"])
    front = torch.tensor([value == "front_lt60" for value in bands])
    side = torch.tensor([value == "side_60_to_lt120" for value in bands])
    rear = torch.tensor([value.startswith("rear_") for value in bands])
    rear_near = torch.tensor([value == "rear_120_to_lt150" for value in bands])
    rear_deep = torch.tensor([value == "rear_150_to_180" for value in bands])
    masks = {
        "overall": torch.ones(len(errors), dtype=torch.bool),
        "front": front,
        "side": side,
        "rear": rear,
        "rear_negative_120_to_lt150": rear_near & (azimuth < 0),
        "rear_negative_150_to_180": rear_deep & (azimuth < 0),
        "rear_positive_120_to_lt150": rear_near & (azimuth > 0),
        "rear_positive_150_to_180": rear_deep & (azimuth > 0),
    }
    return {
        name: _yaw_metric_values(errors[mask])
        for name, mask in masks.items()
        if bool(mask.any())
    }


@torch.inference_mode()
def evaluate_forward_yaw_model(model, loader, device: torch.device) -> dict[str, dict[str, float]]:
    """Evaluate full-range head-forward azimuth on internal rotation-matrix dev data."""
    model.eval()
    errors: list[torch.Tensor] = []
    metadata_rows: dict[str, list[Any]] = {"azimuth_deg": [], "pose_band": []}
    for images, target, metadata in loader:
        prediction = model(images.to(device, non_blocking=True)).float()
        target = target.to(device, non_blocking=True).float()
        pred_yaw = yaw_from_rotation_matrix_rad(prediction)
        target_yaw = yaw_from_rotation_matrix_rad(target)
        delta = torch.atan2(torch.sin(pred_yaw - target_yaw), torch.cos(pred_yaw - target_yaw))
        errors.append(torch.rad2deg(delta.abs()).cpu())
        metadata_rows["azimuth_deg"].extend(
            metadata["azimuth_deg"].cpu().tolist()
            if torch.is_tensor(metadata["azimuth_deg"])
            else list(metadata["azimuth_deg"])
        )
        metadata_rows["pose_band"].extend(list(metadata["pose_band"]))
    if not errors:
        raise ValueError("empty internal dev loader")
    return grouped_forward_yaw_metrics(torch.cat(errors), metadata_rows)
