from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.pose import (
    UndefinedAzimuthError,
    azimuth_side,
    forward_azimuth_degrees,
    pose_band,
)


REAR_BUCKETS = (
    "negative:rear_120_to_lt150",
    "negative:rear_150_to_180",
    "positive:rear_120_to_lt150",
    "positive:rear_150_to_180",
)


@dataclass(frozen=True)
class PoseBuckets:
    rear: dict[str, tuple[int, ...]]
    retention: tuple[int, ...]

    @property
    def rear_count(self) -> int:
        return sum(len(values) for values in self.rear.values())

    @property
    def total_count(self) -> int:
        return self.rear_count + len(self.retention)

    @property
    def natural_rear_fraction(self) -> float:
        return self.rear_count / self.total_count


@dataclass(frozen=True)
class EpochSamplePlan:
    order: list[tuple[int, int]]
    rear_fraction_requested: float
    rear_fraction_observed: float
    rear_bucket_policy: str
    bucket_draws: dict[str, int]
    unique_samples: int
    repeated_draws: int


def scan_pose_buckets(manifests: list[Path]) -> PoseBuckets:
    rear: dict[str, list[int]] = {name: [] for name in REAR_BUCKETS}
    retention: list[int] = []
    global_index = 0
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                try:
                    azimuth = forward_azimuth_degrees(
                        np.asarray(row["rotation_matrix"], dtype=np.float64)
                    )
                except UndefinedAzimuthError:
                    retention.append(global_index)
                    global_index += 1
                    continue
                band = pose_band(azimuth)
                if band.startswith("rear_"):
                    side = azimuth_side(azimuth)
                    key = f"{side}:{band}"
                    if key not in rear:
                        raise ValueError(f"Rear sample has unsupported azimuth side: {key}")
                    rear[key].append(global_index)
                else:
                    retention.append(global_index)
                global_index += 1
    if not retention:
        raise ValueError("No retention samples were found")
    return PoseBuckets(
        rear={name: tuple(values) for name, values in rear.items()},
        retention=tuple(retention),
    )


def _sample_indices(values: tuple[int, ...], count: int, generator: torch.Generator) -> list[int]:
    if count == 0:
        return []
    if not values:
        raise ValueError("Cannot sample from an empty pose bucket")
    source = torch.tensor(values, dtype=torch.int64)
    result: list[int] = []
    while len(result) < count:
        order = torch.randperm(len(source), generator=generator)
        result.extend(source[order].tolist())
    return result[:count]


def _allocate_rear_draws(
    buckets: PoseBuckets,
    rear_total: int,
    policy: str,
) -> dict[str, int]:
    if policy not in {"equal", "proportional"}:
        raise ValueError("rear bucket policy must be equal or proportional")
    if policy == "equal":
        weights = {name: 1.0 for name in REAR_BUCKETS}
    else:
        weights = {name: float(len(buckets.rear[name])) for name in REAR_BUCKETS}
    denominator = sum(weights.values())
    raw = {name: rear_total * weights[name] / denominator for name in REAR_BUCKETS}
    result = {name: int(raw[name]) for name in REAR_BUCKETS}
    remaining = rear_total - sum(result.values())
    priority = sorted(REAR_BUCKETS, key=lambda name: (-(raw[name] - result[name]), name))
    for name in priority[:remaining]:
        result[name] += 1
    return result


def build_epoch_plan(
    buckets: PoseBuckets,
    *,
    num_samples: int,
    rear_fraction: float,
    seed: int,
    epoch: int,
    rear_bucket_policy: str = "equal",
) -> EpochSamplePlan:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 < rear_fraction < 1.0:
        raise ValueError("rear_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)

    rear_total = int(round(num_samples * rear_fraction))
    retention_total = num_samples - rear_total
    bucket_draws = _allocate_rear_draws(buckets, rear_total, rear_bucket_policy)

    order: list[int] = []
    for name in REAR_BUCKETS:
        order.extend(_sample_indices(buckets.rear[name], bucket_draws[name], generator))
    order.extend(_sample_indices(buckets.retention, retention_total, generator))

    permutation = torch.randperm(len(order), generator=generator).tolist()
    shuffled = [order[index] for index in permutation]
    return EpochSamplePlan(
        order=[(index, epoch) for index in shuffled],
        rear_fraction_requested=rear_fraction,
        rear_fraction_observed=rear_total / num_samples,
        rear_bucket_policy=rear_bucket_policy,
        bucket_draws=bucket_draws,
        unique_samples=len(set(shuffled)),
        repeated_draws=len(shuffled) - len(set(shuffled)),
    )


def build_epoch_order(
    buckets: PoseBuckets,
    *,
    num_samples: int,
    rear_fraction: float,
    seed: int,
    epoch: int,
    rear_bucket_policy: str = "equal",
) -> list[tuple[int, int]]:
    return build_epoch_plan(
        buckets,
        num_samples=num_samples,
        rear_fraction=rear_fraction,
        seed=seed,
        epoch=epoch,
        rear_bucket_policy=rear_bucket_policy,
    ).order
