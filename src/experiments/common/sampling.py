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


def build_epoch_order(
    buckets: PoseBuckets,
    *,
    num_samples: int,
    rear_fraction: float,
    seed: int,
    epoch: int,
) -> list[tuple[int, int]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 < rear_fraction < 1.0:
        raise ValueError("rear_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)

    rear_total = int(round(num_samples * rear_fraction))
    retention_total = num_samples - rear_total
    base = rear_total // len(REAR_BUCKETS)
    remainder = rear_total % len(REAR_BUCKETS)

    order: list[int] = []
    for position, name in enumerate(REAR_BUCKETS):
        count = base + (1 if position < remainder else 0)
        order.extend(_sample_indices(buckets.rear[name], count, generator))
    order.extend(_sample_indices(buckets.retention, retention_total, generator))

    permutation = torch.randperm(len(order), generator=generator).tolist()
    return [(order[index], epoch) for index in permutation]
