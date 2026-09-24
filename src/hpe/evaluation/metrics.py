from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


ERROR_COLUMNS = [
    "pitch_error_deg",
    "yaw_error_deg",
    "roll_error_deg",
    "mean_axis_error_deg",
    "geodesic_error_deg",
    "vec1_error_deg",
    "vec2_error_deg",
    "vec3_error_deg",
    "vmae_deg",
]


def metric_row(frame: pd.DataFrame, **labels: Any) -> dict[str, Any]:
    row: dict[str, Any] = {**labels, "count": int(len(frame))}
    for column in ERROR_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64)
        row[f"{column}_mean"] = float(np.mean(values)) if len(values) else None
    geodesic = frame["geodesic_error_deg"].to_numpy(dtype=np.float64)
    for quantile, name in ((0.5, "median"), (0.9, "p90"), (0.95, "p95")):
        row[f"geodesic_error_deg_{name}"] = (
            float(np.quantile(geodesic, quantile)) if len(geodesic) else None
        )
    row["geodesic_error_deg_gt90_count"] = int((geodesic > 90.0).sum())
    row["geodesic_error_deg_gt90_percent"] = (
        float((geodesic > 90.0).mean() * 100.0) if len(geodesic) else None
    )
    return row


def _group_rows(
    frame: pd.DataFrame,
    column: str,
    labels: Iterable[str],
    group_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        subset = frame[frame[column] == label]
        if len(subset):
            rows.append(metric_row(subset, **{group_name: label}))
    return rows


def yaw_bin_table(
    frame: pd.DataFrame,
    *,
    bin_size_deg: int = 30,
    include_empty: bool = False,
) -> pd.DataFrame:
    if bin_size_deg <= 0 or 360 % bin_size_deg:
        raise ValueError("yaw bin size must be a positive divisor of 360")
    edges = np.arange(
        -180,
        180 + bin_size_deg,
        bin_size_deg,
        dtype=float,
    )
    labels = [
        f"{int(left)}_to_{int(right)}"
        for left, right in zip(edges[:-1], edges[1:])
    ]
    data = frame.copy()
    clipped_yaw = data["gt_source_yaw_deg"].clip(
        -180.0,
        np.nextafter(180.0, -np.inf),
    )
    data["yaw_bin"] = pd.cut(
        clipped_yaw,
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")
    rows: list[dict[str, Any]] = []
    for label in labels:
        subset = data[data["yaw_bin"] == label]
        if include_empty or len(subset):
            rows.append(metric_row(subset, yaw_bin=label))
    return pd.DataFrame(rows)


def metric_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = frame.copy()
    absolute_yaw = data["gt_source_yaw_deg"].abs()
    data["yaw_band"] = pd.cut(
        absolute_yaw,
        bins=[-np.inf, 60.0, 120.0, np.inf],
        labels=["front_lt60", "side_60_to_lt120", "rear_ge120"],
        right=False,
    ).astype("string")

    head_scale = np.sqrt(data["bbox_width"] * data["bbox_height"])
    data["head_size_bin"] = pd.cut(
        head_scale,
        bins=[-np.inf, 32.0, 64.0, 128.0, np.inf],
        labels=["lt32", "32_to_lt64", "64_to_lt128", "ge128"],
        right=False,
    ).astype("string")

    data["occlusion_bin"] = pd.cut(
        data["occlusion_percent"],
        bins=[0.0, 25.0, 50.0, 75.0, 90.0, np.inf],
        labels=["0_to_lt25", "25_to_lt50", "50_to_lt75", "75_to_lt90", "ge90"],
        right=False,
        include_lowest=True,
    ).astype("string")

    return {
        "overall": pd.DataFrame([metric_row(data)]),
        "yaw_bands": pd.DataFrame(
            _group_rows(
                data,
                "yaw_band",
                ["front_lt60", "side_60_to_lt120", "rear_ge120"],
                "yaw_band",
            )
        ),
        "yaw_bins": yaw_bin_table(data, bin_size_deg=30),
        "head_size_bins": pd.DataFrame(
            _group_rows(
                data,
                "head_size_bin",
                ["lt32", "32_to_lt64", "64_to_lt128", "ge128"],
                "head_size_bin",
            )
        ),
        "occlusion_bins": pd.DataFrame(
            _group_rows(
                data,
                "occlusion_bin",
                ["0_to_lt25", "25_to_lt50", "50_to_lt75", "75_to_lt90", "ge90"],
                "occlusion_bin",
            )
        ),
    }
