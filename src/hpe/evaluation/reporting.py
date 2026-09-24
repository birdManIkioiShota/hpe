from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hpe-matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )
        stream.write("\n")


def write_metric_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=False, float_format="%.8f")
        write_json(metrics_dir / f"{name}.json", table.to_dict(orient="records"))


def write_yaw_plot(output_dir: Path, yaw_bins: pd.DataFrame, dataset: str) -> None:
    if yaw_bins.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5))
    x_values = range(len(yaw_bins))
    axis.plot(x_values, yaw_bins["geodesic_error_deg_mean"], marker="o", label="SO(3) geodesic")
    axis.plot(x_values, yaw_bins["mean_axis_error_deg_mean"], marker="o", label="Euler-axis MAE")
    axis.set_xticks(list(x_values), yaw_bins["yaw_bin"], rotation=45, ha="right")
    axis.set_xlabel("Ground-truth source yaw bin (degrees)")
    axis.set_ylabel("Mean error (degrees)")
    axis.set_title(f"{dataset}: error by yaw")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / f"{dataset}_yaw_errors.png", dpi=160)
    plt.close(figure)


def write_yaw_radar_plot(
    output_dir: Path,
    yaw_bins: pd.DataFrame,
    dataset: str,
) -> None:
    if yaw_bins.empty:
        return
    required = {
        "yaw_bin",
        "yaw_error_deg_mean",
        "geodesic_error_deg_mean",
    }
    if not required.issubset(yaw_bins.columns):
        raise ValueError("yaw radar table is missing required columns")

    centers = []
    for label in yaw_bins["yaw_bin"].astype(str):
        left, right = label.split("_to_")
        centers.append((float(left) + float(right)) / 2.0)
    theta = np.deg2rad(np.asarray(centers, dtype=np.float64))

    figure, axis = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": "polar"},
    )
    for column, label in (
        ("yaw_error_deg_mean", "Yaw absolute error"),
        ("geodesic_error_deg_mean", "SO(3) geodesic"),
    ):
        values = yaw_bins[column].to_numpy(dtype=np.float64)
        axis.plot(
            np.append(theta, theta[0]),
            np.append(values, values[0]),
            marker="o",
            label=label,
        )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetagrids(
        np.arange(0, 360, 30),
        labels=[f"{degree}°" for degree in range(0, 360, 30)],
    )
    axis.set_title(f"{dataset}: mean error by 15-degree yaw bin")
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    figure.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        plots_dir / f"{dataset}_yaw15_radar.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_yaw_comparison_radar(
    output_path: Path,
    *,
    yaw_centers_deg: list[float],
    baseline_mean_deg: list[float],
    candidate_mean_deg: list[float],
    baseline_label: str,
    candidate_label: str,
    title: str,
) -> None:
    if not yaw_centers_deg:
        return
    if not (
        len(yaw_centers_deg)
        == len(baseline_mean_deg)
        == len(candidate_mean_deg)
    ):
        raise ValueError("yaw comparison radar lengths differ")

    theta = np.deg2rad(np.asarray(yaw_centers_deg, dtype=np.float64))
    figure, axis = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": "polar"},
    )
    for values, label in (
        (baseline_mean_deg, baseline_label),
        (candidate_mean_deg, candidate_label),
    ):
        array = np.asarray(values, dtype=np.float64)
        axis.plot(
            np.append(theta, theta[0]),
            np.append(array, array[0]),
            marker="o",
            label=label,
        )

    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    signed_ticks = np.arange(-180, 180, 30)
    axis.set_thetagrids(
        np.mod(signed_ticks, 360),
        labels=[f"{degree}°" for degree in signed_ticks],
    )
    axis.set_title(title)
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
