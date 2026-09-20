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
