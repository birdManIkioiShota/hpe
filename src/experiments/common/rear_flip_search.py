from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from hpe.datasets.common import write_json_atomic


REAR_FRACTION_LEVELS = ("natural", "0.05", "0.25")
FLIP_WEIGHT_LEVELS = (0.0, 0.2, 1.0)


@dataclass(frozen=True)
class SearchCondition:
    condition_id: str
    rear_fraction_level: str
    rear_fraction: float
    flip_consistency_weight: float


def _fraction_label(level: str) -> str:
    return "natural" if level == "natural" else f"{round(float(level) * 100):03d}"


def _flip_label(weight: float) -> str:
    return f"{round(weight * 100):03d}"


def search_conditions(natural_rear_fraction: float) -> list[SearchCondition]:
    if not 0.0 < natural_rear_fraction < 1.0:
        raise ValueError("natural rear fraction must be between 0 and 1")
    result = []
    for rear_level in REAR_FRACTION_LEVELS:
        rear_fraction = (
            natural_rear_fraction if rear_level == "natural" else float(rear_level)
        )
        for flip_weight in FLIP_WEIGHT_LEVELS:
            result.append(
                SearchCondition(
                    condition_id=(
                        f"rear_{_fraction_label(rear_level)}_"
                        f"flip_{_flip_label(flip_weight)}"
                    ),
                    rear_fraction_level=rear_level,
                    rear_fraction=rear_fraction,
                    flip_consistency_weight=flip_weight,
                )
            )
    return result


def condition_records(conditions: list[SearchCondition]) -> list[dict[str, Any]]:
    return [asdict(condition) for condition in conditions]


def _metric(row: dict[str, Any], group: str, name: str) -> float:
    return float(row["dev"][group][name])


def summarize_search(run_dir: Path) -> tuple[Path, Path, Path]:
    config = json.loads((run_dir / "config.json").read_text())
    rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        condition_dir = run_dir / "conditions" / condition["condition_id"]
        status = json.loads((condition_dir / "status.json").read_text())
        if status.get("status") != "completed":
            raise ValueError(f"Condition is not completed: {condition['condition_id']}")
        final = json.loads(
            (condition_dir / "metrics" / f"epoch_{config['epochs']:03d}.json").read_text()
        )
        baseline = json.loads((condition_dir / "metrics" / "dev_baseline.json").read_text())
        row = {
            **condition,
            "epoch": final["epoch"],
            "rear_fraction_observed": final["train"]["rear_fraction_observed"],
            "rear_mean_deg": _metric(final, "rear", "mean_deg"),
            "rear_p90_deg": _metric(final, "rear", "p90_deg"),
            "rear_over90_percent": _metric(final, "rear", "over90_percent"),
            "front_mean_deg": _metric(final, "front", "mean_deg"),
            "side_mean_deg": _metric(final, "side", "mean_deg"),
            "front_p90_deg": _metric(final, "front", "p90_deg"),
            "side_p90_deg": _metric(final, "side", "p90_deg"),
            "front_delta_deg": (
                _metric(final, "front", "mean_deg")
                - float(baseline["front"]["mean_deg"])
            ),
            "side_delta_deg": (
                _metric(final, "side", "mean_deg")
                - float(baseline["side"]["mean_deg"])
            ),
            "front_p90_delta_deg": (
                _metric(final, "front", "p90_deg")
                - float(baseline["front"]["p90_deg"])
            ),
            "side_p90_delta_deg": (
                _metric(final, "side", "p90_deg")
                - float(baseline["side"]["p90_deg"])
            ),
            "best_epoch": status["best_epoch"],
        }
        rows.append(row)
        for epoch in range(1, int(config["epochs"]) + 1):
            epoch_result = json.loads(
                (condition_dir / "metrics" / f"epoch_{epoch:03d}.json").read_text()
            )
            epoch_rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "rear_fraction_level": condition["rear_fraction_level"],
                    "rear_fraction": condition["rear_fraction"],
                    "flip_consistency_weight": condition["flip_consistency_weight"],
                    "epoch": epoch,
                    "rear_mean_deg": _metric(epoch_result, "rear", "mean_deg"),
                    "rear_p90_deg": _metric(epoch_result, "rear", "p90_deg"),
                    "rear_over90_percent": _metric(
                        epoch_result, "rear", "over90_percent"
                    ),
                    "front_mean_deg": _metric(epoch_result, "front", "mean_deg"),
                    "side_mean_deg": _metric(epoch_result, "side", "mean_deg"),
                    "front_p90_deg": _metric(epoch_result, "front", "p90_deg"),
                    "side_p90_deg": _metric(epoch_result, "side", "p90_deg"),
                    "front_delta_deg": (
                        _metric(epoch_result, "front", "mean_deg")
                        - float(baseline["front"]["mean_deg"])
                    ),
                    "side_delta_deg": (
                        _metric(epoch_result, "side", "mean_deg")
                        - float(baseline["side"]["mean_deg"])
                    ),
                    "front_p90_delta_deg": (
                        _metric(epoch_result, "front", "p90_deg")
                        - float(baseline["front"]["p90_deg"])
                    ),
                    "side_p90_delta_deg": (
                        _metric(epoch_result, "side", "p90_deg")
                        - float(baseline["side"]["p90_deg"])
                    ),
                }
            )

    by_key = {
        (row["rear_fraction_level"], float(row["flip_consistency_weight"])): row
        for row in rows
    }
    effects = []
    for row in rows:
        flip_reference = by_key[(row["rear_fraction_level"], 0.0)]
        natural_reference = by_key[("natural", float(row["flip_consistency_weight"]))]
        effects.append(
            {
                "condition_id": row["condition_id"],
                "rear_fraction_level": row["rear_fraction_level"],
                "flip_consistency_weight": row["flip_consistency_weight"],
                "rear_mean_delta_vs_same_fraction_flip_zero": (
                    row["rear_mean_deg"] - flip_reference["rear_mean_deg"]
                ),
                "front_mean_delta_vs_same_fraction_flip_zero": (
                    row["front_mean_deg"] - flip_reference["front_mean_deg"]
                ),
                "side_mean_delta_vs_same_fraction_flip_zero": (
                    row["side_mean_deg"] - flip_reference["side_mean_deg"]
                ),
                "rear_mean_delta_vs_natural_same_flip": (
                    row["rear_mean_deg"] - natural_reference["rear_mean_deg"]
                ),
                "front_mean_delta_vs_natural_same_flip": (
                    row["front_mean_deg"] - natural_reference["front_mean_deg"]
                ),
                "side_mean_delta_vs_natural_same_flip": (
                    row["side_mean_deg"] - natural_reference["side_mean_deg"]
                ),
            }
        )

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_json = metrics_dir / "comparison.json"
    effects_json = metrics_dir / "effects.json"
    summary_csv = metrics_dir / "comparison.csv"
    epoch_json = metrics_dir / "epoch_comparison.json"
    epoch_csv = metrics_dir / "epoch_comparison.csv"
    write_json_atomic(summary_json, rows)
    write_json_atomic(effects_json, effects)
    write_json_atomic(epoch_json, epoch_rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with epoch_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    rear_levels = list(REAR_FRACTION_LEVELS)
    flip_levels = list(FLIP_WEIGHT_LEVELS)
    for filename, key, title in (
        ("rear_mean_heatmap.png", "rear_mean_deg", "Rear mean geodesic error (deg)"),
        (
            "retention_degradation_heatmap.png",
            "retention_degradation",
            "Worst front/side mean degradation (deg)",
        ),
    ):
        values = []
        for rear_level in rear_levels:
            line = []
            for flip_weight in flip_levels:
                row = by_key[(rear_level, flip_weight)]
                value = (
                    max(
                        row["front_delta_deg"],
                        row["side_delta_deg"],
                        row["front_p90_delta_deg"],
                        row["side_p90_delta_deg"],
                    )
                    if key == "retention_degradation"
                    else row[key]
                )
                line.append(value)
            values.append(line)
        figure, axis = plt.subplots(figsize=(6, 4))
        image = axis.imshow(values, aspect="auto")
        axis.set_xticks(range(len(flip_levels)), labels=[str(value) for value in flip_levels])
        axis.set_yticks(range(len(rear_levels)), labels=rear_levels)
        axis.set_xlabel("Flip-consistency weight")
        axis.set_ylabel("Rear sampling level")
        axis.set_title(title)
        for row_index, line in enumerate(values):
            for column_index, value in enumerate(line):
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(artifacts / filename, dpi=160)
        plt.close(figure)
    return summary_json, summary_csv, effects_json
