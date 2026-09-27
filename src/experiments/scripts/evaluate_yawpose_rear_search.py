"""Evaluate all YawPose rear-search conditions and add full-range yaw metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.common.run_directory import experiment_run_path
from experiments.common.yawpose_search import WEIGHT_SEARCH_KIND, condition_records
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.evaluation import compare_runs
from hpe.evaluation.comparison import _check_compatible
from hpe.evaluation.reporting import write_yaw_comparison_radar
from training.prepare_data import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="yawpose_rear_stratified_search")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument(
        "--checkpoint-choice",
        choices=("best", "final"),
        default="final",
    )
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-sample-limit", type=int, default=10000)
    return parser


def _safe_name(value: str, kind: str) -> str:
    if (
        not value
        or value in {".", "..", "comparisons"}
        or Path(value).name != value
    ):
        raise ValueError(f"{kind} must be one path-safe name")
    return value


def _safe_suffix(value: str) -> str:
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in value
    ):
        raise ValueError("name suffix contains unsupported characters")
    return value


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(delta)


def _metric(error: np.ndarray) -> dict:
    if not len(error):
        return {
            "count": 0,
            "mean_deg": None,
            "median_deg": None,
            "p90_deg": None,
            "over30_percent": None,
            "over60_percent": None,
            "over90_percent": None,
        }
    return {
        "count": int(len(error)),
        "mean_deg": float(np.mean(error)),
        "median_deg": float(np.median(error)),
        "p90_deg": float(np.quantile(error, 0.9)),
        "over30_percent": float(np.mean(error > 30.0) * 100.0),
        "over60_percent": float(np.mean(error > 60.0) * 100.0),
        "over90_percent": float(np.mean(error > 90.0) * 100.0),
    }


def _masks(yaw: np.ndarray) -> dict[str, np.ndarray]:
    absolute = np.abs(yaw)
    return {
        "overall": np.ones(len(yaw), dtype=bool),
        "front_lt60": absolute < 60.0,
        "side_60_to_lt120": (absolute >= 60.0) & (absolute < 120.0),
        "rear_ge120": absolute >= 120.0,
        "rear_negative_120_to_lt150": (
            (yaw <= -120.0) & (yaw > -150.0)
        ),
        "rear_negative_150_to_180": yaw <= -150.0,
        "rear_positive_120_to_lt150": (
            (yaw >= 120.0) & (yaw < 150.0)
        ),
        "rear_positive_150_to_180": yaw >= 150.0,
    }


def _yaw15_masks(yaw: np.ndarray) -> dict[str, np.ndarray]:
    signed = (np.asarray(yaw, dtype=np.float64) + 180.0) % 360.0 - 180.0
    result: dict[str, np.ndarray] = {}
    for left in range(-180, 180, 15):
        right = left + 15
        result[f"yaw_{left}_to_lt{right}"] = (
            (signed >= left) & (signed < right)
        )
    return result


def _bootstrap_ci(
    difference: np.ndarray,
    *,
    repetitions: int,
    sample_limit: int,
) -> tuple[float | None, float | None]:
    if not len(difference):
        return None, None
    rng = np.random.default_rng(0)
    values = np.asarray(difference, dtype=np.float64)
    if len(values) > sample_limit:
        values = rng.choice(
            values,
            size=sample_limit,
            replace=False,
        )
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(values), size=len(values))
        means[index] = values[sample].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _yaw_comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    repetitions: int,
    sample_limit: int,
) -> list[dict]:
    baseline_run = json.loads((baseline_dir / "run.json").read_text())
    candidate_run = json.loads((candidate_dir / "run.json").read_text())
    baseline_datasets = {
        row["dataset"] for row in baseline_run["datasets"]
    }
    candidate_datasets = {
        row["dataset"] for row in candidate_run["datasets"]
    }
    if baseline_datasets != candidate_datasets:
        raise ValueError("baseline/candidate evaluation datasets differ")

    rows = []
    for dataset in sorted(baseline_datasets):
        baseline = pd.read_csv(
            baseline_dir / "predictions" / f"{dataset}.csv.gz"
        )
        candidate = pd.read_csv(
            candidate_dir / "predictions" / f"{dataset}.csv.gz"
        )
        columns = [
            "instance_id",
            "gt_source_yaw_deg",
            "pred_R_02",
            "pred_R_22",
        ]
        merged = baseline[columns].merge(
            candidate[columns],
            on="instance_id",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if len(merged) != len(baseline) or len(merged) != len(candidate):
            raise ValueError(f"prediction instance mismatch for {dataset}")
        gt = merged[
            "gt_source_yaw_deg_baseline"
        ].to_numpy(dtype=np.float64)
        if not np.allclose(
            gt,
            merged["gt_source_yaw_deg_candidate"],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(f"source yaw differs for {dataset}")

        baseline_yaw = np.degrees(
            np.arctan2(
                merged["pred_R_02_baseline"],
                merged["pred_R_22_baseline"],
            )
        )
        candidate_yaw = np.degrees(
            np.arctan2(
                merged["pred_R_02_candidate"],
                merged["pred_R_22_candidate"],
            )
        )
        baseline_error = _circular_error(baseline_yaw, gt)
        candidate_error = _circular_error(candidate_yaw, gt)

        masks = _masks(gt)
        if dataset == "agora_hpe":
            masks.update(_yaw15_masks(gt))
        for group, mask in masks.items():
            is_agora_yaw15 = (
                dataset == "agora_hpe"
                and group.startswith("yaw_")
            )
            if not mask.any() and not is_agora_yaw15:
                continue
            base_metric = _metric(baseline_error[mask])
            candidate_metric = _metric(candidate_error[mask])
            difference = (
                candidate_error[mask] - baseline_error[mask]
            )
            ci_low, ci_high = _bootstrap_ci(
                difference,
                repetitions=repetitions,
                sample_limit=sample_limit,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "baseline": base_metric,
                    "candidate": candidate_metric,
                    "mean_delta_candidate_minus_baseline_deg": (
                        float(np.mean(difference))
                        if len(difference)
                        else None
                    ),
                    "mean_delta_ci95_low": ci_low,
                    "mean_delta_ci95_high": ci_high,
                }
            )
    return rows


def _write_agora_yaw15_comparison_plot(
    rows: list[dict],
    *,
    output_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["dataset"] == "agora_hpe"
        and row["group"].startswith("yaw_")
    ]
    if not selected:
        return
    if len(selected) != 24:
        raise ValueError("AGORA yaw comparison must contain 24 bins")

    centers = []
    baseline_values = []
    candidate_values = []
    for row in selected:
        label = row["group"][len("yaw_"):]
        left, right = label.split("_to_lt")
        centers.append((float(left) + float(right)) / 2.0)
        baseline_values.append(
            float(row["baseline"]["mean_deg"])
            if row["baseline"]["mean_deg"] is not None
            else float("nan")
        )
        candidate_values.append(
            float(row["candidate"]["mean_deg"])
            if row["candidate"]["mean_deg"] is not None
            else float("nan")
        )

    write_yaw_comparison_radar(
        output_path,
        yaw_centers_deg=centers,
        baseline_mean_deg=baseline_values,
        candidate_mean_deg=candidate_values,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        title="AGORA-HPE: head-forward yaw error by 15-degree bin",
    )


def _validate_weight_evaluation(
    result_dir: Path,
    checkpoint: Path,
    reference_metadata: dict,
) -> None:
    metadata = json.loads((result_dir / "run.json").read_text())
    if metadata["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise ValueError(
            f"evaluation checkpoint differs from training: {result_dir.name}"
        )
    _check_compatible(reference_metadata, metadata)


def _weight_comparisons(
    output_root: Path,
    conditions: list[dict],
    *,
    suffix: str,
    reference_key: str,
    repetitions: int,
    sample_limit: int,
) -> None:
    """Compare trained candidates to the no-YawPose control in the same suite."""
    control = next(row for row in conditions if row["subset"] == "base")
    control_name = control["condition_id"] + suffix
    control_dir = output_root / "conditions" / control_name
    control_run = json.loads((control_dir / "run.json").read_text())
    control_datasets = {row["dataset"]: row for row in control_run["datasets"]}
    rows = []
    for condition in conditions:
        name = condition["condition_id"] + suffix
        result_dir = output_root / "conditions" / name
        result_run = json.loads((result_dir / "run.json").read_text())
        _check_compatible(control_run, result_run)
        comparison_name = f"{control_name}_vs_{name}"
        if name != control_name:
            comparison_dir = output_root / "comparisons" / comparison_name
            if comparison_dir.exists():
                previous = json.loads((comparison_dir / "run.json").read_text())
                expected = {
                    "baseline_checkpoint_sha256": control_run["checkpoint"]["sha256"],
                    "candidate_checkpoint_sha256": result_run["checkpoint"]["sha256"],
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_sample_limit": sample_limit,
                }
                if any(previous.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        f"saved control comparison settings changed: {comparison_name}"
                    )
            else:
                compare_runs(
                    control_dir,
                    result_dir,
                    output_root,
                    name=comparison_name,
                    bootstrap_repetitions=repetitions,
                    bootstrap_sample_limit=sample_limit,
                )
        yaw_rows = _yaw_comparison(
            control_dir,
            result_dir,
            repetitions=repetitions,
            sample_limit=sample_limit,
        )
        if name != control_name:
            write_json_atomic(
                output_root / "yaw_comparisons" / f"{comparison_name}.json", yaw_rows
            )
            _write_agora_yaw15_comparison_plot(
                yaw_rows,
                output_path=output_root
                / "yaw_comparisons"
                / "plots"
                / f"{comparison_name}_agora_yaw15.png",
                baseline_label=control_name,
                candidate_label=name,
            )
        so3 = []
        for dataset in result_run["datasets"]:
            dataset_name = dataset["dataset"]
            control_dataset = control_datasets[dataset_name]
            mean = dataset["overall"]["geodesic_error_deg_mean"]
            control_mean = control_dataset["overall"]["geodesic_error_deg_mean"]
            item = {
                "dataset": dataset_name,
                "count": dataset["count"],
                "mean_deg": mean,
                "control_mean_deg": control_mean,
                "delta_candidate_minus_control_deg": mean - control_mean,
            }
            if dataset_name == "agora_hpe":
                relative = Path("datasets/agora_hpe/metrics/yaw_bins_15.json")
                control_bins = {
                    row["yaw_bin"]: row
                    for row in json.loads((control_dir / relative).read_text())
                }
                candidate_bins = json.loads((result_dir / relative).read_text())
                if len(candidate_bins) != 24 or len(control_bins) != 24:
                    raise ValueError("AGORA SO(3) comparison requires 24 yaw bins")
                bins = []
                for row in candidate_bins:
                    base = control_bins[row["yaw_bin"]]
                    if row["count"] != base["count"]:
                        raise ValueError("AGORA yaw-bin counts differ")
                    value = row["geodesic_error_deg_mean"]
                    base_value = base["geodesic_error_deg_mean"]
                    bins.append(
                        {
                            "yaw_bin": row["yaw_bin"],
                            "count": row["count"],
                            "mean_deg": value,
                            "control_mean_deg": base_value,
                            "delta_candidate_minus_control_deg": (
                                value - base_value if row["count"] else None
                            ),
                        }
                    )
                item["yaw_bins_15"] = bins
            so3.append(item)
        rows.append(
            {
                **condition,
                "evaluation_name": name,
                "checkpoint": result_run["checkpoint"],
                "so3": so3,
                "head_forward_yaw": yaw_rows,
            }
        )
    summary_path = output_root / "weight_summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {"kind": WEIGHT_SEARCH_KIND, "checkpoint_choice": "final", "suites": {}}
    )
    summary["suites"][reference_key] = {
        "control_condition": control["condition_id"],
        "control_checkpoint": control_run["checkpoint"],
        "delta_definition": "candidate minus no-YawPose control; negative is improvement",
        "head_forward_yaw_definition": "atan2(R[0,2], R[2,2]) versus manifest source yaw",
        "conditions": rows,
    }
    write_json_atomic(summary_path, summary)


def main() -> None:
    args = build_parser().parse_args()
    suffix = _safe_suffix(args.name_suffix)
    baseline_run = _safe_name(args.baseline_run, "baseline run")
    if args.bootstrap_repetitions <= 0 or args.bootstrap_sample_limit <= 0:
        raise ValueError("bootstrap settings must be positive")

    run_dir = experiment_run_path(ROOT, args.run_id)
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError(
            "complete all search conditions before external evaluation"
        )
    search_config = json.loads((run_dir / "config.json").read_text())
    conditions = search_config["conditions"]
    weight_search = search_config["kind"] == WEIGHT_SEARCH_KIND
    if weight_search:
        if args.checkpoint_choice != "final":
            raise ValueError("loss-weight comparison requires the fixed final checkpoint")
        if conditions != condition_records("loss-weight"):
            raise ValueError("loss-weight comparison requires the fixed six conditions")
    output_root = ROOT / "eval" / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = (
        ROOT
        / "eval"
        / baseline_run
        / "evaluations"
        / "baseline"
    )
    if not baseline.is_dir():
        raise FileNotFoundError(baseline)
    baseline_metadata = json.loads((baseline / "run.json").read_text())
    yaw_root = output_root / "yaw_comparisons"
    yaw_root.mkdir(exist_ok=True)

    summary = []
    progress = tqdm(
        conditions,
        desc="YawPose external evaluation",
        unit="condition",
    )
    for condition in progress:
        condition_id = condition["condition_id"]
        checkpoint_suffix = (
            "_best"
            if args.checkpoint_choice == "best"
            else ""
        )
        evaluation_name = (
            condition_id + checkpoint_suffix + suffix
        )
        result_dir = (
            output_root
            / "conditions"
            / evaluation_name
        )
        existing_result = result_dir.is_dir()
        if weight_search:
            checkpoint = (
                run_dir / "conditions" / condition_id / "checkpoints"
                / f"epoch_{int(search_config['epochs']):03d}.pth"
            )
            if existing_result:
                _validate_weight_evaluation(result_dir, checkpoint, baseline_metadata)
        comparison_name = f"{baseline_metadata['run_name']}_vs_{evaluation_name}"
        comparison_dir = output_root / "comparisons" / comparison_name
        if result_dir.is_dir() and not comparison_dir.exists():
            compare_runs(
                baseline,
                result_dir,
                output_root,
                name=comparison_name,
            )
        elif comparison_dir.exists() and not result_dir.is_dir():
            raise FileNotFoundError(
                f"comparison exists without evaluation result: {comparison_dir}"
            )
        elif not result_dir.is_dir():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.scripts.evaluate_candidate",
                    "--run-id",
                    args.run_id,
                    "--condition-id",
                    condition_id,
                    "--checkpoint-choice",
                    args.checkpoint_choice,
                    "--baseline-run",
                    baseline_run,
                    "--name",
                    evaluation_name,
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )

        if weight_search and not existing_result:
            _validate_weight_evaluation(result_dir, checkpoint, baseline_metadata)

        yaw_rows = _yaw_comparison(
            baseline,
            result_dir,
            repetitions=args.bootstrap_repetitions,
            sample_limit=args.bootstrap_sample_limit,
        )
        yaw_path = yaw_root / f"{evaluation_name}.json"
        write_json_atomic(yaw_path, yaw_rows)
        _write_agora_yaw15_comparison_plot(
            yaw_rows,
            output_path=(
                yaw_root
                / "plots"
                / f"{evaluation_name}_agora_yaw15.png"
            ),
            baseline_label=baseline_metadata["run_name"],
            candidate_label=evaluation_name,
        )
        for row in yaw_rows:
            summary.append(
                {
                    "condition_id": condition_id,
                    "evaluation_name": evaluation_name,
                    "adoption_ratio": condition["adoption_ratio"],
                    "hpe_regime": condition["hpe_regime"],
                    "use_dad": condition["use_dad"],
                    **(
                        {"yawpose_weight": condition["yawpose_weight"]}
                        if weight_search else {}
                    ),
                    **row,
                }
            )

    key = (
        f"{baseline_run}_{args.checkpoint_choice}{suffix}"
    )
    write_json_atomic(
        output_root / f"yaw_summary_{key}.json",
        summary,
    )
    if weight_search:
        _weight_comparisons(
            output_root, conditions, suffix=suffix, reference_key=key,
            repetitions=args.bootstrap_repetitions,
            sample_limit=args.bootstrap_sample_limit,
        )


if __name__ == "__main__":
    main()
