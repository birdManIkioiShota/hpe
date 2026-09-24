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
from hpe.datasets.common import write_json_atomic
from hpe.evaluation import compare_runs
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
) -> tuple[float, float]:
    if not len(difference):
        return float("nan"), float("nan")
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
            if not mask.any():
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
                    "mean_delta_candidate_minus_baseline_deg": float(
                        np.mean(difference)
                    ),
                    "mean_delta_ci95_low": ci_low,
                    "mean_delta_ci95_high": ci_high,
                }
            )
    return rows


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

        yaw_rows = _yaw_comparison(
            baseline,
            result_dir,
            repetitions=args.bootstrap_repetitions,
            sample_limit=args.bootstrap_sample_limit,
        )
        yaw_path = yaw_root / f"{evaluation_name}.json"
        write_json_atomic(yaw_path, yaw_rows)
        for row in yaw_rows:
            summary.append(
                {
                    "condition_id": condition_id,
                    "evaluation_name": evaluation_name,
                    "adoption_ratio": condition["adoption_ratio"],
                    "hpe_regime": condition["hpe_regime"],
                    "use_dad": condition["use_dad"],
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


if __name__ == "__main__":
    main()
