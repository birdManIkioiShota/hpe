from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from hpe.evaluation.metrics import ERROR_COLUMNS
from hpe.evaluation.reporting import write_json


def _read_run(path: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = path.resolve()
    metadata_path = run_dir / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        return run_dir, json.load(stream)


def _manifest_map(metadata: dict[str, Any]) -> dict[str, str]:
    return {item["dataset"]: item["manifest_sha256"] for item in metadata["datasets"]}


def _check_compatible(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    if baseline["protocol_version"] != candidate["protocol_version"]:
        raise ValueError("Runs use different evaluation protocol versions.")
    baseline_settings = baseline["settings"]
    candidate_settings = candidate["settings"]
    comparable_settings = (
        "model",
        "crop_source",
        "resize",
        "center_crop",
        "normalization_mean",
        "normalization_std",
        "max_samples",
        "axis_error_representation",
        "yaw_group_source",
    )
    if baseline["protocol_version"] >= 2:
        comparable_settings += (
            "amp", "batch_size", "deterministic", "rotation_normalization", "metric_dtype",
            "geodesic_formula", "vector_definition", "cuda_matmul_fp32_precision",
            "cudnn_conv_fp32_precision",
        )
    mismatches = [
        key for key in comparable_settings if baseline_settings.get(key) != candidate_settings.get(key)
    ]
    if mismatches:
        raise ValueError(f"Runs have incompatible settings: {', '.join(mismatches)}")
    baseline_manifests = _manifest_map(baseline)
    candidate_manifests = _manifest_map(candidate)
    if baseline_manifests != candidate_manifests:
        raise ValueError("Runs were evaluated against different dataset manifests.")
    if baseline["protocol_version"] >= 2:
        baseline_images = {item["dataset"]: item.get("image_lock_sha256") for item in baseline["datasets"]}
        candidate_images = {item["dataset"]: item.get("image_lock_sha256") for item in candidate["datasets"]}
        if baseline_images != candidate_images:
            raise ValueError("Runs have different or missing image locks.")
    return list(baseline_manifests)


def _bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
    sample_limit: int,
) -> tuple[float, float]:
    if len(differences) == 0:
        return float("nan"), float("nan")
    if len(differences) > sample_limit:
        differences = rng.choice(differences, size=sample_limit, replace=False)
    means = np.empty(repetitions, dtype=np.float64)
    chunk_size = 100
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start : start + count] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _comparison_rows(
    merged: pd.DataFrame,
    dataset: str,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
    bootstrap_sample_limit: int,
) -> list[dict[str, Any]]:
    absolute_yaw = merged["gt_source_yaw_deg_baseline"].abs()
    masks = {
        "overall": np.ones(len(merged), dtype=bool),
        "front_lt60": absolute_yaw < 60.0,
        "side_60_to_lt120": (absolute_yaw >= 60.0) & (absolute_yaw < 120.0),
        "rear_ge120": absolute_yaw >= 120.0,
    }
    rows: list[dict[str, Any]] = []
    for group, mask in masks.items():
        subset = merged.loc[mask]
        if subset.empty:
            continue
        for metric in ERROR_COLUMNS:
            difference = (
                subset[f"{metric}_candidate"] - subset[f"{metric}_baseline"]
            ).to_numpy(dtype=np.float64)
            ci_low, ci_high = _bootstrap_ci(
                difference,
                rng,
                bootstrap_repetitions,
                bootstrap_sample_limit,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "metric": metric,
                    "count": len(subset),
                    "baseline_mean": float(subset[f"{metric}_baseline"].mean()),
                    "candidate_mean": float(subset[f"{metric}_candidate"].mean()),
                    "delta_candidate_minus_baseline": float(difference.mean()),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "improved_count": int((difference < 0).sum()),
                    "tied_count": int((difference == 0).sum()),
                    "worsened_count": int((difference > 0).sum()),
                }
            )
    return rows


def compare_runs(
    baseline_path: Path,
    candidate_path: Path,
    output_root: Path,
    name: str | None = None,
    overwrite: bool = False,
    bootstrap_repetitions: int = 1000,
    bootstrap_sample_limit: int = 10_000,
    seed: int = 0,
) -> Path:
    baseline_dir, baseline = _read_run(baseline_path)
    candidate_dir, candidate = _read_run(candidate_path)
    datasets = _check_compatible(baseline, candidate)
    comparison_name = name or f"{baseline['run_name']}_vs_{candidate['run_name']}"
    if not comparison_name or Path(comparison_name).name != comparison_name:
        raise ValueError("Comparison name must be one path-safe name.")
    if bootstrap_repetitions <= 0 or bootstrap_sample_limit <= 0:
        raise ValueError("Bootstrap settings must be positive.")

    comparisons_root = output_root.resolve() / "comparisons"
    comparisons_root.mkdir(parents=True, exist_ok=True)
    final_dir = comparisons_root / comparison_name
    partial_dir = comparisons_root / f".{comparison_name}.partial"
    if final_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Comparison already exists: {final_dir}")
        shutil.rmtree(final_dir)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, Any]] = []
    try:
        for dataset in tqdm(
            datasets,
            desc="compare runs",
            unit="dataset",
            dynamic_ncols=True,
        ):
            baseline_frame = pd.read_csv(baseline_dir / "predictions" / f"{dataset}.csv.gz")
            candidate_frame = pd.read_csv(candidate_dir / "predictions" / f"{dataset}.csv.gz")
            columns = ["instance_id", "gt_source_yaw_deg", *ERROR_COLUMNS]
            merged = baseline_frame[columns].merge(
                candidate_frame[columns],
                on="instance_id",
                how="outer",
                suffixes=("_baseline", "_candidate"),
                validate="one_to_one",
                indicator=True,
            )
            if not (merged["_merge"] == "both").all():
                raise ValueError(f"The prediction instance sets differ for {dataset}.")
            if not np.allclose(
                merged["gt_source_yaw_deg_baseline"],
                merged["gt_source_yaw_deg_candidate"],
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"Ground-truth yaw differs for {dataset}.")
            all_rows.extend(
                _comparison_rows(
                    merged,
                    dataset,
                    rng,
                    bootstrap_repetitions,
                    bootstrap_sample_limit,
                )
            )

        frame = pd.DataFrame(all_rows)
        frame.to_csv(partial_dir / "comparison.csv", index=False, float_format="%.8f")
        write_json(partial_dir / "comparison.json", frame.to_dict(orient="records"))
        write_json(
            partial_dir / "run.json",
            {
                "comparison_name": comparison_name,
                "baseline": str(baseline_dir),
                "baseline_checkpoint_sha256": baseline["checkpoint"]["sha256"],
                "candidate": str(candidate_dir),
                "candidate_checkpoint_sha256": candidate["checkpoint"]["sha256"],
                "protocol_version": baseline["protocol_version"],
                "datasets": datasets,
                "bootstrap_repetitions": bootstrap_repetitions,
                "bootstrap_sample_limit": bootstrap_sample_limit,
                "seed": seed,
                "delta_definition": "candidate minus baseline; negative is improvement",
            },
        )
        os.replace(partial_dir, final_dir)
    except BaseException:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise
    return final_dir
