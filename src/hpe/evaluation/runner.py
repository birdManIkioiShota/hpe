from __future__ import annotations

import os
import platform
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hpe.data import ManifestDataset
from hpe.datasets.common import sha256_file
from hpe.evaluation.metrics import metric_tables, yaw_bin_table
from hpe.evaluation.reporting import (
    write_json,
    write_metric_tables,
    write_yaw_plot,
    write_yaw_radar_plot,
)
from hpe.geometry import (
    circular_error_degrees,
    euler_degrees_to_matrix,
    geodesic_error_degrees,
    matrix_to_euler_degrees,
    vector_errors_degrees,
)
from hpe.models import SixDRepNet360, load_checkpoint


DEFAULT_DATASETS = ("agora_hpe", "aflw2000", "300w_lp")
MANIFESTS = {
    "agora_hpe": Path("datasets/prepared/agora_hpe/manifest.jsonl"),
    "aflw2000": Path("datasets/prepared/aflw2000/manifest.jsonl"),
    "300w_lp": Path("datasets/prepared/300w_lp/manifest.jsonl"),
    "dad3dheads": Path("datasets/prepared/dad3dheads/manifest.jsonl"),
}


@dataclass(frozen=True)
class EvaluationConfig:
    project_root: Path
    checkpoint: Path
    run_name: str
    output_root: Path | None = None
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    device: str = "cuda:0"
    batch_size: int = 256
    workers: int = 8
    amp: bool = True
    max_samples: int | None = None
    overwrite: bool = False
    deterministic: bool = False
    preserve_partial: bool = False
    image_lock_sha256: dict[str, str] | None = None
    manifest_paths: dict[str, Path] | None = None
    image_hashes: dict[str, str] | None = None


def _validate_config(config: EvaluationConfig) -> tuple[Path, torch.device]:
    if (
        not config.run_name or config.run_name in (".", "..")
        or Path(config.run_name).name != config.run_name
    ):
        raise ValueError("run_name must be one path-safe name.")
    if config.batch_size <= 0 or config.workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative.")
    if not config.datasets or len(set(config.datasets)) != len(config.datasets):
        raise ValueError("Specify at least one dataset, without duplicates.")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if config.image_lock_sha256 is not None and set(config.image_lock_sha256) != set(config.datasets):
        raise ValueError("Image locks must cover exactly the selected datasets.")
    unknown = set(config.datasets) - set(MANIFESTS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if config.manifest_paths and set(config.manifest_paths) - set(config.datasets):
        raise ValueError("Manifest overrides must name selected datasets")
    device = torch.device(config.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA and explicitly selected CPU are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to this process. "
            "The evaluator does not silently fall back to CPU."
        )
    output_root = (config.output_root or (config.project_root / "eval")).resolve()
    return output_root, device


def _to_list(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _evaluate_dataset(
    model: SixDRepNet360,
    config: EvaluationConfig,
    dataset_name: str,
    device: torch.device,
    output_dir: Path,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest = (config.project_root / (config.manifest_paths or {}).get(dataset_name, MANIFESTS[dataset_name])).resolve()
    dataset = ManifestDataset(config.project_root, manifest, config.max_samples,
                              image_hashes=config.image_hashes,
                              read_error_dir=output_dir / "image_read_errors" / dataset_name)
    if not len(dataset):
        raise ValueError(f"Empty evaluation manifest: {manifest}")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )
    columns: dict[str, list[Any]] = {}

    def extend(name: str, values: Any) -> None:
        columns.setdefault(name, []).extend(_to_list(values))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    processed = 0
    error_sum = 0.0
    max_orthogonality_error = 0.0
    max_determinant_error = 0.0
    with torch.inference_mode(), tqdm(
        total=len(dataset), desc=f"evaluate {dataset_name}",
        unit="head", dynamic_ncols=True,
    ) as progress:
        for images, metadata in loader:
            images = images.to(device, non_blocking=True)
            use_amp = config.amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predicted_matrix = model(images)

            source_euler = torch.stack(
                (
                    metadata["pitch_deg"],
                    metadata["yaw_deg"],
                    metadata["roll_deg"],
                ),
                dim=1,
            ).to(device=device, dtype=torch.float64, non_blocking=True)
            predicted_matrix = predicted_matrix.to(dtype=source_euler.dtype)
            identity = torch.eye(3, dtype=source_euler.dtype, device=device)
            ortho = (predicted_matrix.transpose(1, 2) @ predicted_matrix - identity).abs().amax()
            det_error = (torch.linalg.det(predicted_matrix) - 1).abs().amax()
            if not torch.isfinite(predicted_matrix).all() or ortho > 1e-4 or det_error > 1e-4:
                raise RuntimeError(f"Invalid predicted rotation in {dataset_name}.")
            max_orthogonality_error = max(max_orthogonality_error, float(ortho))
            max_determinant_error = max(max_determinant_error, float(det_error))
            if "rotation_matrix" in metadata:
                target_matrix = metadata["rotation_matrix"].to(device=device, dtype=source_euler.dtype, non_blocking=True)
            else:
                target_matrix = euler_degrees_to_matrix(source_euler)
            predicted_euler = matrix_to_euler_degrees(predicted_matrix)
            target_euler = matrix_to_euler_degrees(target_matrix)
            axis_errors = circular_error_degrees(predicted_euler, target_euler)
            geodesic = geodesic_error_degrees(predicted_matrix, target_matrix, stable=True)
            vector_errors = vector_errors_degrees(predicted_matrix, target_matrix, rows=True)
            bbox = metadata["bbox_xyxy"]
            crop = metadata["crop_xyxy"]

            for name in ("dataset", "sample_id", "instance_id", "image_path"):
                extend(name, metadata[name])
            extend("gt_source_pitch_deg", source_euler[:, 0])
            extend("gt_source_yaw_deg", source_euler[:, 1])
            extend("gt_source_roll_deg", source_euler[:, 2])
            extend("gt_eval_pitch_deg", target_euler[:, 0])
            extend("gt_eval_yaw_deg", target_euler[:, 1])
            extend("gt_eval_roll_deg", target_euler[:, 2])
            extend("pred_pitch_deg", predicted_euler[:, 0])
            extend("pred_yaw_deg", predicted_euler[:, 1])
            extend("pred_roll_deg", predicted_euler[:, 2])
            extend("pitch_error_deg", axis_errors[:, 0])
            extend("yaw_error_deg", axis_errors[:, 1])
            extend("roll_error_deg", axis_errors[:, 2])
            extend("mean_axis_error_deg", axis_errors.mean(dim=1))
            extend("geodesic_error_deg", geodesic)
            extend("vec1_error_deg", vector_errors[:, 0])
            extend("vec2_error_deg", vector_errors[:, 1])
            extend("vec3_error_deg", vector_errors[:, 2])
            extend("vmae_deg", vector_errors.mean(dim=1))
            extend("bbox_x1", bbox[:, 0])
            extend("bbox_y1", bbox[:, 1])
            extend("bbox_x2", bbox[:, 2])
            extend("bbox_y2", bbox[:, 3])
            extend("bbox_width", bbox[:, 2] - bbox[:, 0])
            extend("bbox_height", bbox[:, 3] - bbox[:, 1])
            extend("crop_x1", crop[:, 0])
            extend("crop_y1", crop[:, 1])
            extend("crop_x2", crop[:, 2])
            extend("crop_y2", crop[:, 3])
            extend("occlusion_percent", metadata["occlusion_percent"])
            for row in range(3):
                for column in range(3):
                    extend(f"pred_R_{row}{column}", predicted_matrix[:, row, column])
                    extend(f"gt_R_{row}{column}", target_matrix[:, row, column])
            processed += images.shape[0]
            error_sum += float(geodesic.sum())
            progress.set_postfix(geodesic=f"{error_sum / processed:.3f} deg", refresh=False)
            progress.update(images.shape[0])
            if event_callback is not None:
                event_callback({"event": "evaluation_progress", "dataset": dataset_name,
                                "processed": processed, "total": len(dataset),
                                "elapsed_seconds": time.perf_counter() - started,
                                "geodesic_mean_deg": error_sum / processed})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = pd.DataFrame(columns)
    if len(predictions) != len(dataset):
        raise RuntimeError(f"Prediction count mismatch for {dataset_name}.")
    if predictions["instance_id"].duplicated().any():
        raise RuntimeError(f"Duplicate instance IDs in {dataset_name}.")
    numeric_errors = predictions[
        [name for name in predictions if name.endswith("_error_deg") or name == "vmae_deg"]
    ]
    if not np.isfinite(numeric_errors.to_numpy()).all():
        raise RuntimeError(f"Non-finite evaluation error in {dataset_name}.")

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        predictions_dir / f"{dataset_name}.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.8f",
    )
    tables = metric_tables(predictions)
    if dataset_name == "agora_hpe":
        tables["yaw_bins_15"] = yaw_bin_table(
            predictions,
            bin_size_deg=15,
            include_empty=True,
        )
    dataset_metrics = output_dir / "datasets" / dataset_name
    write_metric_tables(dataset_metrics, tables)
    write_yaw_plot(dataset_metrics, tables["yaw_bins"], dataset_name)
    if dataset_name == "agora_hpe":
        write_yaw_radar_plot(
            dataset_metrics,
            tables["yaw_bins_15"],
            dataset_name,
        )
    result = {
        "dataset": dataset_name,
        "count": len(predictions),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(predictions) / elapsed if elapsed else None,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "overall": tables["overall"].iloc[0].to_dict(),
    }
    if dataset_name == "dad3dheads":
        result["ground_truth_representation"] = "model_view_matrix[:3,:3].T (direct matrix)"
        result["yaw_group_definition"] = "derived RzRyRx branch with |pitch|<=90; rear |yaw|>=120"
    result["rotation_checks"] = {
        "max_orthogonality_error": max_orthogonality_error,
        "max_determinant_error": max_determinant_error,
    }
    if config.image_lock_sha256 is not None:
        result["image_lock_sha256"] = config.image_lock_sha256[dataset_name]
    tqdm.write(f"{dataset_name}: {len(predictions)} heads, SO(3)={error_sum / processed:.4f} deg")
    if event_callback is not None:
        event_callback({"event": "dataset_complete", **result})
    dataset.close()
    return result


def _environment(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        details["gpu_name"] = torch.cuda.get_device_name(device)
        details["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return details


def evaluate(
    config: EvaluationConfig,
    *, event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    project_root = config.project_root.resolve()
    output_root, device = _validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / config.run_name
    partial_dir = output_root / f".{config.run_name}.partial"
    if final_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Evaluation run already exists: {final_dir}")
        shutil.rmtree(final_dir)
    if partial_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Incomplete evaluation already exists: {partial_dir}")
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    try:
        model = SixDRepNet360(rotation_fp32=True)
        checkpoint_info = load_checkpoint(model, config.checkpoint)
        model.to(device).eval()
        torch.backends.cudnn.benchmark = not config.deterministic
        torch.backends.cudnn.deterministic = config.deterministic

        results = [
            _evaluate_dataset(model, config, dataset_name, device, partial_dir, event_callback)
            for dataset_name in config.datasets
        ]
        summary = pd.DataFrame(
            [{"dataset": item["dataset"], **item["overall"]} for item in results]
        )
        summary.to_csv(partial_dir / "summary.csv", index=False, float_format="%.8f")
        write_json(partial_dir / "summary.json", summary.to_dict(orient="records"))
        run_metadata = {
            "run_name": config.run_name,
            "checkpoint": checkpoint_info,
            "datasets": results,
            "settings": {
                "model": "SixDRepNet360-ResNet50",
                "crop_source": "ground_truth_manifest",
                "resize": 256,
                "center_crop": 224,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "device": str(device),
                "batch_size": config.batch_size,
                "workers": config.workers,
                "amp": config.amp and device.type == "cuda",
                "max_samples": config.max_samples,
                "axis_error_representation": "canonical RzRyRx Euler for both target and prediction",
                "yaw_group_source": "manifest source yaw",
                "head_size_measure": "sqrt(bbox_width*bbox_height)",
                "rotation_normalization": "fp32",
                "metric_dtype": "float64",
                "geodesic_formula": "atan2",
                "vector_definition": "rows",
                "deterministic": config.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            },
            "environment": _environment(device),
        }
        write_json(partial_dir / "run.json", run_metadata)
        os.replace(partial_dir, final_dir)
    except BaseException:
        if partial_dir.exists() and not config.preserve_partial:
            shutil.rmtree(partial_dir)
        raise
    return final_dir
