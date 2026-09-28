from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.datasets import PoseManifestDataset, validate_rotation
from experiments.common.vgg_dev_targets import VggDevEvaluationTarget
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.geometry.rotations import (
    circular_error_degrees,
    geodesic_error_degrees,
    matrix_to_euler_degrees,
)
from hpe.models import SixDRepNet360, load_checkpoint


EVALUATION_SCHEMA = "vgg_dev_all_conditions_v1"
EXPECTED_DEV_COUNT = 51_914
VGG_DEV_RELATIVE = Path("datasets/prepared/vgg_data/dev.jsonl")
VGG_METADATA_RELATIVE = Path("datasets/prepared/vgg_data/metadata.json")
PREDICTION_FILENAME = "predictions.csv.gz"

METRIC_COLUMNS = (
    "geodesic_error_deg",
    "pitch_error_deg",
    "yaw_error_deg",
    "roll_error_deg",
    "mean_axis_error_deg",
)
SUMMARY_KEYS = (
    "so3_mean_deg",
    "pitch_maae_deg",
    "yaw_maae_deg",
    "roll_maae_deg",
    "mean_axis_maae_deg",
)
SOURCE_FILES = (
    "src/experiments/common/vgg_dev_evaluation.py",
    "src/experiments/common/vgg_dev_targets.py",
    "src/experiments/common/datasets.py",
    "src/experiments/common/pose.py",
    "src/hpe/data/dataset.py",
    "src/hpe/geometry/rotations.py",
    "src/hpe/models/checkpoint.py",
    "src/hpe/models/sixdrepnet360.py",
    "pyproject.toml",
)


@dataclass(frozen=True)
class DevInputIdentity:
    manifest_path: str
    manifest_sha256: str
    instance_ids_sha256: str
    count: int
    dataset: str
    split: str
    metadata_sha256: str


def _sha256_text(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_vgg_dev_input(
    project_root: Path,
    *,
    manifest_relative: Path = VGG_DEV_RELATIVE,
    metadata_relative: Path = VGG_METADATA_RELATIVE,
) -> DevInputIdentity:
    project_root = project_root.resolve()
    manifest = (project_root / manifest_relative).resolve()
    metadata_path = (project_root / metadata_relative).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "vggheads" or metadata.get("status") != "completed":
        raise ValueError("VGGHeads prepared metadata is incomplete or incompatible")
    dev_meta = metadata.get("splits", {}).get("dev")
    if not isinstance(dev_meta, dict):
        raise ValueError("VGGHeads metadata has no dev split")
    if int(dev_meta.get("heads", -1)) != EXPECTED_DEV_COUNT:
        raise ValueError(
            f"Expected VGGHeads dev count {EXPECTED_DEV_COUNT}, "
            f"metadata reports {dev_meta.get('heads')}"
        )
    manifest_sha = sha256_file(manifest)
    if manifest_sha != dev_meta.get("sha256"):
        raise ValueError("VGGHeads dev manifest SHA-256 differs from prepared metadata")

    ids: list[str] = []
    seen: set[str] = set()
    count = 0
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("dataset") != "vggheads":
                raise ValueError(f"{manifest}:{line_number}: dataset is not vggheads")
            if row.get("split") != "dev":
                raise ValueError(f"{manifest}:{line_number}: split is not dev")
            instance_id = str(row.get("instance_id", ""))
            if not instance_id:
                raise ValueError(f"{manifest}:{line_number}: empty instance_id")
            if instance_id in seen:
                raise ValueError(f"{manifest}:{line_number}: duplicate instance_id {instance_id}")
            seen.add(instance_id)
            ids.append(instance_id)
            validate_rotation(row["rotation_matrix"])
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.parts or image_path.is_absolute() or ".." in image_path.parts:
                raise ValueError(f"{manifest}:{line_number}: invalid image_path")
            image_sha = str(row.get("image_sha256", ""))
            if len(image_sha) != 64 or any(ch not in "0123456789abcdef" for ch in image_sha):
                raise ValueError(f"{manifest}:{line_number}: invalid image_sha256")
            crop = row.get("crop_xyxy")
            if not isinstance(crop, list) or len(crop) != 4:
                raise ValueError(f"{manifest}:{line_number}: invalid crop_xyxy")
            count += 1

    if count != EXPECTED_DEV_COUNT:
        raise ValueError(f"Expected {EXPECTED_DEV_COUNT} VGGHeads dev rows, found {count}")
    return DevInputIdentity(
        manifest_path=manifest.relative_to(project_root).as_posix(),
        manifest_sha256=manifest_sha,
        instance_ids_sha256=_sha256_text(ids),
        count=count,
        dataset="vggheads",
        split="dev",
        metadata_sha256=sha256_file(metadata_path),
    )


def verify_target_record(project_root: Path, target: VggDevEvaluationTarget) -> None:
    checkpoint = project_root / target.checkpoint_path
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != target.checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {target.logical_id}: "
            f"expected={target.checkpoint_sha256}, actual={actual_sha}"
        )
    reference_path = project_root / target.reference_run_path
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    recorded_sha = reference.get("checkpoint", {}).get("sha256")
    if recorded_sha != target.checkpoint_sha256:
        raise ValueError(
            f"Saved evaluation record disagrees with target {target.logical_id}: "
            f"{reference_path} has {recorded_sha}"
        )


def evaluation_code_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    return hashes


def git_revision(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("rev-parse", "HEAD")
    dirty_output = run("status", "--porcelain")
    return {
        "commit": revision,
        "dirty": None if dirty_output is None else bool(dirty_output),
    }


def evaluation_settings(
    *,
    device: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_SCHEMA,
        "model": "SixDRepNet360-ResNet50",
        "rotation_normalization": "fp32",
        "inference_precision": "fp32",
        "amp": False,
        "metric_dtype": "float64",
        "geodesic_formula": "stable_atan2",
        "axis_error_representation": "canonical RzRyRx Euler for target and prediction",
        "yaw_bin_source": "GT head-forward +Z azimuth atan2(R[0,2], R[2,2])",
        "yaw_bins": "24 x 15 degree bins from [-180,-165) through [165,180]",
        "undefined_yaw": "horizontal norm of GT head-forward vector below 1e-8",
        "resize": 256,
        "center_crop": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "shuffle": False,
        "device": device,
        "batch_size": batch_size,
        "workers": workers,
        "deterministic": True,
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
    }


def evaluation_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def full_range_yaw_degrees(rotation: np.ndarray, *, eps: float = 1e-8) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    direction = matrix[:, 2]
    horizontal = math.hypot(float(direction[0]), float(direction[2]))
    if horizontal < eps:
        return float("nan")
    return math.degrees(math.atan2(float(direction[0]), float(direction[2])))


def yaw_bin_index(yaw_deg: float) -> int | None:
    yaw = float(yaw_deg)
    if not math.isfinite(yaw):
        return None
    tolerance = 1e-10
    if yaw < -180.0 - tolerance or yaw > 180.0 + tolerance:
        raise ValueError(f"yaw outside [-180,180]: {yaw}")
    yaw = min(180.0, max(-180.0, yaw))
    if math.isclose(yaw, 180.0, abs_tol=tolerance, rel_tol=0):
        return 23
    index = int(math.floor((yaw + 180.0) / 15.0))
    if not 0 <= index < 24:
        raise AssertionError(f"Unexpected yaw bin index {index} for {yaw}")
    return index


def yaw_bin_record(index: int) -> dict[str, Any]:
    if not 0 <= index < 24:
        raise ValueError("yaw bin index must be in [0,23]")
    left = -180 + 15 * index
    right = left + 15
    return {
        "yaw_bin_index": index,
        "yaw_bin": (
            f"yaw_{left}_to_{right}"
            if index == 23
            else f"yaw_{left}_to_lt{right}"
        ),
        "left_deg": left,
        "right_deg": right,
        "left_inclusive": True,
        "right_inclusive": index == 23,
    }


def _metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    count = int(len(frame))
    if not count:
        return {
            "count": 0,
            "so3_mean_deg": None,
            "pitch_maae_deg": None,
            "yaw_maae_deg": None,
            "roll_maae_deg": None,
            "mean_axis_maae_deg": None,
        }
    values = {
        name: frame[column].to_numpy(dtype=np.float64)
        for name, column in (
            ("so3", "geodesic_error_deg"),
            ("pitch", "pitch_error_deg"),
            ("yaw", "yaw_error_deg"),
            ("roll", "roll_error_deg"),
        )
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Evaluation metrics contain non-finite values")
    pitch = float(np.mean(values["pitch"]))
    yaw = float(np.mean(values["yaw"]))
    roll = float(np.mean(values["roll"]))
    return {
        "count": count,
        "so3_mean_deg": float(np.mean(values["so3"])),
        "pitch_maae_deg": pitch,
        "yaw_maae_deg": yaw,
        "roll_maae_deg": roll,
        "mean_axis_maae_deg": (pitch + yaw + roll) / 3.0,
    }


def _assert_weighted_reconstruction(
    overall: dict[str, Any],
    bins: list[dict[str, Any]],
    undefined: dict[str, Any],
) -> None:
    groups = [*bins, undefined]
    total = sum(int(group["count"]) for group in groups)
    if total != int(overall["count"]):
        raise ValueError(
            f"Yaw-bin counts plus undefined count {total} do not equal overall {overall['count']}"
        )
    for key in SUMMARY_KEYS:
        numerator = 0.0
        denominator = 0
        for group in groups:
            value = group[key]
            count = int(group["count"])
            if count:
                if value is None:
                    raise ValueError(f"Non-empty group has missing metric {key}")
                numerator += float(value) * count
                denominator += count
        reconstructed = numerator / denominator
        if not math.isclose(
            reconstructed,
            float(overall[key]),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"Weighted {key} does not reconstruct overall: "
                f"{reconstructed} vs {overall[key]}"
            )


def aggregate_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "instance_id",
        "gt_full_yaw_deg",
        "geodesic_error_deg",
        "pitch_error_deg",
        "yaw_error_deg",
        "roll_error_deg",
        "mean_axis_error_deg",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if frame["instance_id"].duplicated().any():
        raise ValueError("Prediction table contains duplicate instance IDs")
    overall = _metric_summary(frame)
    indices = np.asarray(
        [yaw_bin_index(value) for value in frame["gt_full_yaw_deg"]],
        dtype=object,
    )
    bins: list[dict[str, Any]] = []
    for index in range(24):
        subset = frame[indices == index]
        bins.append({**yaw_bin_record(index), **_metric_summary(subset)})
    undefined_mask = np.asarray([value is None for value in indices], dtype=bool)
    undefined = {
        "group": "undefined_gt_full_yaw",
        **_metric_summary(frame[undefined_mask]),
    }
    result = {
        "schema": EVALUATION_SCHEMA,
        "overall": {
            **overall,
            "defined_yaw_count": int((~undefined_mask).sum()),
            "undefined_yaw_count": int(undefined_mask.sum()),
        },
        "yaw_bins_15": bins,
        "undefined_yaw": undefined,
    }
    _assert_weighted_reconstruction(overall, bins, undefined)
    return result


def _write_aggregate_files(output_dir: Path, aggregate: dict[str, Any]) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(metrics_dir / "aggregate.json", aggregate)
    pd.DataFrame([aggregate["overall"]]).to_csv(
        metrics_dir / "overall.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(aggregate["yaw_bins_15"]).to_csv(
        metrics_dir / "yaw_bins_15.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame([aggregate["undefined_yaw"]]).to_csv(
        metrics_dir / "undefined_yaw.csv",
        index=False,
        float_format="%.17g",
    )


def aggregate_prediction_file(path: Path) -> dict[str, Any]:
    return aggregate_predictions(pd.read_csv(path))


def _aggregate_close(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if (
        left["overall"]["count"] != right["overall"]["count"]
        or left["overall"]["defined_yaw_count"] != right["overall"]["defined_yaw_count"]
        or left["overall"]["undefined_yaw_count"] != right["overall"]["undefined_yaw_count"]
    ):
        return False
    for key in SUMMARY_KEYS:
        if not math.isclose(
            float(left["overall"][key]),
            float(right["overall"][key]),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            return False
    for lrow, rrow in zip(left["yaw_bins_15"], right["yaw_bins_15"], strict=True):
        if lrow["count"] != rrow["count"]:
            return False
        for key in SUMMARY_KEYS:
            lvalue, rvalue = lrow[key], rrow[key]
            if lvalue is None or rvalue is None:
                if lvalue is not None or rvalue is not None:
                    return False
            elif not math.isclose(float(lvalue), float(rvalue), rel_tol=0, abs_tol=1e-10):
                return False
    return True


def _configure_determinism(device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"


def _prediction_frame(
    model: SixDRepNet360,
    *,
    project_root: Path,
    manifest_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    error_dir: Path,
) -> pd.DataFrame:
    dataset = PoseManifestDataset(
        project_root,
        manifest_path,
        augment=False,
        seed=0,
        allowed_datasets={"vggheads"},
        read_error_dir=error_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    columns: dict[str, list[Any]] = {}

    def extend(name: str, values: Any) -> None:
        if torch.is_tensor(values):
            values = values.detach().cpu().tolist()
        columns.setdefault(name, []).extend(list(values))

    identity = torch.eye(3, dtype=torch.float64, device=device)
    processed = 0
    try:
        with torch.inference_mode(), tqdm(
            total=len(dataset),
            desc="VGGHeads dev",
            unit="head",
            dynamic_ncols=True,
        ) as progress:
            for images, target, metadata in loader:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                predicted = model(images)
                predicted64 = predicted.to(dtype=torch.float64)
                target64 = target.to(device=device, dtype=torch.float64, non_blocking=True)

                orthogonality = (
                    predicted64.transpose(1, 2) @ predicted64 - identity
                ).abs().amax()
                determinant_error = (torch.linalg.det(predicted64) - 1.0).abs().amax()
                if (
                    not torch.isfinite(predicted64).all()
                    or orthogonality > 1e-4
                    or determinant_error > 1e-4
                ):
                    raise RuntimeError("Model produced an invalid rotation matrix")

                predicted_euler = matrix_to_euler_degrees(predicted64)
                target_euler = matrix_to_euler_degrees(target64)
                axis_error = circular_error_degrees(predicted_euler, target_euler)
                geodesic = geodesic_error_degrees(predicted64, target64, stable=True)
                if not torch.isfinite(geodesic).all() or not torch.isfinite(axis_error).all():
                    raise RuntimeError("Non-finite VGGHeads dev metric")

                batch_count = images.shape[0]
                extend("instance_id", metadata["instance_id"])
                extend("image_path", metadata["image_path"])
                extend("gt_full_yaw_deg", metadata["azimuth_deg"])
                extend("gt_pitch_deg", target_euler[:, 0])
                extend("gt_euler_yaw_deg", target_euler[:, 1])
                extend("gt_roll_deg", target_euler[:, 2])
                extend("pred_pitch_deg", predicted_euler[:, 0])
                extend("pred_euler_yaw_deg", predicted_euler[:, 1])
                extend("pred_roll_deg", predicted_euler[:, 2])
                extend("pitch_error_deg", axis_error[:, 0])
                extend("yaw_error_deg", axis_error[:, 1])
                extend("roll_error_deg", axis_error[:, 2])
                extend("mean_axis_error_deg", axis_error.mean(dim=1))
                extend("geodesic_error_deg", geodesic)
                for row in range(3):
                    for column in range(3):
                        extend(f"gt_R_{row}{column}", target64[:, row, column])
                        extend(f"pred_R_{row}{column}", predicted64[:, row, column])
                processed += batch_count
                progress.update(batch_count)
                progress.set_postfix(
                    so3_deg=f"{float(geodesic.sum()) / batch_count:.3f}",
                    refresh=False,
                )
    finally:
        dataset.close()

    if processed != len(dataset):
        raise RuntimeError(f"Processed {processed} samples but dataset contains {len(dataset)}")
    frame = pd.DataFrame(columns)
    if len(frame) != len(dataset):
        raise RuntimeError("Prediction row count differs from VGGHeads dev count")
    if frame["instance_id"].duplicated().any():
        raise RuntimeError("Duplicate VGGHeads dev instance ID after inference")
    numeric = frame[list(METRIC_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Saved prediction metrics contain non-finite values")
    return frame


def _runtime_payload(
    *,
    project_root: Path,
    input_identity: DevInputIdentity,
    checkpoint_sha256: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_SCHEMA,
        "input": asdict(input_identity),
        "checkpoint_sha256": checkpoint_sha256,
        "settings": settings,
        "source_hashes": evaluation_code_hashes(project_root),
    }


def _status(path: Path, status: str, **details: Any) -> None:
    write_json_atomic(path / "status.json", {"status": status, **details})


def _regression_check(
    aggregate: dict[str, Any],
    targets: list[VggDevEvaluationTarget],
    *,
    tolerance_deg: float = 1e-4,
) -> list[dict[str, Any]]:
    checks = []
    actual = float(aggregate["overall"]["so3_mean_deg"])
    for target in targets:
        expected = target.reference_vgg_dev_so3_mean_deg
        if expected is None:
            continue
        delta = actual - float(expected)
        record = {
            "logical_id": target.logical_id,
            "metric": "so3_mean_deg",
            "expected_deg": float(expected),
            "actual_deg": actual,
            "delta_deg": delta,
            "tolerance_deg": tolerance_deg,
        }
        checks.append(record)
        if abs(delta) > tolerance_deg:
            raise ValueError(
                f"VGGHeads dev regression check failed for {target.logical_id}: "
                f"expected {expected:.12f}, got {actual:.12f}, delta={delta:.12f}"
            )
    return checks


def evaluate_model_group(
    project_root: Path,
    output_root: Path,
    targets: list[VggDevEvaluationTarget],
    *,
    input_identity: DevInputIdentity,
    device_name: str,
    batch_size: int,
    workers: int,
    overwrite_stale: bool = False,
) -> Path:
    if not targets:
        raise ValueError("A model group must contain at least one logical target")
    checkpoint_sha = targets[0].checkpoint_sha256
    if any(target.checkpoint_sha256 != checkpoint_sha for target in targets):
        raise ValueError("A model group must contain one checkpoint SHA-256")
    for target in targets:
        verify_target_record(project_root, target)

    settings = evaluation_settings(
        device=device_name,
        batch_size=batch_size,
        workers=workers,
    )
    payload = _runtime_payload(
        project_root=project_root,
        input_identity=input_identity,
        checkpoint_sha256=checkpoint_sha,
        settings=settings,
    )
    fingerprint = evaluation_fingerprint(payload)
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    final_dir = model_root / checkpoint_sha
    partial_dir = model_root / f".{checkpoint_sha}.partial"

    if final_dir.is_dir():
        run_path = final_dir / "run.json"
        status_path = final_dir / "status.json"
        if run_path.is_file() and status_path.is_file():
            previous = json.loads(run_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("status") == "completed"
                and previous.get("evaluation_fingerprint") == fingerprint
                and (final_dir / PREDICTION_FILENAME).is_file()
            ):
                return final_dir
        if not overwrite_stale:
            raise ValueError(
                f"Existing completed result is incompatible with current evaluation: {final_dir}"
            )
        shutil.rmtree(final_dir)

    if partial_dir.exists():
        previous_run = partial_dir / "run.json"
        compatible = False
        if previous_run.is_file():
            previous = json.loads(previous_run.read_text(encoding="utf-8"))
            compatible = previous.get("evaluation_fingerprint") == fingerprint
        if not compatible and not overwrite_stale:
            raise ValueError(
                f"Existing partial result has a different fingerprint: {partial_dir}"
            )
        shutil.rmtree(partial_dir)

    partial_dir.mkdir(parents=True)
    aliases = [target.logical_id for target in targets]
    representative = targets[0]
    run_metadata = {
        "schema": EVALUATION_SCHEMA,
        "evaluation_fingerprint": fingerprint,
        "checkpoint": {
            "path": representative.checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
        },
        "logical_targets": aliases,
        "input": asdict(input_identity),
        "settings": settings,
        "source_hashes": payload["source_hashes"],
        "git": git_revision(project_root),
    }
    write_json_atomic(partial_dir / "run.json", run_metadata)
    _status(partial_dir, "running")
    try:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("Only CUDA or explicit CPU evaluation is supported")
        _configure_determinism(device)
        model = SixDRepNet360(rotation_fp32=True)
        checkpoint_info = load_checkpoint(model, project_root / representative.checkpoint_path)
        if checkpoint_info["sha256"] != checkpoint_sha:
            raise ValueError("Loaded checkpoint SHA-256 differs from target registry")
        model.to(device).eval()

        frame = _prediction_frame(
            model,
            project_root=project_root,
            manifest_path=project_root / input_identity.manifest_path,
            device=device,
            batch_size=batch_size,
            workers=workers,
            error_dir=partial_dir / "image_read_errors",
        )
        if len(frame) != input_identity.count:
            raise RuntimeError("Inference did not produce the fixed VGGHeads dev count")
        predictions_path = partial_dir / PREDICTION_FILENAME
        frame.to_csv(
            predictions_path,
            index=False,
            compression="gzip",
            float_format="%.17g",
        )
        aggregate = aggregate_predictions(frame)
        if aggregate["overall"]["count"] != EXPECTED_DEV_COUNT:
            raise RuntimeError("Aggregate count is not the fixed VGGHeads dev count")
        reloaded = aggregate_prediction_file(predictions_path)
        if not _aggregate_close(aggregate, reloaded):
            raise RuntimeError("Saved predictions do not reproduce the in-memory aggregate")
        regression_checks = _regression_check(aggregate, targets)
        _write_aggregate_files(partial_dir, aggregate)
        write_json_atomic(
            partial_dir / "validation.json",
            {
                "prediction_reaggregation_matches": True,
                "weighted_yaw_bins_reconstruct_overall": True,
                "count": aggregate["overall"]["count"],
                "regression_checks": regression_checks,
            },
        )
        _status(
            partial_dir,
            "completed",
            count=aggregate["overall"]["count"],
            so3_mean_deg=aggregate["overall"]["so3_mean_deg"],
        )
        os.replace(partial_dir, final_dir)
        return final_dir
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        _status(partial_dir, status, error=f"{type(error).__name__}: {error}")
        raise


def regroup_targets_by_checkpoint(
    targets: Iterable[VggDevEvaluationTarget],
) -> dict[str, list[VggDevEvaluationTarget]]:
    groups: dict[str, list[VggDevEvaluationTarget]] = {}
    for target in targets:
        groups.setdefault(target.checkpoint_sha256, []).append(target)
    return groups


def regenerate_aggregate(model_dir: Path) -> dict[str, Any]:
    prediction_path = model_dir / PREDICTION_FILENAME
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    aggregate = aggregate_prediction_file(prediction_path)
    _write_aggregate_files(model_dir, aggregate)
    return aggregate


def _format_metric(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.6f}"


def _write_markdown_reports(
    output_root: Path,
    overall_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    experiments = sorted({row["experiment_id"] for row in overall_rows})
    for experiment in experiments:
        selected = [row for row in overall_rows if row["experiment_id"] == experiment]
        lines = [
            f"# {experiment} VGGHeads dev evaluation",
            "",
            "| condition | epoch | count | SO(3) mean | pitch MAAE | yaw MAAE | roll MAAE | 3-axis mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in selected:
            lines.append(
                "| {condition_id} | {epoch} | {count} | {so3} | {pitch} | {yaw} | {roll} | {axis} |".format(
                    condition_id=row["condition_id"],
                    epoch="-" if row["epoch"] is None else row["epoch"],
                    count=row["count"],
                    so3=_format_metric(row["so3_mean_deg"]),
                    pitch=_format_metric(row["pitch_maae_deg"]),
                    yaw=_format_metric(row["yaw_maae_deg"]),
                    roll=_format_metric(row["roll_maae_deg"]),
                    axis=_format_metric(row["mean_axis_maae_deg"]),
                )
            )
        (reports / f"{experiment}_overall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        bins = [row for row in bin_rows if row["experiment_id"] == experiment]
        lines = [
            f"# {experiment} VGGHeads dev 15-degree GT yaw bins",
            "",
            "| condition | yaw bin | count | SO(3) mean | pitch MAAE | yaw MAAE | roll MAAE | 3-axis mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in bins:
            lines.append(
                "| {condition_id} | {yaw_bin} | {count} | {so3} | {pitch} | {yaw} | {roll} | {axis} |".format(
                    condition_id=row["condition_id"],
                    yaw_bin=row["yaw_bin"],
                    count=row["count"],
                    so3=_format_metric(row["so3_mean_deg"]),
                    pitch=_format_metric(row["pitch_maae_deg"]),
                    yaw=_format_metric(row["yaw_maae_deg"]),
                    roll=_format_metric(row["roll_maae_deg"]),
                    axis=_format_metric(row["mean_axis_maae_deg"]),
                )
            )
        (reports / f"{experiment}_yaw_bins_15.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def write_comparison_outputs(
    output_root: Path,
    targets: Iterable[VggDevEvaluationTarget],
) -> None:
    selected = list(targets)
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    by_sha = regroup_targets_by_checkpoint(selected)

    for target in selected:
        model_dir = output_root / "models" / target.checkpoint_sha256
        aggregate_path = model_dir / "metrics" / "aggregate.json"
        if not aggregate_path.is_file():
            raise FileNotFoundError(aggregate_path)
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        overall_rows.append(
            {
                "experiment_id": target.experiment_id,
                "condition_id": target.condition_id,
                "logical_id": target.logical_id,
                "epoch": target.epoch,
                "checkpoint_sha256": target.checkpoint_sha256,
                "shared_prediction_result": len(by_sha[target.checkpoint_sha256]) > 1,
                **aggregate["overall"],
            }
        )
        for row in aggregate["yaw_bins_15"]:
            bin_rows.append(
                {
                    "experiment_id": target.experiment_id,
                    "condition_id": target.condition_id,
                    "logical_id": target.logical_id,
                    "epoch": target.epoch,
                    "checkpoint_sha256": target.checkpoint_sha256,
                    **row,
                }
            )

    overall_frame = pd.DataFrame(overall_rows)
    bins_frame = pd.DataFrame(bin_rows)
    overall_frame.to_csv(
        comparison_dir / "overall.csv",
        index=False,
        float_format="%.17g",
    )
    bins_frame.to_csv(
        comparison_dir / "yaw_bins_15.csv",
        index=False,
        float_format="%.17g",
    )
    write_json_atomic(
        comparison_dir / "overall.json",
        overall_frame.to_dict(orient="records"),
    )
    write_json_atomic(
        comparison_dir / "yaw_bins_15.json",
        bins_frame.to_dict(orient="records"),
    )
    write_json_atomic(
        comparison_dir / "targets.json",
        [
            {
                **asdict(target),
                "checkpoint_path": target.checkpoint_path.as_posix(),
                "reference_run_path": target.reference_run_path.as_posix(),
                "logical_id": target.logical_id,
            }
            for target in selected
        ],
    )
    _write_markdown_reports(output_root, overall_rows, bin_rows)
