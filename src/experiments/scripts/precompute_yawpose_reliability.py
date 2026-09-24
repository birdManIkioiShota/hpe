"""Precompute fixed YawPose rear reliability ranking before any training run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.run_directory import ExperimentRun, experiment_run_path
from experiments.common.yawpose import (
    ADOPTION_RATIOS,
    REAR_YAW_BINS,
    TEACHER_IDS,
    YawPoseDataset,
    reliability_records,
    signed_yaw_degrees,
    subset_records,
    yaw_from_rotation_matrix_rad,
)
from hpe.datasets.common import sha256_file, write_json_atomic, write_jsonl_atomic
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT, BASE_SHA256
from training.prepare_data import ROOT, prepared_data_path


SEMIUHPE_REVISION = "c8f67102bf5aba8869b3f23453ac67599f21aa1f"
WHENET_REVISION = "a0d7bdfb5e2ac97ae6b0ae3eef79fdcf4075ab82"
MIN_CALIBRATION_SAMPLES = 20
SOURCE_FILES = (
    "src/experiments/common/yawpose.py",
    "src/experiments/scripts/precompute_yawpose_reliability.py",
    "src/hpe/models/sixdrepnet360.py",
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_external_teacher(
    prediction_path: Path,
    *,
    teacher_id: str,
    required_revision: str,
    expected_ids: set[str],
    candidate_manifest_sha256: str,
) -> tuple[dict[str, float], dict]:
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    manifest_path = prediction_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"teacher sidecar manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("teacher_id") != teacher_id:
        raise ValueError(f"teacher manifest ID mismatch for {teacher_id}")
    if manifest.get("implementation_revision") != required_revision:
        raise ValueError(f"unexpected {teacher_id} implementation revision")
    if manifest.get("candidate_manifest_sha256") != candidate_manifest_sha256:
        raise ValueError(f"{teacher_id} predictions were generated from a different candidate manifest")
    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if len(checkpoint_sha) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_sha.lower()):
        raise ValueError(f"invalid {teacher_id} checkpoint SHA-256")
    validation = manifest.get("yaw_convention_validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError(f"{teacher_id} yaw convention validation did not pass")
    if int(validation.get("count", 0)) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(f"{teacher_id} yaw convention validation used too few samples")
    if int(validation.get("positive_count", 0)) <= 0 or int(validation.get("negative_count", 0)) <= 0:
        raise ValueError(f"{teacher_id} yaw convention validation must cover both yaw signs")
    if int(validation.get("selected_sign", 0)) not in {-1, 1}:
        raise ValueError(f"{teacher_id} yaw convention validation has an invalid sign")

    predictions: dict[str, float] = {}
    for line_number, row in enumerate(_read_jsonl(prediction_path), 1):
        instance_id = str(row["instance_id"])
        if instance_id in predictions:
            raise ValueError(f"duplicate {teacher_id} prediction at line {line_number}")
        if row.get("valid", True) is not True:
            raise ValueError(f"invalid {teacher_id} prediction for {instance_id}")
        value = float(row["yaw_deg"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {teacher_id} prediction for {instance_id}")
        predictions[instance_id] = signed_yaw_degrees(value)
    if set(predictions) != expected_ids:
        missing = expected_ids - set(predictions)
        extra = set(predictions) - expected_ids
        raise ValueError(
            f"{teacher_id} prediction IDs differ: missing={len(missing)} extra={len(extra)}"
        )
    manifest = {
        **manifest,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "manifest_sha256": sha256_file(manifest_path),
        "count": len(predictions),
    }
    return predictions, manifest


def _infer_sixd(
    manifest_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    output_path: Path,
) -> tuple[dict[str, float], dict]:
    checkpoint = ROOT / BASE_CHECKPOINT
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("base checkpoint SHA-256 mismatch")
    dataset = YawPoseDataset(ROOT, manifest_path, augment=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = SixDRepNet360(rotation_fp32=True)
    load_checkpoint(model, checkpoint)
    model.to(device).eval()
    rows = []
    predictions: dict[str, float] = {}
    with torch.inference_mode():
        for images, _, metadata in tqdm(loader, desc="SixDRepNet360 teacher", unit="batch"):
            rotation = model(images.to(device, non_blocking=True)).float()
            yaw = torch.rad2deg(yaw_from_rotation_matrix_rad(rotation)).cpu().tolist()
            for instance_id, value in zip(metadata["instance_id"], yaw):
                signed = signed_yaw_degrees(float(value))
                predictions[str(instance_id)] = signed
                rows.append({"instance_id": str(instance_id), "yaw_deg": signed, "valid": True})
    write_jsonl_atomic(output_path, rows)
    manifest = {
        "teacher_id": "sixdrepnet360_base",
        "implementation": "birdManIkioiShota/hpe",
        "implementation_revision": "recorded-by-source-sha256",
        "checkpoint_path": BASE_CHECKPOINT,
        "checkpoint_sha256": BASE_SHA256,
        "input_preprocess": "Resize256-CenterCrop224-ImageNet normalization",
        "yaw_adapter": "atan2(R[0,2],R[2,2]) -> degrees -> [-180,180)",
        "prediction_sha256": sha256_file(output_path),
        "count": len(predictions),
    }
    write_json_atomic(output_path.with_suffix(".manifest.json"), manifest)
    dataset = None
    return predictions, manifest


def _summary(rows: list[dict], subsets: dict[str, list[dict]]) -> dict:
    result = {
        "count": len(rows),
        "score": {
            "min": min(row["reliability_score"] for row in rows),
            "median": float(np.median([row["reliability_score"] for row in rows])),
            "max": max(row["reliability_score"] for row in rows),
        },
        "subsets": {},
    }
    for name, subset in subsets.items():
        result["subsets"][name] = {
            "count": len(subset),
            "by_source": dict(sorted(Counter(str(row["source"]) for row in subset).items())),
            "by_rear_bucket": dict(
                sorted(Counter(str(row["rear_bucket"]) for row in subset).items())
            ),
            "by_rear_yaw_bin": {
                name: sum(str(row["rear_yaw_bin"]) == name for row in subset)
                for name in REAR_YAW_BINS
            },
            "human_corrected": sum(
                bool(row.get("human_corrected")) for row in subset
            ),
            "score_max": max(row["reliability_score"] for row in subset),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="yawpose_reliability")
    parser.add_argument("--data-id", default="yawpose_rear")
    parser.add_argument("--semiuhpe-predictions", required=True)
    parser.add_argument("--whenet-predictions", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid batch size/workers")
    run_path = experiment_run_path(ROOT, args.run_id)
    if run_path.exists():
        raise FileExistsError(run_path)
    prepared = prepared_data_path(ROOT, args.data_id)
    candidate_manifest = prepared / "rear_candidates.jsonl"
    prepared_metadata = prepared / "metadata.json"
    if not candidate_manifest.is_file() or not prepared_metadata.is_file():
        raise FileNotFoundError("prepare YawPose before reliability precomputation")
    candidates = _read_jsonl(candidate_manifest)
    expected_ids = {str(row["instance_id"]) for row in candidates}
    if len(expected_ids) != len(candidates):
        raise ValueError("duplicate YawPose candidate instance IDs")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SixD teacher precomputation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    semi_path = Path(args.semiuhpe_predictions).resolve()
    whenet_path = Path(args.whenet_predictions).resolve()
    candidate_sha256 = sha256_file(candidate_manifest)
    semi, semi_manifest = _load_external_teacher(
        semi_path,
        teacher_id="semiuhpe_effnetv2s",
        required_revision=SEMIUHPE_REVISION,
        expected_ids=expected_ids,
        candidate_manifest_sha256=candidate_sha256,
    )
    whenet, whenet_manifest = _load_external_teacher(
        whenet_path,
        teacher_id="whenet",
        required_revision=WHENET_REVISION,
        expected_ids=expected_ids,
        candidate_manifest_sha256=candidate_sha256,
    )

    config = {
        "kind": "yawpose_reliability_precompute",
        "data_id": args.data_id,
        "teacher_ids": list(TEACHER_IDS),
        "adoption_ratios": list(ADOPTION_RATIOS),
        "device": args.device,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "score": (
            "mean(percentile(gt_median_error), percentile(teacher_dispersion), "
            "percentile(gt_max_error)) within signed 15-degree rear-yaw strata"
        ),
        "reliability_strata": list(REAR_YAW_BINS),
    }
    provenance = {
        "candidate_manifest": {
            "path": str(candidate_manifest.relative_to(ROOT)),
            "sha256": sha256_file(candidate_manifest),
        },
        "prepared_metadata": {
            "path": str(prepared_metadata.relative_to(ROOT)),
            "sha256": sha256_file(prepared_metadata),
        },
        "external_teachers": {
            "semiuhpe_effnetv2s": semi_manifest,
            "whenet": whenet_manifest,
        },
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
    }
    run = ExperimentRun.create(ROOT, args.run_id, config=config, provenance=provenance)
    try:
        sixd_path = run.path / "predictions" / "sixdrepnet360_base.jsonl"
        sixd, sixd_manifest = _infer_sixd(
            candidate_manifest,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            output_path=sixd_path,
        )
        predictions = {
            "sixdrepnet360_base": sixd,
            "semiuhpe_effnetv2s": semi,
            "whenet": whenet,
        }
        rows = reliability_records(candidates, predictions)
        reliability_path = run.path / "predictions" / "reliability.jsonl"
        write_jsonl_atomic(reliability_path, rows)
        subsets: dict[str, list[dict]] = {}
        subset_dir = run.path / "predictions" / "subsets"
        subset_dir.mkdir()
        for ratio in ADOPTION_RATIOS:
            label = f"top{int(round(ratio * 100)):03d}"
            subset = subset_records(rows, ratio)
            subsets[label] = subset
            write_jsonl_atomic(subset_dir / f"{label}.jsonl", subset)
        summary = _summary(rows, subsets)
        summary["teachers"] = {
            "sixdrepnet360_base": sixd_manifest,
            "semiuhpe_effnetv2s": semi_manifest,
            "whenet": whenet_manifest,
        }
        summary["reliability_sha256"] = sha256_file(reliability_path)
        write_json_atomic(run.path / "metrics" / "summary.json", summary)
        run.complete(
            candidates=len(rows),
            reliability_sha256=summary["reliability_sha256"],
            subsets={name: len(rows_) for name, rows_ in subsets.items()},
        )
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
