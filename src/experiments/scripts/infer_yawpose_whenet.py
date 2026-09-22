"""Generate fixed WHENet yaw predictions for YawPose rear candidates.

Run this file with the Python environment used by the fixed WHENet repository revision.
It intentionally does not import the HPE package so TensorFlow/Keras dependencies stay
outside the HPE uv environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm


REQUIRED_REVISION = "a0d7bdfb5e2ac97ae6b0ae3eef79fdcf4075ab82"
CALIBRATION_LABEL_SOURCE = "intent_operator_promoted"
MIN_CALIBRATION_SAMPLES = 20
CALIBRATION_MAX_ABS_YAW = 165.0
ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if abs(result) < 1e-12 else result


def _circular_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((prediction - target + 180.0) % 360.0 - 180.0)


def _calibrate_sign(
    rows: list[dict],
    predictions: dict[str, float],
) -> tuple[int, dict]:
    calibration = [
        row
        for row in rows
        if row.get("label_source") == CALIBRATION_LABEL_SOURCE
        and 120.0 <= abs(float(row["canonical_yaw_deg"])) <= CALIBRATION_MAX_ABS_YAW
    ]
    if len(calibration) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"WHENet yaw calibration requires at least {MIN_CALIBRATION_SAMPLES} "
            f"operator-verified rear samples; found {len(calibration)}"
        )
    target = np.asarray(
        [float(row["canonical_yaw_deg"]) for row in calibration],
        dtype=np.float64,
    )
    if not np.any(target < 0.0) or not np.any(target > 0.0):
        raise ValueError("WHENet yaw calibration requires both yaw signs")
    raw = np.asarray(
        [predictions[str(row["instance_id"])] for row in calibration],
        dtype=np.float64,
    )
    plus_error = float(_circular_error(raw, target).mean())
    minus_error = float(_circular_error(-raw, target).mean())
    if plus_error == minus_error:
        raise ValueError("WHENet yaw sign calibration is ambiguous")
    sign = 1 if plus_error < minus_error else -1
    return sign, {
        "passed": True,
        "method": "operator-verified YawPose rear samples",
        "label_source": CALIBRATION_LABEL_SOURCE,
        "count": len(calibration),
        "positive_count": int(np.sum(target > 0.0)),
        "negative_count": int(np.sum(target < 0.0)),
        "max_abs_yaw_deg": CALIBRATION_MAX_ABS_YAW,
        "mean_error_sign_plus_deg": plus_error,
        "mean_error_sign_minus_deg": minus_error,
        "selected_sign": sign,
    }


def _git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temp.replace(path)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="datasets/prepared/yawpose_rear/rear_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default="datasets/prepared/yawpose_teachers/whenet.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    repo = Path(args.repo).resolve()
    if _git_revision(repo) != REQUIRED_REVISION:
        raise ValueError(f"WHENet repository must be at {REQUIRED_REVISION}")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = (
        (ROOT / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest).resolve()
    )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    sys.path.insert(0, str(repo))
    from whenet import WHENet  # type: ignore

    model = WHENet(str(checkpoint))
    source_rows = _read_jsonl(manifest)
    raw_predictions: dict[str, float] = {}
    for start in tqdm(
        range(0, len(source_rows), args.batch_size),
        desc="WHENet teacher",
        unit="batch",
    ):
        batch_rows = source_rows[start : start + args.batch_size]
        images = []
        for row in batch_rows:
            bgr = cv2.imread(str(ROOT / row["image_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(ROOT / row["image_path"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(
                cv2.resize(
                    rgb,
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        yaw, _, _ = model.get_angle(np.asarray(images))
        for row, value in zip(batch_rows, yaw):
            raw_predictions[str(row["instance_id"])] = float(value)

    sign, calibration = _calibrate_sign(source_rows, raw_predictions)
    rows = [
        {
            "instance_id": str(row["instance_id"]),
            "yaw_deg": _signed(
                sign * raw_predictions[str(row["instance_id"])]
            ),
            "valid": True,
        }
        for row in source_rows
    ]

    output = (
        (ROOT / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output).resolve()
    )
    sidecar_path = output.with_suffix(".manifest.json")
    if output.exists() or sidecar_path.exists():
        raise FileExistsError(f"teacher output already exists: {output}")
    _write_jsonl(output, rows)
    sidecar = {
        "teacher_id": "whenet",
        "model": "WHENet wide-range yaw",
        "implementation": "Ascend-Research/HeadPoseEstimation-WHENet",
        "implementation_revision": REQUIRED_REVISION,
        "repository_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "input_preprocess": (
            "upstream WHENet RGB 224x224 and ImageNet normalization"
        ),
        "yaw_adapter": (
            "upstream yaw output; data-calibrated sign; wrap [-180,180)"
        ),
        "yaw_convention_validation": calibration,
        "candidate_manifest_sha256": _sha256(manifest),
        "prediction_sha256": _sha256(output),
        "count": len(rows),
    }
    _write_json(sidecar_path, sidecar)
    print(output)


if __name__ == "__main__":
    main()
