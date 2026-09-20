from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hpe.datasets.common import (
    SCHEMA_VERSION,
    relative_posix,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


def landmark_crop_xyxy(landmarks: list[list[float]]) -> list[float]:
    """Reproduce the loose landmark crop used by 6DRepNet/6DRepNet360."""
    if len(landmarks) != 2 or not landmarks[0] or not landmarks[1]:
        raise ValueError("Expected landmarks in [x_coordinates, y_coordinates] form")
    x_min = min(landmarks[0])
    y_min = min(landmarks[1])
    x_max = max(landmarks[0])
    y_max = max(landmarks[1])
    k = 0.20
    x_min -= 2 * k * abs(x_max - x_min)
    y_min -= 2 * k * abs(y_max - y_min)
    x_max += 2 * k * abs(x_max - x_min)
    y_max += 0.6 * k * abs(y_max - y_min)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def prepare_single_pose_dataset(
    *,
    project_root: Path,
    dataset_name: str,
    split: str,
    images_dir: Path,
    labels_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    if not overwrite and (manifest_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Prepared output already exists for {dataset_name}; pass --overwrite"
        )

    with labels_path.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    if not isinstance(labels, dict):
        raise TypeError(f"Expected an object in {labels_path}")

    image_names = {path.name for path in images_dir.glob("*.jpg")}
    label_names = set(labels)
    missing_images = sorted(label_names - image_names)
    unexpected_images = sorted(image_names - label_names)
    if missing_images or unexpected_images:
        raise ValueError(
            f"{dataset_name} image/label mismatch: "
            f"missing={len(missing_images)}, unexpected={len(unexpected_images)}"
        )

    angle_min = {axis: float("inf") for axis in ("pitch", "yaw", "roll")}
    angle_max = {axis: float("-inf") for axis in ("pitch", "yaw", "roll")}

    def records():
        for image_name in sorted(labels):
            item = labels[image_name]
            required = {"bbox", "height", "landmarks", "pose", "width"}
            missing = required - item.keys()
            if missing:
                raise ValueError(f"{image_name} is missing fields: {sorted(missing)}")
            image_path = images_dir / image_name
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise ValueError(f"Missing or empty image: {image_path}")

            yaw, pitch, roll = (float(value) for value in item["pose"])
            for axis, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
                angle_min[axis] = min(angle_min[axis], value)
                angle_max[axis] = max(angle_max[axis], value)

            bbox = [float(value) for value in item["bbox"]]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Invalid bbox for {image_name}: {bbox}")
            yield {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset_name,
                "split": split,
                "sample_id": Path(image_name).stem,
                "instance_id": Path(image_name).stem,
                "image_path": relative_posix(image_path, project_root),
                "image_width": int(item["width"]),
                "image_height": int(item["height"]),
                "bbox_xyxy": bbox,
                "crop_xyxy": landmark_crop_xyxy(item["landmarks"]),
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "roll_deg": roll,
            }

    count = write_jsonl_atomic(manifest_path, records())
    if count != len(labels):
        raise RuntimeError(f"Wrote {count} records for {len(labels)} labels")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "split": split,
        "images": count,
        "instances": count,
        "image_root": relative_posix(images_dir, project_root),
        "manifest": relative_posix(manifest_path, project_root),
        "source_labels": relative_posix(labels_path, project_root),
        "source_labels_sha256": sha256_file(labels_path),
        "angle_range_deg": {
            axis: {"min": angle_min[axis], "max": angle_max[axis]}
            for axis in angle_min
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata
