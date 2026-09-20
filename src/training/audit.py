from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from hpe.data.dataset import evaluation_transform
from hpe.datasets.common import sha256_file
from hpe.datasets.single_pose import landmark_crop_xyxy
from hpe.geometry import (
    circular_error_degrees,
    euler_degrees_to_matrix,
    geodesic_error_degrees,
    matrix_to_euler_degrees,
    rotation_matrix_from_6d,
    vector_errors_degrees,
)


BASE_CHECKPOINT = "checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
BASE_SHA256 = "3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6"
BENCHMARKS = {
    "agora_hpe": {
        "sha256": "c4b70189a17ea4ad7d79112f66c2562672395c8dd969e85b1bd0804357c6abf2",
        "instances": 7505, "images": 1070, "split": "validation",
    },
    "aflw2000": {
        "sha256": "218f3dc212ccfe7b6f88475649febd0124a81670de462ad9caab5e9040e98470",
        "instances": 1988, "images": 1988, "split": "validation",
    },
    "300w_lp": {
        "sha256": "9e425513d30747d1fc1c9d5e6fab93ced9c6d900ac0cc457d0d4d94c57cf1619",
        "instances": 122217, "images": 122217, "split": "train",
    },
    "dad3dheads": {"instances": 4312, "images": 4312, "split": "validation"},
}
REFERENCES = {
    "architecture_transform_and_row_vectors":
        "https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/test.py",
    "rotation_convention":
        "https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/utils.py",
    "landmark_crop":
        "https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/datasets.py",
    "agora_label_provenance": "https://github.com/hnuzhy/DirectMHP",
}


def geometry_audit() -> dict[str, Any]:
    """Check conventions against independent analytic/SciPy references, not images."""
    checks: list[str] = []
    # Include rear views, wraparound, and exact Euler singularities.
    angles = np.array([
        [0, 0, 0], [90, 0, 0], [0, 90, 0], [0, -90, 0], [0, 180, 0],
        [21, 153, -37], [-48, -172, 23], [24, 90, -32], [-31, -90, 44],
    ], dtype=np.float64)
    reference = torch.from_numpy(Rotation.from_euler("xyz", angles, degrees=True).as_matrix())
    matrices = euler_degrees_to_matrix(torch.from_numpy(angles))
    torch.testing.assert_close(matrices, reference, atol=1e-12, rtol=0)
    checks.append("Rz @ Ry @ Rx agrees with independent SciPy xyz rotations")

    canonical = matrix_to_euler_degrees(matrices)
    torch.testing.assert_close(euler_degrees_to_matrix(canonical), matrices, atol=1e-12, rtol=0)
    checks.append("rear and singular Euler round trips preserve the rotation")

    # Horizontal reflection conjugates R; this changes yaw/roll signs, not pitch.
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    reflected_angles = torch.from_numpy(angles * [1, -1, -1])
    torch.testing.assert_close(mirror @ matrices @ mirror,
                               euler_degrees_to_matrix(reflected_angles), atol=1e-12, rtol=0)
    checks.append("horizontal reflection obeys S R S and reverses yaw/roll")

    six_d = torch.cat((matrices[:, :, 0], matrices[:, :, 1]), dim=1)
    torch.testing.assert_close(rotation_matrix_from_6d(six_d), matrices, atol=1e-12, rtol=0)
    checks.append("6D output basis is stored in matrix columns")

    targets = torch.eye(3, dtype=torch.float64).expand_as(matrices)
    expected = np.rad2deg(Rotation.from_matrix(reference.numpy()).magnitude())
    torch.testing.assert_close(geodesic_error_degrees(matrices, targets, stable=True),
                               torch.from_numpy(expected), atol=1e-10, rtol=0)
    torch.testing.assert_close(geodesic_error_degrees(matrices, matrices, stable=True),
                               torch.zeros(len(matrices), dtype=torch.float64), atol=1e-10, rtol=0)
    checks.append("SO(3) distances match SciPy, including 0 and 180 degrees")

    # Non-identity targets distinguish row-vector and column-vector definitions.
    shifted = matrices.roll(1, dims=0)
    row_reference = torch.rad2deg(torch.acos((matrices * shifted).sum(dim=2).clamp(-1, 1)))
    torch.testing.assert_close(vector_errors_degrees(matrices, shifted, rows=True),
                               row_reference, atol=2e-6, rtol=0)
    checks.append("Vec1/2/3 and VMAE use rows as in upstream test.py")
    torch.testing.assert_close(circular_error_degrees(torch.tensor([179., 359.]),
                                                     torch.tensor([-179., 1.])),
                               torch.tensor([2., 2.]))
    checks.append("circular angle error wraps at 180/360 degrees")

    np.testing.assert_allclose(landmark_crop_xyxy([[10., 110.], [20., 220.]]),
                               [-30., -60., 166., 253.6], atol=1e-12)
    # A constant image checks RGB, scale, normalization, and output dimensions.
    pixel = (127, 64, 32)
    tensor = evaluation_transform()(Image.new("RGB", (300, 300), pixel))
    expected_pixel = (torch.tensor(pixel) / 255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(tensor, expected_pixel[:, None, None].expand(3, 224, 224))
    checks.append("sequential landmark margin, RGB, Resize256/CenterCrop224, ImageNet normalization")
    return {"status": "passed", "checks": checks, "references": REFERENCES}


def validate_record(record: dict[str, Any], dataset: str, root: Path) -> tuple[str, tuple[int, int]]:
    """Validate input integrity without choosing samples based on prediction errors."""
    if record["dataset"] != dataset or record["schema_version"] != 1:
        raise ValueError("Unexpected dataset or schema_version")
    if record["split"] != BENCHMARKS[dataset]["split"]:
        raise ValueError("Unexpected source split")
    for key in ("sample_id", "instance_id"):
        if not str(record[key]):
            raise ValueError(f"Empty {key}")
    angles = [float(record[f"{axis}_deg"]) for axis in ("pitch", "yaw", "roll")]
    if not all(math.isfinite(value) and abs(value) <= 180 for value in angles):
        raise ValueError("Invalid source Euler angles")
    if dataset == "dad3dheads":
        from hpe.datasets.dad3dheads import validate_rotation
        matrix = validate_rotation(record["rotation_matrix"])
        reconstructed = euler_degrees_to_matrix(torch.tensor([angles], dtype=torch.float64))[0].numpy()
        if not np.allclose(matrix, reconstructed, atol=1e-4, rtol=0):
            raise ValueError("Derived Euler labels do not match the DAD rotation")
    crop = np.asarray(record["crop_xyxy"], dtype=np.float64)
    if crop.shape != (4,) or not np.isfinite(crop).all():
        raise ValueError("Invalid crop coordinates")
    x1, y1, x2, y2 = (int(value) for value in crop)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty integer crop")
    # Negative coordinates are intentional: PIL pads these regions with black.
    if "bbox_xyxy" in record:
        bbox = np.asarray(record["bbox_xyxy"], dtype=np.float64)
    else:
        x, y, width, height = record["bbox_xywh"]
        bbox = np.asarray([x, y, x + width, y + height], dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("Invalid bounding box")
    if dataset == "agora_hpe":
        if not np.array_equal(crop, bbox):
            raise ValueError("AGORA crop no longer matches the frozen ground-truth box")
        occlusion = float(record["occlusion_percent"])
        if not math.isfinite(occlusion) or not 0 <= occlusion < 90:
            raise ValueError("Invalid AGORA occlusion")
    image = Path(record["image_path"])
    if image.is_absolute() or ".." in image.parts:
        raise ValueError("image_path must be relative to the project")
    resolved = (root / image).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Image resolves outside the project")
    size = (int(record["image_width"]), int(record["image_height"]))
    if min(size) <= 0:
        raise ValueError("Invalid image dimensions")
    return image.as_posix(), size


def audit_manifest(root: Path, dataset: str, output: Path, *, manifest: Path | None = None) -> dict[str, Any]:
    from hpe.evaluation.runner import MANIFESTS
    manifest = (root / (manifest or MANIFESTS[dataset])).resolve()
    expected = dict(BENCHMARKS[dataset])
    if dataset == "dad3dheads":
        from hpe.datasets.dad3dheads import SCHEMA
        metadata = json.loads((manifest.parent / "metadata.json").read_text())
        status = json.loads((manifest.parent / "status.json").read_text())
        if (metadata.get("schema") != SCHEMA or metadata.get("status") != "completed"
                or status.get("status") != "completed" or metadata.get("split") != "validation"
                or metadata.get("dataset") != dataset or metadata.get("role") != "benchmark_only"
                or metadata.get("instances") != expected["instances"] or metadata.get("images") != expected["images"]):
            raise ValueError("DAD validation preparation is incomplete or incompatible")
        expected["sha256"] = metadata["manifest_sha256"]
    digest = sha256_file(manifest)
    if digest != expected["sha256"]:
        raise ValueError(f"Frozen manifest SHA-256 mismatch: {dataset}")
    ids: set[str] = set()
    images: dict[str, tuple[int, int]] = {}
    bands = {"front_lt60": 0, "side_60_to_lt120": 0, "rear_ge120": 0}
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(tqdm(stream, total=expected["instances"],
                                               desc=f"audit labels {dataset}", unit="head"), 1):
            try:
                record = json.loads(line)
                image, size = validate_record(record, dataset, root)
                instance = str(record["instance_id"])
                if instance in ids:
                    raise ValueError("Duplicate instance_id")
                ids.add(instance)
                if image in images and images[image] != size:
                    raise ValueError("Conflicting image dimensions")
                images[image] = size
                yaw = abs(record["yaw_deg"])
                band = "front_lt60" if yaw < 60 else "side_60_to_lt120" if yaw < 120 else "rear_ge120"
                bands[band] += 1
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
    if len(ids) != expected["instances"] or len(images) != expected["images"]:
        raise ValueError(f"Frozen dataset count mismatch: {dataset}")

    # Freeze input bytes for later reproducibility checks, not duplicate screening.
    output.mkdir(parents=True, exist_ok=True)
    image_lock = output / f"{dataset}_images.jsonl"
    dataset_digest = hashlib.sha256()
    with image_lock.open("x", encoding="utf-8") as stream:
        for image, size in tqdm(sorted(images.items()), desc=f"hash images {dataset}", unit="image"):
            path = root / image
            digest = sha256_file(path)
            with Image.open(path) as source:
                if source.size != size:
                    raise ValueError(f"Image dimensions differ from manifest: {image}")
                source.verify()
            entry = {"image_path": image, "sha256": digest, "width": size[0], "height": size[1]}
            encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(encoded)
            dataset_digest.update(encoded.encode())
    return {
        "dataset": dataset, "manifest": str(manifest),
        "manifest_sha256": expected["sha256"], "instances": len(ids), "images": len(images),
        "yaw_bands": bands, "image_lock": str(image_lock),
        "image_lock_sha256": dataset_digest.hexdigest(),
        "role": "seen_pretraining_regression" if dataset == "300w_lp" else "benchmark",
    }
