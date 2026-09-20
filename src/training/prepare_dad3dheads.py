"""Prepare the DAD-3DHeads validation archive for benchmark evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.datasets.dad3dheads import (
    SCHEMA, VALIDATION_COUNT, REFERENCES, rotation_from_model_view, full_range_euler,
)
from training.prepare_data import ROOT, prepared_data_path


def extract_validation(archive: Path, output: Path) -> None:
    """Extract only regular validation files into a fresh output directory."""
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        selected, seen = [], set()
        # Validate all names/types before writing anything from the archive.
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if any(part.startswith("._") or part == "__MACOSX" for part in path.parts):
                continue
            if not path.parts or path.parts[0] != "val":
                raise ValueError(f"Archive contains a non-validation path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links/special files are not supported: {member.name}")
            if path in seen:
                raise ValueError(f"Repeated archive member: {member.name}")
            seen.add(path)
            if member.isfile():
                selected.append((member, path))
        for member, path in tqdm(selected, desc="Extract DAD validation", unit="file"):
            target = output.joinpath(*path.parts)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError("Archive target escapes output")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)


def item_file(item: dict, key: str, folder: str, suffix: str, output: Path) -> Path:
    item_id = str(item["item_id"])
    if not item_id or PurePosixPath(item_id).name != item_id or item_id in {".", ".."}:
        raise ValueError("Invalid DAD item_id")
    path = PurePosixPath(item[key])
    expected = ("val", folder, item_id + suffix)
    if path.is_absolute() or ".." in path.parts or tuple(path.parts[-3:]) != expected:
        raise ValueError(f"Unexpected validation source path: {item[key]}")
    return output.joinpath(*expected)


def prepare(root: Path, archive: Path, output: Path, *, expected_count: int = VALIDATION_COUNT) -> Path:
    root, archive, output = root.resolve(), archive.resolve(), output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    prepared_root = (root / "datasets/prepared").resolve()
    if not output.is_relative_to(prepared_root) or output == prepared_root:
        raise ValueError("Prepared output must be a new directory inside datasets/prepared")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        extract_validation(archive, output)
        index = output / "val/val.json"
        items = json.loads(index.read_text())
        if not isinstance(items, list) or len(items) != expected_count:
            raise ValueError(f"Expected {expected_count} validation entries")
        manifest = output / "manifest.jsonl"
        ids, images = set(), set()
        with manifest.open("x", encoding="utf-8") as stream:
            for item in tqdm(items, desc="Prepare DAD validation", unit="head"):
                item_id = str(item["item_id"])
                if item_id in ids:
                    raise ValueError(f"Repeated validation item_id: {item_id}")
                ids.add(item_id)
                image_path = item_file(item, "img_path", "images", ".png", output)
                annotation_path = item_file(item, "annotation_path", "annotations", ".json", output)
                annotation = json.loads(annotation_path.read_text())
                rotation = rotation_from_model_view(annotation["model_view_matrix"])
                pitch, yaw, roll = full_range_euler(rotation)
                box = np.asarray(item["bbox"], dtype=np.float64)
                if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
                    raise ValueError(f"Invalid bbox: {item_id}")
                x, y, width, height = box.tolist()
                crop = [x, y, x + width, y + height]
                if int(crop[2]) <= int(crop[0]) or int(crop[3]) <= int(crop[1]):
                    raise ValueError(f"Empty integer crop: {item_id}")
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                    image.load()
                if min(x + width, image_width) <= max(x, 0) or min(y + height, image_height) <= max(y, 0):
                    raise ValueError(f"Bounding box has no image intersection: {item_id}")
                images.add(image_path)
                row = {
                    "schema_version": 1, "dataset": "dad3dheads", "split": "validation",
                    "sample_id": item_id, "instance_id": item_id,
                    "image_path": image_path.relative_to(root).as_posix(),
                    "annotation_path": annotation_path.relative_to(root).as_posix(),
                    "annotation_sha256": sha256_file(annotation_path),
                    "image_width": image_width, "image_height": image_height,
                    "bbox_xywh": box.tolist(), "crop_xyxy": crop,
                    "rotation_matrix": rotation.tolist(),
                    "pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll,
                    "pose_source": "model_view_rotation_transpose",
                    "yaw_source": "derived_RzRyRx_pitch_in_minus90_plus90",
                    "attributes": item.get("attributes", {}),
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        metadata = {
            "schema": SCHEMA, "status": "completed", "dataset": "dad3dheads", "split": "validation",
            "role": "benchmark_only", "instances": len(ids), "images": len(images),
            "manifest_sha256": sha256_file(manifest), "index_sha256": sha256_file(index),
            "archive": str(archive), "rotation_conversion": "model_view_matrix[:3,:3].T",
            "crop_policy": "provided xywh bbox; integer PIL crop; Resize256/CenterCrop224",
            "yaw_grouping": "derived RzRyRx branch with |pitch|<=90; ambiguous for inverted/singular poses",
            "references": REFERENCES, "duplicate_screening": "not_performed",
        }
        write_json_atomic(output / "metadata.json", metadata)
        write_json_atomic(output / "status.json", {"status": "completed", "instances": len(ids)})
        return manifest
    except BaseException as error:
        write_json_atomic(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-id", default="dad3dheads")
    parser.add_argument("--archive", type=Path, default=Path("datasets/downloads/DAD-3DHeads/val.tar"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = prepare(root, root / args.archive, prepared_data_path(root, args.data_id))
    print(f"Validation manifest: {manifest}")


if __name__ == "__main__":
    main()
