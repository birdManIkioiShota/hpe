"""Prepare DAD-3DHeads train data for experiment-only supervised training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.common.pose import UndefinedAzimuthError, forward_azimuth_degrees, pose_band
from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.datasets.dad3dheads import REFERENCES, full_range_euler, rotation_from_model_view
from training.prepare_data import ROOT, prepared_data_path


SCHEMA = "dad3dheads-train-v1"


def _canonical_train_path(name: str, *, directory: bool) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    if any(part.startswith("._") or part == "__MACOSX" for part in path.parts):
        return None
    positions = [index for index, part in enumerate(path.parts) if part == "train"]
    if not positions:
        if directory:
            return None
        raise ValueError(f"Archive file is outside train/: {name}")
    if len(positions) != 1:
        raise ValueError(f"Archive member is not uniquely under train/: {name}")
    return PurePosixPath(*path.parts[positions[0]:])


def extract_train(archive: Path, output: Path) -> None:
    with tarfile.open(archive, "r:*") as source:
        selected: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        seen: set[PurePosixPath] = set()
        for member in source.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links/special files are not supported: {member.name}")
            canonical = _canonical_train_path(member.name, directory=member.isdir())
            if canonical is None:
                continue
            if canonical in seen:
                raise ValueError(f"Repeated canonical archive member: {canonical}")
            seen.add(canonical)
            if member.isfile():
                selected.append((member, canonical))
        if not selected:
            raise ValueError("No train files were found in the archive")
        for member, canonical in tqdm(selected, desc="Extract DAD train", unit="file"):
            target = output.joinpath(*canonical.parts)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError("Archive target escapes output")
            target.parent.mkdir(parents=True, exist_ok=True)
            reader = source.extractfile(member)
            if reader is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)


def _item_file(item: dict, key: str, folder: str, suffix: str, output: Path) -> Path:
    item_id = str(item["item_id"])
    if not item_id or PurePosixPath(item_id).name != item_id or item_id in {".", ".."}:
        raise ValueError("Invalid DAD item_id")
    path = PurePosixPath(str(item[key]))
    expected = ("train", folder, item_id + suffix)
    if path.is_absolute() or ".." in path.parts or tuple(path.parts[-3:]) != expected:
        raise ValueError(f"Unexpected train source path: {item[key]}")
    return output.joinpath(*expected)


def _split(item_id: str, seed: int, dev_fraction: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "dev" if value < dev_fraction else "train"


def prepare(
    root: Path,
    archive: Path,
    output: Path,
    *,
    seed: int,
    dev_fraction: float,
    expected_count: int | None = None,
) -> tuple[Path, Path]:
    root, archive, output = root.resolve(), archive.resolve(), output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")
    prepared_root = (root / "datasets/prepared").resolve()
    if not output.is_relative_to(prepared_root) or output == prepared_root:
        raise ValueError("Prepared output must be a new directory inside datasets/prepared")

    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        extract_train(archive, output)
        index = output / "train/train.json"
        items = json.loads(index.read_text())
        if not isinstance(items, list) or not items:
            raise ValueError("DAD train index must be a non-empty list")
        if expected_count is not None and len(items) != expected_count:
            raise ValueError(f"Expected {expected_count} DAD train entries")

        streams = {
            split: (output / f"{split}.jsonl").open("x", encoding="utf-8")
            for split in ("train", "dev")
        }
        counts = {"train": 0, "dev": 0}
        ids: set[str] = set()
        images: set[str] = set()
        try:
            for item in tqdm(items, desc="Prepare DAD train", unit="head"):
                item_id = str(item["item_id"])
                if item_id in ids:
                    raise ValueError(f"Repeated train item_id: {item_id}")
                ids.add(item_id)
                split = _split(item_id, seed, dev_fraction)

                image_path = _item_file(item, "img_path", "images", ".png", output)
                annotation_path = _item_file(item, "annotation_path", "annotations", ".json", output)
                annotation = json.loads(annotation_path.read_text())
                rotation = rotation_from_model_view(annotation["model_view_matrix"])
                pitch, yaw, roll = full_range_euler(rotation)
                try:
                    azimuth = forward_azimuth_degrees(rotation)
                except UndefinedAzimuthError:
                    azimuth = None

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

                relative_image = image_path.relative_to(root).as_posix()
                images.add(relative_image)
                row = {
                    "schema_version": 1,
                    "dataset": "dad3dheads",
                    "split": split,
                    "sample_id": item_id,
                    "instance_id": item_id,
                    "image_path": relative_image,
                    "image_sha256": sha256_file(image_path),
                    "annotation_path": annotation_path.relative_to(root).as_posix(),
                    "annotation_sha256": sha256_file(annotation_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    "bbox_xywh": box.tolist(),
                    "crop_xyxy": crop,
                    "rotation_matrix": rotation.tolist(),
                    "pitch_deg": pitch,
                    "yaw_deg": yaw,
                    "roll_deg": roll,
                    "forward_azimuth_deg": azimuth,
                    "pose_band": "undefined" if azimuth is None else pose_band(azimuth),
                    "pose_source": "model_view_rotation_transpose",
                    "attributes": item.get("attributes", {}),
                }
                streams[split].write(json.dumps(row, allow_nan=False) + "\n")
                counts[split] += 1
        finally:
            for stream in streams.values():
                stream.close()

        if min(counts.values()) == 0:
            raise ValueError("DAD train split produced an empty partition")
        metadata = {
            "schema": SCHEMA,
            "status": "completed",
            "dataset": "dad3dheads",
            "role": "experiment_training",
            "source_split": "train",
            "seed": seed,
            "dev_fraction": dev_fraction,
            "instances": len(ids),
            "images": len(images),
            "splits": {
                split: {
                    "instances": counts[split],
                    "sha256": sha256_file(output / f"{split}.jsonl"),
                }
                for split in ("train", "dev")
            },
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "index_sha256": sha256_file(index),
            "rotation_conversion": "model_view_matrix[:3,:3].T",
            "split_policy": "seeded hash of item_id; no subject/video identifier is exposed by DAD metadata",
            "references": REFERENCES,
            "limitations": [
                "Internal train/dev separation is item-level because the released metadata exposes no subject or video grouping identifier.",
                "Official DAD validation is not read or modified by this preparation.",
            ],
        }
        write_json_atomic(output / "metadata.json", metadata)
        write_json_atomic(output / "status.json", {"status": "completed", "splits": counts})
        return output / "train.jsonl", output / "dev.jsonl"
    except BaseException as error:
        write_json_atomic(
            output / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-id", default="dad3dheads_train")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("datasets/downloads/DAD-3DHeads/train.tar"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    archive = args.archive if args.archive.is_absolute() else ROOT / args.archive
    train, dev = prepare(
        ROOT,
        archive,
        prepared_data_path(ROOT, args.data_id),
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        expected_count=args.expected_count,
    )
    print(f"Train manifest: {train}")
    print(f"Dev manifest: {dev}")


if __name__ == "__main__":
    main()
