"""Prepare VGGHeads manifests without modifying source datasets."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from hpe.datasets.common import sha256_file, write_json_atomic
from training.vgg import annotation_heads, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "vggheads"


def _named_path(root: Path, parent: Path, name: str, kind: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Use a single directory name for the {kind} ID")
    output = root / parent / name
    if not output.resolve().is_relative_to((root / parent).resolve()):
        raise ValueError(f"{kind.capitalize()} must stay inside {parent}")
    return output


def run_output_path(root: Path, name: str) -> Path:
    """Return an FT result directory; ft_runs is reserved for these outputs."""
    return _named_path(root, Path("ft_runs"), name, "run")


def prepared_data_path(root: Path, name: str) -> Path:
    """Return a prepared-data directory outside the FT result namespace."""
    return _named_path(root, Path("datasets/prepared"), name, "data")


def split_group(group: str, seed: int) -> str:
    fraction = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if fraction < .8 else "dev" if fraction < .9 else "holdout"


def source_group(annotation: Path, vgg: Path) -> str:
    """Group heads/shards of a named source without image comparison."""
    collection = annotation.relative_to(vgg).parts[0]
    return f"{collection}:{annotation.stem}"


def prepare(root: Path, output: Path, seed: int):
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    counts = Counter()
    try:
        vgg = root / "datasets/VGGHeads"
        annotations = sorted(vgg.rglob("*.npz"))
        annotations = [path for path in annotations if path.parent.name == "annotations"]
        if not annotations:
            raise ValueError("No VGGHeads annotations found")
        image_groups = {}
        paired_images = set()
        with (output / "candidates.jsonl").open("x") as candidates, (output / "exclusions.jsonl").open("x") as excluded:
            def reject(path, reason, **details):
                counts[reason] += 1
                excluded.write(json.dumps({"path": str(path.relative_to(root)), "reason": reason, **details}) + "\n")

            for annotation in tqdm(annotations, desc="VGGHeads: annotations and images", unit="image"):
                image_dir = annotation.parent.parent / "images"
                matches = [image_dir / (annotation.stem + suffix) for suffix in (".jpg", ".jpeg", ".png")]
                matches = [path for path in matches if path.is_file()]
                if len(matches) != 1:
                    reject(annotation, "missing_or_ambiguous_image")
                    continue
                image_path = matches[0]
                paired_images.add(image_path)
                key = image_path.relative_to(root).as_posix()
                try:
                    heads = annotation_heads(annotation)
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    # Reject empty/padding-dominated or enormous invalid crops, not extreme poses.
                    width, height = image.size
                    for head in heads:
                        x1, y1, x2, y2 = head["crop_xyxy"]
                        intersection = max(0, min(x2, width) - max(0, x1)) * max(0, min(y2, height) - max(0, y1))
                        area = (x2 - x1) * (y2 - y1)
                        if intersection < .5 * area or area > 4 * width * height:
                            raise ValueError("Crop has less than 50% visible area or is oversized")
                    if np.asarray(image.convert("L").resize((32, 32)), dtype=float).std() < 5:
                        raise ValueError("Near-uniform image")
                except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
                    reject(annotation, "invalid_annotation_or_image", detail=str(error))
                    continue
                raw_hash = sha256_file(image_path)
                image_groups[key] = source_group(annotation, vgg)
                annotation_hash = sha256_file(annotation)
                for head in heads:
                    candidates.write(json.dumps({"dataset": "vggheads", "instance_id": f"{key}#{head['head_index']}",
                        "image_path": key, "image_sha256": raw_hash,
                        "annotation_path": annotation.relative_to(root).as_posix(),
                        "annotation_sha256": annotation_hash, **head}) + "\n")
            for path in tqdm(sorted(vgg.rglob("*")), desc="Unpaired image inventory", unit="file"):
                if path.parent.name == "images" and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path not in paired_images:
                    reject(path, "missing_annotation")

        with (output / "image_groups.jsonl").open("x") as stream:
            for key, group in sorted(image_groups.items()):
                stream.write(json.dumps({"image_path": key, "group_id": group,
                    "excluded": False,
                    "split": split_group(group, seed)}) + "\n")
        split_counts, split_groups = Counter(), {name: set() for name in ("train", "dev", "holdout")}
        streams = {name: (output / f"{name}.jsonl").open("x") for name in split_groups}
        try:
            for row in tqdm(read_jsonl(output / "candidates.jsonl"), desc="Write grouped splits", unit="head"):
                group = image_groups[row["image_path"]]
                split = split_group(group, seed)
                row.update(group_id=group, split=split)
                streams[split].write(json.dumps(row) + "\n")
                split_counts[split] += 1
                split_groups[split].add(group)
        finally:
            for stream in streams.values():
                stream.close()
        if any(split_counts[name] == 0 for name in streams):
            raise ValueError("An empty split resulted; inspect exclusions/groups before training")
        metadata = {"schema": SCHEMA, "status": "completed", "seed": seed,
            "split_policy": "80/10/10 seeded source-group hash; same collection/stem and all heads stay together",
            "pose_conversion": "R_hpe = R_flame.T @ diag(1,-1,-1); rotation slice 403:409; extended_bbox xywh",
            "duplicate_screening": "not_performed", "exclusions": dict(counts),
            "splits": {name: {"heads": split_counts[name], "groups": len(split_groups[name]),
                              "sha256": sha256_file(output / f"{name}.jsonl")} for name in streams},
            "limitations": ["No cross-dataset or image-similarity screening is performed.",
                "VGGHeads pose annotations are generated estimates, not motion-capture truth.",
                "Holdout is reserved and is not read by the training loop."]}
        write_json_atomic(output / "metadata.json", metadata)
        write_json_atomic(output / "status.json", {"status": "completed", "heads": dict(split_counts)})
        print(json.dumps({"output": str(output), "heads": dict(split_counts), "exclusions": dict(counts)}, ensure_ascii=False, indent=2))
    except BaseException as error:
        write_json_atomic(output / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-id", default="vgg_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    prepare(ROOT, prepared_data_path(ROOT, args.data_id), args.seed)


if __name__ == "__main__":
    main()
