"""Prepare rear-only YawPose candidates without modifying the source dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tqdm import tqdm

from experiments.common.yawpose import rear_bucket, signed_yaw_degrees
from hpe.datasets.common import sha256_file, write_json_atomic, write_jsonl_atomic
from training.prepare_data import ROOT, prepared_data_path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_corrections(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        image = str(row["image"])
        if image in result:
            raise ValueError(f"duplicate manual correction at line {line_number}: {image}")
        result[image] = row
    return result


def prepare(
    root: Path,
    *,
    dataset_root: Path,
    output: Path,
    corrections_path: Path | None,
) -> Path:
    dataset_root = dataset_root.resolve()
    labels_path = dataset_root / "labels_fixed.jsonl"
    images_root = dataset_root / "images"
    if not labels_path.is_file() or not images_root.is_dir():
        raise FileNotFoundError("YawPose root must contain labels_fixed.jsonl and images/")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_atomic(output / "status.json", {"status": "running"})

    corrections = _load_corrections(corrections_path)
    labels = _read_jsonl(labels_path)
    by_image: dict[str, dict] = {}
    for line_number, row in enumerate(labels, 1):
        image = str(row["image"])
        if image in by_image:
            raise ValueError(f"duplicate YawPose image at line {line_number}: {image}")
        by_image[image] = row

    unknown_corrections = set(corrections) - set(by_image)
    if unknown_corrections:
        raise ValueError(
            "manual corrections reference unknown images: "
            + ", ".join(sorted(unknown_corrections)[:5])
        )

    correction_snapshot = []
    for image in sorted(corrections):
        correction = corrections[image]
        label_yaw = float(by_image[image]["yaw_deg"]) % 360.0
        original_yaw = float(correction["original_yaw"]) % 360.0
        if abs(signed_yaw_degrees(label_yaw - original_yaw)) > 1e-6:
            raise ValueError(f"manual correction original yaw mismatch: {image}")
        correction_snapshot.append(correction)
    write_jsonl_atomic(output / "manual_corrections_snapshot.jsonl", correction_snapshot)

    candidates: list[dict] = []
    bucket_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    corrected_count = 0
    for row in tqdm(labels, desc="Prepare YawPose rear", unit="sample"):
        image = str(row["image"])
        correction = corrections.get(image)
        original_raw = float(row["yaw_deg"]) % 360.0
        canonical_raw = (
            float(correction["corrected_yaw"]) % 360.0 if correction else original_raw
        )
        canonical = signed_yaw_degrees(canonical_raw)
        if abs(canonical) < 120.0:
            continue
        image_path = (dataset_root / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        try:
            relative_image = image_path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                "YawPose dataset must be inside the HPE project root so manifests remain portable"
            ) from exc
        bucket = rear_bucket(canonical)
        source = str(row.get("source", "unknown"))
        record = {
            "dataset": "yawpose",
            "instance_id": image,
            "image_path": relative_image,
            "image_sha256": sha256_file(image_path),
            "source": source,
            "label_source": str(row.get("label_source", "")),
            "original_yaw_deg": signed_yaw_degrees(original_raw),
            "canonical_yaw_deg": canonical,
            "human_corrected": correction is not None,
            "rear_bucket": bucket,
        }
        candidates.append(record)
        bucket_counts[bucket] += 1
        source_counts[source] += 1
        corrected_count += int(correction is not None)

    if not candidates:
        raise ValueError("YawPose contains no rear candidates")
    manifest = output / "rear_candidates.jsonl"
    write_jsonl_atomic(manifest, candidates)
    correction_snapshot_path = output / "manual_corrections_snapshot.jsonl"
    metadata = {
        "schema": "yawpose_rear_v1",
        "status": "completed",
        "source_dataset_root": dataset_root.relative_to(root.resolve()).as_posix(),
        "labels": {
            "path": labels_path.relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(labels_path),
            "count": len(labels),
        },
        "manual_corrections": {
            "source_path": None if corrections_path is None or not corrections_path.exists() else corrections_path.resolve().relative_to(root.resolve()).as_posix(),
            "snapshot": correction_snapshot_path.relative_to(root.resolve()).as_posix(),
            "snapshot_sha256": sha256_file(correction_snapshot_path),
            "count": len(corrections),
        },
        "rear_candidates": {
            "path": manifest.relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(manifest),
            "count": len(candidates),
            "human_corrected": corrected_count,
            "by_source": dict(sorted(source_counts.items())),
            "by_bucket": dict(sorted(bucket_counts.items())),
        },
        "yaw_convention": "signed=((yaw_deg+180)%360)-180; rear=abs(yaw)>=120",
        "image_policy": "source 320x320 crop; files are referenced, not copied",
    }
    write_json_atomic(output / "metadata.json", metadata)
    write_json_atomic(output / "status.json", {"status": "completed", "rear_candidates": len(candidates)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-id", default="yawpose_rear")
    parser.add_argument("--dataset-root", default="datasets/yawpose")
    parser.add_argument("--corrections", default="datasets/yawpose/manual_corrections.jsonl")
    args = parser.parse_args()
    dataset_root = (ROOT / args.dataset_root).resolve()
    corrections = (ROOT / args.corrections).resolve() if args.corrections else None
    output = prepared_data_path(ROOT, args.data_id)
    try:
        manifest = prepare(
            ROOT,
            dataset_root=dataset_root,
            output=output,
            corrections_path=corrections,
        )
    except BaseException as error:
        if output.is_dir():
            write_json_atomic(
                output / "status.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
    print(manifest.relative_to(ROOT))


if __name__ == "__main__":
    main()
