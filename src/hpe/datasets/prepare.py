from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hpe.datasets.agora import prepare_agora
from hpe.datasets.common import relative_posix, write_json_atomic
from hpe.datasets.single_pose import prepare_single_pose_dataset


def prepare_all(project_root: Path, *, overwrite: bool) -> dict[str, Any]:
    extracted = project_root / "datasets" / "extracted"
    annotations = project_root / "datasets" / "downloads" / "annotations"
    prepared = project_root / "datasets" / "prepared"

    results = {
        "aflw2000": prepare_single_pose_dataset(
            project_root=project_root,
            dataset_name="aflw2000",
            split="validation",
            images_dir=extracted / "AFLW2000",
            labels_path=annotations / "val_AFLW2000.json",
            output_dir=prepared / "aflw2000",
            overwrite=overwrite,
        ),
        "300w_lp": prepare_single_pose_dataset(
            project_root=project_root,
            dataset_name="300w_lp",
            split="train",
            images_dir=extracted / "300W_LP",
            labels_path=annotations / "train_300W_LP.json",
            output_dir=prepared / "300w_lp",
            overwrite=overwrite,
        ),
        "agora_hpe": prepare_agora(
            project_root=project_root,
            images_dir=extracted / "AGORA" / "images" / "validation",
            pickle_dir=(
                extracted
                / "AGORA"
                / "annotations"
                / "validation_SMPLX"
                / "SMPLX"
            ),
            output_dir=prepared / "agora_hpe",
            overwrite=overwrite,
        ),
    }
    report_path = prepared / "preprocessing_report.json"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": results,
        "report_path": relative_posix(report_path, project_root),
    }
    write_json_atomic(report_path, report)
    return report
