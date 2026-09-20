from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpe.datasets.common import write_json_atomic


RUN_SUBDIRECTORIES = ("checkpoints", "metrics", "predictions", "artifacts")


def validate_run_id(run_id: str) -> str:
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or run_id == "comparisons"
    ):
        raise ValueError("run_id must be one path-safe name other than 'comparisons'")
    return run_id


def experiment_run_path(project_root: Path, run_id: str) -> Path:
    return project_root / "experiments" / "runs" / validate_run_id(run_id)


@dataclass(frozen=True)
class ExperimentRun:
    path: Path

    @classmethod
    def create(
        cls,
        project_root: Path,
        run_id: str,
        *,
        config: dict[str, Any],
        provenance: dict[str, Any],
    ) -> "ExperimentRun":
        path = experiment_run_path(project_root, run_id)
        path.mkdir(parents=True, exist_ok=False)
        for name in RUN_SUBDIRECTORIES:
            (path / name).mkdir()
        write_json_atomic(path / "config.json", config)
        write_json_atomic(path / "provenance.json", provenance)
        write_json_atomic(path / "status.json", {"status": "running"})
        (path / "events.jsonl").touch()
        return cls(path)

    def write_status(self, status: str, **details: Any) -> None:
        if status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Unsupported run status: {status}")
        write_json_atomic(self.path / "status.json", {"status": status, **details})

    def complete(self, **details: Any) -> None:
        self.write_status("completed", **details)

    def fail(self, error: BaseException) -> None:
        self.write_status(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
