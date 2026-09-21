from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from hpe.datasets.common import write_json_atomic


RUN_SUBDIRECTORIES = ("checkpoints", "metrics", "predictions", "artifacts")
SEARCH_SUBDIRECTORIES = ("conditions", "metrics", "artifacts")


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


def validate_condition_id(condition_id: str) -> str:
    if (
        not condition_id
        or condition_id in {".", ".."}
        or Path(condition_id).name != condition_id
    ):
        raise ValueError("condition_id must be one path-safe name")
    return condition_id


def experiment_condition_path(project_root: Path, run_id: str, condition_id: str) -> Path:
    return (
        experiment_run_path(project_root, run_id)
        / "conditions"
        / validate_condition_id(condition_id)
    )


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
        return cls.create_at(path, config=config, provenance=provenance)

    @classmethod
    def create_at(
        cls,
        path: Path,
        *,
        config: dict[str, Any],
        provenance: dict[str, Any],
        subdirectories: tuple[str, ...] = RUN_SUBDIRECTORIES,
    ) -> "ExperimentRun":
        path.mkdir(parents=True, exist_ok=False)
        for name in subdirectories:
            (path / name).mkdir()
        write_json_atomic(path / "config.json", config)
        write_json_atomic(path / "provenance.json", provenance)
        write_json_atomic(path / "status.json", {"status": "running"})
        (path / "events.jsonl").touch()
        run = cls(path)
        run.event("started")
        return run

    @classmethod
    def open(cls, path: Path) -> "ExperimentRun":
        required = ("config.json", "provenance.json", "status.json", "events.jsonl")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Experiment state is incomplete at {path}: {', '.join(missing)}"
            )
        return cls(path)

    def event(self, kind: str, **details: Any) -> None:
        payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event": kind,
            **details,
        }
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")

    def write_status(self, status: str, **details: Any) -> None:
        if status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Unsupported run status: {status}")
        write_json_atomic(self.path / "status.json", {"status": status, **details})

    def complete(self, **details: Any) -> None:
        self.write_status("completed", **details)
        self.event("completed", **details)

    def fail(self, error: BaseException) -> None:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        details = {"error": f"{type(error).__name__}: {error}"}
        self.write_status(status, **details)
        self.event(status, **details)
