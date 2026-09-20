"""Audit inputs and evaluate the fixed 6DRepNet360 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from hpe.datasets.common import sha256_file, write_json_atomic
from hpe.evaluation.runner import EvaluationConfig, evaluate
from hpe.models import SixDRepNet360, load_checkpoint
from training.audit import BASE_CHECKPOINT, BASE_SHA256, BENCHMARKS, audit_manifest, geometry_audit
from training.prepare_data import prepared_data_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-id", required=True, help="New directory name under eval; never overwritten.")
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--dad-data-id", default="dad3dheads", help="Prepared validation directory under datasets/prepared.")
    parser.add_argument("--audit-only", action="store_true", help="Audit without benchmark inference; CPU is sufficient.")
    parser.add_argument("--device", default="cuda:0", help="CUDA required for evaluation unless CPU is explicit.")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, help="First N heads per dataset; marked smoke, never a full baseline.")
    return parser


def validate_args(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run-id must start with a letter/digit and contain only letters, digits, _, -, .")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed.")
    if args.batch_size <= 0 or args.workers < 0 or args.seed < 0 or args.seed >= 2**32:
        raise ValueError("Invalid batch-size, workers, or seed (must be in [0, 2**32)).")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive.")
    if args.audit_only and args.max_samples is not None:
        raise ValueError("audit-only always audits complete manifests; omit max-samples.")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use a CUDA device or explicitly select CPU.")
    if not args.audit_only:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Evaluation does not fall back to CPU.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device index is out of range.")
        elif args.precision != "fp32":
            raise ValueError("CPU evaluation supports only fp32.")
    root = args.root.resolve()
    output = root / "eval" / args.run_id
    if not output.resolve().is_relative_to(root / "eval"):
        raise ValueError("Output must remain inside the project's eval directory.")
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}. Choose a new run-id.")
    return output


class EventLog:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8")
        self.last_progress = 0.0

    def __call__(self, event: dict[str, Any]) -> None:
        now = time.monotonic()
        if event["event"] == "evaluation_progress":
            if now - self.last_progress < 10 and event["processed"] != event["total"]:
                return
            self.last_progress = now
        payload = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
        self.stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def source_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "src" / "hpe").rglob("*.py"))
    paths += sorted((root / "src" / "training").rglob("*.py"))
    paths += [root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()}


def run(args: argparse.Namespace) -> Path:
    output = validate_args(args)
    root = args.root.resolve()
    checkpoint = root / BASE_CHECKPOINT
    # Validate the exact initial weight before any benchmark inference.
    if sha256_file(checkpoint) != BASE_SHA256:
        raise ValueError("Baseline requires the exact original 300W-LP+Panoptic checkpoint (SHA-256 mismatch).")
    manifests = {}
    if "dad3dheads" in args.datasets:
        manifests["dad3dheads"] = prepared_data_path(root, args.dad_data_id) / "manifest.jsonl"
    output.mkdir(parents=True, exist_ok=False)
    log = EventLog(output / "events.jsonl")
    started = time.monotonic()
    status: dict[str, Any] = {"operation": "baseline_evaluation", "status": "running", "training_performed": False}
    write_json_atomic(output / "status.json", status)
    try:
        settings = {**vars(args), "root": str(root), "checkpoint": str(checkpoint),
                    "scope": "audit" if args.audit_only else "smoke" if args.max_samples else "full",
                    "source_sha256": source_hashes(root)}
        write_json_atomic(output / "config.json", settings)
        log({"event": "started", "settings": settings})
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.use_deterministic_algorithms(True)

        with tqdm(total=2 + len(args.datasets), desc="Baseline input audit", unit="check") as progress:
            audit = {"geometry": geometry_audit(), "datasets": []}
            progress.update()
            model = SixDRepNet360()
            audit["checkpoint"] = load_checkpoint(model, checkpoint)
            audit["checkpoint"]["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            audit["checkpoint"]["strict_load"] = True
            del model
            progress.update()
            for dataset in args.datasets:
                result = audit_manifest(root, dataset, output / "input_lock", manifest=manifests.get(dataset))
                audit["datasets"].append(result)
                log({"event": "dataset_audited", **result})
                progress.update()
        audit["limitations"] = [
            "Image hashes lock the files used by this evaluation.",
            "Input integrity and synthetic geometry checks do not certify annotation accuracy or absence of training-data leakage.",
            "AGORA/AFLW/300W-LP labels and sample membership are not altered.",
            "300W-LP was already used to train the initial checkpoint; it is a regression reference.",
        ]
        write_json_atomic(output / "audit.json", audit)
        tqdm.write(f"Input audit passed. Inputs: {output / 'input_lock'}")
        if not args.audit_only:
            evaluation = evaluate(EvaluationConfig(
                project_root=root, checkpoint=checkpoint, run_name="baseline",
                output_root=output / "evaluations", datasets=tuple(args.datasets),
                device=args.device, batch_size=args.batch_size, workers=args.workers,
                amp=args.precision == "fp16", max_samples=args.max_samples,
                deterministic=True, preserve_partial=True,
                image_lock_sha256={item["dataset"]: item["image_lock_sha256"] for item in audit["datasets"]},
                manifest_paths=manifests,
            ), event_callback=log)
            status["evaluation"] = str(evaluation)
        status["status"] = "completed"
        status["scope"] = settings["scope"]
        status["full_baseline"] = not args.audit_only and args.max_samples is None
        status["datasets"] = list(args.datasets)
        log({"event": "completed", **status})
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        with (output / "error.txt").open("w", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        log({"event": "failed", **status})
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(output / "status.json", status)
        log.close()
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except KeyboardInterrupt:
        print("Baseline interrupted. Logs and incomplete results are preserved; use a new run-id to rerun.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Baseline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Baseline {('audit' if args.audit_only else 'evaluation')} completed: {output}")


if __name__ == "__main__":
    main()
