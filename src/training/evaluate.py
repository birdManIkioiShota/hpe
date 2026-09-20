"""Explicit post-training benchmark evaluation; never invoked by train.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from hpe.datasets.common import sha256_file
from hpe.evaluation import EvaluationConfig, compare_runs, evaluate
from training.audit import BENCHMARKS
from training.evaluate_baseline import EventLog
from training.prepare_data import ROOT, run_output_path
from training.vgg import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run", required=True, help="Completed baseline directory under eval")
    parser.add_argument("--name", help="Directory name under eval; defaults to run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("Invalid batch size/workers")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required")
    run_dir = run_output_path(ROOT, args.run_id)
    if (not args.baseline_run or Path(args.baseline_run).name != args.baseline_run
            or args.baseline_run in {".", "..", "comparisons"}):
        parser.error("Baseline run must be one path-safe name other than 'comparisons'")
    reference = ROOT / "eval" / args.baseline_run
    evaluation_name = args.name or args.run_id
    if (not evaluation_name or Path(evaluation_name).name != evaluation_name
            or evaluation_name in {".", "..", "comparisons"}):
        parser.error("Evaluation name must be one path-safe name other than 'comparisons'")
    if json.loads((run_dir / "status.json").read_text())["status"] != "completed":
        raise ValueError("Finish training before this final benchmark evaluation")
    status = json.loads((reference / "status.json").read_text())
    if status.get("status") != "completed" or status.get("scope") != "full":
        raise ValueError("Reference must be a completed, full baseline evaluation")
    baseline = reference / "evaluations/baseline"
    baseline_run = json.loads((baseline / "run.json").read_text())
    if (baseline_run["protocol_version"] != 2 or baseline_run["settings"]["amp"]
            or baseline_run["settings"]["max_samples"] is not None):
        raise ValueError("Reference must use full protocol v2 / FP32")
    evaluation_root = ROOT / "eval"
    comparison_name = f"{baseline_run['run_name']}_vs_{evaluation_name}"
    occupied = [evaluation_root / evaluation_name,
                evaluation_root / f".{evaluation_name}.partial",
                evaluation_root / f"{evaluation_name}_events.jsonl",
                evaluation_root / "comparisons" / comparison_name]
    if any(path.exists() for path in occupied):
        raise FileExistsError(f"Evaluation output {evaluation_name!r} already exists under eval/. "
                              "Remove its incomplete output before rerunning it.")
    if args.batch_size != baseline_run["settings"]["batch_size"]:
        raise ValueError("Comparison requires the same evaluation batch size as the baseline")
    entries = {row["dataset"]: row for row in json.loads((reference / "audit.json").read_text())["datasets"]}
    selected = tuple(row["dataset"] for row in baseline_run["datasets"])
    if not selected or set(selected) != set(entries) or set(selected) - set(BENCHMARKS):
        raise ValueError("Baseline audit/evaluation dataset selection differs")
    locks, manifests, image_hashes = {}, {}, {}
    # Verify unchanged inputs from the reference run; no cross-image matching.
    for name in selected:
        entry = entries[name]
        manifest, lock = Path(entry["manifest"]), Path(entry["image_lock"])
        result = next(row for row in baseline_run["datasets"] if row["dataset"] == name)
        if (sha256_file(manifest) != entry["manifest_sha256"]
                or entry["manifest_sha256"] != result["manifest_sha256"]
                or sha256_file(lock) != entry["image_lock_sha256"]
                or entry["image_lock_sha256"] != result.get("image_lock_sha256")
                or result["count"] != entry["instances"]):
            raise ValueError(f"Baseline input metadata changed: {name}")
        for row in tqdm(read_jsonl(lock), total=entry["images"], desc=f"Verify {name}", unit="image"):
            if sha256_file(ROOT / row["image_path"]) != row["sha256"]:
                raise ValueError(f"Benchmark image changed: {row['image_path']}")
            image_hashes[row["image_path"]] = row["sha256"]
        locks[name] = entry["image_lock_sha256"]
        manifests[name] = manifest
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    log = EventLog(evaluation_root / f"{evaluation_name}_events.jsonl")
    try:
        result = evaluate(EvaluationConfig(project_root=ROOT, checkpoint=run_dir / "best.pth",
            run_name=evaluation_name, output_root=evaluation_root, datasets=selected,
            device=args.device, batch_size=args.batch_size, workers=args.workers, amp=False,
            protocol_version=2, deterministic=True, preserve_partial=True, image_lock_sha256=locks,
            manifest_paths=manifests, image_hashes=image_hashes), event_callback=log)
        comparison = compare_runs(baseline, result, evaluation_root, name=comparison_name)
        print(f"Evaluation: {result}\nComparison: {comparison}")
    except BaseException as error:
        log({"event": "evaluation_failed", "error": str(error)})
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
