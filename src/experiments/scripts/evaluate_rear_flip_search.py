"""Evaluate every completed condition in one rear/flip experiment directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

from experiments.common.run_directory import experiment_run_path
from hpe.datasets.common import write_json_atomic
from hpe.evaluation import compare_runs
from training.prepare_data import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rear_flip_search")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--checkpoint-choice", choices=("best", "final"), default="final")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def _safe_suffix(value: str) -> str:
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise ValueError("name suffix may contain only lowercase letters, digits, '_' and '-'")
    return value


def _safe_name(value: str, kind: str) -> str:
    if not value or value in {".", "..", "comparisons"} or Path(value).name != value:
        raise ValueError(f"{kind} must be one path-safe name other than 'comparisons'")
    return value


def main() -> None:
    args = build_parser().parse_args()
    suffix = _safe_suffix(args.name_suffix)
    baseline_run = _safe_name(args.baseline_run, "baseline run")
    run_dir = experiment_run_path(ROOT, args.run_id)
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("Complete all search conditions before external evaluation")
    search_config = json.loads((run_dir / "config.json").read_text())
    conditions = search_config["conditions"]
    output_root = ROOT / "eval" / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / "eval" / baseline_run / "evaluations" / "baseline"
    baseline_metadata = json.loads((baseline / "run.json").read_text())
    evaluation_key = f"{baseline_run}_{args.checkpoint_choice}{suffix}"
    status_path = output_root / f"status_{evaluation_key}.json"
    write_json_atomic(
        status_path,
        {
            "status": "running",
            "baseline_run": baseline_run,
            "checkpoint_choice": args.checkpoint_choice,
            "name_suffix": suffix,
        },
    )

    rows = []
    progress = tqdm(conditions, desc="External evaluation", unit="condition")
    try:
        for condition in progress:
            condition_id = condition["condition_id"]
            checkpoint_suffix = "_best" if args.checkpoint_choice == "best" else ""
            evaluation_name = condition_id + checkpoint_suffix + suffix
            progress.set_postfix(condition=condition_id)
            result_dir = output_root / "conditions" / evaluation_name
            comparison_name = f"{baseline_metadata['run_name']}_vs_{evaluation_name}"
            comparison_dir = (
                output_root
                / "comparisons"
                / comparison_name
            )
            if result_dir.is_dir() and not comparison_dir.exists():
                compare_runs(
                    baseline,
                    result_dir,
                    output_root,
                    name=comparison_name,
                )
            elif comparison_dir.exists() and not result_dir.is_dir():
                raise FileNotFoundError(
                    f"Comparison exists without its evaluation result: {comparison_dir}"
                )
            elif not result_dir.is_dir():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.scripts.evaluate_candidate",
                        "--run-id",
                        args.run_id,
                        "--condition-id",
                        condition_id,
                        "--checkpoint-choice",
                        args.checkpoint_choice,
                        "--baseline-run",
                        baseline_run,
                        "--name",
                        evaluation_name,
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.batch_size),
                        "--workers",
                        str(args.workers),
                    ],
                    check=True,
                )
            summaries = json.loads((result_dir / "summary.json").read_text())
            for summary in summaries:
                rows.append(
                    {
                        "condition_id": condition_id,
                        "evaluation_name": evaluation_name,
                        "rear_fraction_level": condition["rear_fraction_level"],
                        "rear_fraction": condition["rear_fraction"],
                        "flip_consistency_weight": condition["flip_consistency_weight"],
                        **summary,
                    }
                )

        summary_json = output_root / f"summary_{evaluation_key}.json"
        summary_csv = output_root / f"summary_{evaluation_key}.csv"
        write_json_atomic(summary_json, rows)
        with summary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json_atomic(
            status_path,
            {
                "status": "completed",
                "baseline_run": baseline_run,
                "checkpoint_choice": args.checkpoint_choice,
                "conditions": len(conditions),
                "summary": str(summary_json.relative_to(ROOT)),
            },
        )
    except BaseException as error:
        write_json_atomic(
            status_path,
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    main()
