from __future__ import annotations

import argparse
from pathlib import Path

from hpe.datasets.prepare import prepare_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-datasets",
        help="Build evaluation-ready manifests from the extracted datasets.",
    )
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing prepared manifests.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a 6DRepNet360 checkpoint against prepared datasets.",
    )
    evaluate.add_argument("--root", type=Path, default=Path.cwd())
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--output-root", type=Path)
    evaluate.add_argument(
        "--datasets",
        nargs="+",
        choices=("agora_hpe", "aflw2000", "300w_lp"),
        default=("agora_hpe", "aflw2000", "300w_lp"),
    )
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--no-amp", action="store_true")
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser(
        "compare",
        help="Make a paired comparison between two evaluation runs.",
    )
    compare.add_argument("--root", type=Path, default=Path.cwd())
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--name")
    compare.add_argument("--bootstrap-repetitions", type=int, default=1000)
    compare.add_argument("--bootstrap-sample-limit", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-datasets":
        report = prepare_all(args.root.resolve(), overwrite=args.overwrite)
        for dataset, result in report["datasets"].items():
            print(
                f"{dataset}: {result['images']} images, "
                f"{result['instances']} instances"
            )
        print(f"Report: {report['report_path']}")
    elif args.command == "evaluate":
        from hpe.evaluation import EvaluationConfig, evaluate

        root = args.root.resolve()
        output = evaluate(
            EvaluationConfig(
                project_root=root,
                checkpoint=args.checkpoint.resolve(),
                run_name=args.run_name,
                output_root=args.output_root.resolve() if args.output_root else None,
                datasets=tuple(args.datasets),
                device=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
                amp=not args.no_amp,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
            )
        )
        print(f"Evaluation: {output}")
    elif args.command == "compare":
        from hpe.evaluation import compare_runs

        output = compare_runs(
            args.baseline,
            args.candidate,
            args.root.resolve() / "eval",
            name=args.name,
            overwrite=args.overwrite,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_sample_limit=args.bootstrap_sample_limit,
            seed=args.seed,
        )
        print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
