from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.vgg_dev_evaluation import (
    EXPECTED_DEV_COUNT,
    evaluate_model_group,
    regenerate_aggregate,
    regroup_targets_by_checkpoint,
    verify_target_record,
    verify_vgg_dev_input,
    write_comparison_outputs,
)
from experiments.common.vgg_dev_targets import (
    TARGETS,
    VggDevEvaluationTarget,
    validate_target_registry,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("eval/vgg_dev_all_conditions")


def _select_targets(args: argparse.Namespace) -> list[VggDevEvaluationTarget]:
    if args.all:
        return list(TARGETS)
    if args.experiment is not None:
        selected = [target for target in TARGETS if target.experiment_id == args.experiment]
        if not selected:
            choices = ", ".join(sorted({target.experiment_id for target in TARGETS}))
            raise ValueError(
                f"Unknown experiment {args.experiment!r}. Available experiments: {choices}"
            )
        return selected
    if args.condition is not None:
        selected = [target for target in TARGETS if target.logical_id == args.condition]
        if not selected:
            raise ValueError(
                "--condition must be an exact logical ID in the form "
                "experiment_id:condition_id"
            )
        return selected
    raise AssertionError("One target selector is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed historical HPE checkpoints on the same VGGHeads dev "
            "51,914-instance split and aggregate by GT head-forward yaw."
        )
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 38 logical conditions (36 unique checkpoints).",
    )
    selector.add_argument(
        "--experiment",
        help="Evaluate every condition in one experiment_id.",
    )
    selector.add_argument(
        "--condition",
        help="Evaluate one exact experiment_id:condition_id logical target.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Project-relative output directory.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate aggregate files from completed saved predictions without inference.",
    )
    parser.add_argument(
        "--overwrite-stale",
        action="store_true",
        help="Replace an incompatible completed/partial model result instead of refusing reuse.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")

    validate_target_registry()
    targets = _select_targets(args)
    output_root = (ROOT / args.output_root).resolve()
    if not output_root.is_relative_to(ROOT.resolve()):
        raise ValueError("--output-root must stay inside the project root")
    output_root.mkdir(parents=True, exist_ok=True)

    input_identity = verify_vgg_dev_input(ROOT)
    if input_identity.count != EXPECTED_DEV_COUNT:
        raise AssertionError("Verified VGGHeads dev count changed unexpectedly")

    groups = regroup_targets_by_checkpoint(targets)
    print(
        f"VGGHeads dev: {input_identity.count} instances; "
        f"{len(targets)} logical condition(s); {len(groups)} unique checkpoint(s)"
    )

    if args.aggregate_only:
        for target in targets:
            verify_target_record(ROOT, target)
        for checkpoint_sha in groups:
            model_dir = output_root / "models" / checkpoint_sha
            regenerate_aggregate(model_dir)
    else:
        for checkpoint_sha, aliases in groups.items():
            print(
                f"evaluate {checkpoint_sha[:12]} "
                f"({', '.join(target.logical_id for target in aliases)})"
            )
            evaluate_model_group(
                ROOT,
                output_root,
                aliases,
                input_identity=input_identity,
                device_name=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
                overwrite_stale=args.overwrite_stale,
            )

    write_comparison_outputs(output_root, targets)


if __name__ == "__main__":
    main()
