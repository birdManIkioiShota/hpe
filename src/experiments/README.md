# Experiment tooling

This package contains experiment-only orchestration. Reusable model, geometry, dataset,
loss, and evaluation primitives belong in their normal production modules.

Generated experiment state is stored outside the source tree:

```text
experiments/runs/<run-id>/
  config.json
  provenance.json
  status.json
  events.jsonl
  checkpoints/
  metrics/
  predictions/
  artifacts/
```

The source tree must not encode conversational history, temporary phase numbers, or
experiment conclusions in module, class, or function names. Run conditions belong in
configuration/provenance files and generated run directories.

## Pose-distribution audit

Prepared rotation-matrix manifests can be audited with a full-range head-forward
azimuth that does not depend on canonical Euler yaw:

```bash
uv run python -m experiments.scripts.analyze_pose_distribution \
  --run-id vgg_pose_distribution \
  --data-id vgg_data
```

The script writes sample-level azimuths under `artifacts/` and aggregate counts under
`metrics/`.

Rear crops can be exported for direct annotation inspection with deterministic sampling:

```bash
uv run python -m experiments.scripts.export_pose_audit_samples \
  --run-id vgg_rear_audit_samples \
  --data-id vgg_data \
  --split dev \
  --samples-per-bucket 16
```

The four exported buckets are negative/positive 120–150 degrees and negative/positive
150–180 degrees. Each crop is accompanied by its source path, crop coordinates,
full-range azimuth, and rotation matrix.

## Horizontal-flip equivariance

A compatible checkpoint can be checked for left/right equivariance on a prepared
VGGHeads split:

```bash
uv run python -m experiments.scripts.analyze_flip_equivariance \
  --run-id base_flip_equivariance \
  --checkpoint checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth \
  --data-id vgg_data \
  --split dev
```

The metric is the SO(3) distance between the prediction on the original image and the
prediction on the horizontally flipped image after conjugating the latter by the
horizontal-reflection matrix.

## DAD-3DHeads train preparation

Official DAD validation remains benchmark-only. The train archive can be prepared
independently for experiment training:

```bash
uv run python -m experiments.scripts.prepare_dad3dheads_train \
  --data-id dad3dheads_train \
  --archive datasets/downloads/DAD-3DHeads/train.tar
```

The output is stored under `datasets/prepared/dad3dheads_train/` as deterministic
internal `train.jsonl` and `dev.jsonl` manifests. The released DAD metadata does not
expose a subject/video group identifier, so this split is item-level and is recorded as
a limitation in metadata.

## Rear-balanced distillation training

The experiment trainer starts from the audited base checkpoint, combines VGGHeads and
DAD train, balances the four rear azimuth groups, and distills the base model only on
non-rear retention samples:

```bash
uv run python -m experiments.scripts.train_rear_balanced \
  --run-id dad_vgg_rear_distill \
  --vgg-data-id vgg_data \
  --dad-data-id dad3dheads_train \
  --update-scope layer4 \
  --rear-fraction 0.25 \
  --backbone-lr 1e-6 \
  --head-lr 3e-5 \
  --retain-distill-weight 1.0 \
  --rear-flip-consistency-weight 0.0
```

The flip-consistency weight is required explicitly. Set it to `0.0` to reproduce the
original rear-balanced distillation objective. A positive value adds an SO(3)
consistency loss between each rear prediction and the restored prediction from the
horizontally flipped view.

Checkpoint selection uses only the internal VGGHeads/DAD dev manifests. Front and side
means must independently satisfy the configured retention tolerance. Feasible
checkpoints are then ordered by the worst P90 among the four signed rear groups,
overall rear P90, worst signed-group >90-degree rate, and rear mean. The protected
benchmark datasets are not read by this training script.

## Rear-sampling / flip-consistency search

The fixed experiment matrix uses source-proportional rear buckets and evaluates three
rear sampling levels (the manifest's natural rate, 5%, and 25%) against three
flip-consistency weights (0, 0.2, and 1.0). All nine conditions start independently
from the audited base checkpoint and run for ten epochs by default:

```bash
uv run python -m experiments.scripts.run_rear_flip_search \
  --run-id rear_flip_search
```

Inspect the resolved matrix without creating output:

```bash
uv run python -m experiments.scripts.run_rear_flip_search \
  --run-id rear_flip_search \
  --dry-run
```

The experiment is one run at `experiments/runs/rear_flip_search/`. Individual
conditions are stored below its `conditions/` directory. Re-running the same command
skips completed conditions and resumes interrupted conditions from their last completed
epoch. The aggregate fixed-epoch comparison is written to the parent run's `metrics/`
directory. It does not read protected benchmarks.

After all conditions complete, evaluate their final ten-epoch checkpoints explicitly:

```bash
uv run python -m experiments.scripts.evaluate_rear_flip_search \
  --run-id rear_flip_search \
  --baseline-run baseline_fp32
```

DAD official validation uses its independent baseline and a suffix so that both suites
remain in the same evaluation directory:

```bash
uv run python -m experiments.scripts.evaluate_rear_flip_search \
  --run-id rear_flip_search \
  --baseline-run baseline_dad_fp32 \
  --name-suffix _dad
```

External outputs are grouped under `eval/rear_flip_search/conditions/` and paired
baseline comparisons under `eval/rear_flip_search/comparisons/`. External results are
not fed back into checkpoint or hyperparameter selection.

## Checkpoint interpolation

An unchanged SixDRepNet360 checkpoint can be created between the audited base weights
and another compatible checkpoint:

```bash
uv run python -m experiments.scripts.interpolate_checkpoints \
  --run-id weight_interp_a025 \
  --alpha 0.25 \
  --candidate-checkpoint ft_runs/vgg_ft/best.pth
```

The resulting strict-loadable checkpoint is
`experiments/runs/<run-id>/checkpoints/best.pth`.

## Final benchmark evaluation

A completed experiment run with `checkpoints/best.pth` can be evaluated against an
existing audited full baseline:

```bash
uv run python -m experiments.scripts.evaluate_candidate \
  --run-id weight_interp_a025 \
  --baseline-run baseline_fp32
```

Final benchmark output remains under `eval/<run-id>/`, with paired comparisons under
`eval/comparisons/`.
