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
  --retain-distill-weight 1.0
```

Checkpoint selection uses only the internal VGGHeads/DAD dev manifests. A candidate
must satisfy the configured retention tolerance before rear P90, >90-degree rate, and
rear mean are considered. The protected benchmark datasets are not read by this
training script.

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
