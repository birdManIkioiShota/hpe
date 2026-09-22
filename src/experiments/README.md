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

## Project terminology

Experiment documentation and CLI help use two project-specific terms.

- `audited base checkpoint` means `checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` with SHA-256 `3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6`, loaded by the fixed SixDRepNet360-ResNet50 architecture after the baseline audit described in the repository README.
- `protected benchmark` means an external evaluation dataset that is not read by experiment training or checkpoint selection. Current protected evaluations are AGORA-HPE, AFLW2000, 300W-LP, and DAD-3DHeads official validation when the corresponding baseline run is supplied. Their results are produced only by explicit evaluation commands after training.

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
mean and P90 metrics must each independently satisfy the configured retention tolerance.
Feasible checkpoints are then ordered by the worst P90 among the four signed rear
groups, overall rear P90, worst signed-group >90-degree rate, and rear mean. Internal
metrics also include dataset-specific front, side, and rear groups for VGGHeads and
DAD-3DHeads diagnostics. The protected benchmark datasets are not read by this training
script.

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


## YawPose rear-yaw reliability search

YawPose rear-yaw training is split into dataset preparation, fixed teacher inference,
reliability precomputation, training, and protected external evaluation. Teacher inference
is completed before any training run; training conditions only read immutable ranked
subset manifests.

Prepare the rear-only YawPose manifest from the local dataset:

```bash
uv run python -m experiments.scripts.prepare_yawpose \
  --data-id yawpose_rear \
  --dataset-root datasets/yawpose
```

If `datasets/yawpose/manual_corrections.jsonl` does not exist, preparation records an
empty correction snapshot. Source YawPose files are never modified.

Generate the fixed SemiUHPE EfficientNetV2-S predictions after checking out the revision
recorded in `docs/experiments/yawpose_rear_yaw_experiment_plan.md` and downloading the
published `DAD-WildHead-EffNetV2-S-best.pth` checkpoint:

```bash
uv run python -m experiments.scripts.infer_yawpose_semiuhpe \
  --data-id yawpose_rear \
  --repo /path/to/SemiUHPE \
  --checkpoint /path/to/DAD-WildHead-EffNetV2-S-best.pth
```

WHENet keeps its TensorFlow/Keras environment separate from this project's PyTorch
environment. Run the standalone adapter with the Python interpreter for the fixed WHENet
checkout:

```bash
/path/to/whenet/python src/experiments/scripts/infer_yawpose_whenet.py \
  --repo /path/to/HeadPoseEstimation-WHENet \
  --checkpoint /path/to/WHENet.h5
```

Both external teacher adapters write a prediction JSONL and a sidecar
`.manifest.json` containing the fixed implementation revision, checkpoint SHA-256,
preprocessing description, yaw adapter, and adapter-fixture result.

Precompute SixDRepNet360 predictions, the three-teacher reliability score, ranking, and
all five nested subset manifests in one immutable experiment run:

```bash
uv run python -m experiments.scripts.precompute_yawpose_reliability \
  --run-id yawpose_reliability \
  --data-id yawpose_rear \
  --semiuhpe-predictions datasets/prepared/yawpose_teachers/semiuhpe_effnetv2s.jsonl \
  --whenet-predictions datasets/prepared/yawpose_teachers/whenet.jsonl
```

The resulting sample-level ranking and `top020`, `top040`, `top060`, `top080`,
and `top100` manifests are stored under
`experiments/runs/yawpose_reliability/predictions/`. Teacher models are not loaded
during subsequent training.

Inspect the fixed six-condition search matrix without starting training:

```bash
uv run python -m experiments.scripts.run_yawpose_rear_search \
  --run-id yawpose_rear_search \
  --reliability-run yawpose_reliability \
  --dry-run
```

Run the baseline plus five adoption-ratio conditions:

```bash
uv run python -m experiments.scripts.run_yawpose_rear_search \
  --run-id yawpose_rear_search \
  --reliability-run yawpose_reliability
```

The condition runs are stored below
`experiments/runs/yawpose_rear_search/conditions/`. The existing VGGHeads and
DAD-3DHeads stream keeps natural rear sampling, base-model non-rear distillation, and
SO(3) rear flip consistency. YawPose uses a separate same-sized stream with yaw-only
circular loss and no horizontal flip.

After all six conditions complete, evaluate the fixed final checkpoint against the
existing protected baselines:

```bash
uv run python -m experiments.scripts.evaluate_yawpose_rear_search \
  --run-id yawpose_rear_search \
  --baseline-run baseline_fp32

uv run python -m experiments.scripts.evaluate_yawpose_rear_search \
  --run-id yawpose_rear_search \
  --baseline-run baseline_dad_fp32 \
  --name-suffix _dad
```

Normal SO(3) evaluation continues to use `evaluate_candidate`. The YawPose evaluation
wrapper additionally computes full-range yaw error from the saved prediction rotation
matrices and writes paired yaw summaries under `eval/yawpose_rear_search/`.


The YawPose-specific unit checks are intentionally limited to yaw periodicity/loss,
reliability ranking with nested subsets, and the fixed six-condition matrix:

```bash
uv run python -m unittest discover -s src/tests -p test_yawpose_experiment.py -v
```
