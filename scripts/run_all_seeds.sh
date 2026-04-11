#!/usr/bin/env bash
#
# End-to-end three-seed pipeline. Equivalent to `make all` but with the
# seeds spelled out, per-seed output directories, and
# the optional Bayesian-dynamics path enabled.
#
# Usage:
#   IMAGENET_VAL=/path/to/imagenet/val bash scripts/run_all_seeds.sh
#
# Set DEVICE=cpu for a small smoke run on CPU (still slow; intended for
# correctness verification, not for the final numbers).

set -euo pipefail

IMAGENET_VAL="${IMAGENET_VAL:-/data/imagenet/val}"
DEVICE="${DEVICE:-cuda}"
MANIFEST="${MANIFEST:-manifest.json}"
DYNAMICS_CKPT="${DYNAMICS_CKPT:-checkpoints/dynamics_v1.pt}"
LAPLACE_CKPT="${LAPLACE_CKPT:-checkpoints/laplace_v1.pt}"
SEEDS=(42 1337 2024)
AGENTS=(A B C D)
RUNS_ROOT="${RUNS_ROOT:-runs}"
EVAL_ROOT="${EVAL_ROOT:-runs/eval}"

mkdir -p checkpoints

echo "[1/6] manifest"
python scripts/build_manifest.py "$IMAGENET_VAL" > "$MANIFEST"

echo "[2/6] dynamics pretraining"
python -m foveated.algos.dynamics_train \
  --manifest "$MANIFEST" --device "$DEVICE" --out "$DYNAMICS_CKPT"

echo "[3/6] Laplace posterior fit"
python -m foveated.experiments.fit_laplace \
  --manifest "$MANIFEST" --device "$DEVICE" \
  --dynamics-ckpt "$DYNAMICS_CKPT" --out "$LAPLACE_CKPT"

echo "[4/6] train 4 agents x 3 seeds"
for seed in "${SEEDS[@]}"; do
  out_dir="$RUNS_ROOT/seed_${seed}"
  mkdir -p "$out_dir"
  for agent in "${AGENTS[@]}"; do
    echo "  agent=$agent seed=$seed"
    python -m foveated.experiments.train_agent \
      --agent "$agent" --seed "$seed" \
      --manifest "$MANIFEST" --device "$DEVICE" \
      --dynamics-ckpt "$DYNAMICS_CKPT" \
      --laplace-ckpt "$LAPLACE_CKPT" \
      --out-dir "$out_dir"
  done
done

echo "[5/6] evaluate each seed"
for seed in "${SEEDS[@]}"; do
  eval_dir="$EVAL_ROOT/seed_${seed}"
  mkdir -p "$eval_dir"
  python -m foveated.experiments.evaluate \
    --manifest "$MANIFEST" --device "$DEVICE" \
    --ckpt-dir "$RUNS_ROOT/seed_${seed}" \
    --dynamics-ckpt "$DYNAMICS_CKPT" \
    --out-dir "$eval_dir" \
    --seed "$seed"
done

echo "[6/6] aggregate and composite figure"
python -m foveated.experiments.aggregate_seeds \
  --eval-root "$EVAL_ROOT" --out-dir "$EVAL_ROOT"

python -m foveated.experiments.calibration \
  --manifest "$MANIFEST" --device "$DEVICE" \
  --dynamics-ckpt "$DYNAMICS_CKPT" --out-dir "$EVAL_ROOT"

python -m foveated.experiments.severity_sweep \
  --manifest "$MANIFEST" --device "$DEVICE" \
  --dynamics-ckpt "$DYNAMICS_CKPT" --out-dir "$EVAL_ROOT"

python -m foveated.experiments.feature_ablation \
  --manifest "$MANIFEST" --device "$DEVICE" \
  --agent-c-ckpt "$RUNS_ROOT/seed_42/agent_C_seed42_final.zip" \
  --dynamics-ckpt "$DYNAMICS_CKPT" \
  --out "$EVAL_ROOT/feature_ablation.json"

python -m foveated.experiments.lipschitz_estimate \
  --manifest "$MANIFEST" --device "$DEVICE" \
  --out "$EVAL_ROOT/lipschitz_constants.csv"

python -m foveated.experiments.validate_bound \
  --eval-dir "$EVAL_ROOT" --out "$EVAL_ROOT/bound_validation.md"

python -m foveated.experiments.composite_figure \
  --eval-dir "$EVAL_ROOT" --out "$EVAL_ROOT/composite_figure.pdf"

echo "done. results under $EVAL_ROOT/"
