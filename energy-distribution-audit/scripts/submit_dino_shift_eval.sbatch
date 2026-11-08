#!/bin/bash
# Example:
# SHIFT_NAME=in_a SHIFT_DATA=/path/to/imagenet-a IMAGENET1K=/path/to/imagenet \
#   sbatch scripts/submit_dino_shift_eval.sbatch

#SBATCH --job-name=dino-shift
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/configs/dino_sources.csv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/dino/finetune}"
DINO_REPO="${DINO_REPO:-$PROJECT_ROOT/external/dino}"
SHIFT_NAME="${SHIFT_NAME:?Set SHIFT_NAME, for example in_c or in_a}"
SHIFT_DATA="${SHIFT_DATA:?Set SHIFT_DATA to the benchmark image root}"
IMAGENET1K="${IMAGENET1K:?Set IMAGENET1K to the ImageNet root used for class mapping}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/results/dino_${SHIFT_NAME}.csv}"
PER_ROOT_OUTPUT="${PER_ROOT_OUTPUT:-$PROJECT_ROOT/results/dino_${SHIFT_NAME}_per_root.csv}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$(dirname "$OUTPUT")"
cd "$PROJECT_ROOT"

python -m density_audit.eval_dino_shift \
  --manifest "$MANIFEST" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --shift-name "$SHIFT_NAME" \
  --data "$SHIFT_DATA" \
  --imagenet1k "$IMAGENET1K" \
  --output "$OUTPUT" \
  --per-root-output "$PER_ROOT_OUTPUT" \
  --dino-repo "$DINO_REPO" \
  --arch "${ARCH:-vit_small}" \
  --patch-size "${PATCH_SIZE:-16}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-12}" \
  --device "${DEVICE:-cuda}"
