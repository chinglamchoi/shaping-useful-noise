#!/bin/bash
# Submit from project root after linear eval:
#   SHIFT_NAME=in_c SHIFT_DATA=/path/to/imagenet-c-subset IMAGENET100=/path/to/imagenet100 sbatch scripts/submit_l2s_small_shift_eval_array.sbatch
#   SHIFT_NAME=in_a SHIFT_DATA=/path/to/imagenet-a-subset IMAGENET100=/path/to/imagenet100 sbatch scripts/submit_l2s_small_shift_eval_array.sbatch
#
# SHIFT_DATA must be an ImageFolder tree with the same class indexing as the
# linear classifier, e.g. an ImageNet100-compatible subset for the small-scale
# Learning-to-See protocol.

#SBATCH --job-name=l2s-shift
#SBATCH --array=0-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
LEARNING_REPO="${LEARNING_REPO:-$PROJECT_ROOT/external/learning_with_noise}"
SHIFT_NAME="${SHIFT_NAME:?Set SHIFT_NAME=in_c or in_a}"
SHIFT_DATA="${SHIFT_DATA:?Set SHIFT_DATA=/path/to/ImageFolder split}"
IMAGENET100="${IMAGENET100:?Set IMAGENET100=/path/to/imagenet100 with train/val folders for class mapping}"
SHIFT_LAYOUT="${SHIFT_LAYOUT:-single}"
OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_ROOT/results/l2s_shift_eval.csv}"
PER_ROOT_CSV="${PER_ROOT_CSV:-$PROJECT_ROOT/results/l2s_shift_eval_per_root.csv}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-12}"

DATASETS=(
  dead_leaves-squares
  dead_leaves-oriented
  dead_leaves-mixed
  dead_leaves-textures
  stat-spectrum
  stat-wmm
  stat-spectrum_color
  stat-spectrum_color_wmm
  stylegan-random
  stylegan-highfreq
  stylegan-sparse
  stylegan-oriented
  feature_vis-random
  feature_vis-dead_leaves
)

ALIASES=(
  dead_squares
  dead_oriented
  dead_mixed
  dead_textures
  stat_spectrum
  stat_wmm
  stat_spectrum_color
  stat_spectrum_color_wmm
  stylegan_random
  stylegan_highfreq
  stylegan_sparse
  stylegan_oriented
  feature_vis_random
  feature_vis_dead_leaves
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
ALIAS="${ALIASES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$LEARNING_REPO/lincls_imagenet/small_scale/$DATASET/encoder_classifier.pth"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing linear classifier checkpoint: $CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

python -m density_audit.eval_l2s_shift_suite \
  --learning-repo "$LEARNING_REPO" \
  --checkpoint "$CHECKPOINT" \
  --data "$SHIFT_DATA" \
  --class-map-from "$IMAGENET100/val" \
  --layout "$SHIFT_LAYOUT" \
  --dataset "$ALIAS" \
  --split-name "$SHIFT_NAME" \
  --output-csv "$OUTPUT_CSV" \
  --per-root-csv "$PER_ROOT_CSV" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gpu 0
