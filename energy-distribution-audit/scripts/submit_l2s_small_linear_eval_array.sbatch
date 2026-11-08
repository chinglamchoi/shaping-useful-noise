#!/bin/bash
# Submit from project root:
#   IMAGENET100=/path/to/imagenet100 sbatch scripts/submit_l2s_small_linear_eval_array.sbatch

#SBATCH --job-name=l2s-lin
#SBATCH --array=0-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
LEARNING_REPO="${LEARNING_REPO:-$PROJECT_ROOT/external/learning_with_noise}"
IMAGENET100="${IMAGENET100:?Set IMAGENET100=/path/to/imagenet100 with train/val folders}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SEED="${SEED:-$(python -m density_audit.random_seed)}"

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

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$LEARNING_REPO/lincls_imagenet/small_scale/$DATASET"

if [[ -s "$OUT_DIR/val_acc.txt" && -s "$OUT_DIR/encoder_classifier.pth" ]]; then
  echo "Linear eval already exists: $OUT_DIR"
  exit 0
fi

cd "$LEARNING_REPO"

CUDA_VISIBLE_DEVICES=0 python -m density_audit.run_seeded --seed "$SEED" --script "$LEARNING_REPO/align_uniform/linear_eval_imagenet100.py" \
  -d "$DATASET" \
  --imagenet100_path "$IMAGENET100" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --gpu 0
