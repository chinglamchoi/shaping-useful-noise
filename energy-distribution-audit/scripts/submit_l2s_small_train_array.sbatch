#!/bin/bash
# Submit from project root:
#   sbatch scripts/submit_l2s_small_train_array.sbatch
#
# Optional:
#   EPOCHS=200 BATCH_SIZE=768 sbatch scripts/submit_l2s_small_train_array.sbatch

#SBATCH --job-name=l2s-train
#SBATCH --array=0-13
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
LEARNING_REPO="${LEARNING_REPO:-$PROJECT_ROOT/external/learning_with_noise}"
DATA_ROOT="${DATA_ROOT:-$LEARNING_REPO/data/small_scale}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-20}"
LR="${LR:-}"
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
DATASET_DIR="$DATA_ROOT/$DATASET"
OUT_DIR="$LEARNING_REPO/encoders/small_scale/$DATASET"

if [[ -s "$OUT_DIR/encoder.pth" ]]; then
  echo "Encoder already exists: $OUT_DIR/encoder.pth"
  exit 0
fi

if [[ ! -d "$DATASET_DIR/train" ]]; then
  echo "Missing dataset train folder: $DATASET_DIR/train" >&2
  exit 1
fi

cd "$LEARNING_REPO"

LR_ARGS=()
if [[ -n "$LR" ]]; then
  LR_ARGS=(--lr "$LR")
fi

python -m density_audit.run_seeded --seed "$SEED" --script "$LEARNING_REPO/align_uniform/main.py" \
  --imagefolder "$DATASET_DIR" \
  --result_folder "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --gpus 0 1 \
  "${LR_ARGS[@]}"
