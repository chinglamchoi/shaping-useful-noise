#!/bin/bash
# Submit from project root after encoder pretraining:
#   IMAGENET100=/path/to/imagenet100 sbatch scripts/submit_l2s_small_finetune_array.sbatch
#
# This fills the IN FT column for Table~\ref{tab:metric-landscape-l2s}.

#SBATCH --job-name=l2s-ft
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
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
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
ENCODER="$LEARNING_REPO/encoders/small_scale/$DATASET/encoder.pth"
OUT_DIR="$LEARNING_REPO/finetune_imagenet/small_scale/$DATASET"

if [[ -s "$OUT_DIR/val_acc.txt" && -s "$OUT_DIR/finetuned_classifier.pth" ]]; then
  echo "Fine-tune already exists: $OUT_DIR"
  exit 0
fi
if [[ ! -s "$ENCODER" ]]; then
  echo "Missing encoder checkpoint: $ENCODER" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

python -m density_audit.train_l2s_finetune \
  --learning-repo "$LEARNING_REPO" \
  --encoder "$ENCODER" \
  --imagenet100 "$IMAGENET100" \
  --output-dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --gpu 0
