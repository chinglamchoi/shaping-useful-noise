#!/bin/bash
# Configure site-specific Slurm account, partition, and QoS on submission.
# Example: IMAGENET1K=/path/to/imagenet sbatch --array=0-8 scripts/submit_dino_finetune_array.sbatch

#SBATCH --job-name=dino-finetune
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/configs/dino_sources.csv}"
DINO_REPO="${DINO_REPO:-$PROJECT_ROOT/external/dino}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-$PROJECT_ROOT/checkpoints/dino/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/checkpoints/dino/finetune}"
IMAGENET1K="${IMAGENET1K:?Set IMAGENET1K to an ImageNet root containing train/ and val/}"
SEED="${SEED:-$(python -m density_audit.random_seed)}"

readarray -t DATASETS < <(python - "$MANIFEST" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
if "include" in frame:
    include = frame["include"].fillna(1).astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
    frame = frame[include]
print("\n".join(frame["dataset"].astype(str)))
PY
)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$PRETRAIN_ROOT/$DATASET/checkpoint.pth"
OUTPUT_DIR="$OUTPUT_ROOT/$DATASET"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python -m density_audit.train_dino_finetune \
  --dino-repo "$DINO_REPO" \
  --checkpoint "$CHECKPOINT" \
  --imagenet1k "$IMAGENET1K" \
  --output-dir "$OUTPUT_DIR" \
  --arch "${ARCH:-vit_small}" \
  --patch-size "${PATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-12}" \
  --lr "${LR:-5e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --seed "$SEED" \
  --resume
