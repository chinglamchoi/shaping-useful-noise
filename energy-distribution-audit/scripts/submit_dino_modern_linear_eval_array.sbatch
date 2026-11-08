#!/bin/bash
# Submit after DINO pretraining:
#   IMAGENET1K=/path/to/imagenet_pytorch sbatch scripts/submit_dino_modern_linear_eval_array.sbatch

#SBATCH --job-name=dino-lin
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/configs/dino_sources.csv}"
DINO_REPO="${DINO_REPO:-$PROJECT_ROOT/external/dino}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/checkpoints/dino/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/dino_downstream/linear_eval}"
IMAGENET1K="${IMAGENET1K:?Set IMAGENET1K=/path/to/ImageNet1K with train/val folders}"
ARCH="${ARCH:-vit_small}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-128}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-$(python -m density_audit.random_seed)}"

cd "$PROJECT_ROOT"

readarray -t ROWS < <(python - <<'PY' "$MANIFEST"
import pandas as pd
import sys
from pathlib import Path
manifest = pd.read_csv(sys.argv[1])
if "include" in manifest.columns:
    include = manifest["include"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
    manifest = manifest[include]
for _, row in manifest.reset_index(drop=True).iterrows():
    image_dir = str(row.get("image_dir", "")).strip()
    dataset = str(row["dataset"]).strip()
    display = str(row.get("display", dataset)).strip()
    if image_dir and Path(image_dir).exists():
        print(f"{dataset}\t{display}\t{image_dir}")
PY
)

if (( SLURM_ARRAY_TASK_ID >= ${#ROWS[@]} )); then
  echo "Array task $SLURM_ARRAY_TASK_ID outside manifest row count ${#ROWS[@]}; exiting."
  exit 0
fi

IFS=$'\t' read -r DATASET DISPLAY DATA_PATH <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
PRETRAIN_DIR="$MODEL_ROOT/$DATASET"
LINEAR_DIR="$OUTPUT_ROOT/$DATASET"

if [[ ! -s "$PRETRAIN_DIR/checkpoint.pth" ]]; then
  echo "Missing DINO checkpoint: $PRETRAIN_DIR/checkpoint.pth" >&2
  exit 1
fi
if [[ -s "$LINEAR_DIR/log.txt" || -s "$LINEAR_DIR/checkpoint.pth.tar" ]]; then
  echo "Linear eval output already exists: $LINEAR_DIR"
  exit 0
fi

EXTRA_ARGS=()
if [[ -n "${DINO_LINEAR_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($DINO_LINEAR_EXTRA_ARGS)
fi

python -m density_audit.train_dino_modern linear-eval \
  --dino-repo "$DINO_REPO" \
  --arch "$ARCH" \
  --imagenet1k "$IMAGENET1K" \
  --output-dir "$PRETRAIN_DIR" \
  --linear-output-dir "$LINEAR_DIR" \
  --epochs "$EPOCHS" \
  --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
