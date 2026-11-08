#!/bin/bash
# Submit from the project root after configs/dino_sources.csv is populated:
#   DINO_REPO=external/dino EPOCHS=100 ARCH=vit_small sbatch scripts/submit_dino_modern_array.sbatch

#SBATCH --job-name=dino-modern
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/configs/dino_sources.csv}"
DINO_REPO="${DINO_REPO:-$PROJECT_ROOT/external/dino}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/checkpoints/dino/pretrain}"
ARCH="${ARCH:-vit_small}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SAVECKP_FREQ="${SAVECKP_FREQ:-20}"
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

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "No included manifest rows with existing image_dir in $MANIFEST" >&2
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#ROWS[@]} )); then
  echo "Array task $SLURM_ARRAY_TASK_ID outside manifest row count ${#ROWS[@]}; exiting."
  exit 0
fi

IFS=$'\t' read -r DATASET DISPLAY DATA_PATH <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$OUTPUT_ROOT/$DATASET"

if [[ -s "$OUT_DIR/checkpoint.pth" ]]; then
  echo "DINO checkpoint already exists: $OUT_DIR/checkpoint.pth"
  exit 0
fi

EXTRA_ARGS=()
if [[ -n "${DINO_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($DINO_EXTRA_ARGS)
fi

python -m density_audit.train_dino_modern train \
  --dino-repo "$DINO_REPO" \
  --arch "$ARCH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
  --num-workers "$NUM_WORKERS" \
  --saveckp-freq "$SAVECKP_FREQ" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
