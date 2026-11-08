#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
IMAGENET_ROOT="${IMAGENET_ROOT:?Set IMAGENET_ROOT to an ImageNet root containing train/ and val/}"
GUTH_REPO="${GUTH_REPO:-$PROJECT_ROOT/external/DualScoreMatching}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/checkpoints/guth/models/finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays}"
NORMALIZATION="${NORMALIZATION:-$PROJECT_ROOT/checkpoints/guth/outputs/energies_test.pt}"
REAL_ROOT="${REAL_ROOT:-$DATA_ROOT/real_reference}"
OUT="${OUT:-$RESULTS_ROOT/real_reference_eld_imagenet64_gray}"
LIMIT="${LIMIT:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT"

DATASETS=(
  "imagenet_ref=$IMAGENET_ROOT/train"
  "imagenet_heldout=$IMAGENET_ROOT/val"
)
SUMMARY_DATASETS=(imagenet_heldout)

if [[ -d "$REAL_ROOT/coco/val2017" ]]; then
  DATASETS+=("coco_val=$REAL_ROOT/coco/val2017")
  SUMMARY_DATASETS+=(coco_val)
else
  echo "Skipping COCO; missing $REAL_ROOT/coco/val2017" >&2
fi

if [[ -d "$REAL_ROOT/places365/val_256" ]]; then
  DATASETS+=("places365_val=$REAL_ROOT/places365/val_256")
  SUMMARY_DATASETS+=(places365_val)
else
  echo "Skipping Places365; missing $REAL_ROOT/places365/val_256" >&2
fi

if [[ -d "$REAL_ROOT/oxford_pets/test" ]]; then
  DATASETS+=("oxford_pets_test=$REAL_ROOT/oxford_pets/test")
  SUMMARY_DATASETS+=(oxford_pets_test)
fi

python -m density_audit.guth_histogram \
  --guth-repo "$GUTH_REPO" \
  --model-dir "$MODEL_DIR" \
  --normalization-constant "$NORMALIZATION" \
  --real-dataset imagenet_ref \
  --datasets "${DATASETS[@]}" \
  --output-dir "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --device "$DEVICE"

python -m density_audit.real_reference_eld \
  --score-dir "$OUT" \
  --real-dataset imagenet_ref \
  --datasets "${SUMMARY_DATASETS[@]}" \
  --seed "${SEED:-$(python -m density_audit.random_seed)}" \
  --output-csv "$OUT/real_reference_eld.csv" \
  --output-md "$OUT/real_reference_eld.md"
