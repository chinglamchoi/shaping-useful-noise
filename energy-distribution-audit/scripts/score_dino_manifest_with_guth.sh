#!/usr/bin/env bash
# Score the modern/DINO table manifest with the Guth ImageNet density model.
#
# Fill image_dir paths in MANIFEST first, then run:
#   MANIFEST=configs/dino_sources.csv \
#   REAL_IMAGE_DIR=data/imagenet_hf_64x64_val \
#   GUTH_REPO=external/DualScoreMatching \
#   bash scripts/score_dino_manifest_with_guth.sh

set -euo pipefail

MANIFEST="${MANIFEST:-configs/dino_sources.csv}"
GUTH_REPO="${GUTH_REPO:-external/DualScoreMatching}"
MODEL_DIR="${MODEL_DIR:-checkpoints/guth/models/finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays}"
NORMALIZATION="${NORMALIZATION:-checkpoints/guth/outputs/energies_test.pt}"
REAL_IMAGE_DIR="${REAL_IMAGE_DIR:?Set REAL_IMAGE_DIR=/path/to/ImageNet64/reference/images}"
OUTPUT_DIR="${OUTPUT_DIR:-results/guth_histogram_dino_manifest_50k_gray}"
LIMIT="${LIMIT:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

ARGS=(
  --manifest "$MANIFEST"
  --guth-repo "$GUTH_REPO"
  --model-dir "$MODEL_DIR"
  --normalization-constant "$NORMALIZATION"
  --real-image-dir "$REAL_IMAGE_DIR"
  --output-dir "$OUTPUT_DIR"
  --limit "$LIMIT"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

python -m density_audit.score_manifest_with_guth "${ARGS[@]}"
