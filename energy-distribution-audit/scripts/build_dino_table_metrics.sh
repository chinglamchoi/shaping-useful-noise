#!/usr/bin/env bash
# Build the left-side metrics for Table~\ref{tab:metric-landscape-dino}.
#
# The manifest must contain dataset, image_dir, and score_csv columns. Set the
# three required paths explicitly so the command is portable across machines.

set -euo pipefail

MANIFEST="${MANIFEST:-results/guth_dino/manifest_scored.csv}"
: "${REAL_SCORES:?Set REAL_SCORES to scores_imagenet.csv}"
: "${REAL_FEATURES:?Set REAL_FEATURES to the ImageNet feature archive}"
FEATURE_DIR="${FEATURE_DIR:-results/dino_table/features}"
FEATURE_MODEL="${FEATURE_MODEL:-resnet50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dino_table}"
DOWNSTREAM_CSV="${DOWNSTREAM_CSV:-results/dino_downstream.csv}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_FEATURE_ROWS="${MAX_FEATURE_ROWS:-20000}"
PCA_DIM="${PCA_DIM:-128}"
PCA_FIT_SAMPLES="${PCA_FIT_SAMPLES:-50000}"
RECALL_K="${RECALL_K:-3}"
LPIPS_SAMPLES="${LPIPS_SAMPLES:-5000}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-256}"
LPIPS_DEVICE="${LPIPS_DEVICE:-cuda}"

ARGS=(
  --manifest "$MANIFEST"
  --real-scores "$REAL_SCORES"
  --real-features "$REAL_FEATURES"
  --feature-dir "$FEATURE_DIR"
  --feature-model "$FEATURE_MODEL"
  --feature-device "$DEVICE"
  --feature-batch-size "$BATCH_SIZE"
  --output-dir "$OUTPUT_DIR"
  --max-feature-rows "$MAX_FEATURE_ROWS"
  --pca-dim "$PCA_DIM"
  --pca-fit-samples "$PCA_FIT_SAMPLES"
  --recall-k "$RECALL_K"
)

if [[ "${EXTRACT_MISSING_FEATURES:-0}" == "1" ]]; then
  ARGS+=(--extract-missing-features)
fi
if [[ "${FORCE_FEATURE_EXTRACT:-0}" == "1" ]]; then
  ARGS+=(--force-feature-extract)
fi
if [[ "${COMPUTE_LPIPS_VAR:-0}" == "1" ]]; then
  ARGS+=(
    --compute-lpips-var
    --lpips-samples "$LPIPS_SAMPLES"
    --lpips-device "$LPIPS_DEVICE"
    --lpips-batch-size "$LPIPS_BATCH_SIZE"
    --lpips-cache "${LPIPS_CACHE:-$OUTPUT_DIR/lpips_var_cache.csv}"
  )
fi
if [[ "${REFRESH_LPIPS_CACHE:-0}" == "1" ]]; then
  ARGS+=(--refresh-lpips-cache)
fi
if [[ -s "$DOWNSTREAM_CSV" ]]; then
  ARGS+=(--downstream-csv "$DOWNSTREAM_CSV")
fi

python -m density_audit.modern_metrics "${ARGS[@]}"
