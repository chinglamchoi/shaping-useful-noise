#!/usr/bin/env bash
set -euo pipefail

# Builds the Learning-to-See table metric CSV from existing Guth scores and
# ResNet feature .npz files. Downstream columns are joined if
# results/l2s_downstream.csv exists.

DOWNSTREAM_CSV="${DOWNSTREAM_CSV:-results/l2s_downstream.csv}"
SCORE_DIRS="${SCORE_DIRS:-results/l2s_scores}"
FEATURE_DIR="${FEATURE_DIR:-results/l2s_features}"
OUTPUT_DIR="${OUTPUT_DIR:-results/l2s_table}"
DOWNSTREAM_ARGS=()
if [[ -f "$DOWNSTREAM_CSV" ]]; then
  DOWNSTREAM_ARGS=(--downstream-csv "$DOWNSTREAM_CSV")
fi

LPIPS_ARGS=()
if [[ "${COMPUTE_LPIPS_VAR:-0}" == "1" ]]; then
  LPIPS_ARGS=(
    --compute-lpips-var
    --lpips-device "${LPIPS_DEVICE:-cuda}"
    --lpips-samples "${LPIPS_SAMPLES:-2000}"
    --lpips-batch-size "${LPIPS_BATCH_SIZE:-64}"
    --lpips-cache "${LPIPS_CACHE:-$OUTPUT_DIR/lpips_var_cache.csv}"
  )
  if [[ "${REFRESH_LPIPS_CACHE:-0}" == "1" ]]; then
    LPIPS_ARGS+=(--refresh-lpips-cache)
  fi
fi

python -m density_audit.l2s_metrics \
  --score-dirs \
    "$SCORE_DIRS" \
  --feature-dir "$FEATURE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-feature-rows "${MAX_FEATURE_ROWS:-10000}" \
  --pca-dim "${PCA_DIM:-128}" \
  --pca-fit-samples "${PCA_FIT_SAMPLES:-30000}" \
  --recall-k "${RECALL_K:-3}" \
  "${DOWNSTREAM_ARGS[@]}" \
  "${LPIPS_ARGS[@]}"
