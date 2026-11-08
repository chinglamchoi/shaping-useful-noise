#!/bin/bash
# Build results/l2s_downstream.csv after linear, fine-tune, and shift evals.
#
# Usage from project root:
#   bash scripts/collect_l2s_downstream.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
LEARNING_REPO="${LEARNING_REPO:-$PROJECT_ROOT/external/learning_with_noise}"
LINEAR_ROOT="${LINEAR_ROOT:-$LEARNING_REPO/lincls_imagenet/small_scale}"
FINETUNE_ROOT="${FINETUNE_ROOT:-$LEARNING_REPO/finetune_imagenet/small_scale}"
SHIFT_CSV="${SHIFT_CSV:-$PROJECT_ROOT/results/l2s_shift_eval.csv}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/results/l2s_downstream.csv}"

ARGS=(
  --learning-repo "$LEARNING_REPO"
  --linear-root "$LINEAR_ROOT"
  --output "$OUTPUT"
)

if [[ -d "$FINETUNE_ROOT" ]]; then
  ARGS+=(--finetune-root "$FINETUNE_ROOT")
fi
if [[ -s "$SHIFT_CSV" ]]; then
  ARGS+=(--shift-csv "$SHIFT_CSV")
fi

python -m density_audit.collect_l2s_downstream "${ARGS[@]}"
