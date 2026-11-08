#!/usr/bin/env bash
# Collect DINO/ViT-S downstream columns after linear/fine-tune/shift evals.

set -euo pipefail

MANIFEST="${MANIFEST:-configs/dino_sources.csv}"
LINEAR_ROOT="${LINEAR_ROOT:-results/dino_downstream/linear_eval}"
OUTPUT="${OUTPUT:-results/dino_downstream.csv}"

ARGS=(
  --manifest "$MANIFEST"
  --linear-root "$LINEAR_ROOT"
  --output "$OUTPUT"
)

if [[ -s "${FINETUNE_CSV:-}" ]]; then
  ARGS+=(--finetune-csv "$FINETUNE_CSV")
fi
if [[ -s "${IN_C_CSV:-}" ]]; then
  ARGS+=(--in-c-csv "$IN_C_CSV")
fi
if [[ -s "${IN_A_CSV:-}" ]]; then
  ARGS+=(--in-a-csv "$IN_A_CSV")
fi

python -m density_audit.collect_dino_downstream "${ARGS[@]}"
