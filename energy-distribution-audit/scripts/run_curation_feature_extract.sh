#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCORE_ROOT="${SCORE_ROOT:?Set SCORE_ROOT to a directory containing scores_imagenet.csv and scores_<source>.csv}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the root containing modern_synthetic/<source>}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/curation/features}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
SOURCES=(imagenet biggan stylegan3 diffusion_low_cfg diffusion_mid_cfg high_cfg_diffusion)
LIMITS=(50000 10000 10000 10000 10000 10000)

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

for index in "${!SOURCES[@]}"; do
  source="${SOURCES[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  scores="$SCORE_ROOT/scores_${source}.csv"
  output="$OUTPUT_ROOT/${source}_resnet50.npz"
  if [[ -s "$output" ]]; then
    echo "skip $source: $output exists"
    continue
  fi
  if [[ ! -s "$scores" ]]; then
    echo "missing $scores" >&2
    exit 1
  fi
  extra=()
  if [[ "$source" != "imagenet" ]]; then
    extra+=(--search-root "$DATA_ROOT/modern_synthetic/$source/train")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -m density_audit.extract_features \
    --scores-csv "$scores" \
    --output "$output" \
    --model resnet50 \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --limit "${LIMITS[$index]}" \
    --recover-missing-paths \
    "${extra[@]}" \
    > "$OUTPUT_ROOT/logs/${source}.log" 2>&1 &
done

wait
echo "wrote features to $OUTPUT_ROOT"
