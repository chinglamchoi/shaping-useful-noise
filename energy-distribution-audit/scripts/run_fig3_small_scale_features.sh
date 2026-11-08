#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after the Guth histogram score CSVs exist.
#
# Optional overrides:
#   DEVICE=cuda BATCH_SIZE=256 bash scripts/run_fig3_small_scale_features.sh
#   FORCE=1 DEVICE=cuda bash scripts/run_fig3_small_scale_features.sh

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FEATURE_MODEL="${FEATURE_MODEL:-resnet50}"
FEATURE_DIR="${FEATURE_DIR:-results/fig3_features}"
OUT_DIR="${OUT_DIR:-results/figures/fig3_small_scale}"
FORCE="${FORCE:-0}"
N_PAIRS="${N_PAIRS:-5000}"
PCA_DIM="${PCA_DIM:-64}"
PCA_FIT_SAMPLES="${PCA_FIT_SAMPLES:-50000}"
TAIL_FRACTION="${TAIL_FRACTION:-0.40}"
MAX_REAL_ROWS="${MAX_REAL_ROWS:-50000}"
MAX_CANDIDATE_ROWS_PER_DATASET="${MAX_CANDIDATE_ROWS_PER_DATASET:-10000}"

REAL_SCORES="${REAL_SCORES:-results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_imagenet.csv}"
LEFT_SYNTHETIC="${LEFT_SYNTHETIC:-results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color_wmm.csv}"

mkdir -p "$FEATURE_DIR" "$OUT_DIR"

declare -a SCORE_FILES=(
  "results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_squares.csv"
  "results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_oriented.csv"
  "results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_mixed.csv"
  "results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_textures.csv"
  "results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum.csv"
  "results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_wmm.csv"
  "results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color.csv"
  "results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color_wmm.csv"
  "results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_random.csv"
  "results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_highfreq.csv"
  "results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_sparse.csv"
  "results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_oriented.csv"
  "results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_random.csv"
  "results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_dead_leaves.csv"
)

feature_path_for_score() {
  local score_csv="$1"
  local base
  base="$(basename "$score_csv")"
  base="${base#scores_}"
  base="${base%.csv}"
  printf "%s/%s_%s.npz" "$FEATURE_DIR" "$base" "$FEATURE_MODEL"
}

extract_or_skip() {
  local score_csv="$1"
  local feature_npz="$2"
  local label="$3"

  if [[ "$FORCE" != "1" && -s "$feature_npz" ]]; then
    echo "Skipping $label features; found existing $feature_npz"
    return
  fi

  echo "Extracting features for $label"
  python -m density_audit.extract_features \
    --scores-csv "$score_csv" \
    --output "$feature_npz" \
    --model "$FEATURE_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
}

extract_or_skip "$REAL_SCORES" "$FEATURE_DIR/imagenet_${FEATURE_MODEL}.npz" "ImageNet"

declare -a FEATURE_FILES=()
for score_csv in "${SCORE_FILES[@]}"; do
  if [[ ! -f "$score_csv" ]]; then
    echo "Missing score CSV: $score_csv" >&2
    exit 1
  fi
  feature_npz="$(feature_path_for_score "$score_csv")"
  FEATURE_FILES+=("$feature_npz")
  extract_or_skip "$score_csv" "$feature_npz" "$score_csv"
done

echo "Generating pooled all-small-scale Figure 3"
python -m density_audit.fig3_eld_method_real \
  --real-scores "$REAL_SCORES" \
  --synthetic-scores "$LEFT_SYNTHETIC" \
  --real-features "$FEATURE_DIR/imagenet_${FEATURE_MODEL}.npz" \
  --candidate-scores "${SCORE_FILES[@]}" \
  --candidate-features "${FEATURE_FILES[@]}" \
  --n-pairs "$N_PAIRS" \
  --pca-dim "$PCA_DIM" \
  --pca-fit-samples "$PCA_FIT_SAMPLES" \
  --tail-fraction "$TAIL_FRACTION" \
  --max-real-rows "$MAX_REAL_ROWS" \
  --max-candidate-rows-per-dataset "$MAX_CANDIDATE_ROWS_PER_DATASET" \
  --output-dir "$OUT_DIR"
