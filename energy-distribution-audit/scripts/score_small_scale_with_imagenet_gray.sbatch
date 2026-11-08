#!/bin/bash
# Manuscript Experiment 1: score ImageNet reference plus all 14 Baradad small-scale
# synthetic datasets under Guth's grayscale ImageNet64 density model.
#
# Submit:
#   mkdir -p logs/slurm
#   IMAGENET_DIR=data/imagenet_hf_64x64_val sbatch scripts/score_small_scale_with_imagenet_gray.sbatch

#SBATCH --job-name=score-eld-gray
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_ROOT/data/imagenet_hf_64x64_val}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/external/learning_with_noise/data/small_scale}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/manuscript_small_scale_50k_gray}"
GUTH_REPO="${GUTH_REPO:-$PROJECT_ROOT/external/DualScoreMatching}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/checkpoints/guth/models/finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays}"
NORMALIZATION="${NORMALIZATION:-$PROJECT_ROOT/checkpoints/guth/outputs/energies_test.pt}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LIMIT="${LIMIT:-50000}"

cd "$PROJECT_ROOT"

python -m density_audit.guth_histogram \
  --guth-repo "$GUTH_REPO" \
  --model-dir "$MODEL_DIR" \
  --normalization-constant "$NORMALIZATION" \
  --real-dataset imagenet \
  --datasets \
    imagenet="$IMAGENET_DIR" \
    dead_squares="$DATA_ROOT/dead_leaves-squares/train" \
    dead_oriented="$DATA_ROOT/dead_leaves-oriented/train" \
    dead_mixed="$DATA_ROOT/dead_leaves-mixed/train" \
    dead_textures="$DATA_ROOT/dead_leaves-textures/train" \
    stat_spectrum="$DATA_ROOT/stat-spectrum/train" \
    stat_wmm="$DATA_ROOT/stat-wmm/train" \
    stat_spectrum_color="$DATA_ROOT/stat-spectrum_color/train" \
    stat_spectrum_color_wmm="$DATA_ROOT/stat-spectrum_color_wmm/train" \
    stylegan_random="$DATA_ROOT/stylegan-random/train" \
    stylegan_highfreq="$DATA_ROOT/stylegan-highfreq/train" \
    stylegan_sparse="$DATA_ROOT/stylegan-sparse/train" \
    stylegan_oriented="$DATA_ROOT/stylegan-oriented/train" \
    feature_vis_random="$DATA_ROOT/feature_vis-random/train" \
    feature_vis_dead_leaves="$DATA_ROOT/feature_vis-dead_leaves/train" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --output-dir "$OUTPUT_DIR"

python -m density_audit.checks scores --result-dir "$OUTPUT_DIR" --min-n "$LIMIT"
python -m density_audit.eld_table --score-dirs "$OUTPUT_DIR" --real-dataset imagenet --min-n "$LIMIT" --output-dir "$OUTPUT_DIR/eld"
