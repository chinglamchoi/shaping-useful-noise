#!/usr/bin/env bash
# Generate the simple local datasets for the DINO/modern metric table.
#
# Usage:
#   N=50000 SIZE=64 bash scripts/generate_simple_modern_datasets.sh

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-data/modern_synthetic}"
N="${N:-50000}"
SIZE="${SIZE:-64}"
SEED="${SEED:-$(python -m density_audit.random_seed)}"

python -m density_audit.generate_modern_datasets gaussian \
  --output-dir "$OUT_ROOT/gaussian_noise" \
  --n "$N" \
  --size "$SIZE" \
  --seed "$SEED"

python -m density_audit.generate_modern_datasets perlin \
  --output-dir "$OUT_ROOT/perlin_noise" \
  --n "$N" \
  --size "$SIZE" \
  --seed "$((SEED + 1))"

echo "Wrote:"
echo "  $OUT_ROOT/gaussian_noise/train"
echo "  $OUT_ROOT/perlin_noise/train"
