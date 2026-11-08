#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
REAL_ROOT="${REAL_ROOT:-$DATA_ROOT/real_reference}"
TORCHVISION_ROOT="${TORCHVISION_ROOT:-$DATA_ROOT/torchvision}"
WORKERS="${WORKERS:-16}"

mkdir -p "$REAL_ROOT" "$TORCHVISION_ROOT"

echo "Preparing torchvision datasets under $REAL_ROOT"
python -m density_audit.prepare_real_reference_datasets \
  --torchvision-root "$TORCHVISION_ROOT" \
  --output-root "$REAL_ROOT" \
  --datasets oxford_pets \
  --pets-split test \
  --workers "$WORKERS"

cat <<EOF

Next download the larger datasets. Put them here:

  COCO val2017 images:
    $REAL_ROOT/coco/val2017

  Places365 val images:
    $REAL_ROOT/places365/val_256

Suggested commands:

  mkdir -p "$REAL_ROOT/coco" "$REAL_ROOT/places365"
  cd "$REAL_ROOT/coco"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -n val2017.zip

  cd "$REAL_ROOT/places365"
  wget -c http://data.csail.mit.edu/places/places365/val_256.tar
  mkdir -p val_256
  tar -xf val_256.tar -C val_256

Afterwards, score with:

  bash scripts/score_real_reference_datasets.sh

EOF
