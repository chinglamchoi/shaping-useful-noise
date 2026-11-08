#!/usr/bin/env bash
# Prepare the source image folders for Table~\ref{tab:metric-landscape-dino}.
#
# This script is intentionally resumable. Set RUN_* flags to 1 for the sources
# you want to create on the current machine.
#
# Install optional deps first:
#   pip install -r requirements-modern.txt

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-data/modern_synthetic}"
N="${N:-50000}"
SIZE_SIMPLE="${SIZE_SIMPLE:-64}"
SEED="${SEED:-$(python -m density_audit.random_seed)}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_ROOT" manifests

if [[ "${RUN_SIMPLE:-1}" == "1" ]]; then
  python -m density_audit.prepare_modern_sources simple \
    --output-dir "$OUT_ROOT" \
    --n "$N" \
    --size "$SIZE_SIMPLE" \
    --seed "$SEED"
fi

if [[ "${RUN_FRACTALDB:-0}" == "1" ]]; then
  python -m density_audit.prepare_modern_sources fractaldb \
    --output-dir "$OUT_ROOT/fractaldb" \
    --n "$N" \
    --size "$SIZE_SIMPLE" \
    --seed "$SEED" \
    ${FRACTALDB_STREAMING:+--streaming}
fi

if [[ "${RUN_BIGGAN:-0}" == "1" ]]; then
  python -m density_audit.prepare_modern_sources biggan \
    --output-dir "$OUT_ROOT/biggan" \
    --model "${BIGGAN_MODEL:-biggan-deep-256}" \
    --n "$N" \
    --batch-size "${BIGGAN_BATCH_SIZE:-64}" \
    --truncation "${BIGGAN_TRUNCATION:-0.4}" \
    --device "$DEVICE" \
    --seed "$SEED"
fi

if [[ "${RUN_DIFFUSION:-0}" == "1" ]]; then
  COMMON_DIFFUSION=(
    --model "${DIFFUSION_MODEL:-stabilityai/stable-diffusion-2-base}"
    --n "$N"
    --size "${DIFFUSION_SIZE:-512}"
    --batch-size "${DIFFUSION_BATCH_SIZE:-16}"
    --device "$DEVICE"
    --fp16
  )
  if [[ -n "${PROMPTS_FILE:-}" ]]; then
    COMMON_DIFFUSION+=(--prompts-file "$PROMPTS_FILE")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    COMMON_DIFFUSION+=(--hf-token "$HF_TOKEN")
  fi

  python -m density_audit.prepare_modern_sources diffusion \
    --name diffusion_low_cfg \
    --output-dir "$OUT_ROOT/diffusion_low_cfg" \
    --steps "${DIFFUSION_STEPS_LOW:-30}" \
    --cfg "${DIFFUSION_CFG_LOW:-1.5}" \
    --seed "$((SEED + 10))" \
    "${COMMON_DIFFUSION[@]}"

  python -m density_audit.prepare_modern_sources diffusion \
    --name diffusion_mid_cfg \
    --output-dir "$OUT_ROOT/diffusion_mid_cfg" \
    --steps "${DIFFUSION_STEPS_MID:-30}" \
    --cfg "${DIFFUSION_CFG_MID:-7.5}" \
    --seed "$((SEED + 20))" \
    "${COMMON_DIFFUSION[@]}"

  python -m density_audit.prepare_modern_sources diffusion \
    --name high_cfg_diffusion \
    --output-dir "$OUT_ROOT/high_cfg_diffusion" \
    --steps "${DIFFUSION_STEPS_HIGH:-30}" \
    --cfg "${DIFFUSION_CFG_HIGH:-14.0}" \
    --seed "$((SEED + 30))" \
    "${COMMON_DIFFUSION[@]}"

  python -m density_audit.prepare_modern_sources diffusion \
    --name ldm_mid_steps \
    --output-dir "$OUT_ROOT/ldm_mid_steps" \
    --steps "${LDM_STEPS_MID:-50}" \
    --cfg "${LDM_CFG_MID:-7.5}" \
    --seed "$((SEED + 40))" \
    "${COMMON_DIFFUSION[@]}"

  python -m density_audit.prepare_modern_sources diffusion \
    --name stablerep \
    --output-dir "$OUT_ROOT/stablerep" \
    --steps "${STABLEREP_STEPS:-50}" \
    --cfg "${STABLEREP_CFG:-7.5}" \
    --seed "$((SEED + 50))" \
    "${COMMON_DIFFUSION[@]}"

  python -m density_audit.prepare_modern_sources diffusion \
    --name stablerep_high \
    --output-dir "$OUT_ROOT/stablerep_high" \
    --steps "${STABLEREP_HIGH_STEPS:-50}" \
    --cfg "${STABLEREP_HIGH_CFG:-14.0}" \
    --seed "$((SEED + 60))" \
    "${COMMON_DIFFUSION[@]}"
fi

if [[ "${RUN_STYLEGAN3:-0}" == "1" ]]; then
  python -m density_audit.prepare_modern_sources stylegan3 \
    --output-dir "$OUT_ROOT/stylegan3" \
    --stylegan3-repo "${STYLEGAN3_REPO:-external/stylegan3}" \
    --network "${STYLEGAN3_NETWORK:-https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl}" \
    --n "$N" \
    --truncation "${STYLEGAN3_TRUNCATION:-1.0}" \
    --seed "$SEED" \
    ${STYLEGAN3_DISABLE_CUSTOM_OPS:+--disable-custom-ops}
fi

if [[ "${RUN_UNIFORM:-1}" == "1" ]]; then
  COMPONENTS=()
  for name in fractaldb biggan stylegan3 diffusion_mid_cfg stablerep perlin_noise gaussian_noise; do
    if [[ -d "$OUT_ROOT/$name/train" ]]; then
      COMPONENTS+=("$name=$OUT_ROOT/$name/train")
    fi
  done
  if [[ "${#COMPONENTS[@]}" -ge 2 ]]; then
    python -m density_audit.prepare_modern_sources uniform-mixture \
      --output-dir "$OUT_ROOT/uniform_mixture" \
      --n "$N" \
      --seed "$SEED" \
      --components "${COMPONENTS[@]}"
  else
    echo "Skipping uniform mixture; need at least two prepared components."
  fi
fi

python -m density_audit.prepare_modern_sources manifest \
  --root "$OUT_ROOT" \
  --output "${MANIFEST_OUT:-configs/dino_sources.csv}" \
  --auto-include
