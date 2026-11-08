# Energy Distribution Audits for Synthetic Pretraining

This repository contains the minimal research code used to compute energy-level
distribution (ELD) diagnostics, reproduce the DINO/ViT-S and Learning-to-See
experiments, and run the additional analyses reported during review.

The release has two layers:

- **Main paper:** ImageNet-referenced Guth scoring, ELD/pELD, FID/LPIPS/recall,
  DINO/ViT-S and Learning-to-See downstream evaluation, and paper figures.
- **Extended analyses:** cross-density-scorer agreement, real-to-real reference
  comparisons, metric complementarity, and controlled curation.

The experiments use ImageNet together with procedurally generated and diffusion-based synthetic datasets. The reproduction guides provide instructions to download or sample each dataset and obtain the required upstream checkpoints. A pretrained grayscale ImageNet64 Guth scorer is included under `checkpoints/guth`. Wrappers are provided to generate DINO checkpoints from configured synthetic datasets from the upstream DINO implementation.

## Installation

Create a Python 3.10 environment and install a PyTorch build suitable for the
local CUDA runtime first. Then run:

```bash
python -m pip install -e '.[metrics,test]'
git clone https://github.com/FlorentinGuth/DualScoreMatching external/DualScoreMatching
git -C external/DualScoreMatching checkout c84182f312a2b8234d4a78bc4c15e2434cb0a14c
git clone https://github.com/facebookresearch/dino external/dino
git clone https://github.com/mbaradad/learning_with_noise external/learning_with_noise
git -C external/learning_with_noise checkout a31abe38eac2bc9071c639dd925e7bfa7bf8dde9
```

Optional modern generator dependencies are listed in
`requirements-modern.txt`.

## Seed Policy

Experiment commands accept `--seed`. When it is omitted, the release draws a
random seed from the operating system and prints `seed=<value>` before work
begins. Record that value to repeat a run exactly. Slurm wrappers similarly use
`SEED` when supplied and otherwise generate and log one.

```bash
python -m density_audit.random_seed
SEED=123456 bash scripts/prepare_modern_dino_sources.sh
```

## Reproduction Guides

- Main paper: [`docs/REPRODUCE_MAIN.md`](docs/REPRODUCE_MAIN.md)
- Extended/rebuttal experiments: [`docs/REBUTTAL_EXPERIMENTS.md`](docs/REBUTTAL_EXPERIMENTS.md)
- Models and checkpoints: [`checkpoints/README.md`](checkpoints/README.md)

## Release Audit

```bash
python scripts/audit_release.py .
python -m compileall -q density_audit scripts
pytest -q
```

The audit scans source, documentation, configuration, and checkpoint bytes for
credentials, private infrastructure paths, personal addresses, and model-output
attribution text.

## Citation

If you use this code or its results, please cite:

```bibtex
@inproceedings{choi2026shaping,
  title={Shaping Useful Noise: Energy Distributions Predict Visual Pretraining Quality},
  author={Choi, Ching Lam and Torralba, Antonio and Isola, Phillip and Jegelka, Stefanie},
  booktitle={Advances in Neural Information Processing Systems},
  url={https://chinglamchoi.github.io/cchoi/docs/noiseAudits.pdf},
  year={2026}
}
```
