# Checkpoints

## Included

`guth/` contains the grayscale ImageNet64 density scorer used for the main
energy audit and its normalization energies. The accompanying upstream license
is in `../third_party/DualScoreMatching.LICENSE`; hashes are recorded in
`SHA256SUMS`.

## DINO/ViT-S

The paper-specific DINO checkpoints are outputs of training on each synthetic
source and are not redistributed. Reproduce them from the configured datasets
using Section 4 of [`../docs/REPRODUCE_MAIN.md`](../docs/REPRODUCE_MAIN.md).

Download the DINO implementation during installation:

```bash
git clone https://github.com/facebookresearch/dino external/dino
```

Then launch pretraining, linear evaluation, fine-tuning, and ImageNet-C/A
evaluation with the supplied Slurm wrappers. No pretrained DINO checkpoint is
required to begin the paper's synthetic-source pretraining runs.

The generated checkpoints follow this layout:

```text
<dino_pretrain_root>/<dataset>/checkpoint.pth
<dino_finetune_root>/<dataset>/finetuned_classifier.pth
```

The paths are passed between stages through `MODEL_ROOT`, `PRETRAIN_ROOT`, and
`CHECKPOINT_ROOT`, as shown in the reproduction guide.

## Learning-to-See and Optional Scorers

Learning-to-See encoders and classifiers are regenerated with the supplied
wrappers and pinned upstream implementation. DenseFlow and improved-diffusion
weights are third-party artifacts and should be downloaded from their original
projects rather than redistributed here.

## User-Supplied Checkpoints

Evaluation commands can also consume compatible checkpoints from previous
runs. Before sharing such files publicly, remove absolute paths, credentials,
scheduler metadata, and other private state. The included helper produces a
sanitized copy:

```bash
python scripts/sanitize_checkpoint.py \
  --input /path/to/checkpoint.pth \
  --output /path/to/sanitized_checkpoint.pth
python scripts/audit_release.py .
```
