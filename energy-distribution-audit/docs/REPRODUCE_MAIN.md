# Reproducing the Main Results

Run commands from the repository root. Paths below are examples; use local
storage for large image datasets and result directories.

## 1. Data and ImageNet Reference

Set common roots and either choose a seed or let each command print a newly
generated one:

```bash
export IMAGENET_ROOT=/path/to/imagenet
export DATA_ROOT=/path/to/data
export RESULTS_ROOT=/path/to/results
export SEED="${SEED:-$(python -m density_audit.random_seed)}"
```

The ImageNet root must contain `train/<WNID>` and `val/<WNID>`. Prepare the
modern nine-source suite with:

```bash
DATA_ROOT="$DATA_ROOT" SEED="$SEED" bash scripts/generate_simple_modern_datasets.sh
DATA_ROOT="$DATA_ROOT" SEED="$SEED" bash scripts/prepare_modern_dino_sources.sh
```

Generator-backed sources require the optional packages and upstream weights
described by their providers. The paper source list is in
`configs/dino_sources.csv`.

## 2. Score Images and Compute ELD/pELD

The bundled grayscale ImageNet64 scorer is used by default:

```bash
MANIFEST=configs/dino_sources.csv \
OUTPUT_DIR="$RESULTS_ROOT/guth_dino" BATCH_SIZE=256 DEVICE=cuda \
bash scripts/score_dino_manifest_with_guth.sh
```

Each `scores_<dataset>.csv` contains per-image normalized log probability. The
ELD implementation converts it to energy and computes raw one-dimensional
Wasserstein distance and percentile-calibrated ELD:

```bash
python -m density_audit.eld_table \
  --score-dirs "$RESULTS_ROOT/guth_dino" \
  --real-dataset imagenet \
  --output-dir "$RESULTS_ROOT/eld_dino"
```

## 3. Dataset Metrics

The scoring step writes `manifest_scored.csv`, adding the generated `score_csv`
paths to the source manifest. Extract ImageNet features once, then run the
matched metric table:

```bash
python -m density_audit.extract_features \
  --scores-csv "$RESULTS_ROOT/guth_dino/scores_imagenet.csv" \
  --output "$RESULTS_ROOT/dino_table/features/imagenet_resnet50.npz" \
  --model resnet50 --device cuda --batch-size 512

MANIFEST="$RESULTS_ROOT/guth_dino/manifest_scored.csv" \
REAL_SCORES="$RESULTS_ROOT/guth_dino/scores_imagenet.csv" \
REAL_FEATURES="$RESULTS_ROOT/dino_table/features/imagenet_resnet50.npz" \
OUTPUT_DIR="$RESULTS_ROOT/dino_table" \
EXTRACT_MISSING_FEATURES=1 COMPUTE_LPIPS_VAR=1 \
bash scripts/build_dino_table_metrics.sh
```

This computes ELD/pELD, Gaussian feature-moment FID, LPIPS variation, and
improved recall from matched input sets.

## 4. DINO/ViT-S Pretraining and Evaluation

Point `configs/dino_sources.csv` at the prepared image roots. The Slurm wrappers
contain resource requests but no site-specific account or partition settings;
provide those at submission time if required.

```bash
SEED="$SEED" CONFIG=configs/dino_sources.csv \
OUTPUT_ROOT="$RESULTS_ROOT/dino_pretrain" \
sbatch --array=0-8 scripts/submit_dino_modern_array.sbatch

SEED="$SEED" CONFIG=configs/dino_sources.csv \
MODEL_ROOT="$RESULTS_ROOT/dino_pretrain" \
OUTPUT_ROOT="$RESULTS_ROOT/dino_linear" IMAGENET1K="$IMAGENET_ROOT" \
sbatch --array=0-8 scripts/submit_dino_modern_linear_eval_array.sbatch

SEED="$SEED" MANIFEST=configs/dino_sources.csv \
PRETRAIN_ROOT="$RESULTS_ROOT/dino_pretrain" \
OUTPUT_ROOT="$RESULTS_ROOT/dino_finetune" IMAGENET1K="$IMAGENET_ROOT" \
sbatch --array=0-8 scripts/submit_dino_finetune_array.sbatch
```

Evaluate fine-tuned classifiers on ImageNet-C or ImageNet-A:

```bash
SHIFT_NAME=in_c SHIFT_DATA=/path/to/imagenet-c \
MANIFEST=configs/dino_sources.csv CHECKPOINT_ROOT="$RESULTS_ROOT/dino_finetune" \
IMAGENET1K="$IMAGENET_ROOT" OUTPUT="$RESULTS_ROOT/dino_in_c.csv" \
sbatch scripts/submit_dino_shift_eval.sbatch

SHIFT_NAME=in_a SHIFT_DATA=/path/to/imagenet-a \
MANIFEST=configs/dino_sources.csv CHECKPOINT_ROOT="$RESULTS_ROOT/dino_finetune" \
IMAGENET1K="$IMAGENET_ROOT" OUTPUT="$RESULTS_ROOT/dino_in_a.csv" \
sbatch scripts/submit_dino_shift_eval.sbatch
```

The evaluator maps dataset WNIDs to full ImageNet-1K classifier indices and
strictly rejects incompatible checkpoints.

## 5. Learning-to-See Replication

The wrappers expect the pinned `external/learning_with_noise` repository:

```bash
SEED="$SEED" sbatch --array=0-13 scripts/submit_l2s_small_train_array.sbatch
SEED="$SEED" sbatch --array=0-13 scripts/submit_l2s_small_linear_eval_array.sbatch
SEED="$SEED" sbatch --array=0-13 scripts/submit_l2s_small_finetune_array.sbatch
sbatch --array=0-13 scripts/submit_l2s_small_shift_eval_array.sbatch

bash scripts/collect_l2s_downstream.sh
SCORE_DIRS="$RESULTS_ROOT/l2s_scores" \
FEATURE_DIR="$RESULTS_ROOT/l2s_features" OUTPUT_DIR="$RESULTS_ROOT/l2s_table" \
bash scripts/build_l2s_table_metrics.sh
```

ImageNet-C/A evaluation is restricted to the ImageNet-100 label subset used by
the original protocol.

## 6. Figures

Figure scripts consume the metric CSVs created above and write PNG/PDF outputs:

```bash
python scripts/plot_figure1_energy_prism.py --help
python scripts/plot_fig3_family_energy_ridges.py --help
python scripts/plot_figure4_dino_eld_residual.py --help
python scripts/plot_eld_robustness.py --help
python scripts/make_synthetic_sample_grids.py --help
```
