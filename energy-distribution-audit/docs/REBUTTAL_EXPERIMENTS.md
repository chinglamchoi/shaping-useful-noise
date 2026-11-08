# Extended and Rebuttal Experiments

These analyses reuse the scored images and downstream table from the main
pipeline. They are deliberately separated from the minimal paper path.

## 1. Cross-Scorer pELD

Construct a common source table, then score the same sampled images with each
density estimator. The lightweight DCT-PCA-GMM scorer is available directly:

```bash
python -m density_audit.make_score_source_table \
  --score-root results/guth_dino \
  --output results/cross_scorer/source_scores.csv --absolute

python -m density_audit.surrogate_density_scores \
  --mode dct_gmm --scorer-name dct_pca_gmm \
  --real-scores results/guth_dino/scores_imagenet.csv \
  --table results/cross_scorer/source_scores.csv \
  --output results/cross_scorer/scores_dct_pca_gmm.csv \
  --limit-real 5000 --limit-synthetic 2000 \
  --keep-dct 1024 --pca-dim 96 --gmm-components 16
```

For DenseFlow, prepare fixed 64x64 inputs, score them with a downloaded
ImageNet64 model, and convert the per-image BPD output:

```bash
python -m density_audit.prepare_denseflow_scoring --help
python scripts/score_denseflow_manifest.py --help
python -m density_audit.convert_denseflow_scores --help
```

For the improved-diffusion VLB scorer, obtain the upstream ImageNet64
checkpoint and run `python -m density_audit.score_improved_diffusion_vlb
--help`.

Combine score files using a separate empirical ImageNet reference CDF for each
scorer. Raw likelihood scales are never compared across model families:

```bash
python -m density_audit.cross_scorer_eld \
  --score guth=results/cross_scorer/scores_guth.csv:logp_db_dim:logp \
  --score dct_pca_gmm=results/cross_scorer/scores_dct_pca_gmm.csv:raw_score:energy \
  --score denseflow=results/cross_scorer/scores_denseflow.csv:raw_score:energy \
  --score diffusion_vlb=results/cross_scorer/scores_diffusion_vlb.csv:raw_score:energy \
  --bootstrap 500 --output-dir results/cross_scorer/report
```

## 2. Existing Metrics and Complementarity

Use the generated table for matched correlations, restricted subsets, partial
rank correlations, residual analysis, pairwise ordering, and one- or
two-predictor leave-one-dataset-out evaluation:

```bash
python -m density_audit.rebuttal_metric_correlations \
  --table results/dino_table/metrics.csv \
  --output-dir results/rebuttal_metrics

python -m density_audit.rebuttal_complementarity \
  --table results/dino_table/metrics.csv \
  --output-dir results/rebuttal_complementarity
```

The same programs accept `--preset l2s` or a Learning-to-See table. Because the
number of source datasets is small, multivariate results should be treated as
exploratory and limited to at most two predictors.

## 3. Real-to-Real Reference Comparisons

Prepare COCO val2017, Places365 validation, and Oxford-IIIT Pets test under one
root, then score them with the same ImageNet density model:

```bash
DATA_ROOT=/path/to/data bash scripts/prepare_real_reference_datasets.sh

IMAGENET_ROOT=/path/to/imagenet DATA_ROOT=/path/to/data \
RESULTS_ROOT=/path/to/results LIMIT=50000 BATCH_SIZE=128 DEVICE=cuda \
bash scripts/score_real_reference_datasets.sh

python -m density_audit.real_reference_eld \
  --score-dir /path/to/results/real_reference_eld_imagenet64_gray \
  --real-dataset imagenet_ref \
  --datasets imagenet_heldout coco_val places365_val oxford_pets_test \
  --output-csv results/real_reference_eld.csv \
  --output-md results/real_reference_eld.md
```

## 4. Controlled Curation

The proof of concept fixes the source pool and final sample budget. It compares
source-balanced uniform sampling, top-likelihood filtering, feature-moment
FID-guided mixture weights, and pELD quantile matching. FID guidance optimizes
nonnegative source weights against real-data Gaussian feature moments; pELD
guidance allocates individual samples to approximate the uniform real-energy
percentile distribution.

```bash
python -m density_audit.curation_poc \
  --real-scores results/guth_dino/scores_imagenet.csv \
  --score-csv biggan=results/guth_dino/scores_biggan.csv \
  --score-csv stylegan3=results/guth_dino/scores_stylegan3.csv \
  --score-csv diffusion_low_cfg=results/guth_dino/scores_diffusion_low_cfg.csv \
  --score-csv diffusion_mid_cfg=results/guth_dino/scores_diffusion_mid_cfg.csv \
  --score-csv high_cfg_diffusion=results/guth_dino/scores_high_cfg_diffusion.csv \
  --fid-guidance feature_moments \
  --real-features results/features/imagenet_resnet50.npz \
  --feature-npz biggan=results/features/biggan_resnet50.npz \
  --feature-npz stylegan3=results/features/stylegan3_resnet50.npz \
  --feature-npz diffusion_low_cfg=results/features/diffusion_low_cfg_resnet50.npz \
  --feature-npz diffusion_mid_cfg=results/features/diffusion_mid_cfg_resnet50.npz \
  --feature-npz high_cfg_diffusion=results/features/high_cfg_diffusion_resnet50.npz \
  --eld-guidance peld_quantile --dataset-size 25000 --per-source-cap 10000 \
  --output-root results/curation_poc --make-imagefolders --link-mode symlink
```

Run `python -m density_audit.curation_feature_proxy --help` for the frozen
feature diagnostic. It compares nearest-neighbor distance to ImageNet,
effective neighbor-class diversity, and feature covariance effective rank.
These diagnostics test whether energy matching avoids likelihood collapse and
retains broad natural-image support; they do not replace downstream retraining.
