# pELD

Fast raw and percentile energy-level distribution distances for evaluating how
a candidate dataset occupies the likelihood landscape of a reference dataset.

```bash
pip install peld
```

## Quick Start

Score the reference and candidate images with the same normalized density
model, obtaining one energy per image. Then:

```python
import peld

metric = peld.PELD(reference_energies)
distance = metric(candidate_energies)  # percentile ELD; lower is better

result = metric.both(candidate_energies)
print(result.percentile, result.raw)
```

For log probabilities, pELD converts `log p(x)` to `E(x) = -log p(x)`:

```python
metric = peld.PELD(reference_logp, score_kind="logp")
distance = metric(candidate_logp)
```

The functional API is convenient for one comparison:

```python
peld.percentile_eld(reference_energies, candidate_energies)
peld.raw_eld(reference_energies, candidate_energies)
```

The fitted object sorts the reference once, making repeated comparisons fast:

```python
scores = metric.score_many({
    "generator_a": energies_a,
    "generator_b": energies_b,
})
metric.save("imagenet_reference_peld.npz")
metric = peld.PELD.load("imagenet_reference_peld.npz")
```

NumPy arrays, pandas values, and PyTorch tensors are accepted. PyTorch is not a
dependency; tensor conversion uses `detach().cpu().numpy()` when available.

## From Images to pELD

pELD is density-model-relative. It deliberately does not prescribe a single
image scorer. The reference and candidate images must be evaluated by the same
normalized density model and preprocessing pipeline.

For a PyTorch-style loader and callable scorer:

```python
from peld import PELD, collect_scores

reference_energies = collect_scores(energy_model, reference_loader)
candidate_energies = collect_scores(energy_model, candidate_loader)
distance = PELD(reference_energies)(candidate_energies)
```

Or evaluate both loaders directly:

```python
result = peld.dataset_eld(energy_model, reference_loader, candidate_loader)
print(result.percentile)
```

The scorer must return one scalar per input image. `collect_scores` accepts
plain batches, `(images, labels)` batches, or dictionaries containing `images`,
`image`, `pixel_values`, or `input`.

## Command Line

```bash
peld reference.csv candidate.csv \
  --reference-column energy \
  --candidate-column energy \
  --metric both --json
```

NPY, NPZ, CSV, TSV, and whitespace-delimited text are supported.

## Definitions

Given reference data `P`, candidate data `Q`, and a normalized reference
density model with energy `E(x) = -log p(x)`:

- **Raw ELD** is the one-dimensional Wasserstein-1 distance between the energy
  distributions of `P` and `Q`. Its units and scale depend on the scorer.
- **Percentile ELD (pELD)** maps candidate energies through the empirical CDF
  of the reference energies, then computes their Wasserstein-1 distance to the
  uniform distribution on `[0, 1]`. Its range is `[0, 0.5]`.

pELD is invariant to strictly increasing transformations of energy and is the
recommended default for comparisons across datasets under one scorer. Raw ELD
is useful when the energy scale itself is scientifically meaningful.

For a reference of size `n` and candidate of size `m`, fitting costs
`O(n log n)`. Each comparison costs `O(m log n + m log m)` and stores `O(n+m)`
values. The package uses exact empirical calculations and has no SciPy
dependency.

## Interpretation

- Lower pELD means the candidate better covers the reference likelihood levels.
- A low value does not guarantee semantic coverage, label balance, or visual
  quality; report complementary metrics when those properties matter.
- The result is relative to the chosen density model and its reference data.
- Use a held-out reference sample when estimating real-to-real baselines.
- Midranks are used for tied energies. Large numbers of exact ties indicate a
  quantized scorer and should be reported.

## Citation

If you use pELD in your work, please cite:

```bibtex
@inproceedings{choi2026shaping,
  title={Shaping Useful Noise: Energy Distributions Predict Visual Pretraining Quality},
  author={Choi, Ching Lam and Torralba, Antonio and Isola, Phillip and Jegelka, Stefanie},
  booktitle={Advances in Neural Information Processing Systems},
  url={https://chinglamchoi.github.io/cchoi/docs/noiseAudits.pdf},
  year={2026}
}
```

## Development

```bash
python -m pip install -e '.[test,publish]'
pytest
python -m build
python -m twine check dist/*
```
