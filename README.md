# Shaping Useful Noise

Code and metric implementations for the NeurIPS 2026 paper
*Shaping Useful Noise: Energy Distributions Predict Visual Pretraining Quality*.

## Repository Contents

| Directory | Contents |
|---|---|
| [`peld/`](peld/) | Lightweight NumPy package for raw energy-level distribution distance and percentile ELD (pELD), with Python and command-line APIs. |
| [`energy-distribution-audit/`](energy-distribution-audit/) | Research code, configurations, scripts, checkpoints, and documentation for the main paper and extended review experiments. |

## Use pELD

Install the local package:

```bash
python -m pip install ./peld
```

After its PyPI release, the equivalent command is:

```bash
pip install peld
```

Score reference and candidate images with the same normalized density model,
then compare their scalar energies:

```python
import peld

metric = peld.PELD(reference_energies)
percentile_distance = metric(candidate_energies)
both_distances = metric.both(candidate_energies)
```

See the [`peld` README](peld/README.md) for image-scoring adapters, the
functional API, file formats, and interpretation guidance.

## Reproduce the Experiments

```bash
cd energy-distribution-audit
python -m pip install -e '.[metrics,test]'
```

The research release provides separate guides for:

- [Main paper experiments](energy-distribution-audit/docs/REPRODUCE_MAIN.md)
- [Extended and rebuttal experiments](energy-distribution-audit/docs/REBUTTAL_EXPERIMENTS.md)
- [Model and checkpoint setup](energy-distribution-audit/checkpoints/README.md)

Datasets with redistribution restrictions, including ImageNet, are not bundled.
The included scripts document the expected layouts and expose seed arguments;
omitting a seed generates and logs one rather than silently fixing it.

## Metric Scope

Raw ELD compares one-dimensional energy distributions using the
Wasserstein-1 distance. pELD first maps energies through the empirical CDF of a
reference dataset, producing a scale-free distance in `[0, 0.5]`. Both metrics
are reference-relative and require the reference and candidate datasets to be
scored by the same normalized density model and preprocessing pipeline.

## Citation

```bibtex
@inproceedings{choi2026shaping,
  title={Shaping Useful Noise: Energy Distributions Predict Visual Pretraining Quality},
  author={Choi, Ching Lam and Torralba, Antonio and Isola, Phillip and Jegelka, Stefanie},
  booktitle={Advances in Neural Information Processing Systems},
  url={https://chinglamchoi.github.io/cchoi/docs/noiseAudits.pdf},
  year={2026}
}
```

## License

Add the intended project license before publication. Bundled third-party code
and checkpoints remain governed by their upstream licenses and terms.
