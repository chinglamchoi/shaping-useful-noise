from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


ScoreKind = Literal["energy", "logp"]


def _as_numpy(values: Any) -> NDArray[np.float64]:
    """Convert NumPy-, pandas-, or PyTorch-like values without requiring torch."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1 and int(np.prod(array.shape[1:])) == 1:
        array = array.reshape(array.shape[0])
    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError("Score arrays must not be empty.")
    if not np.isfinite(array).all():
        count = int((~np.isfinite(array)).sum())
        raise ValueError(f"Scores contain {count} non-finite value(s).")
    return array


def to_energy(scores: ArrayLike, score_kind: ScoreKind = "energy") -> NDArray[np.float64]:
    """Return energies, converting log probabilities with E(x) = -log p(x)."""
    values = _as_numpy(scores)
    if score_kind == "energy":
        return values
    if score_kind == "logp":
        return -values
    raise ValueError("score_kind must be 'energy' or 'logp'.")


def _validate_sorted_reference(reference: ArrayLike) -> NDArray[np.float64]:
    values = _as_numpy(reference)
    if values.size > 1 and np.any(values[:-1] > values[1:]):
        raise ValueError("sorted_reference must be sorted in ascending order.")
    return values


def percentile_ranks(
    reference_scores: ArrayLike,
    scores: ArrayLike,
    *,
    score_kind: ScoreKind = "energy",
) -> NDArray[np.float64]:
    """Map scores through the empirical energy CDF of the reference sample.

    Ties receive their midrank. Values below or above the observed reference
    range map to 0 or 1, respectively.
    """
    reference = np.sort(to_energy(reference_scores, score_kind))
    values = to_energy(scores, score_kind)
    return percentile_ranks_from_sorted(reference, values)


def percentile_ranks_from_sorted(
    sorted_reference: ArrayLike,
    energy_scores: ArrayLike,
) -> NDArray[np.float64]:
    """Fast rank transform when the ascending reference energies are cached."""
    reference = _validate_sorted_reference(sorted_reference)
    values = _as_numpy(energy_scores)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2.0 * reference.size)


def uniform_wasserstein(ranks: ArrayLike) -> float:
    """Exact empirical W1 distance between percentile ranks and Uniform(0, 1)."""
    values = np.sort(_as_numpy(ranks))
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("Percentile ranks must lie in [0, 1].")
    target = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return float(np.mean(np.abs(values - target)))


def percentile_eld(
    reference_scores: ArrayLike,
    candidate_scores: ArrayLike,
    *,
    score_kind: ScoreKind = "energy",
) -> float:
    """Compute percentile ELD (pELD); lower is better and the range is [0, 0.5]."""
    reference = np.sort(to_energy(reference_scores, score_kind))
    candidate = to_energy(candidate_scores, score_kind)
    ranks = percentile_ranks_from_sorted(reference, candidate)
    return uniform_wasserstein(ranks)


def peld(
    reference_scores: ArrayLike,
    candidate_scores: ArrayLike,
    *,
    score_kind: ScoreKind = "energy",
) -> float:
    """Alias for :func:`percentile_eld`."""
    return percentile_eld(reference_scores, candidate_scores, score_kind=score_kind)


def raw_eld(
    reference_scores: ArrayLike,
    candidate_scores: ArrayLike,
    *,
    score_kind: ScoreKind = "energy",
) -> float:
    """Compute exact one-dimensional empirical W1 in the scorer's energy units."""
    reference = np.sort(to_energy(reference_scores, score_kind))
    candidate = np.sort(to_energy(candidate_scores, score_kind))
    support = np.sort(np.concatenate([reference, candidate]))
    if support.size < 2:
        return 0.0
    widths = np.diff(support)
    ref_cdf = np.searchsorted(reference, support[:-1], side="right") / reference.size
    cand_cdf = np.searchsorted(candidate, support[:-1], side="right") / candidate.size
    return float(np.sum(np.abs(ref_cdf - cand_cdf) * widths))


@dataclass(frozen=True)
class ELDResult:
    """Raw and percentile ELD computed from the same two score samples."""

    percentile: float
    raw: float
    n_reference: int
    n_candidate: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "percentile_eld": self.percentile,
            "raw_eld": self.raw,
            "n_reference": self.n_reference,
            "n_candidate": self.n_candidate,
        }


class PELD:
    """Reusable pELD metric with a cached empirical reference distribution.

    Parameters
    ----------
    reference_scores:
        Per-image energies or log probabilities from the reference dataset.
    score_kind:
        ``"energy"`` for E(x), or ``"logp"`` for log p(x). Candidate scores
        use the same convention unless overridden at call time.
    """

    def __init__(self, reference_scores: ArrayLike, *, score_kind: ScoreKind = "energy") -> None:
        self.score_kind: ScoreKind = score_kind
        self._reference = np.sort(to_energy(reference_scores, score_kind))

    @property
    def n_reference(self) -> int:
        return int(self._reference.size)

    @property
    def sorted_reference(self) -> NDArray[np.float64]:
        view = self._reference.view()
        view.flags.writeable = False
        return view

    def transform(
        self,
        candidate_scores: ArrayLike,
        *,
        score_kind: ScoreKind | None = None,
    ) -> NDArray[np.float64]:
        """Return candidate energy percentiles under the fitted reference CDF."""
        kind = self.score_kind if score_kind is None else score_kind
        candidate = to_energy(candidate_scores, kind)
        return percentile_ranks_from_sorted(self._reference, candidate)

    def __call__(
        self,
        candidate_scores: ArrayLike,
        *,
        score_kind: ScoreKind | None = None,
    ) -> float:
        """Return pELD for one candidate dataset."""
        return uniform_wasserstein(self.transform(candidate_scores, score_kind=score_kind))

    def raw(
        self,
        candidate_scores: ArrayLike,
        *,
        score_kind: ScoreKind | None = None,
    ) -> float:
        """Return raw ELD in the underlying energy units."""
        kind = self.score_kind if score_kind is None else score_kind
        return raw_eld(self._reference, to_energy(candidate_scores, kind), score_kind="energy")

    def both(
        self,
        candidate_scores: ArrayLike,
        *,
        score_kind: ScoreKind | None = None,
    ) -> ELDResult:
        """Return pELD and raw ELD from the same candidate scores."""
        kind = self.score_kind if score_kind is None else score_kind
        candidate = to_energy(candidate_scores, kind)
        ranks = percentile_ranks_from_sorted(self._reference, candidate)
        return ELDResult(
            percentile=uniform_wasserstein(ranks),
            raw=raw_eld(self._reference, candidate, score_kind="energy"),
            n_reference=self.n_reference,
            n_candidate=int(candidate.size),
        )

    def score_many(
        self,
        candidates: Mapping[str, ArrayLike],
        *,
        score_kind: ScoreKind | None = None,
    ) -> dict[str, float]:
        """Compute pELD for several named candidate datasets."""
        return {name: self(values, score_kind=score_kind) for name, values in candidates.items()}

    def save(self, path: str | Path) -> Path:
        """Save the fitted empirical reference CDF as a portable NPZ file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                sorted_reference=self._reference,
                score_kind=np.asarray(self.score_kind),
                format_version=np.asarray(1, dtype=np.int64),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PELD":
        """Load a metric previously written with :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            version = int(archive["format_version"])
            if version != 1:
                raise ValueError(f"Unsupported PELD format version: {version}")
            reference = _validate_sorted_reference(archive["sorted_reference"])
            score_kind = str(archive["score_kind"])
            if score_kind not in {"energy", "logp"}:
                raise ValueError(f"Unsupported score kind: {score_kind}")
        metric = cls.__new__(cls)
        metric.score_kind = score_kind
        metric._reference = reference.copy()
        return metric
