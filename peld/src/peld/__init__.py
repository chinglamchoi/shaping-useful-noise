"""Raw and percentile energy-level distribution distances."""

from ._core import (
    ELDResult,
    PELD,
    ScoreKind,
    peld,
    percentile_eld,
    percentile_ranks,
    percentile_ranks_from_sorted,
    raw_eld,
    to_energy,
    uniform_wasserstein,
)
from .scoring import collect_scores, dataset_eld

__all__ = [
    "ELDResult",
    "PELD",
    "ScoreKind",
    "collect_scores",
    "dataset_eld",
    "peld",
    "percentile_eld",
    "percentile_ranks",
    "percentile_ranks_from_sorted",
    "raw_eld",
    "to_energy",
    "uniform_wasserstein",
]

__version__ = "0.1.0"
