from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ENERGY_BANDS = [0, 5, 25, 75, 95, 100]


@dataclass(frozen=True)
class ScoreTable:
    dataset: str
    source: Path
    scores: pd.DataFrame


def logp_to_energy(logp_db_dim: np.ndarray) -> np.ndarray:
    """Convert log probability in dB/dim to energy in dB/dim."""
    return -np.asarray(logp_db_dim, dtype=np.float64)


def empirical_percentile_ranks(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return mid-rank empirical percentiles in [0, 1].

    Mid-ranks make the finite-sample real-vs-real pELD baseline zero when the
    same reference sample is passed on both sides, matching the manuscript's
    population statement that real energy ranks are uniform.
    """
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2 * reference.size)


def uniform_w1_from_ranks(ranks: np.ndarray) -> float:
    """W1(Unif(0,1), empirical ranks), the manuscript percentile ELD."""
    ranks = np.asarray(ranks, dtype=np.float64)
    uniform_quantiles = (np.arange(ranks.size, dtype=np.float64) + 0.5) / ranks.size
    return float(np.mean(np.abs(np.sort(ranks) - uniform_quantiles)))


def _score_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right):
        return False
    left_values = left["logp_db_dim"].to_numpy(np.float64)
    right_values = right["logp_db_dim"].to_numpy(np.float64)
    return bool(np.allclose(np.sort(left_values), np.sort(right_values), rtol=0, atol=1e-8))


def load_score_tables(
    score_dirs: Iterable[Path],
    allow_duplicate_identical: bool = True,
    keep_first_duplicate: set[str] | None = None,
) -> dict[str, ScoreTable]:
    """Load scores_<dataset>.csv files from one or more histogram output dirs."""
    tables: dict[str, ScoreTable] = {}
    for score_dir in score_dirs:
        score_dir = score_dir.expanduser()
        for csv_path in sorted(score_dir.glob("scores_*.csv")):
            df = pd.read_csv(csv_path)
            required = {"dataset", "path", "logp_db_dim"}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
            names = sorted(df["dataset"].unique())
            if len(names) != 1:
                raise ValueError(f"{csv_path} contains multiple datasets: {names}")
            dataset = names[0]
            if dataset in tables:
                if keep_first_duplicate is not None and dataset in keep_first_duplicate:
                    continue
                if allow_duplicate_identical and _score_values_equal(tables[dataset].scores, df):
                    continue
                raise ValueError(
                    f"Duplicate non-identical dataset {dataset!r}: {tables[dataset].source} and {csv_path}. "
                    "Use distinct dataset aliases or analyze one model setting at a time."
                )
            tables[dataset] = ScoreTable(dataset=dataset, source=csv_path, scores=df)
    return tables


def compute_eld_rows(tables: dict[str, ScoreTable], real_dataset: str) -> list[dict[str, float | str]]:
    from scipy import stats

    if real_dataset not in tables:
        raise ValueError(f"Real dataset {real_dataset!r} not found in {sorted(tables)}")

    real_logp = tables[real_dataset].scores["logp_db_dim"].to_numpy(np.float64)
    real_energy = logp_to_energy(real_logp)
    real_mu = float(np.mean(real_energy))
    real_sigma = float(np.std(real_energy))

    rows: list[dict[str, float | str]] = []
    for dataset, table in sorted(tables.items()):
        logp = table.scores["logp_db_dim"].to_numpy(np.float64)
        energy = logp_to_energy(logp)
        ranks = empirical_percentile_ranks(real_energy, energy)

        row: dict[str, float | str] = {
            "dataset": dataset,
            "source_scores": str(table.source),
            "n": int(energy.size),
            "mean_energy_db_dim": float(np.mean(energy)),
            "std_energy_db_dim": float(np.std(energy)),
            "skew_energy_db_dim": float(stats.skew(energy, bias=False)) if energy.size > 2 else float("nan"),
            "mean_logp_db_dim": float(np.mean(logp)),
            "std_logp_db_dim": float(np.std(logp)),
            "energy_shift_db_dim": float(np.mean(energy) - real_mu),
            "energy_spread_ratio": float(np.std(energy) / real_sigma) if real_sigma > 0 else float("nan"),
            "raw_eld_db_dim": float(stats.wasserstein_distance(real_energy, energy)),
            "peld": uniform_w1_from_ranks(ranks),
            "real_energy_percentile_mean": float(np.mean(ranks)),
            "real_energy_percentile_std": float(np.std(ranks)),
        }
        for lo, hi in zip(ENERGY_BANDS[:-1], ENERGY_BANDS[1:]):
            if hi == 100:
                mask = (ranks >= lo / 100) & (ranks <= hi / 100)
            else:
                mask = (ranks >= lo / 100) & (ranks < hi / 100)
            row[f"real_energy_band_{lo}_{hi}_pct"] = float(np.mean(mask))
        rows.append(row)
    return rows


def regime_label(row: pd.Series, shift_threshold: float = 0.35, spread_low: float = 0.7, spread_high: float = 1.3) -> str:
    """Heuristic labels matching the manuscript's energy-regime language."""
    shift = row["energy_shift_db_dim"]
    spread = row["energy_spread_ratio"]
    if row["dataset"] == "imagenet":
        return "real-reference"
    if shift > shift_threshold and spread < spread_low:
        return "low-likelihood compressed"
    if shift > shift_threshold:
        return "noise-like / high-energy shifted"
    if shift < -shift_threshold and spread < spread_low:
        return "over-concentrated high-likelihood"
    if abs(shift) <= shift_threshold and spread < spread_low:
        return "over-concentrated"
    if abs(shift) > shift_threshold and spread_low <= spread <= spread_high:
        return "broad-but-shifted"
    if spread > spread_high:
        return "over-broad"
    return "near-real energy shape"
