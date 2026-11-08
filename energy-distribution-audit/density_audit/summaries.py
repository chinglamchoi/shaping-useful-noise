from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REAL_DENSITY_BANDS = [0, 5, 25, 75, 95, 100]


def summarize_logp(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    loc, scale = stats.gumbel_r.fit(values)
    nll = -float(np.mean(stats.gumbel_r.logpdf(values, loc=loc, scale=scale)))
    ks = stats.kstest(values, "gumbel_r", args=(loc, scale))
    return {
        "n": int(values.size),
        "mean_logp_db_dim": float(np.mean(values)),
        "var_logp_db_dim": float(np.var(values)),
        "std_logp_db_dim": float(np.std(values)),
        "median_logp_db_dim": float(np.median(values)),
        "skew_logp_db_dim": float(stats.skew(values, bias=False)) if values.size > 2 else float("nan"),
        "kurtosis_logp_db_dim": float(stats.kurtosis(values, bias=False)) if values.size > 3 else float("nan"),
        "min_logp_db_dim": float(np.min(values)),
        "max_logp_db_dim": float(np.max(values)),
        "gumbel_location": float(loc),
        "gumbel_scale": float(scale),
        "gumbel_mean_nll": nll,
        "gumbel_ks_stat": float(ks.statistic),
        "gumbel_ks_pvalue": float(ks.pvalue),
    }


def add_real_percentiles(summary: dict[str, float], values: np.ndarray, real_values: np.ndarray) -> dict[str, float]:
    ranks = stats.percentileofscore(real_values, values, kind="weak")
    for lo, hi in zip(REAL_DENSITY_BANDS[:-1], REAL_DENSITY_BANDS[1:]):
        if hi == 100:
            mask = (ranks >= lo) & (ranks <= hi)
        else:
            mask = (ranks >= lo) & (ranks < hi)
        summary[f"real_density_band_{lo}_{hi}_pct"] = float(np.mean(mask))
    summary["real_percentile_mean"] = float(np.mean(ranks))
    summary["real_percentile_std"] = float(np.std(ranks))
    return summary


def write_summary_table(rows: list[dict[str, float]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("dataset")
    df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    return df
