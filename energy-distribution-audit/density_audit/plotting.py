from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_histograms(score_tables: dict[str, pd.DataFrame], summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_values = np.concatenate([df["logp_db_dim"].to_numpy() for df in score_tables.values()])
    xmin = min(-2.5, float(np.nanpercentile(all_values, 0.5)))
    xmax = max(50.0, float(np.nanpercentile(all_values, 99.5)))

    plt.figure(figsize=(7.5, 4.5), layout="constrained")
    for name, df in score_tables.items():
        plt.hist(
            df["logp_db_dim"],
            bins=50,
            range=(xmin, xmax),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=name,
        )
    plt.yscale("log")
    plt.xlabel(r"$\log p_{\theta}(x)$ (dB/dim)")
    plt.ylabel("Frequency")
    plt.legend(fontsize=8)
    plt.savefig(output_dir / "logp_histograms_overlay.png", dpi=220)
    plt.savefig(output_dir / "logp_histograms_overlay.pdf")
    plt.close()

    for name, df in score_tables.items():
        row = summary.loc[summary["dataset"] == name].iloc[0]
        values = df["logp_db_dim"].to_numpy()
        plt.figure(figsize=(4.0, 2.5), layout="constrained")
        plt.hist(values, bins=35, density=True, log=True, alpha=0.75, label="Data")
        x = np.linspace(xmin, xmax, 1000)
        y = stats.gumbel_r.pdf(x, loc=row["gumbel_location"], scale=row["gumbel_scale"])
        plt.plot(x, y, color="tab:red", label="Gumbel fit")
        plt.xlim(xmin, xmax)
        plt.xlabel(r"$\log p_{\theta}(x)$ (dB/dim)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=8)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        plt.savefig(output_dir / f"logp_histogram_{safe}.png", dpi=220)
        plt.savefig(output_dir / f"logp_histogram_{safe}.pdf")
        plt.close()
