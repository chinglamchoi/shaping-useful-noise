from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from density_audit.eld import (
    compute_eld_rows,
    empirical_percentile_ranks,
    load_score_tables,
    logp_to_energy,
    uniform_w1_from_ranks,
)
from density_audit.random_seed import resolve_seed


DEFAULT_SCORE_DIRS = [
    Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray"),
    Path("results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray"),
    Path("results/guth_histogram_stat_plus_hf_imagenet64_50k_gray"),
    Path("results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray"),
]

LABELS = {
    "dead_squares": "dead squares",
    "dead_oriented": "dead oriented",
    "dead_mixed": "dead mixed",
    "dead_textures": "dead textures",
    "stat_spectrum": "spectrum",
    "stat_wmm": "WMM",
    "stat_spectrum_color": "spectrum+color",
    "stat_spectrum_color_wmm": "spectrum+color+WMM",
    "stylegan_random": "StyleGAN random",
    "stylegan_highfreq": "StyleGAN high-freq.",
    "stylegan_sparse": "StyleGAN sparse",
    "stylegan_oriented": "StyleGAN oriented",
    "feature_vis_random": "FV random",
    "feature_vis_dead_leaves": "FV dead leaves",
}


def read_scores(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "logp_db_dim" not in df.columns:
        raise ValueError(f"{path} missing logp_db_dim")
    return logp_to_energy(df["logp_db_dim"].to_numpy(np.float64))


def load_eld_table(score_dirs: list[Path], real_dataset: str) -> pd.DataFrame:
    tables = load_score_tables(score_dirs, keep_first_duplicate={real_dataset})
    df = pd.DataFrame(compute_eld_rows(tables, real_dataset=real_dataset))
    df = df[df["dataset"] != real_dataset].copy()
    df["label"] = df["dataset"].map(LABELS).fillna(df["dataset"])
    return df.sort_values("peld").reset_index(drop=True)


def convergence_curve(real_energy: np.ndarray, synth_energy: np.ndarray, sample_sizes: list[int], repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    max_n = min(real_energy.size, synth_energy.size)
    rows = []
    for n in sample_sizes:
        if n > max_n:
            continue
        values = []
        for _ in range(repeats):
            r = rng.choice(real_energy.size, n, replace=False)
            s = rng.choice(synth_energy.size, n, replace=False)
            ranks = empirical_percentile_ranks(real_energy[r], synth_energy[s])
            values.append(uniform_w1_from_ranks(ranks))
        rows.append(
            {
                "n": n,
                "mean": float(np.mean(values)),
                "lo": float(np.percentile(values, 5)),
                "hi": float(np.percentile(values, 95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustness checks for ELD/pELD estimates.")
    parser.add_argument("--score-dirs", type=Path, nargs="+", default=DEFAULT_SCORE_DIRS)
    parser.add_argument("--real-dataset", default="imagenet")
    parser.add_argument("--convergence-dataset", default="stat_spectrum_color_wmm")
    parser.add_argument("--sample-sizes", default="250,500,1000,2000,5000,10000,20000,50000")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/figures/fig5_eld_robustness"))
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    main_df = load_eld_table(args.score_dirs, args.real_dataset)

    tables = load_score_tables(args.score_dirs, keep_first_duplicate={args.real_dataset})
    real_energy = logp_to_energy(tables[args.real_dataset].scores["logp_db_dim"].to_numpy(np.float64))
    if args.convergence_dataset not in tables:
        raise ValueError(f"--convergence-dataset {args.convergence_dataset!r} not found")
    synth_energy = logp_to_energy(tables[args.convergence_dataset].scores["logp_db_dim"].to_numpy(np.float64))
    sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x.strip()]
    conv = convergence_curve(real_energy, synth_energy, sample_sizes, args.repeats, seed=args.seed)
    full_ranks = empirical_percentile_ranks(real_energy, synth_energy)
    full_peld = uniform_w1_from_ranks(full_ranks)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.35), dpi=180)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    rho = stats.spearmanr(main_df["peld"], main_df["raw_eld_db_dim"]).statistic
    ax.scatter(main_df["peld"], main_df["raw_eld_db_dim"], s=52, color="#C7772C", edgecolor="white", linewidth=0.7)
    ax.set_xlabel("percentile ELD")
    ax.set_ylabel("raw ELD (dB/dim)")
    ax.set_title("(a) raw vs percentile ELD", loc="left", weight="bold")
    ax.text(0.05, 0.92, f"Spearman $\\rho$ = {rho:.2f}", transform=ax.transAxes, fontsize=9.5, weight="bold")

    ax = axes[1]
    ax.fill_between(conv["n"], conv["lo"], conv["hi"], color="#5AAE61", alpha=0.22, linewidth=0)
    ax.plot(conv["n"], conv["mean"], color="#1B8A5A", lw=2.0, marker="o", ms=3.5)
    ax.axhline(full_peld, color="#222222", lw=1.0, ls="--", label="50k estimate")
    ax.set_xscale("log")
    ax.set_xlabel("samples per distribution")
    ax.set_ylabel("pELD")
    ax.set_title("(b) sample convergence", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes:
        ax.grid(True, color="#DDDDDD", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.5)

    fig.suptitle("ELD Is Stable Under Calibration and Sampling", x=0.06, ha="left", fontsize=15.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    main_df.to_csv(args.output.with_name(args.output.name + "_table.csv"), index=False)
    conv.to_csv(args.output.with_name(args.output.name + "_convergence.csv"), index=False)
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
