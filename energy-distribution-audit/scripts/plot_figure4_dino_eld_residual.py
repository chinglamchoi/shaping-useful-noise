from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from adjustText import adjust_text
except Exception:  # pragma: no cover - optional plotting dependency
    adjust_text = None


DATA = [
    # regime, dataset, label, eld, fid, lpips, recall, lin, ft, in_c, in_a
    ("Noise-like", "fractaldb", "FractalDB", 0.426, 1254, 0.260, 0.018, 23.91, 56.66, 18.70, 10.30),
    ("Noise-like", "perlin_noise", "Perlin", 0.427, 1922, 0.208, 0.000, 21.53, 55.38, 16.79, 9.11),
    ("Noise-like", "gaussian_noise", "Gaussian", 0.500, 2145, 0.237, 0.000, 20.26, 52.26, 15.53, 5.01),
    ("Over-concentrated", "high_cfg_diffusion", "high CFG", 0.138, 481, 0.342, 0.020, 28.18, 59.34, 21.41, 10.84),
    ("Broad / shifted", "stylegan3", "StyleGAN3", 0.309, 1148, 0.305, 0.001, 24.71, 57.49, 18.34, 10.59),
    ("Broad / shifted", "biggan", "BigGAN", 0.112, 566, 0.303, 0.101, 28.36, 58.97, 18.10, 10.35),
    ("Broad / shifted", "diffusion_mid_cfg", "mid CFG", 0.088, 459, 0.338, 0.030, 29.17, 59.57, 21.44, 12.35),
    ("Broad / shifted", "diffusion_low_cfg", "low CFG", 0.153, 1129, 0.297, 0.022, 28.67, 59.60, 21.38, 11.99),
    ("Curated mixtures", "uniform_mixture", "mixture", 0.112, 616, 0.294, 0.023, 27.45, 57.61, 20.80, 11.07),
]

COLORS = {
    "Noise-like": "#6E6E6E",
    "Over-concentrated": "#C85A3A",
    "Broad / shifted": "#2C7BB6",
    "Curated mixtures": "#2A9D55",
}
MARKERS = {
    "Noise-like": "o",
    "Over-concentrated": "s",
    "Broad / shifted": "^",
    "Curated mixtures": "D",
}


def make_frame() -> pd.DataFrame:
    columns = [
        "regime",
        "dataset",
        "label",
        "eld",
        "fid",
        "lpips",
        "recall",
        "in1k_lin",
        "in1k_ft",
        "in_c",
        "in_a",
    ]
    return pd.DataFrame(DATA, columns=columns)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs = x[order]
    slope, intercept = np.polyfit(x, y, 1)
    return xs, slope * xs + intercept


def residualize(y: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    z = (covariate - covariate.mean()) / covariate.std(ddof=0)
    X = np.c_[np.ones_like(z), z]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    offsets_override: dict[str, tuple[int, int]] | None = None,
) -> None:
    # Hand-tuned offsets for a 9-point plot; avoids adding a heavy dependency.
    offsets = {
        "FractalDB": (-18, 8),
        "Perlin": (-8, -14),
        "Gaussian": (-35, -2),
        "high CFG": (-18, 8),
        "StyleGAN3": (-22, -14),
        "BigGAN": (-18, -16),
        "mid CFG": (7, 8),
        "low CFG": (8, 2),
        "mixture": (6, -14),
    }
    if offsets_override:
        offsets.update(offsets_override)
    texts = []
    for _, row in df.iterrows():
        dx, dy = offsets.get(row["label"], (5, 5))
        ann = ax.annotate(
            row["label"],
            (row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.0,
            color="#242424",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.72},
            arrowprops={"arrowstyle": "-", "color": "#B0B0B0", "lw": 0.55, "alpha": 0.8},
        )
        texts.append(ann)
    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "y", "texts": "xy"},
            arrowprops={"arrowstyle": "-", "color": "#B0B0B0", "lw": 0.55, "alpha": 0.75},
            expand_text=(1.08, 1.12),
            expand_points=(1.08, 1.12),
            force_text=(0.35, 0.45),
            force_points=(0.15, 0.25),
        )


def add_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    offsets_override: dict[str, tuple[int, int]] | None = None,
) -> None:
    for regime, sub in df.groupby("regime", sort=False):
        ax.scatter(
            sub["eld"],
            sub[y_col],
            s=78,
            marker=MARKERS[regime],
            color=COLORS[regime],
            edgecolor="white",
            linewidth=0.8,
            label=regime,
            zorder=3,
        )
    xs, ys = fit_line(df["eld"].to_numpy(float), df[y_col].to_numpy(float))
    ax.plot(xs, ys, color="#222222", lw=1.15, alpha=0.8, zorder=2)
    rho, p = stats.spearmanr(df["eld"], df[y_col])
    ax.text(
        0.03,
        0.05,
        rf"Spearman $\rho={rho:.2f}$",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D8D8D8", "alpha": 0.9},
    )
    annotate_points(ax, df, "eld", y_col, offsets_override=offsets_override)
    ax.set_title(title, loc="left", fontsize=15.5, weight="bold")
    ax.set_xlabel("Eld ↓")
    ax.set_ylabel(ylabel)
    ax.grid(color="#E4E4E4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_residual_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df.copy()
    df["lin_resid_fid"] = residualize(df["in1k_lin"].to_numpy(float), df["fid"].to_numpy(float))
    for regime, sub in df.groupby("regime", sort=False):
        ax.scatter(
            sub["eld"],
            sub["lin_resid_fid"],
            s=58,
            marker=MARKERS[regime],
            color=COLORS[regime],
            edgecolor="white",
            linewidth=0.8,
            label=regime,
            zorder=3,
        )
    xs, ys = fit_line(df["eld"].to_numpy(float), df["lin_resid_fid"].to_numpy(float))
    ax.plot(xs, ys, color="#222222", lw=1.15, alpha=0.8, zorder=2)
    ax.axhline(0, color="#555555", lw=0.9, linestyle=(0, (4, 3)), alpha=0.8)
    rho, _ = stats.spearmanr(df["eld"], df["lin_resid_fid"])
    ax.text(
        0.03,
        0.05,
        rf"Residual $\rho={rho:.2f}$",
        transform=ax.transAxes,
        fontsize=9,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D8D8D8", "alpha": 0.9},
    )
    annotate_points(ax, df, "eld", "lin_resid_fid")
    ax.set_title("(d) variation not explained by FID", loc="left", fontsize=11.5, weight="bold")
    ax.set_xlabel("Eld ↓")
    ax.set_ylabel("IN-1K linear residual")
    ax.grid(color="#E4E4E4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    df = make_frame()
    out = Path("results/figures/fig4_dino_eld_residual")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25), constrained_layout=True)
    fig.suptitle("Eld Predicts DINO/ViT-S Transfer and Robustness", x=0.02, ha="left", fontsize=15, weight="bold")

    add_scatter(
        axes[0],
        df,
        "in1k_lin",
        "IN-1K linear (%)",
        "(a) frozen representation",
        offsets_override={
            "mid CFG": (8, 10),
            "low CFG": (12, 2),
            "high CFG": (-42, 9),
            "BigGAN": (-52, -6),
            "mixture": (8, -26),
            "FractalDB": (-28, 7),
            "Perlin": (-18, -16),
            "Gaussian": (-42, 4),
        },
    )
    add_scatter(
        axes[1],
        df,
        "in_c",
        "ImageNet-C (%)",
        "(b) corruption shift",
        offsets_override={
            "mid CFG": (-14, 12),
            "low CFG": (12, 5),
            "high CFG": (-42, 17),
            "mixture": (6, -24),
            "BigGAN": (-50, -6),
            "StyleGAN3": (-32, -13),
            "FractalDB": (-26, 12),
            "Perlin": (-10, -18),
            "Gaussian": (-42, -1),
        },
    )
    add_scatter(
        axes[2],
        df,
        "in_a",
        "ImageNet-A (%)",
        "(c) natural shift",
        offsets_override={
            "mid CFG": (10, 10),
            "low CFG": (10, 3),
            "high CFG": (-8, 8),
            "mixture": (4, -15),
            "BigGAN": (-28, -18),
            "StyleGAN3": (-22, -15),
            "FractalDB": (-16, 7),
            "Perlin": (-10, -18),
            "Gaussian": (-44, -2),
        },
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16))

    for ax in axes:
        ax.set_xlim(0.055, 0.525)
    axes[1].set_ylim(15.25, 21.85)
    axes[2].set_ylim(4.65, 12.75)

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out.with_suffix('.pdf')}")
    print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {out.with_suffix('.csv')}")
    for col in ["in1k_lin", "in_c", "in_a"]:
        rho, p = stats.spearmanr(df["eld"], df[col])
        print(f"{col}: Spearman rho={rho:.3f}, p={p:.3g}")
    res = residualize(df["in1k_lin"].to_numpy(float), df["fid"].to_numpy(float))
    rho, p = stats.spearmanr(df["eld"], res)
    print(f"in1k_lin residual after FID: Spearman rho={rho:.3f}, p={p:.3g}")


if __name__ == "__main__":
    main()
