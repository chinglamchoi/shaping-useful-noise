"""Figure 3: family-level energy ridgelines.

This simplified Figure 3 keeps only the energy-density view. It overlays all
small-scale synthetic datasets within their generator family, plus ImageNet as
the real reference row.

Rows:
  ImageNet
  Dead leaves
  Statistical
  StyleGAN
  Feature visualizations
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks


DEFAULT_REAL = Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_imagenet.csv")
DEFAULT_SYNTHETICS = [
    Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_squares.csv"),
    Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_oriented.csv"),
    Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_mixed.csv"),
    Path("results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_textures.csv"),
    Path("results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum.csv"),
    Path("results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_wmm.csv"),
    Path("results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color.csv"),
    Path("results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color_wmm.csv"),
    Path("results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_random.csv"),
    Path("results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_highfreq.csv"),
    Path("results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_sparse.csv"),
    Path("results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_oriented.csv"),
    Path("results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_random.csv"),
    Path("results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_dead_leaves.csv"),
]

LABELS = {
    "imagenet": "ImageNet",
    "dead_squares": "squares",
    "dead_oriented": "oriented",
    "dead_mixed": "mixed",
    "dead_textures": "textures",
    "stat_spectrum": "spectrum",
    "stat_wmm": "WMM",
    "stat_spectrum_color": "spectrum+color",
    "stat_spectrum_color_wmm": "spectrum+color+WMM",
    "stylegan_random": "random",
    "stylegan_highfreq": "high-freq.",
    "stylegan_sparse": "sparse",
    "stylegan_oriented": "oriented",
    "feature_vis_random": "random",
    "feature_vis_dead_leaves": "dead leaves",
}

LINE_STYLES = ["-", "--", "-.", ":"]

FAMILY_ORDER = [
    "ImageNet",
    "Dead leaves",
    "Statistical",
    "StyleGAN",
    "Feature visualizations",
]

FAMILY_DISPLAY = {
    "ImageNet": "ImageNet",
    "Dead leaves": "Dead leaves",
    "Statistical": "Statistical",
    "StyleGAN": "StyleGAN",
    "Feature visualizations": "Feature\nvisualizations",
}

FAMILY_COLOR = {
    "ImageNet": "#1B1B1B",
    "Dead leaves": "#2368B8",
    "Statistical": "#E17C05",
    "StyleGAN": "#009E73",
    "Feature visualizations": "#7B3294",
}


@dataclass(frozen=True)
class EnergyCurve:
    dataset: str
    family: str
    label: str
    source: Path
    energy: np.ndarray
    raw_eld: float
    peld: float
    fid: float | None = None


def family_for(dataset: str) -> str:
    if dataset == "imagenet":
        return "ImageNet"
    if dataset.startswith("dead_"):
        return "Dead leaves"
    if dataset.startswith("stat_"):
        return "Statistical"
    if dataset.startswith("stylegan_"):
        return "StyleGAN"
    if dataset.startswith("feature_vis_"):
        return "Feature visualizations"
    return "Other"


def read_energy(path: Path, limit: int | None) -> tuple[str, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"dataset", "logp_db_dim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    names = sorted(frame["dataset"].unique())
    if len(names) != 1:
        raise ValueError(f"{path} contains multiple datasets: {names}")
    if limit is not None:
        frame = frame.iloc[:limit]
    energy = logp_to_energy(frame["logp_db_dim"].to_numpy(np.float64))
    if not np.isfinite(energy).all():
        raise ValueError(f"{path} contains non-finite energies")
    return names[0], energy


def make_curve(path: Path, real_energy: np.ndarray, limit: int | None, fid_by_dataset: dict[str, float]) -> EnergyCurve:
    dataset, energy = read_energy(path, limit)
    if dataset == "imagenet":
        raw_eld = 0.0
        peld = 0.0
    else:
        raw_eld = float(stats.wasserstein_distance(real_energy, energy))
        peld = uniform_w1_from_ranks(empirical_percentile_ranks(real_energy, energy))
    return EnergyCurve(
        dataset=dataset,
        family=family_for(dataset),
        label=LABELS.get(dataset, dataset.replace("_", " ")),
        source=path,
        energy=energy,
        raw_eld=raw_eld,
        peld=peld,
        fid=fid_by_dataset.get(dataset),
    )


def kde_profile(energy: np.ndarray, grid: np.ndarray, bandwidth_scale: float) -> np.ndarray:
    kde = stats.gaussian_kde(energy)
    kde.set_bandwidth(kde.factor * bandwidth_scale)
    values = kde(grid)
    peak = float(np.max(values))
    return values / peak if peak > 0 else values


def line_style_for(index: int) -> str:
    return LINE_STYLES[index % len(LINE_STYLES)]


def load_metric_map(path: Path | None, dataset_column: str, fid_column: str) -> dict[str, float]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    missing = {dataset_column, fid_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        value = row[fid_column]
        if pd.isna(value):
            continue
        out[str(row[dataset_column])] = float(value)
    return out


def format_fid(value: float | None) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.1f}"


def draw_ridges(
    curves: list[EnergyCurve],
    families: list[str],
    grid: np.ndarray,
    xmin: float,
    xmax: float,
    output_path: Path,
    bandwidth_scale: float,
    with_summary: bool,
) -> None:
    y_positions = {family: i for i, family in enumerate(families[::-1])}

    fig = plt.figure(figsize=(12.4, 6.8), dpi=180)
    fig.patch.set_facecolor("white")
    if with_summary:
        ax = fig.add_axes([0.10, 0.16, 0.62, 0.72])
        side = fig.add_axes([0.75, 0.12, 0.23, 0.78])
        side.axis("off")
    else:
        ax = fig.add_axes([0.10, 0.16, 0.64, 0.72])
        side = fig.add_axes([0.77, 0.15, 0.21, 0.73])
        side.axis("off")

    ax.set_facecolor("#FBFBFA")
    ax.grid(axis="x", color="#DADADA", linewidth=0.8, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="x", labelsize=10, colors="#333333")
    ax.tick_params(axis="y", length=0)

    row_height = 0.78
    for family in families:
        y0 = y_positions[family]
        family_curves = [curve for curve in curves if curve.family == family]
        color = FAMILY_COLOR[family]
        ax.hlines(y0, xmin, xmax, color="#E7E7E7", lw=1.0, zorder=1)

        if family == "ImageNet":
            curve = family_curves[0]
            profile = kde_profile(curve.energy, grid, bandwidth_scale)
            ax.fill_between(grid, y0, y0 + profile * row_height, color="#BDBDBD", alpha=0.45, zorder=2)
            ax.plot(grid, y0 + profile * row_height, color="#1B1B1B", lw=2.4, zorder=4)
            ax.text(xmin, y0 + row_height * 0.72, "ImageNet reference", ha="left", va="center", fontsize=10.2, weight="bold", color="#1B1B1B")
        else:
            family_curves = sorted(family_curves, key=lambda curve: curve.peld)
            for j, curve in enumerate(family_curves):
                profile = kde_profile(curve.energy, grid, bandwidth_scale)
                alpha = 0.10 + 0.08 * (j / max(1, len(family_curves) - 1))
                ax.fill_between(grid, y0, y0 + profile * row_height, color=color, alpha=alpha, zorder=2)
                ax.plot(
                    grid,
                    y0 + profile * row_height,
                    color=color,
                    lw=1.8,
                    alpha=0.84,
                    linestyle=line_style_for(j),
                    zorder=3,
                )

    ax.set_yticks([y_positions[family] + row_height * 0.33 for family in families])
    ax.set_yticklabels([FAMILY_DISPLAY.get(family, family) for family in families], fontsize=10.5, weight="bold")
    ax.set_ylim(-0.20, len(families) - 1 + row_height + 0.30)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("real-density energy $E_P(x)$ in dB/dim  (lower = higher likelihood)", fontsize=11)
    ax.set_title("Energy distributions scored by the same ImageNet density model", loc="left", fontsize=12.8, weight="bold")

    title_x = 0.10 if with_summary else 0.11
    fig.text(title_x, 0.94, "Synthetic Families Occupy Different Energy Profiles", fontsize=19.2, weight="bold", ha="left", color="#171717")
    fig.text(
        title_x,
        0.902,
        "Each row overlays all small-scale datasets in a family; ELD compares these profiles to the ImageNet reference.",
        fontsize=11.5,
        ha="left",
        color="#666666",
    )

    if side is not None:
        side.text(0.0, 1.00, "Dataset Summary" if with_summary else "Curve Legend", fontsize=11.6, weight="bold", va="top")
        side.text(
            0.0,
            0.965,
            "sorted by pELD within family" if with_summary else "line style identifies family member",
            fontsize=8.8,
            color="#666666",
            va="top",
        )
        if with_summary:
            side.text(0.60, 0.965, "pELD", fontsize=7.6, color="#555555", va="top", ha="right", weight="bold")
            side.text(0.96, 0.965, "FID", fontsize=7.6, color="#555555", va="top", ha="right", weight="bold")
        y = 0.915
        for family in [f for f in families if f != "ImageNet"]:
            family_curves = sorted([curve for curve in curves if curve.family == family], key=lambda curve: curve.peld)
            color = FAMILY_COLOR[family]
            side.text(0.0, y, FAMILY_DISPLAY.get(family, family).replace("\n", " "), fontsize=9.3, weight="bold", va="center", color="#222222")
            side.plot([0.43, 0.64], [y, y], color=color, lw=3.0, solid_capstyle="round")
            y -= 0.036
            for j, curve in enumerate(family_curves):
                style = line_style_for(j)
                side.plot([0.02, 0.095], [y, y], color=color, lw=1.8, linestyle=style, solid_capstyle="round")
                weight = "bold" if j == 0 else "normal"
                side.text(0.115, y, curve.label, fontsize=7.8, va="center", color="#333333", weight=weight)
                if with_summary:
                    side.text(0.60, y, f"{curve.peld:.2f}", fontsize=7.6, va="center", ha="right", color="#666666", weight=weight)
                    side.text(0.96, y, format_fid(curve.fid), fontsize=7.6, va="center", ha="right", color="#666666", weight=weight)
                y -= 0.027
            y -= 0.035

        if with_summary:
            side.text(0.0, y - 0.005, "Reading the x-axis", fontsize=9.6, weight="bold", va="top")
            side.text(
                0.0,
                y - 0.045,
                "left = lower energy,\nhigher likelihood\n\nright = higher energy,\nlower likelihood\n\nFID shown when a metrics CSV is provided.",
                fontsize=7.6,
                color="#666666",
                va="top",
                linespacing=1.23,
            )
        else:
            side.text(0.0, y - 0.005, "Axis", fontsize=9.6, weight="bold", va="top")
            side.text(
                0.0,
                y - 0.045,
                "left = lower energy / higher likelihood\nright = higher energy / lower likelihood",
                fontsize=7.8,
                color="#666666",
                va="top",
                linespacing=1.25,
            )

    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot family-level energy ridgelines.")
    parser.add_argument("--real-scores", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--synthetic-scores", type=Path, nargs="*", default=DEFAULT_SYNTHETICS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/fig3_family_ridges"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--metrics-csv", type=Path, default=None, help="Optional CSV with dataset-level FID scores.")
    parser.add_argument("--metrics-dataset-column", default="dataset")
    parser.add_argument("--fid-column", default="fid")
    args = parser.parse_args()

    fid_by_dataset = load_metric_map(args.metrics_csv, args.metrics_dataset_column, args.fid_column)
    real_dataset, real_energy = read_energy(args.real_scores, args.limit)
    if real_dataset != "imagenet":
        raise ValueError(f"Expected real dataset 'imagenet', got {real_dataset!r}")

    curves = [make_curve(args.real_scores, real_energy, args.limit, fid_by_dataset)]
    curves.extend(make_curve(path, real_energy, args.limit, fid_by_dataset) for path in args.synthetic_scores)

    families = [family for family in FAMILY_ORDER if any(curve.family == family for curve in curves)]
    all_energy = np.concatenate([curve.energy for curve in curves])
    xmin, xmax = np.percentile(all_energy, [0.15, 99.85])
    pad = 0.05 * (xmax - xmin)
    xmin -= pad
    xmax += pad
    grid = np.linspace(xmin, xmax, 700)

    summary_rows = []
    for curve in curves:
        if curve.family == "ImageNet":
            continue
        summary_rows.append(
            {
                "family": curve.family,
                "dataset": curve.dataset,
                "label": curve.label,
                "peld": curve.peld,
                "fid": curve.fid,
                "raw_eld_db_dim": curve.raw_eld,
                "mean_energy_db_dim": float(np.mean(curve.energy)),
                "std_energy_db_dim": float(np.std(curve.energy)),
                "source": str(curve.source),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / "fig3_family_energy_ridges"
    draw_ridges(curves, families, grid, xmin, xmax, base.with_name(base.name + "_with_summary"), args.bandwidth_scale, with_summary=True)
    draw_ridges(curves, families, grid, xmin, xmax, base.with_name(base.name + "_clean"), args.bandwidth_scale, with_summary=False)

    summary = pd.DataFrame(summary_rows).sort_values(["family", "peld"])
    summary.to_csv(args.output_dir / "fig3_family_energy_ridges_summary.csv", index=False)
    print(f"Saved {base.with_name(base.name + '_with_summary').with_suffix('.png')}")
    print(f"Saved {base.with_name(base.name + '_with_summary').with_suffix('.pdf')}")
    print(f"Saved {base.with_name(base.name + '_clean').with_suffix('.png')}")
    print(f"Saved {base.with_name(base.name + '_clean').with_suffix('.pdf')}")
    print(f"Saved {args.output_dir / 'fig3_family_energy_ridges_summary.csv'}")


if __name__ == "__main__":
    main()
