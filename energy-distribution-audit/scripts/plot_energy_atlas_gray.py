"""Publication-style Energy Atlas for the grayscale small-scale audit.

The script is intentionally self-contained: it embeds the ELD table exported by
``density_audit.eld_table`` and writes a compact atlas with three failure
regimes plus a near-real reference basin.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle


csv_text = """dataset,source_scores,n,mean_energy_db_dim,std_energy_db_dim,skew_energy_db_dim,mean_logp_db_dim,std_logp_db_dim,energy_shift_db_dim,energy_spread_ratio,raw_eld_db_dim,peld,real_energy_percentile_mean,real_energy_percentile_std,real_energy_band_0_5_pct,real_energy_band_5_25_pct,real_energy_band_25_75_pct,real_energy_band_75_95_pct,real_energy_band_95_100_pct,regime
imagenet,results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_imagenet.csv,50000,-10.452302010517121,3.814985299377751,-1.0828221538117648,10.452302010517121,3.814985299377751,0.0,1.0,0.0,7.20000000000047e-08,0.5,0.28867513453581695,0.05,0.2,0.5,0.2,0.05,real-reference
stat_spectrum_color_wmm,results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color_wmm.csv,50000,-10.63228655863285,4.189846093576946,-0.9309423335790269,10.63228655863285,4.189846093576946,-0.17998454811572984,1.0982600887768394,0.6883377960824967,0.049054080199999996,0.5109813741999999,0.313708566734958,0.06964,0.23066,0.37132,0.31074,0.01764,near-real energy shape
stylegan_highfreq,results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_highfreq.csv,50000,-10.558765948505402,2.3485901897793457,-0.590644022411549,10.558765948505402,2.3485901897793457,-0.10646393798828058,0.6156223433318121,1.1521052968215943,0.0847063582,0.4496316934,0.21316487260173614,0.00598,0.19518,0.69448,0.10082,0.00354,over-concentrated
stat_wmm,results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_wmm.csv,50000,-9.47437080239296,0.846391081270784,-0.47574493938225926,9.47437080239296,0.846391081270784,0.9779312081241613,0.2218595918072963,2.2273094593334193,0.1715557018,0.5375753234,0.09937731074052776,0.0,0.00094,0.99332,0.00574,0.0,low-likelihood compressed
stat_spectrum_color,results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum_color.csv,50000,-16.317900548686982,6.425116044117623,0.20821954927716907,16.317900548686982,6.425116044117623,-5.865598538169861,1.6841784541517373,5.8663607540512075,0.2505496986,0.2494503562,0.3209555767948317,0.51732,0.14622,0.19182,0.13022,0.01442,over-broad
dead_textures,results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_textures.csv,50000,-7.771844926571846,1.416402337418305,-2.896786294271758,7.771844926571846,1.416402337418305,2.6804570839452753,0.37127334085647185,2.9152032332086564,0.2614258018,0.7474226754,0.1476324209628232,0.00088,0.01794,0.31982,0.66076,0.0006,low-likelihood compressed
stylegan_sparse,results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_sparse.csv,50000,-13.568744415521621,2.561140068662948,-0.24706342828830816,13.568744415521621,2.561140068662948,-3.1164424050045003,0.6713368119873717,3.259766941595078,0.27778671759999995,0.22306636800000001,0.1494107900137757,0.05754,0.59944,0.33774,0.00526,2e-05,over-concentrated high-likelihood
dead_mixed,results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_mixed.csv,50000,-16.503577713479995,6.910394544811783,-1.5584180209447351,16.503577713479995,6.910394544811783,-6.051275702962874,1.811381696788953,6.051275702962875,0.31012571899999997,0.189874281,0.17468680572806589,0.3001,0.36878,0.3299,0.00122,0.0,over-broad
stat_spectrum,results/guth_histogram_stat_plus_hf_imagenet64_50k_gray/scores_stat_spectrum.csv,50000,-16.85323867290497,5.558473047991302,0.1898270525492392,16.85323867290497,5.558473047991302,-6.400936662387847,1.4570103452031453,6.416173349704743,0.31389395640000006,0.18611860039999997,0.22600990749462538,0.49462,0.19816,0.30364,0.00358,0.0,over-broad
dead_oriented,results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_oriented.csv,50000,-16.287393098726273,6.048336815601745,-1.2829717507850764,16.287393098726273,6.048336815601745,-5.835091088209152,1.5854154972991032,5.835091088209153,0.3186656504,0.18133434959999994,0.16799769004508655,0.30434,0.38574,0.3093,0.00062,0.0,over-broad
stylegan_oriented,results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_oriented.csv,50000,-16.066286713190078,3.9544235657621436,-0.37262289060584836,16.066286713190078,3.9544235657621436,-5.613984702672957,1.0365501451361125,5.614867333440781,0.3545471022000001,0.1454529522,0.1493280944019126,0.31756,0.49432,0.18354,0.00456,2e-05,broad-but-shifted
feature_vis_dead_leaves,results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_dead_leaves.csv,50000,-6.714270252399444,0.48341326473656065,-0.20272514959858134,6.714270252399444,0.48341326473656065,3.7380317581176765,0.12671431913916117,3.8502340292739867,0.36459701120000004,0.859957822,0.04514045847277491,0.0,2e-05,0.0184,0.97602,0.00556,low-likelihood compressed
dead_squares,results/guth_histogram_deadleaves_plus_hf_imagenet64_50k_gray/scores_dead_squares.csv,50000,-18.3433045271492,6.288966306871269,-1.4566706923472357,18.3433045271492,6.288966306871269,-7.89100251663208,1.648490312111305,7.89100251663208,0.3926559966,0.1073440034,0.11017634471011818,0.41928,0.46044,0.12028,0.0,0.0,over-broad
feature_vis_random,results/guth_histogram_feature_vis_plus_hf_imagenet64_50k_gray/scores_feature_vis_random.csv,50000,-6.180538505878449,0.6728157201788485,-1.4720135102584004,6.180538505878449,0.6728157201788485,4.271763504638672,0.17636128775872062,4.302675481834412,0.40194106039999994,0.9012865915999999,0.057531298446591744,0.0,0.0002,0.01866,0.85408,0.12706,low-likelihood compressed
stylegan_random,results/guth_histogram_stylegan_plus_hf_imagenet64_50k_gray/scores_stylegan_random.csv,50000,-20.52244066455841,1.0780318249901761,-0.5140657042083571,20.52244066455841,1.0780318249901761,-10.070138654041289,0.28257823828731665,10.090066381549837,0.47926998179999997,0.0207517342,0.007444473654634715,0.99918,0.00082,0.0,0.0,0.0,over-concentrated high-likelihood
"""


LABEL = {
    "imagenet": "ImageNet",
    "stat_spectrum_color_wmm": "Spectrum+Color+WMM",
    "stylegan_highfreq": "StyleGAN high-freq.",
    "stat_wmm": "WMM",
    "stat_spectrum_color": "Spectrum+Color",
    "dead_textures": "Dead textures",
    "stylegan_sparse": "StyleGAN sparse",
    "dead_mixed": "Dead mixed",
    "stat_spectrum": "Spectrum",
    "dead_oriented": "Dead oriented",
    "stylegan_oriented": "StyleGAN oriented",
    "feature_vis_dead_leaves": "FV dead leaves",
    "dead_squares": "Dead squares",
    "feature_vis_random": "FV random",
    "stylegan_random": "StyleGAN random",
}

FAMILY = {
    "dead": "Dead leaves",
    "stat": "Statistical",
    "stylegan": "StyleGAN",
    "feature_vis": "Feature visualizations",
}

COLORS = {
    "Dead leaves": "#2368B8",
    "Statistical": "#F27D00",
    "StyleGAN": "#009E73",
    "Feature visualizations": "#8A2FA8",
}

MARKERS = {
    "Dead leaves": "o",
    "Statistical": "s",
    "StyleGAN": "^",
    "Feature visualizations": "D",
}

REGIME_COLORS = {
    "shifted": "#DDEAF7",
    "concentrated": "#E8E1F4",
    "noise_like": "#F3DDDD",
}


def dataset_family(name: str) -> str:
    if name.startswith("dead_"):
        return FAMILY["dead"]
    if name.startswith("stat_"):
        return FAMILY["stat"]
    if name.startswith("stylegan_"):
        return FAMILY["stylegan"]
    if name.startswith("feature_vis_"):
        return FAMILY["feature_vis"]
    return "Reference"


def point_sizes(peld: pd.Series) -> pd.Series:
    p = np.clip(peld.to_numpy(float), 0.0, 0.50)
    return pd.Series(90 + 860 * (p / 0.50) ** 0.80, index=peld.index)


def add_zone(ax, x0, y0, width, height, color, title, xy, ha="center"):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=color,
            edgecolor="none",
            alpha=0.62,
            zorder=0,
        )
    )
    ax.text(
        xy[0],
        xy[1],
        title,
        ha=ha,
        va="center",
        fontsize=11.8,
        weight="bold",
        color="#536070",
        zorder=1,
    )


def main() -> None:
    df = pd.read_csv(StringIO(csv_text.strip()))
    df["label"] = df["dataset"].map(LABEL).fillna(df["dataset"])
    df["family"] = df["dataset"].map(dataset_family)
    df["size"] = point_sizes(df["peld"])

    synth = df[df["dataset"] != "imagenet"].copy()
    real = df[df["dataset"] == "imagenet"].iloc[0]

    label_offsets = {
        "stat_spectrum_color_wmm": (-0.82, 0.17, "right"),
        "stylegan_random": (0.34, -0.02, "left"),
        "dead_squares": (0.28, -0.02, "left"),
        "feature_vis_random": (-0.68, 0.18, "right"),
        "dead_mixed": (0.26, 0.06, "left"),
        "stylegan_highfreq": (0.42, -0.10, "left"),
    }

    fig = plt.figure(figsize=(13.0, 7.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.15, 0.68, 0.70])
    side = fig.add_axes([0.79, 0.17, 0.18, 0.66])
    side.axis("off")

    ax.set_facecolor("#FBFBFA")
    ax.set_xlim(-10.9, 5.15)
    ax.set_ylim(0.00, 2.02)

    # Three manuscript regimes. These are regions of the energy plane, not
    # decorative islands, so every synthetic point has a clear interpretation.
    add_zone(
        ax,
        x0=-10.9,
        y0=0.80,
        width=9.9,
        height=1.22,
        color=REGIME_COLORS["shifted"],
        title="BROAD / SHIFTED",
        xy=(-6.0, 1.93),
    )
    add_zone(
        ax,
        x0=-10.9,
        y0=0.00,
        width=11.15,
        height=0.80,
        color=REGIME_COLORS["concentrated"],
        title="OVER-CONCENTRATED",
        xy=(-6.2, 0.74),
    )
    add_zone(
        ax,
        x0=0.25,
        y0=0.00,
        width=4.90,
        height=0.80,
        color=REGIME_COLORS["noise_like"],
        title="NOISE-LIKE / HIGH-ENERGY",
        xy=(2.50, 0.74),
    )

    # Near-real target basin is context, not a fourth failure regime.
    ax.add_patch(
        Ellipse(
            (0.0, 1.0),
            width=1.65,
            height=0.36,
            facecolor="#F0F2F1",
            edgecolor="#9AA09C",
            linewidth=0.9,
            alpha=0.65,
            zorder=0.4,
        )
    )
    ax.text(
        0.44,
        0.88,
        "near-real",
        ha="left",
        va="center",
        fontsize=9.8,
        color="#4B5350",
        zorder=1,
    )

    ax.axvline(0, color="#242424", linewidth=1.7, zorder=1.2)
    ax.axhline(1, color="#242424", linewidth=1.7, zorder=1.2)
    ax.grid(color="#D8D8D8", linewidth=0.9, zorder=0)

    for family, sub in synth.groupby("family", sort=False):
        ax.scatter(
            sub["energy_shift_db_dim"],
            sub["energy_spread_ratio"],
            s=sub["size"],
            marker=MARKERS[family],
            facecolor=COLORS[family],
            edgecolor="white",
            linewidth=1.7,
            alpha=0.96,
            zorder=3,
        )

    ax.scatter(
        real["energy_shift_db_dim"],
        real["energy_spread_ratio"],
        marker="*",
        s=540,
        facecolor="#111111",
        edgecolor="white",
        linewidth=1.6,
        zorder=5,
    )
    ax.text(
        0.16,
        1.055,
        "ImageNet",
        ha="left",
        va="bottom",
        fontsize=14,
        weight="bold",
        color="#111111",
        zorder=6,
    )

    for name, (dx, dy, ha) in label_offsets.items():
        row = df[df["dataset"] == name].iloc[0]
        x = row["energy_shift_db_dim"]
        y = row["energy_spread_ratio"]
        ax.annotate(
            row["label"],
            xy=(x, y),
            xytext=(x + dx, y + dy),
            ha=ha,
            va="center",
            fontsize=9.3,
            color="#202020",
            arrowprops=dict(
                arrowstyle="-",
                color="#6C6C6C",
                lw=0.8,
                shrinkA=5,
                shrinkB=7,
            ),
            zorder=7,
        )

    ax.set_xlabel(
        "Mean energy shift relative to ImageNet (dB/dim)",
        fontsize=15,
        labelpad=12,
    )
    ax.set_ylabel(
        "Energy spread ratio relative to ImageNet",
        fontsize=15,
        labelpad=12,
    )
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4])
    ax.set_yticks([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
    ax.tick_params(labelsize=11.5)
    ax.text(
        0.0,
        -0.13,
        "higher likelihood \u2190        energy shift        \u2192 lower likelihood",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        color="#5A5A5A",
        style="italic",
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)

    fig.text(
        0.08,
        0.94,
        "Energy Atlas of Small-Scale Synthetic Data",
        fontsize=22,
        weight="bold",
        ha="left",
        color="#141414",
    )
    fig.text(
        0.08,
        0.895,
        "Guth ImageNet64 grayscale density model. Point size encodes pELD; ImageNet is fixed at (0, 1).",
        fontsize=12.5,
        ha="left",
        color="#565656",
    )

    side.text(0.0, 0.98, "Dataset family", fontsize=12.5, weight="bold", va="top")
    y = 0.90
    for family in ["Dead leaves", "Statistical", "StyleGAN", "Feature visualizations"]:
        side.scatter(
            0.04,
            y,
            s=110,
            marker=MARKERS[family],
            facecolor=COLORS[family],
            edgecolor="white",
            linewidth=1.0,
        )
        side.text(0.13, y, family, fontsize=11.5, va="center")
        y -= 0.075

    side.text(0.0, 0.57, "Regime map", fontsize=12.5, weight="bold", va="top")
    regime_items = [
        ("BROAD / SHIFTED", REGIME_COLORS["shifted"]),
        ("OVER-CONCENTRATED", REGIME_COLORS["concentrated"]),
        ("NOISE-LIKE / HIGH-ENERGY", REGIME_COLORS["noise_like"]),
    ]
    y = 0.49
    for label, color in regime_items:
        side.add_patch(Rectangle((0.00, y - 0.018), 0.055, 0.036, facecolor=color, edgecolor="#B7B7B7"))
        side.text(0.08, y, label, fontsize=9.8, va="center", color="#404040")
        y -= 0.065

    side.text(0.0, 0.23, "Point size", fontsize=12.5, weight="bold")
    for x, val, size in [(0.09, "0.05", 140), (0.29, "0.25", 420), (0.54, "0.48", 850)]:
        side.scatter(x, 0.145, s=size * 0.32, facecolor="#9E9E9E", edgecolor="white")
        side.text(x, 0.075, val, ha="center", va="center", fontsize=9.5)
    side.text(0.0, 0.02, "pELD", fontsize=9.5, color="#666666")

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "energy_atlas_gray_clean_three_regimes.png"
    pdf = out_dir / "energy_atlas_gray_clean_three_regimes.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
