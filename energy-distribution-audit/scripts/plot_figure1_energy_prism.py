"""Figure 1 concept: feature shadows vs energy height.

This is a geometric thesis figure for the paper, not an empirical result plot.
It is designed to play the same role as strong first figures in conceptual ML
papers: make the core blind spot memorable before the metrics and experiments.

Visual claim:
    Feature-space metrics compare the shadow of a dataset.
    ELD compares how that dataset occupies the real-image energy axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon


INK = "#151515"
MUTED = "#60646B"
GRID = "#D7DADF"
REAL = "#262626"
REAL_FILL = "#BFC3CA"
OVER = "#2F6DB3"
NOISE = "#D55E00"
SHIFTED = "#7B3294"
CURATED = "#0F8B63"
PLANE = "#F4F6F8"
PLANE_EDGE = "#B8BEC7"


def iso(x: float, y: float, z: float) -> tuple[float, float]:
    """Simple isometric projection from feature x/y plus energy z."""
    return x + 0.74 * y, 0.42 * y + z


def add_arrow(ax, xy0, xy1, color=INK, lw=1.5, rad=0.0, zorder=5, alpha=1.0):
    arrow = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle="-|>,head_length=7,head_width=4.5",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def add_blob(ax, center, width, height, angle, color, alpha, zorder, edge="white", lw=1.0):
    e = Ellipse(
        center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=edge,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(e)
    return e


def add_energy_column(ax, xy_ground, z_center, z_spread, color, label, label_xy, width=0.32, alpha=0.86):
    """Draw a stylized vertical energy distribution above a shared feature shadow."""
    x0, y0 = xy_ground
    levels = np.linspace(z_center - 2.1 * z_spread, z_center + 2.1 * z_spread, 15)
    levels = levels[(levels >= 0.12) & (levels <= 2.35)]
    for z in levels:
        density = np.exp(-0.5 * ((z - z_center) / z_spread) ** 2)
        w = width * (0.25 + 0.75 * density)
        cx, cy = iso(x0, y0, z)
        add_blob(
            ax,
            (cx, cy),
            width=w,
            height=0.055,
            angle=0,
            color=color,
            alpha=alpha * (0.30 + 0.70 * density),
            zorder=8 + z,
            edge="white",
            lw=0.55,
        )
    bottom = iso(x0, y0, max(0.12, levels.min()))
    top = iso(x0, y0, min(2.35, levels.max()))
    ax.plot([bottom[0], top[0]], [bottom[1], top[1]], color=color, lw=2.0, alpha=0.70, zorder=7)
    ax.annotate(
        label,
        xy=top,
        xytext=label_xy,
        fontsize=9.8,
        color=INK,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=2, shrinkB=5),
        zorder=20,
    )


def ridge(ax, y0, color, label, kind="real", scale=1.0):
    x = np.linspace(0, 1, 300)
    if kind == "real":
        dens = 0.95 * np.exp(-0.5 * ((x - 0.50) / 0.20) ** 2) + 0.35 * np.exp(-0.5 * ((x - 0.78) / 0.10) ** 2)
    elif kind == "over":
        dens = np.exp(-0.5 * ((x - 0.22) / 0.055) ** 2)
    elif kind == "noise":
        dens = np.exp(-0.5 * ((x - 0.82) / 0.075) ** 2)
    elif kind == "shifted":
        dens = np.exp(-0.5 * ((x - 0.64) / 0.16) ** 2)
    else:
        dens = 0.85 * np.exp(-0.5 * ((x - 0.48) / 0.18) ** 2)
    dens = dens / dens.max() * 0.18 * scale
    ax.fill_between(x, y0, y0 + dens, color=color, alpha=0.24, zorder=4)
    ax.plot(x, y0 + dens, color=color, lw=2.0, zorder=5)
    ax.text(1.04, y0 + 0.05, label, ha="left", va="center", fontsize=9.4, color=INK)


def main() -> None:
    fig = plt.figure(figsize=(12.8, 6.9), dpi=180)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.045, 0.11, 0.64, 0.78])
    ax.set_aspect("equal")
    ax.axis("off")

    # Ground feature plane.
    p0 = iso(-2.2, -1.35, 0)
    p1 = iso(2.2, -1.35, 0)
    p2 = iso(2.2, 1.35, 0)
    p3 = iso(-2.2, 1.35, 0)
    plane = Polygon([p0, p1, p2, p3], closed=True, facecolor=PLANE, edgecolor=PLANE_EDGE, lw=1.2, zorder=0)
    ax.add_patch(plane)

    for x in np.linspace(-2, 2, 5):
        a, b = iso(x, -1.35, 0), iso(x, 1.35, 0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=0.7, zorder=1)
    for y in np.linspace(-1, 1, 5):
        a, b = iso(-2.2, y, 0), iso(2.2, y, 0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=0.7, zorder=1)

    # Shared feature shadow: FID-like summaries live here.
    shadow_center = iso(-0.08, -0.05, 0.018)
    add_blob(ax, shadow_center, 1.95, 0.60, 22, "#AEB6C2", 0.34, 2, edge="#7F8996", lw=1.0)
    add_blob(ax, (shadow_center[0] + 0.04, shadow_center[1] + 0.005), 1.88, 0.55, 22, "#E5E8ED", 0.52, 3, edge="#A9B0BA", lw=0.8)
    ax.text(
        shadow_center[0],
        shadow_center[1] - 0.48,
        "same feature shadow\nsimilar moments / FID",
        ha="center",
        va="top",
        fontsize=10.2,
        color=MUTED,
        zorder=10,
    )

    # Energy axis.
    e0 = iso(-2.45, -1.48, 0)
    e1 = iso(-2.45, -1.48, 2.45)
    ax.plot([e0[0], e1[0]], [e0[1], e1[1]], color=INK, lw=1.7, zorder=6)
    add_arrow(ax, (e1[0], e1[1] - 0.12), (e1[0], e1[1] + 0.18), color=INK, lw=1.7, zorder=7)
    ax.text(e1[0] - 0.08, e1[1] + 0.28, "energy\n$E_P(x)$", ha="right", va="bottom", fontsize=11.2, color=INK)
    ax.text(e0[0] - 0.10, e0[1] + 0.04, "high likelihood", ha="right", va="center", fontsize=8.8, color=MUTED)
    ax.text(e1[0] - 0.10, e1[1] - 0.08, "low likelihood", ha="right", va="center", fontsize=8.8, color=MUTED)

    # Real broad band and three synthetic failures.
    add_energy_column(ax, (-0.08, -0.05), 1.15, 0.43, REAL, "real images\nbroad energy range", (1.50, 2.18), width=0.42, alpha=0.55)
    add_energy_column(ax, (-0.45, -0.15), 0.43, 0.09, OVER, "over-concentrated\nlikely but narrow", (-0.85, 0.86), width=0.36, alpha=0.93)
    add_energy_column(ax, (0.22, -0.03), 2.02, 0.12, NOISE, "noise-like\nhigh-energy tail", (1.32, 2.86), width=0.36, alpha=0.90)
    add_energy_column(ax, (0.40, 0.20), 1.48, 0.25, SHIFTED, "broad / shifted\nwrong levels", (2.22, 1.66), width=0.34, alpha=0.88)

    # A small curation arrow toward the real column, not the main story.
    start = iso(1.05, 0.35, 1.75)
    end = iso(0.18, 0.02, 1.22)
    add_arrow(ax, start, end, color=CURATED, lw=2.0, rad=-0.17, zorder=22)
    ax.text(start[0] + 0.18, start[1] + 0.15, "ELD-guided\nresampling", ha="left", va="center", fontsize=9.6, color=CURATED, weight="bold")

    # Panel tag and feature-plane axes.
    ax.text(-2.85, 2.92, "A", fontsize=12, weight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.23", fc=INK, ec="none"))
    ax.text(-2.42, 2.94, "Feature metrics see a projection; ELD sees the probability-level height", fontsize=13.5, weight="bold", color=INK, ha="left", va="center")
    fxa0, fxa1 = iso(1.18, -1.35, 0.03), iso(2.15, -1.35, 0.03)
    fya0, fya1 = iso(2.15, -1.25, 0.03), iso(2.15, -0.35, 0.03)
    add_arrow(ax, fxa0, fxa1, color="#7B8491", lw=1.0, zorder=4)
    add_arrow(ax, fya0, fya1, color="#7B8491", lw=1.0, zorder=4)
    ax.text(fxa1[0] + 0.05, fxa1[1] - 0.02, "$\\phi_1$", fontsize=9, color=MUTED)
    ax.text(fya1[0] + 0.03, fya1[1] + 0.03, "$\\phi_2$", fontsize=9, color=MUTED)

    ax.set_xlim(-3.05, 3.65)
    ax.set_ylim(-0.72, 3.34)

    # Right-side probability-level profiles.
    rx = fig.add_axes([0.72, 0.17, 0.24, 0.63])
    rx.set_xlim(0, 1.30)
    rx.set_ylim(0, 1)
    rx.axis("off")
    rx.text(
        0.0,
        1.03,
        "B",
        fontsize=12,
        weight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.23", fc=INK, ec="none"),
        transform=rx.transAxes,
    )
    rx.text(0.12, 1.035, "ELD compares energy profiles", transform=rx.transAxes, fontsize=13.0, weight="bold", color=INK, va="center")
    rx.text(0.12, 0.975, "not just average likelihood or feature moments", transform=rx.transAxes, fontsize=9.6, color=MUTED, va="center")

    for y in [0.18, 0.36, 0.54, 0.72]:
        rx.plot([0, 1.0], [y, y], color="#ECECEC", lw=0.8, zorder=0)
    ridge(rx, 0.72, REAL, "real", "real", 1.0)
    ridge(rx, 0.54, OVER, "over-concentrated", "over", 1.0)
    ridge(rx, 0.36, NOISE, "noise-like", "noise", 1.0)
    ridge(rx, 0.18, SHIFTED, "broad / shifted", "shifted", 1.0)
    rx.text(0.00, 0.045, "low energy\nhigh likelihood", ha="left", va="top", fontsize=8.8, color=MUTED)
    rx.text(1.00, 0.045, "high energy\nlow likelihood", ha="right", va="top", fontsize=8.8, color=MUTED)
    add_arrow(rx, (0.04, 0.09), (0.96, 0.09), color="#777777", lw=1.0, zorder=7)
    rx.text(
        0.50,
        -0.03,
        "$\\mathrm{ELD}=W_1((E_P)_\\#P,(E_P)_\\#Q)$",
        ha="center",
        va="top",
        fontsize=11.0,
        color=INK,
        transform=rx.transAxes,
    )

    fig.text(0.045, 0.955, "Energy Distributions Reveal What Feature Metrics Miss", fontsize=21.5, weight="bold", ha="left", color=INK)
    fig.text(
        0.045,
        0.915,
        "Synthetic datasets can share feature-space shadows while occupying different likelihood levels under a real-image density model.",
        fontsize=12.0,
        ha="left",
        color=MUTED,
    )

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "figure1_energy_prism.png"
    pdf = out_dir / "figure1_energy_prism.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
