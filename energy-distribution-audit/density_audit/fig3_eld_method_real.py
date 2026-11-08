from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, spatial, stats

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks
from density_audit.random_seed import resolve_seed


@dataclass(frozen=True)
class JoinedTable:
    dataset: str
    path: np.ndarray
    energy: np.ndarray
    features: np.ndarray | None = None


def _read_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path)
    required = {"dataset", "path", "logp_db_dim"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if scores["dataset"].nunique() != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(scores['dataset'].unique())}")
    scores = scores.copy()
    scores["path_key"] = scores["path"].astype(str).map(_path_key)
    scores["energy_db_dim"] = logp_to_energy(scores["logp_db_dim"].to_numpy(np.float64))
    if not np.isfinite(scores["energy_db_dim"]).all():
        raise ValueError(f"{path} contains non-finite energies")
    return scores


def _path_key(path: str) -> str:
    p = Path(path)
    try:
        return str(p.expanduser().resolve())
    except OSError:
        return str(p)


def _load_feature_npz(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    if "paths" not in data or "features" not in data:
        raise ValueError(f"{path} must contain 'paths' and 'features' arrays")
    features = np.asarray(data["features"], dtype=np.float32)
    paths = np.asarray(data["paths"]).astype(str)
    if features.ndim != 2:
        raise ValueError(f"{path} features must be a 2D array, got {features.shape}")
    frame = pd.DataFrame({"path": paths, "path_key": [_path_key(p) for p in paths]})
    frame["feature_row"] = list(features)
    return frame


def energy_stratified_indices(energy: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Deterministically keep an energy-stratified subset.

    This keeps every dataset represented across its energy range while bounding
    RAM for pooled feature experiments.
    """
    energy = np.asarray(energy, dtype=np.float64)
    n = energy.size
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    order = np.argsort(energy)
    bins = np.array_split(order, min(50, max_rows))
    per_bin = np.full(len(bins), max_rows // len(bins), dtype=int)
    per_bin[: max_rows % len(bins)] += 1
    kept: list[np.ndarray] = []
    for bin_indices, k in zip(bins, per_bin):
        if k <= 0:
            continue
        if bin_indices.size <= k:
            kept.append(bin_indices)
        else:
            kept.append(rng.choice(bin_indices, size=k, replace=False))
    out = np.concatenate(kept)
    out.sort()
    return out


def load_joined(scores_csv: Path, features_npz: Path | None = None) -> JoinedTable:
    scores = _read_scores(scores_csv)
    dataset = str(scores["dataset"].iloc[0])
    if features_npz is None:
        return JoinedTable(
            dataset=dataset,
            path=scores["path"].astype(str).to_numpy(),
            energy=scores["energy_db_dim"].to_numpy(np.float64),
            features=None,
        )

    features = _load_feature_npz(features_npz)
    joined = scores.merge(features[["path_key", "feature_row"]], on="path_key", how="inner")
    if joined.empty:
        raise ValueError(f"No matching paths between {scores_csv} and {features_npz}")
    if len(joined) < 0.90 * len(scores):
        print(f"WARNING: joined only {len(joined)} of {len(scores)} score rows for {dataset}")
    feat = np.vstack(joined["feature_row"].to_numpy()).astype(np.float64)
    return JoinedTable(
        dataset=dataset,
        path=joined["path"].astype(str).to_numpy(),
        energy=joined["energy_db_dim"].to_numpy(np.float64),
        features=feat,
    )


def concat_joined(tables: list[JoinedTable], dataset: str) -> JoinedTable:
    if not tables:
        raise ValueError("No tables to concatenate")
    have_features = [table.features is not None for table in tables]
    if any(have_features) and not all(have_features):
        raise ValueError("Either all pooled tables must have features or none should")
    paths = []
    energies = []
    features = []
    for table in tables:
        paths.append(np.asarray([f"{table.dataset}::{p}" for p in table.path], dtype=object))
        energies.append(table.energy)
        if table.features is not None:
            features.append(table.features)
    return JoinedTable(
        dataset=dataset,
        path=np.concatenate(paths),
        energy=np.concatenate(energies).astype(np.float64),
        features=np.vstack(features).astype(np.float32) if features else None,
    )


def w1_equal_or_interp(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = max(a.size, b.size)
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))


def peld(reference_energy: np.ndarray, values_energy: np.ndarray) -> float:
    ranks = empirical_percentile_ranks(reference_energy, values_energy)
    return uniform_w1_from_ranks(ranks)


def fid_from_features(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ca = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * eps
    cb = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * eps
    covmean = linalg.sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    value = diff @ diff + np.trace(ca + cb - 2 * covmean)
    return float(max(value, 0.0))


def standardize_and_pca(
    real_features: np.ndarray,
    pool_features: np.ndarray,
    pca_dim: int,
    pca_fit_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project features with bounded-memory PCA.

    PCA directions are fit on a random subset, then all rows are projected. This
    avoids SVD on the full ImageNet+synthetic feature matrix.
    """
    real_features = np.asarray(real_features, dtype=np.float32)
    pool_features = np.asarray(pool_features, dtype=np.float32)
    all_features = np.vstack([real_features, pool_features]).astype(np.float32, copy=False)
    mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    rng = np.random.default_rng(seed)
    fit_n = min(pca_fit_samples, all_features.shape[0])
    fit_idx = rng.choice(all_features.shape[0], size=fit_n, replace=False)
    fit_z = ((all_features[fit_idx] - mean) / std).astype(np.float32, copy=False)
    _, _, vt = np.linalg.svd(fit_z, full_matrices=False)
    k = min(pca_dim, vt.shape[0])
    components = vt[:k].T.astype(np.float32, copy=False)
    real_z = ((real_features - mean) / std).astype(np.float32, copy=False)
    pool_z = ((pool_features - mean) / std).astype(np.float32, copy=False)
    return real_z @ components, pool_z @ components


def make_feature_matched_energy_split(
    pool: JoinedTable,
    pca_features: np.ndarray,
    tail_fraction: float,
    n_pairs: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if pool.features is None:
        raise ValueError("Feature-matched construction requires candidate features")
    n = pool.energy.size
    tail_n = int(max(n_pairs * 2, np.floor(n * tail_fraction)))
    tail_n = min(tail_n, n // 2)
    if tail_n < n_pairs:
        raise ValueError(f"Need at least {n_pairs} low/high tail samples, got {tail_n}")

    order = np.argsort(pool.energy)
    low_idx = order[:tail_n]
    high_idx = order[-tail_n:]

    low_feat = pca_features[low_idx]
    high_feat = pca_features[high_idx]
    tree = spatial.cKDTree(high_feat)
    distances, nearest = tree.query(low_feat, k=min(8, len(high_idx)))
    if distances.ndim == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]

    candidate_pairs: list[tuple[float, int, int]] = []
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            candidate_pairs.append((float(distances[i, j]), int(low_idx[i]), int(high_idx[nearest[i, j]])))
    candidate_pairs.sort(key=lambda item: item[0])

    used_low: set[int] = set()
    used_high: set[int] = set()
    selected: list[tuple[float, int, int]] = []
    for dist, lo, hi in candidate_pairs:
        if lo in used_low or hi in used_high:
            continue
        selected.append((dist, lo, hi))
        used_low.add(lo)
        used_high.add(hi)
        if len(selected) >= n_pairs:
            break
    if len(selected) < n_pairs:
        raise ValueError(f"Could only construct {len(selected)} unique matched pairs")

    pair_df = pd.DataFrame(
        {
            "feature_distance": [p[0] for p in selected],
            "low_index": [p[1] for p in selected],
            "high_index": [p[2] for p in selected],
            "low_path": [pool.path[p[1]] for p in selected],
            "high_path": [pool.path[p[2]] for p in selected],
            "low_energy_db_dim": [pool.energy[p[1]] for p in selected],
            "high_energy_db_dim": [pool.energy[p[2]] for p in selected],
        }
    )
    return pair_df["low_index"].to_numpy(int), pair_df["high_index"].to_numpy(int), pair_df


def energy_tail_indices(energy: np.ndarray, tail_fraction: float, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    n = energy.size
    tail_n = int(max(n_pairs * 2, np.floor(n * tail_fraction)))
    tail_n = min(tail_n, n // 2)
    order = np.argsort(energy)
    return order[:tail_n], order[-tail_n:]


def plot_figure(
    real: JoinedTable,
    synthetic: JoinedTable,
    low: JoinedTable | None,
    high: JoinedTable | None,
    output_dir: Path,
    metrics: dict[str, float | int | str],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Ellipse

    ink = "#171717"
    muted = "#666666"
    grid = "#D9D9D9"
    real_color = "#1B1B1B"
    synth_color = "#2368B8"
    low_color = "#007A5A"
    high_color = "#D55E00"

    def clean(ax):
        ax.grid(color=grid, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9.5, colors="#333333")

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values)
        y = (np.arange(x.size) + 1) / x.size
        return x, y

    def covariance_ellipse(points: np.ndarray, color: str, label: str) -> Ellipse:
        points = np.asarray(points, dtype=np.float64)
        cov = np.cov(points[:, :2], rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * np.sqrt(np.maximum(vals, 1e-12))
        center = points[:, :2].mean(axis=0)
        return Ellipse(
            center,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            lw=2.0,
            alpha=0.18,
            label=label,
            zorder=2,
        )

    fig = plt.figure(figsize=(12.8, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.98, bottom=0.12, top=0.84, wspace=0.22, hspace=0.38)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_cdf = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    lo = np.percentile(np.concatenate([real.energy, synthetic.energy]), 0.5)
    hi = np.percentile(np.concatenate([real.energy, synthetic.energy]), 99.5)
    bins = np.linspace(lo, hi, 46)

    ax_hist.hist(real.energy, bins=bins, density=True, color="#D9D9D9", edgecolor="white", label="real ImageNet", zorder=2)
    ax_hist.hist(synthetic.energy, bins=bins, density=True, histtype="stepfilled", color="#8AB9EA", edgecolor=synth_color, linewidth=1.3, alpha=0.72, label=synthetic.dataset, zorder=3)
    ax_hist.set_xlabel("real-density energy $E_P(x)$ (lower = higher likelihood)")
    ax_hist.set_ylabel("density")
    ax_hist.legend(frameon=False, fontsize=8.8)
    ax_hist.text(
        0.03,
        0.92,
        f"ELD = {metrics['synthetic_raw_eld']:.2f}\npELD = {metrics['synthetic_peld']:.2f}",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BFBFBF"),
    )
    ax_hist.set_title("A  ELD: compare one-dimensional energy distributions", loc="left", fontsize=12.5, weight="bold")
    clean(ax_hist)

    xr, yr = ecdf(real.energy)
    xs, ys = ecdf(synthetic.energy)
    ax_cdf.plot(xr, yr, color=real_color, lw=2.0, label="real CDF")
    ax_cdf.plot(xs, ys, color=synth_color, lw=2.0, label="synthetic CDF")
    qn = 70
    q = (np.arange(qn) + 0.5) / qn
    rq = np.quantile(real.energy, q)
    sq = np.quantile(synthetic.energy, q)
    for x0, x1, y0 in zip(rq, sq, q):
        ax_cdf.plot([x0, x1], [y0, y0], color=synth_color, lw=0.8, alpha=0.18)
    ax_cdf.set_xlabel("real-density energy $E_P(x)$")
    ax_cdf.set_ylabel("empirical CDF")
    ax_cdf.legend(frameon=False, fontsize=8.8, loc="lower right")
    clean(ax_cdf)

    if low is None or high is None or low.features is None or high.features is None:
        for ax in [ax_feat, ax_energy]:
            ax.axis("off")
            ax.text(0.5, 0.5, "Run with --features-* to build\nfeature-matched controls.", ha="center", va="center", fontsize=12, color=muted)
    else:
        rng = np.random.default_rng(0)
        max_points = min(1600, low.features.shape[0], high.features.shape[0])
        low_plot_idx = rng.choice(low.features.shape[0], size=max_points, replace=False)
        high_plot_idx = rng.choice(high.features.shape[0], size=max_points, replace=False)
        low_xy = low.features[low_plot_idx, :2]
        high_xy = high.features[high_plot_idx, :2]
        ax_feat.scatter(
            low_xy[:, 0],
            low_xy[:, 1],
            s=8,
            color=low_color,
            alpha=0.16,
            linewidths=0,
            label="high-likelihood subset",
            zorder=3,
        )
        ax_feat.scatter(
            high_xy[:, 0],
            high_xy[:, 1],
            s=8,
            color=high_color,
            alpha=0.16,
            linewidths=0,
            label="low-likelihood subset",
            zorder=3,
        )
        ax_feat.add_patch(covariance_ellipse(low.features[:, :2], low_color, "high-likelihood feature ellipse"))
        ax_feat.add_patch(covariance_ellipse(high.features[:, :2], high_color, "low-likelihood feature ellipse"))
        ax_feat.set_xlabel("ResNet feature PCA 1")
        ax_feat.set_ylabel("ResNet feature PCA 2")
        ax_feat.legend(frameon=False, fontsize=8.2, loc="lower right")
        tail_fid = metrics.get("unmatched_tail_fid", float("nan"))
        fid_text = f"feature FID\nbefore matching: {tail_fid:.1f}\nafter matching: {metrics['matched_fid']:.1f}"
        ax_feat.text(
            0.04,
            0.92,
            fid_text,
            transform=ax_feat.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BFBFBF"),
        )
        ax_feat.set_title("B  Match features first", loc="left", fontsize=12.5, weight="bold")
        clean(ax_feat)

        control_bins = np.linspace(
            np.percentile(np.concatenate([real.energy, low.energy, high.energy]), 0.5),
            np.percentile(np.concatenate([real.energy, low.energy, high.energy]), 99.5),
            46,
        )
        ax_energy.hist(real.energy, bins=control_bins, density=True, color="#D9D9D9", edgecolor="white", alpha=0.92, label="real")
        ax_energy.hist(
            low.energy,
            bins=control_bins,
            density=True,
            histtype="step",
            color=low_color,
            linewidth=2.4,
            label="matched high-likelihood subset",
        )
        ax_energy.hist(
            high.energy,
            bins=control_bins,
            density=True,
            histtype="stepfilled",
            color="#F2A65A",
            edgecolor=high_color,
            linewidth=1.4,
            alpha=0.56,
            label="matched low-likelihood subset",
        )
        ax_energy.set_xlabel("real-density energy $E_P(x)$ (lower = higher likelihood)")
        ax_energy.set_ylabel("density")
        ax_energy.legend(frameon=False, fontsize=8.1, loc="upper left")
        energy_w1 = metrics.get("matched_energy_w1_between_subsets", float("nan"))
        ax_energy.text(
            0.58,
            0.92,
            f"same matched samples,\nseparated energies\n$W_1$ = {energy_w1:.2f}",
            transform=ax_energy.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BFBFBF"),
        )
        ax_energy.set_title("C  Then compare probability levels", loc="left", fontsize=12.5, weight="bold")
        clean(ax_energy)

    fig.text(0.07, 0.94, "ELD Measures the Energy Axis Missed by Feature Matching", fontsize=20, weight="bold", ha="left", color=ink)
    fig.text(0.07, 0.902, "The same images can be close in ResNet feature space yet occupy different likelihood levels under the real-image density model.", fontsize=11.8, ha="left", color=muted)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig3_eld_method_real.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "fig3_eld_method_real.pdf", bbox_inches="tight", facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-data Figure 3 ELD method experiment.")
    parser.add_argument("--real-scores", type=Path, required=True)
    parser.add_argument("--synthetic-scores", type=Path, required=True, help="Synthetic score CSV for the left ELD panel.")
    parser.add_argument("--real-features", type=Path, default=None)
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        nargs="*",
        default=None,
        help="One or more score CSVs for the feature-matched construction. Defaults to --synthetic-scores.",
    )
    parser.add_argument(
        "--candidate-features",
        type=Path,
        nargs="*",
        default=None,
        help="Feature .npz files matching --candidate-scores, in the same order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--pca-fit-samples", type=int, default=50000)
    parser.add_argument("--tail-fraction", type=float, default=0.40)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--max-real-rows", type=int, default=50000)
    parser.add_argument("--max-candidate-rows-per-dataset", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-limit", type=int, default=None, help="Optional deterministic row limit for quick local plotting.")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    real = load_joined(args.real_scores, args.real_features)
    synthetic = load_joined(args.synthetic_scores, None)
    if args.plot_limit is not None:
        real = JoinedTable(real.dataset, real.path[: args.plot_limit], real.energy[: args.plot_limit], None if real.features is None else real.features[: args.plot_limit])
        synthetic = JoinedTable(synthetic.dataset, synthetic.path[: args.plot_limit], synthetic.energy[: args.plot_limit], None)
    elif args.real_features is not None and args.max_real_rows > 0:
        idx = energy_stratified_indices(real.energy, args.max_real_rows, args.seed)
        real = JoinedTable(real.dataset, real.path[idx], real.energy[idx], None if real.features is None else real.features[idx])

    metrics: dict[str, float | int | str] = {
        "real_dataset": real.dataset,
        "synthetic_dataset": synthetic.dataset,
        "synthetic_raw_eld": w1_equal_or_interp(real.energy, synthetic.energy),
        "synthetic_peld": peld(real.energy, synthetic.energy),
        "real_n": int(real.energy.size),
        "synthetic_n": int(synthetic.energy.size),
    }

    low_table = None
    high_table = None
    if args.real_features is not None and args.candidate_features is not None:
        candidate_scores = args.candidate_scores or [args.synthetic_scores]
        candidate_features = args.candidate_features
        if len(candidate_scores) != len(candidate_features):
            raise ValueError(
                f"--candidate-scores and --candidate-features must have the same length; "
                f"got {len(candidate_scores)} and {len(candidate_features)}"
            )
        candidate_tables = [
            load_joined(score_csv, feature_npz)
            for score_csv, feature_npz in zip(candidate_scores, candidate_features)
        ]
        if args.max_candidate_rows_per_dataset > 0:
            limited_tables = []
            for i, table in enumerate(candidate_tables):
                idx = energy_stratified_indices(table.energy, args.max_candidate_rows_per_dataset, args.seed + i + 1)
                limited_tables.append(
                    JoinedTable(
                        table.dataset,
                        table.path[idx],
                        table.energy[idx],
                        None if table.features is None else table.features[idx],
                    )
                )
            candidate_tables = limited_tables
        if len(candidate_tables) == 1:
            candidate = candidate_tables[0]
        else:
            candidate = concat_joined(candidate_tables, dataset=f"pooled_{len(candidate_tables)}_small_scale")
        if args.plot_limit is not None:
            candidate = JoinedTable(candidate.dataset, candidate.path[: args.plot_limit], candidate.energy[: args.plot_limit], candidate.features[: args.plot_limit])
        if real.features is None or candidate.features is None:
            raise ValueError("Feature construction requires both real and candidate features")
        real_pca, cand_pca = standardize_and_pca(
            real.features,
            candidate.features,
            args.pca_dim,
            args.pca_fit_samples,
            args.seed,
        )
        n_pairs = min(args.n_pairs, candidate.energy.size // 4)
        low_tail_idx, high_tail_idx = energy_tail_indices(candidate.energy, args.tail_fraction, n_pairs)
        low_idx, high_idx, pairs = make_feature_matched_energy_split(candidate, cand_pca, args.tail_fraction, n_pairs)
        low_table = JoinedTable(
            dataset=f"{candidate.dataset}_feature_matched_low_energy",
            path=candidate.path[low_idx],
            energy=candidate.energy[low_idx],
            features=cand_pca[low_idx],
        )
        high_table = JoinedTable(
            dataset=f"{candidate.dataset}_feature_matched_high_energy",
            path=candidate.path[high_idx],
            energy=candidate.energy[high_idx],
            features=cand_pca[high_idx],
        )
        metrics.update(
            {
                "candidate_dataset": candidate.dataset,
                "candidate_n_after_subsample": int(candidate.energy.size),
                "matched_pairs": int(len(pairs)),
                "unmatched_tail_fid": fid_from_features(cand_pca[low_tail_idx], cand_pca[high_tail_idx]),
                "unmatched_tail_energy_w1": w1_equal_or_interp(candidate.energy[low_tail_idx], candidate.energy[high_tail_idx]),
                "matched_fid": fid_from_features(low_table.features, high_table.features),
                "matched_energy_w1_between_subsets": w1_equal_or_interp(low_table.energy, high_table.energy),
                "low_raw_eld": w1_equal_or_interp(real.energy, low_table.energy),
                "high_raw_eld": w1_equal_or_interp(real.energy, high_table.energy),
                "low_peld": peld(real.energy, low_table.energy),
                "high_peld": peld(real.energy, high_table.energy),
                "mean_pair_feature_distance": float(pairs["feature_distance"].mean()),
                "median_pair_feature_distance": float(pairs["feature_distance"].median()),
                "mean_low_energy": float(low_table.energy.mean()),
                "mean_high_energy": float(high_table.energy.mean()),
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.output_dir / "feature_matched_energy_pairs.csv", index=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fig3_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    plot_figure(real, synthetic, low_table, high_table, args.output_dir, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote {args.output_dir / 'fig3_eld_method_real.png'}")
    print(f"Wrote {args.output_dir / 'fig3_eld_method_real.pdf'}")


if __name__ == "__main__":
    main()
