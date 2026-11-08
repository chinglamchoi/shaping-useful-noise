from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks
from density_audit.random_seed import resolve_seed


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_score_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected DATASET=CSV, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty dataset name in {spec!r}")
    return name, Path(path).expanduser()


def parse_float_spec(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise ValueError(f"Expected DATASET=VALUE, got {spec!r}")
    name, value = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty dataset name in {spec!r}")
    return name, float(value)


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_scores(name: str, path: Path, limit: int | None, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "logp_db_dim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_dataset"] = name
    frame["source_score_csv"] = str(path)
    frame["energy"] = logp_to_energy(frame["logp_db_dim"].to_numpy(np.float64))
    frame = frame[frame["path"].astype(str).map(lambda p: Path(p).suffix.lower() in IMAGE_SUFFIXES)].copy()
    if limit is not None and len(frame) > limit:
        frame = frame.iloc[rng.choice(len(frame), size=limit, replace=False)].copy()
    return frame.reset_index(drop=True)


def apply_path_rewrites(paths: pd.Series, rewrites: list[tuple[str, str]]) -> pd.Series:
    out = paths.astype(str)
    for old, new in rewrites:
        out = out.str.replace(old, new, regex=False)
    return out


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_dataset_root(path: Path) -> Path:
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "modern_synthetic" and i + 3 < len(parts) and parts[i + 2] == "train":
            return Path(*parts[: i + 3])
    return path.parent


def build_stem_index(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        return {}
    all_images = find_images(root)
    index: dict[str, list[Path]] = {"__all__": all_images}
    for image_path in all_images:
        index.setdefault(image_path.stem, []).append(image_path)
    return index


def ordinal_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def recover_missing_path(
    path: Path,
    index_cache: dict[Path, dict[str, list[Path]]],
    search_roots: list[Path],
) -> Path:
    if path.exists():
        return path
    candidates = []
    for suffix in IMAGE_SUFFIXES:
        candidates.append(path.with_suffix(suffix))
        candidates.append(path.parent / "synthetic" / f"{path.stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    ordinal = ordinal_from_stem(path.stem)
    for root in [infer_dataset_root(path), *search_roots]:
        if root not in index_cache:
            index_cache[root] = build_stem_index(root)
        matches = index_cache[root].get(path.stem, [])
        if matches:
            return sorted(matches)[0]
        all_images = index_cache[root].get("__all__", [])
        if ordinal is not None and ordinal < len(all_images):
            return all_images[ordinal]
    return path


def sample_rows(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) == 0:
        return frame.copy()
    replace = len(frame) < n
    return frame.iloc[rng.choice(len(frame), size=n, replace=replace)].copy()


def load_npz_features(path: Path, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "features" not in data:
        raise ValueError(f"{path} missing 'features' array")
    features = np.asarray(data["features"], dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"{path} features must be 2D, got {features.shape}")
    if max_rows is not None and features.shape[0] > max_rows:
        idx = rng.choice(features.shape[0], size=max_rows, replace=False)
        features = features[idx]
    return features


def pca_project_feature_sets(feature_sets: dict[str, np.ndarray], dim: int) -> dict[str, np.ndarray]:
    if dim <= 0:
        return feature_sets
    all_features = np.vstack(list(feature_sets.values()))
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (all_features - mean) / std
    cov = np.cov(z, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    components = vectors[:, order[: min(dim, vectors.shape[1])]]
    return {name: ((features - mean) / std) @ components for name, features in feature_sets.items()}


def gaussian_moments(features: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * eps
    return mu, cov


def fid_from_moments(
    real_mu: np.ndarray,
    real_cov: np.ndarray,
    synth_mu: np.ndarray,
    synth_cov: np.ndarray,
) -> float:
    covmean = linalg.sqrtm(real_cov @ synth_cov)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = real_mu - synth_mu
    return float(max(diff @ diff + np.trace(real_cov + synth_cov - 2 * covmean), 0.0))


def write_moments(path: Path, name: str, mu: np.ndarray, cov: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, name=np.array(name), mu=mu, cov=cov)


def mixture_moments(weights: np.ndarray, moments: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    mus = np.stack([mu for mu, _ in moments])
    mu = weights @ mus
    second = np.zeros_like(moments[0][1])
    for weight, (source_mu, source_cov) in zip(weights, moments):
        second += weight * (source_cov + np.outer(source_mu, source_mu))
    cov = second - np.outer(mu, mu)
    return mu, cov


def optimize_fid_mixture_weights(
    real_features: np.ndarray,
    source_features: dict[str, np.ndarray],
    pca_dim: int,
    eps: float,
    seed: int,
    restarts: int,
    output_dir: Path | None = None,
) -> pd.Series:
    projected = pca_project_feature_sets({"__real__": real_features, **source_features}, pca_dim)
    real_mu, real_cov = gaussian_moments(projected["__real__"], eps)
    sources = sorted(source_features)
    source_moments = [gaussian_moments(projected[source], eps) for source in sources]
    n_sources = len(sources)

    def objective(weights: np.ndarray) -> float:
        synth_mu, synth_cov = mixture_moments(weights, source_moments)
        return fid_from_moments(real_mu, real_cov, synth_mu, synth_cov)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * n_sources
    starts = [np.full(n_sources, 1.0 / n_sources)]
    rng = np.random.default_rng(seed)
    starts.extend(rng.dirichlet(np.ones(n_sources), size=max(restarts, 0)))

    best = None
    for start in starts:
        result = optimize.minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            continue
        weights = np.clip(result.x, 0.0, 1.0)
        weights = weights / weights.sum()
        value = objective(weights)
        if best is None or value < best[0]:
            best = (value, weights)
    if best is None:
        raise RuntimeError("Could not optimize FID-guided mixture weights.")
    if output_dir is not None:
        moment_dir = output_dir / "fid_feature_moments"
        write_moments(moment_dir / "real.npz", "real", real_mu, real_cov)
        for source, (mu, cov) in zip(sources, source_moments):
            write_moments(moment_dir / f"{source}.npz", source, mu, cov)
        pd.DataFrame({"source": sources, "weight": best[1]}).to_csv(
            output_dir / "fid_guided_feature_moment_weights_long.csv", index=False
        )
    return pd.Series(best[1], index=sources, dtype=float)


def select_uniform_source_balanced(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    sources = sorted(pool["source_dataset"].unique())
    base = n // len(sources)
    extra = n % len(sources)
    parts = []
    for i, source in enumerate(sources):
        source_frame = pool[pool["source_dataset"] == source]
        parts.append(sample_rows(source_frame, base + (1 if i < extra else 0), rng))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def select_top_likelihood(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    return pool.nsmallest(n, "energy").copy().reset_index(drop=True)


def allocate_counts(weights: pd.Series, capacities: pd.Series, n: int) -> pd.Series:
    weights = weights.reindex(capacities.index).astype(float)
    weights = weights / weights.sum()
    raw = weights * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder > 0:
        order = (raw - counts).sort_values(ascending=False).index
        for source in order[:remainder]:
            counts[source] += 1

    counts = pd.Series(counts, index=capacities.index, dtype=int)
    overflow = int(np.maximum(counts - capacities, 0).sum())
    counts = np.minimum(counts, capacities).astype(int)
    while overflow > 0:
        room = capacities - counts
        room = room[room > 0]
        if room.empty:
            raise ValueError(f"Not enough source capacity to allocate {n} non-duplicate samples.")
        fill_weights = weights.reindex(room.index)
        fill_weights = fill_weights / fill_weights.sum()
        source = fill_weights.idxmax()
        counts[source] += 1
        overflow -= 1
    return counts


def select_source_weighted(pool: pd.DataFrame, n: int, weights: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    sources = sorted(pool["source_dataset"].unique())
    capacities = pool["source_dataset"].value_counts().reindex(sources).astype(int)
    weights = weights.reindex(sources).fillna(0.0).astype(float)
    if weights.sum() <= 0:
        raise ValueError("Source weights must sum to a positive value.")
    counts = allocate_counts(weights, capacities, n)
    parts = []
    for source, count in counts.items():
        if count > 0:
            parts.append(sample_rows(pool[pool["source_dataset"] == source], int(count), rng))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def fid_softmax_weights(pool: pd.DataFrame, fid_by_source: dict[str, float], temperature: float) -> pd.Series:
    sources = sorted(pool["source_dataset"].unique())
    missing = sorted(set(sources).difference(fid_by_source))
    if missing:
        raise ValueError(f"FID-guided selection missing --fid values for: {missing}")
    fid = pd.Series({source: fid_by_source[source] for source in sources}, dtype=float)
    if temperature <= 0:
        weights = pd.Series(0.0, index=sources)
        weights[fid.idxmin()] = 1.0
    else:
        weights = np.exp(-(fid - fid.min()) / temperature)
        weights = pd.Series(weights, index=sources, dtype=float)
    return weights / weights.sum()


def select_eld_histogram(pool: pd.DataFrame, n: int, bins: int, rng: np.random.Generator) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    target = np.full(bins, n // bins, dtype=int)
    target[: n % bins] += 1
    bin_id = np.clip(np.searchsorted(edges, pool["real_percentile"].to_numpy(), side="right") - 1, 0, bins - 1)
    work = pool.assign(_bin=bin_id)

    selected_parts = []
    selected_indices: set[int] = set()
    shortfall = 0
    for b, count in enumerate(target):
        candidates = work[work["_bin"] == b]
        take = min(count, len(candidates))
        shortfall += count - take
        if take:
            part = sample_rows(candidates, take, rng)
            selected_parts.append(part)
            selected_indices.update(part.index.tolist())

    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else work.iloc[[]].copy()
    if shortfall > 0:
        remaining = work.loc[~work.index.isin(selected_indices)].copy()
        if len(remaining) > 0:
            chosen_hist = np.bincount(selected["_bin"].to_numpy(dtype=int), minlength=bins)
            deficit = target - chosen_hist
            deficit = np.maximum(deficit, 0)
            if deficit.sum() == 0:
                weights = np.ones(len(remaining), dtype=np.float64)
            else:
                weights = deficit[remaining["_bin"].to_numpy(dtype=int)].astype(np.float64) + 1e-6
            weights /= weights.sum()
            take = min(shortfall, len(remaining))
            fill = remaining.iloc[rng.choice(len(remaining), size=take, replace=False, p=weights)].copy()
            selected = pd.concat([selected, fill], ignore_index=False)

    if len(selected) < n:
        remaining = work.loc[~work.index.isin(selected.index)].copy()
        if len(remaining) > 0:
            selected = pd.concat([selected, sample_rows(remaining, n - len(selected), rng)], ignore_index=False)
    if len(selected) > n:
        selected = sample_rows(selected, n, rng)
    return selected.drop(columns=["_bin"], errors="ignore").sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def select_peld_quantile_match(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Select images whose ImageNet energy percentiles approximate Unif(0, 1).

    The percentile pELD objective is a one-dimensional W1 distance to the
    uniform distribution. This greedy matcher targets the N uniform quantiles
    directly, using only sorted scalar percentile values.
    """
    values = pool["real_percentile"].to_numpy(np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    targets = (np.arange(n, dtype=np.float64) + 0.5) / n
    probes = np.searchsorted(sorted_values, targets, side="left")
    probes = np.clip(probes, 0, len(order) - 1)

    selected_sorted: set[int] = set()
    for target, probe in zip(targets, probes):
        left = int(probe)
        right = int(probe) + 1
        while True:
            left_ok = left >= 0 and left not in selected_sorted
            right_ok = right < len(order) and right not in selected_sorted
            if left_ok and right_ok:
                choose_left = abs(sorted_values[left] - target) <= abs(sorted_values[right] - target)
                selected_sorted.add(left if choose_left else right)
                break
            if left_ok:
                selected_sorted.add(left)
                break
            if right_ok:
                selected_sorted.add(right)
                break
            left -= 1
            right += 1
            if left < 0 and right >= len(order):
                break
        if len(selected_sorted) == n:
            break

    selected_indices = order[np.fromiter(selected_sorted, dtype=int, count=len(selected_sorted))]
    selected = pool.iloc[selected_indices].copy()
    if len(selected) < n:
        remaining = pool.drop(index=selected.index)
        selected = pd.concat([selected, sample_rows(remaining, n - len(selected), rng)], ignore_index=False)
    if len(selected) > n:
        selected = sample_rows(selected, n, rng)
    return selected.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def summarize_selection(name: str, selected: pd.DataFrame, real_energy: np.ndarray) -> dict[str, object]:
    energy = selected["energy"].to_numpy(np.float64)
    ranks = selected["real_percentile"].to_numpy(np.float64)
    counts = selected["source_dataset"].value_counts().sort_index()
    return {
        "strategy": name,
        "dataset_size": int(len(selected)),
        "source_pool": ",".join(sorted(selected["source_dataset"].unique())),
        "raw_eld_db_dim": float(stats.wasserstein_distance(real_energy, energy)),
        "peld": uniform_w1_from_ranks(ranks),
        "energy_shift_db_dim": float(np.mean(energy) - np.mean(real_energy)),
        "energy_spread_ratio": float(np.std(energy) / np.std(real_energy)),
        "median_percentile": float(np.median(ranks)),
        "iqr_percentile": float(np.quantile(ranks, 0.75) - np.quantile(ranks, 0.25)),
        **{f"n_{source}": int(count) for source, count in counts.items()},
    }


def write_score_csv(selected: pd.DataFrame, strategy: str, output_dir: Path) -> Path:
    output = output_dir / f"scores_{strategy}.csv"
    frame = selected.copy()
    frame["dataset"] = strategy
    keep = ["dataset", "path", "logp_db_dim", "source_dataset", "energy", "real_percentile"]
    frame[keep].to_csv(output, index=False)
    return output


def materialize_imagefolder(selected: pd.DataFrame, root: Path, mode: str) -> None:
    class_dir = root / "train" / "synthetic"
    class_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i, row in selected.reset_index(drop=True).iterrows():
        src = Path(str(row["path"]))
        suffix = src.suffix.lower() if src.suffix else ".jpg"
        dst = class_dir / f"{i:08d}_{row['source_dataset']}_{stable_id(str(src))}{suffix}"
        if dst.exists():
            pass
        elif mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown link mode {mode!r}")
        manifest_rows.append(
            {
                "strategy": root.name,
                "path": str(dst),
                "source_path": str(src),
                "source_dataset": row["source_dataset"],
                "logp_db_dim": row["logp_db_dim"],
                "real_percentile": row["real_percentile"],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(root / "selection_manifest.csv", index=False)


def markdown_table(summary: pd.DataFrame) -> str:
    show = summary.copy()
    columns = ["strategy", "dataset_size", "raw_eld_db_dim", "peld", "energy_spread_ratio", "median_percentile", "iqr_percentile"]
    source_cols = [c for c in show.columns if c.startswith("n_")]
    for column in source_cols:
        show[column] = show[column].fillna(0).astype(int)
    show = show[columns + source_cols]
    rename = {
        "strategy": "Sampling strategy",
        "dataset_size": "Dataset size",
        "raw_eld_db_dim": "Raw ELD",
        "peld": "pELD",
        "energy_spread_ratio": "Spread",
        "median_percentile": "Median percentile",
        "iqr_percentile": "IQR",
    }
    show = show.rename(columns=rename)
    return show.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-pool ELD-guided curation proof-of-concept datasets.")
    parser.add_argument("--real-scores", type=Path, required=True)
    parser.add_argument("--score-csv", action="append", required=True, help="Repeat as DATASET=/path/to/scores_DATASET.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=25000)
    parser.add_argument("--per-source-cap", type=int, default=None)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--eld-guidance",
        choices=["peld_quantile", "histogram"],
        default="peld_quantile",
        help="peld_quantile directly targets uniform ImageNet energy percentiles; histogram matches coarse bins.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=["uniform", "top_likelihood", "fid_guided", "eld_guided"],
        default=None,
        help="Strategies to build. Default: uniform, top_likelihood, fid_guided, eld_guided.",
    )
    parser.add_argument("--fid", action="append", default=[], help="Repeat as DATASET=FID for --strategy fid_guided.")
    parser.add_argument(
        "--fid-guidance",
        choices=["feature_moments", "source_fid"],
        default="feature_moments",
        help="Use feature-moment FID optimization when features are provided; source_fid uses the scalar --fid values.",
    )
    parser.add_argument("--real-features", type=Path, default=None, help="Real feature .npz for feature-moment FID guidance.")
    parser.add_argument("--feature-npz", action="append", default=[], help="Repeat as DATASET=/path/to/features.npz.")
    parser.add_argument("--fid-feature-max-rows", type=int, default=10000)
    parser.add_argument("--fid-pca-dim", type=int, default=128)
    parser.add_argument("--fid-cov-eps", type=float, default=1e-6)
    parser.add_argument("--fid-opt-restarts", type=int, default=8)
    parser.add_argument(
        "--fid-temperature",
        type=float,
        default=300.0,
        help="Softmax temperature for source-level FID weights. Set <=0 to allocate to the lowest-FID source first.",
    )
    parser.add_argument("--rewrite-path", action="append", default=[], help="Rewrite OLD=NEW in image paths.")
    parser.add_argument("--recover-missing-paths", action="store_true", help="Recover stale score paths by stem or ordinal position.")
    parser.add_argument("--search-root", action="append", type=Path, default=[], help="Additional root to search for stale image paths.")
    parser.add_argument("--make-imagefolders", action="store_true")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    rewrites = [tuple(item.split("=", 1)) for item in args.rewrite_path]
    fid_by_source = dict(parse_float_spec(item) for item in args.fid)
    feature_npz_by_source = {name: path for name, path in (parse_score_spec(item) for item in args.feature_npz)}
    strategies = args.strategy or ["uniform", "top_likelihood", "fid_guided", "eld_guided"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "scores").mkdir(exist_ok=True)

    real = pd.read_csv(args.real_scores)
    if "logp_db_dim" not in real.columns:
        raise ValueError(f"{args.real_scores} missing logp_db_dim")
    real_energy = logp_to_energy(real["logp_db_dim"].to_numpy(np.float64))

    parts = []
    for spec in args.score_csv:
        name, path = parse_score_spec(spec)
        part = read_scores(name, path, args.per_source_cap, rng)
        part["path"] = apply_path_rewrites(part["path"], rewrites)
        if args.recover_missing_paths:
            cache: dict[Path, dict[str, list[Path]]] = {}
            source_search_roots = [
                root
                for root in args.search_root
                if root.name == name or root.parent.name == name or root.parent.parent.name == name
            ]
            part["path"] = part["path"].map(
                lambda p: str(recover_missing_path(Path(str(p)), cache, source_search_roots or args.search_root))
            )
        part["real_percentile"] = empirical_percentile_ranks(real_energy, part["energy"].to_numpy(np.float64))
        parts.append(part)
    pool = pd.concat(parts, ignore_index=True)
    if len(pool) < args.dataset_size:
        raise ValueError(f"Pool has only {len(pool)} images, fewer than requested dataset size {args.dataset_size}")
    pool.to_csv(args.output_root / "fixed_pool_scores.csv", index=False)

    selected_by_strategy: dict[str, pd.DataFrame] = {}
    if "uniform" in strategies:
        selected_by_strategy["uniform"] = select_uniform_source_balanced(pool, args.dataset_size, rng)
    if "top_likelihood" in strategies:
        selected_by_strategy["top_likelihood"] = select_top_likelihood(pool, args.dataset_size)
    if "fid_guided" in strategies:
        if args.fid_guidance == "feature_moments" and args.real_features is not None and feature_npz_by_source:
            source_names = sorted(pool["source_dataset"].unique())
            missing_features = sorted(set(source_names).difference(feature_npz_by_source))
            if missing_features:
                raise ValueError(f"FID feature guidance missing --feature-npz values for: {missing_features}")
            real_features = load_npz_features(args.real_features, args.fid_feature_max_rows, rng)
            source_features = {
                source: load_npz_features(feature_npz_by_source[source], args.fid_feature_max_rows, rng)
                for source in source_names
            }
            fid_weights = optimize_fid_mixture_weights(
                real_features,
                source_features,
                args.fid_pca_dim,
                args.fid_cov_eps,
                args.seed,
                args.fid_opt_restarts,
                args.output_root,
            )
            fid_weights.to_csv(args.output_root / "fid_guided_feature_moment_weights.csv", header=["weight"])
        else:
            fid_weights = fid_softmax_weights(pool, fid_by_source, args.fid_temperature)
            fid_weights.to_csv(args.output_root / "fid_guided_source_fid_weights.csv", header=["weight"])
        selected_by_strategy["fid_guided"] = select_source_weighted(pool, args.dataset_size, fid_weights, rng)
    if "eld_guided" in strategies:
        if args.eld_guidance == "peld_quantile":
            selected_by_strategy["eld_guided"] = select_peld_quantile_match(pool, args.dataset_size, rng)
        else:
            selected_by_strategy["eld_guided"] = select_eld_histogram(pool, args.dataset_size, args.bins, rng)

    rows = []
    for strategy, selected in selected_by_strategy.items():
        write_score_csv(selected, strategy, args.output_root / "scores")
        if args.make_imagefolders:
            materialize_imagefolder(selected, args.output_root / "imagefolders" / strategy, args.link_mode)
        rows.append(summarize_selection(strategy, selected, real_energy))

    summary = pd.DataFrame(rows)
    source_cols = [column for column in summary.columns if column.startswith("n_")]
    for column in source_cols:
        summary[column] = summary[column].fillna(0).astype(int)
    summary.to_csv(args.output_root / "curation_summary.csv", index=False)
    md = markdown_table(summary)
    (args.output_root / "curation_summary.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {args.output_root}")


if __name__ == "__main__":
    main()
