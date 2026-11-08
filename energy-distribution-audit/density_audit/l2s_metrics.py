from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, spatial, stats

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks
from density_audit.random_seed import resolve_seed


DISPLAY_ORDER = [
    ("Dead leaves", "dead_squares", "dead-leaves squares"),
    ("Dead leaves", "dead_oriented", "dead-leaves oriented"),
    ("Dead leaves", "dead_mixed", "dead-leaves mixed"),
    ("Dead leaves", "dead_textures", "dead-leaves textures"),
    ("Statistical image models", "stat_spectrum", "spectrum"),
    ("Statistical image models", "stat_wmm", "WMM"),
    ("Statistical image models", "stat_spectrum_color", "spectrum+color"),
    ("Statistical image models", "stat_spectrum_color_wmm", "spectrum+color+WMM"),
    ("Untrained StyleGAN", "stylegan_random", "StyleGAN random"),
    ("Untrained StyleGAN", "stylegan_highfreq", "StyleGAN high-freq."),
    ("Untrained StyleGAN", "stylegan_sparse", "StyleGAN sparse"),
    ("Untrained StyleGAN", "stylegan_oriented", "StyleGAN oriented"),
    ("Feature visualizations", "feature_vis_random", "feature-vis random"),
    ("Feature visualizations", "feature_vis_dead_leaves", "feature-vis dead leaves"),
]

FEATURE_FILE = {
    "imagenet": "imagenet_resnet50.npz",
    "dead_squares": "dead_squares_resnet50.npz",
    "dead_oriented": "dead_oriented_resnet50.npz",
    "dead_mixed": "dead_mixed_resnet50.npz",
    "dead_textures": "dead_textures_resnet50.npz",
    "stat_spectrum": "stat_spectrum_resnet50.npz",
    "stat_wmm": "stat_wmm_resnet50.npz",
    "stat_spectrum_color": "stat_spectrum_color_resnet50.npz",
    "stat_spectrum_color_wmm": "stat_spectrum_color_wmm_resnet50.npz",
    "stylegan_random": "stylegan_random_resnet50.npz",
    "stylegan_highfreq": "stylegan_highfreq_resnet50.npz",
    "stylegan_sparse": "stylegan_sparse_resnet50.npz",
    "stylegan_oriented": "stylegan_oriented_resnet50.npz",
    "feature_vis_random": "feature_vis_random_resnet50.npz",
    "feature_vis_dead_leaves": "feature_vis_dead_leaves_resnet50.npz",
}


def _path_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path)


def find_score_csv(score_dirs: list[Path], dataset: str) -> Path:
    matches = []
    for root in score_dirs:
        matches.extend(root.glob(f"scores_{dataset}.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find scores_{dataset}.csv under {[str(p) for p in score_dirs]}")
    return sorted(matches)[0]


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"dataset", "path", "logp_db_dim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["dataset"].nunique() != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(frame['dataset'].unique())}")
    frame = frame.copy()
    frame["energy"] = logp_to_energy(frame["logp_db_dim"].to_numpy(np.float64))
    frame["path_key"] = frame["path"].astype(str).map(_path_key)
    return frame


def load_features(path: Path, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    paths = np.asarray(data["paths"]).astype(str)
    features = np.asarray(data["features"], dtype=np.float64)
    if max_rows is not None and features.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(features.shape[0], size=max_rows, replace=False)
        idx.sort()
        paths = paths[idx]
        features = features[idx]
    return np.asarray([_path_key(p) for p in paths], dtype=object), features


def align_features_to_scores(scores: pd.DataFrame, feature_npz: Path, max_rows: int | None, seed: int) -> np.ndarray:
    keys, features = load_features(feature_npz, max_rows=max_rows, seed=seed)
    feature_frame = pd.DataFrame({"path_key": keys, "feature_idx": np.arange(len(keys))})
    joined = scores[["path_key"]].merge(feature_frame, on="path_key", how="inner")
    if joined.empty:
        raise ValueError(f"No overlapping paths between scores and {feature_npz}")
    return features[joined["feature_idx"].to_numpy()]


def fid_from_features(real: np.ndarray, synth: np.ndarray, eps: float = 1e-6) -> float:
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    mu_r = real.mean(axis=0)
    mu_s = synth.mean(axis=0)
    cov_r = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * eps
    cov_s = np.cov(synth, rowvar=False) + np.eye(synth.shape[1]) * eps
    covmean = linalg.sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_s
    return float(max(diff @ diff + np.trace(cov_r + cov_s - 2 * covmean), 0.0))


def standardize_pca(real: np.ndarray, synth: np.ndarray, dim: int, seed: int, fit_samples: int) -> tuple[np.ndarray, np.ndarray]:
    all_features = np.vstack([real, synth]).astype(np.float64)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (all_features - mean) / std
    rng = np.random.default_rng(seed)
    fit_n = min(fit_samples, z.shape[0])
    fit_idx = rng.choice(z.shape[0], size=fit_n, replace=False)
    _, _, vt = np.linalg.svd(z[fit_idx], full_matrices=False)
    components = vt[: min(dim, vt.shape[0])].T
    real_z = (real - mean) / std
    synth_z = (synth - mean) / std
    return real_z @ components, synth_z @ components


def improved_recall(real: np.ndarray, synth: np.ndarray, k: int) -> float:
    real_tree = spatial.cKDTree(real)
    # k+1 because each real point has itself as nearest neighbor.
    distances, _ = real_tree.query(real, k=k + 1)
    radii = distances[:, -1]
    synth_dist, synth_nn = real_tree.query(synth, k=1)
    return float(np.mean(synth_dist <= radii[synth_nn]))


def compute_lpips_var_from_scores(
    scores: pd.DataFrame,
    samples: int,
    device: str,
    seed: int,
    batch_size: int,
) -> tuple[float, str]:
    """Estimate LPIPS variation between two crops of the same image.

    Uses the optional ``lpips`` package when installed. If unavailable, falls
    back to AlexNet feature L2 distance with ImageNet weights; the output column
    remains a crop-variation proxy and should be labelled accordingly if used.
    """
    import torch
    import torchvision
    from PIL import Image

    rng = np.random.default_rng(seed)
    paths = scores["path"].astype(str).to_numpy()
    if paths.size > samples:
        idx = rng.choice(paths.size, size=samples, replace=False)
        paths = paths[idx]

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device).eval()
        implementation = "lpips-alex"

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # LPIPS expects [-1, 1].
            return model(a * 2 - 1, b * 2 - 1).flatten()

    except Exception:
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        alex = torchvision.models.alexnet(weights=weights).features.to(device).eval()
        norm = torchvision.transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        implementation = "alexnet-feature-l2"

        def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            fa = alex(norm(torch.nn.functional.interpolate(a, size=(224, 224), mode="bilinear", align_corners=False))).flatten(1)
            fb = alex(norm(torch.nn.functional.interpolate(b, size=(224, 224), mode="bilinear", align_corners=False))).flatten(1)
            return torch.linalg.vector_norm(fa - fb, dim=1)

    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            crops_a = []
            crops_b = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    crops_a.append(transform(image))
                    crops_b.append(transform(image))
            a = torch.stack(crops_a).to(device)
            b = torch.stack(crops_b).to(device)
            values.append(distance(a, b).detach().cpu().numpy())
    return float(np.mean(np.concatenate(values))), implementation


def cache_lookup(cache_path: Path, key: dict[str, object]) -> float | None:
    if not cache_path.exists():
        return None
    frame = pd.read_csv(cache_path)
    if frame.empty:
        return None
    key_json = json.dumps(key, sort_keys=True)
    match = frame[frame["key_json"].astype(str) == key_json]
    if match.empty:
        return None
    return float(match.iloc[-1]["lpips_var"])


def cache_store(cache_path: Path, key: dict[str, object], value: float, implementation: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "key_json": json.dumps(key, sort_keys=True),
                "dataset": key.get("dataset"),
                "samples": key.get("samples"),
                "seed": key.get("seed"),
                "implementation": implementation,
                "lpips_var": value,
            }
        ]
    )
    if cache_path.exists():
        row.to_csv(cache_path, mode="a", header=False, index=False)
    else:
        row.to_csv(cache_path, index=False)


def load_downstream(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["dataset", "in_lin", "in_ft", "in_c", "in_a"])
    frame = pd.read_csv(path)
    required = {"dataset"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute/assemble Learning-to-See table metrics.")
    parser.add_argument("--score-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("results/fig3_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/l2s_table"))
    parser.add_argument("--downstream-csv", type=Path, default=None, help="Optional CSV with dataset,in_lin,in_ft,in_c,in_a.")
    parser.add_argument("--max-feature-rows", type=int, default=10000)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--pca-fit-samples", type=int, default=30000)
    parser.add_argument("--recall-k", type=int, default=3)
    parser.add_argument("--compute-lpips-var", action="store_true")
    parser.add_argument("--lpips-samples", type=int, default=2000)
    parser.add_argument("--lpips-device", default="cuda")
    parser.add_argument("--lpips-batch-size", type=int, default=64)
    parser.add_argument("--lpips-cache", type=Path, default=None)
    parser.add_argument("--refresh-lpips-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downstream = load_downstream(args.downstream_csv)

    real_scores = read_scores(find_score_csv(args.score_dirs, "imagenet"))
    real_energy = real_scores["energy"].to_numpy(np.float64)
    real_features = align_features_to_scores(
        real_scores,
        args.feature_dir / FEATURE_FILE["imagenet"],
        max_rows=args.max_feature_rows,
        seed=args.seed,
    )

    rows = []
    for i, (family, dataset, display) in enumerate(DISPLAY_ORDER):
        score_csv = find_score_csv(args.score_dirs, dataset)
        scores = read_scores(score_csv)
        energy = scores["energy"].to_numpy(np.float64)
        eld = uniform_w1_from_ranks(empirical_percentile_ranks(real_energy, energy))

        synth_features = align_features_to_scores(
            scores,
            args.feature_dir / FEATURE_FILE[dataset],
            max_rows=args.max_feature_rows,
            seed=args.seed + i + 1,
        )
        real_pca, synth_pca = standardize_pca(
            real_features,
            synth_features,
            dim=args.pca_dim,
            seed=args.seed + i + 100,
            fit_samples=args.pca_fit_samples,
        )
        fid = fid_from_features(real_pca, synth_pca)
        recall = improved_recall(real_pca, synth_pca, k=args.recall_k)
        lpips_var = np.nan
        if args.compute_lpips_var:
            cache_path = args.lpips_cache or (args.output_dir / "lpips_var_cache.csv")
            lpips_seed = args.seed + i + 200
            cache_key = {
                "dataset": dataset,
                "score_csv": str(score_csv),
                "samples": int(args.lpips_samples),
                "seed": int(lpips_seed),
                "image_count": int(len(scores)),
                "metric": "crop-variation",
                "crop_size": 64,
            }
            cached = None if args.refresh_lpips_cache else cache_lookup(cache_path, cache_key)
            if cached is not None:
                lpips_var = cached
                print(f"{dataset}: using cached LPIPS-var {lpips_var:.6g}")
            else:
                print(f"{dataset}: computing LPIPS-var on {args.lpips_samples} samples")
                lpips_var, implementation = compute_lpips_var_from_scores(
                    scores,
                    samples=args.lpips_samples,
                    device=args.lpips_device,
                    seed=lpips_seed,
                    batch_size=args.lpips_batch_size,
                )
                cache_store(cache_path, cache_key, lpips_var, implementation)
                print(f"{dataset}: LPIPS-var {lpips_var:.6g} ({implementation})")

        row = {
            "family": family,
            "dataset": dataset,
            "display": display,
            "eld": eld,
            "fid": fid,
            "lpips_var": lpips_var,
            "recall": recall,
            "n_scores": int(energy.size),
            "score_csv": str(score_csv),
        }
        if not downstream.empty:
            match = downstream[downstream["dataset"].astype(str) == dataset]
            if not match.empty:
                for column in ["in_lin", "in_ft", "in_c", "in_a"]:
                    row[column] = match.iloc[0].get(column, np.nan)
        rows.append(row)

    table = pd.DataFrame(rows)
    for column in ["in_lin", "in_ft", "in_c", "in_a"]:
        if column not in table:
            table[column] = np.nan
    table.to_csv(args.output_dir / "l2s_metric_landscape.csv", index=False)

    latex = table.copy()
    for column in ["eld", "fid", "lpips_var", "recall", "in_lin", "in_ft", "in_c", "in_a"]:
        latex[column] = latex[column].map(lambda x: "TBD" if pd.isna(x) else f"{float(x):.3g}")
    latex[["family", "display", "eld", "fid", "lpips_var", "recall", "in_lin", "in_ft", "in_c", "in_a"]].to_csv(
        args.output_dir / "l2s_metric_landscape_latex_rows.csv",
        index=False,
    )
    print(f"Wrote {args.output_dir / 'l2s_metric_landscape.csv'}")
    print(f"Wrote {args.output_dir / 'l2s_metric_landscape_latex_rows.csv'}")


if __name__ == "__main__":
    main()
