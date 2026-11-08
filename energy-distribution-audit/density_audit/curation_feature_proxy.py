from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from density_audit.random_seed import resolve_seed


def parse_feature_spec(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Expected NAME=/path/to/features.npz, got {item!r}")
    name, path = item.split("=", 1)
    return name, Path(path)


def load_features(path: Path, max_rows: int | None, seed: int) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    features = np.asarray(data["features"], dtype=np.float32)
    paths = [str(p) for p in data["paths"].tolist()]
    if max_rows is not None and len(features) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(features), size=max_rows, replace=False))
        features = features[idx]
        paths = [paths[i] for i in idx]
    return features, paths


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def imagenet_label_from_path(path: str) -> str:
    p = Path(path)
    for parent in [p.parent, *p.parents]:
        name = parent.name
        if len(name) == 9 and name.startswith("n") and name[1:].isdigit():
            return name
    return p.parent.name


def entropy_and_effective_count(labels: list[str]) -> tuple[float, float, int]:
    if not labels:
        return float("nan"), float("nan"), 0
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    return entropy, float(math.exp(entropy)), int(len(counts))


def feature_effective_rank(features: np.ndarray) -> float:
    if len(features) < 2:
        return float("nan")
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eig = singular**2 / max(len(features) - 1, 1)
    denom = float(np.sum(eig**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(eig) ** 2) / denom)


def covariance_trace(features: np.ndarray) -> float:
    if len(features) < 2:
        return float("nan")
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered) / max(len(features) - 1, 1))


def nearest_neighbor_metrics(
    real_features: np.ndarray,
    real_paths: list[str],
    query_features: np.ndarray,
    k: int,
) -> dict[str, float]:
    from sklearn.neighbors import NearestNeighbors

    real_norm = l2_normalize(real_features)
    query_norm = l2_normalize(query_features)
    k = min(k, len(real_norm))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(real_norm)
    distances, indices = nn.kneighbors(query_norm, return_distance=True)

    nearest_labels = [imagenet_label_from_path(real_paths[int(i)]) for i in indices[:, 0]]
    topk_labels = [imagenet_label_from_path(real_paths[int(i)]) for i in indices.reshape(-1)]
    h1, eff1, unique1 = entropy_and_effective_count(nearest_labels)
    hk, effk, uniquek = entropy_and_effective_count(topk_labels)

    return {
        "nn_cosine_mean": float(np.mean(distances[:, 0])),
        "nn_cosine_median": float(np.median(distances[:, 0])),
        "nn_cosine_p90": float(np.quantile(distances[:, 0], 0.90)),
        "nn_cosine_p95": float(np.quantile(distances[:, 0], 0.95)),
        "nn_unique_classes_top1": unique1,
        "nn_class_entropy_top1": h1,
        "nn_effective_classes_top1": eff1,
        "nn_unique_classes_topk": uniquek,
        "nn_class_entropy_topk": hk,
        "nn_effective_classes_topk": effk,
    }


def evaluate_one(
    name: str,
    feature_path: Path,
    real_features: np.ndarray,
    real_paths: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    features, _ = load_features(feature_path, args.max_query, args.seed)
    metrics = nearest_neighbor_metrics(real_features, real_paths, features, args.knn)
    row: dict[str, object] = {
        "strategy": name,
        "n_eval": int(len(features)),
        "feature_effective_rank": feature_effective_rank(features),
        "feature_trace": covariance_trace(features),
    }
    row.update(metrics)
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    keep = [
        "strategy",
        "n_eval",
        "nn_cosine_mean",
        "nn_cosine_p95",
        "nn_unique_classes_top1",
        "nn_effective_classes_top1",
        "feature_effective_rank",
        "feature_trace",
    ]
    show = show[keep]
    show = show.rename(
        columns={
            "strategy": "Strategy",
            "n_eval": "n",
            "nn_cosine_mean": "NN cosine dist. mean ↓",
            "nn_cosine_p95": "NN cosine dist. p95 ↓",
            "nn_unique_classes_top1": "NN classes",
            "nn_effective_classes_top1": "Effective NN classes ↑",
            "feature_effective_rank": "Feature eff. rank ↑",
            "feature_trace": "Feature trace",
        }
    )
    return show.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-feature proxy diagnostics for curated synthetic selections.")
    parser.add_argument("--real-features", type=Path, required=True)
    parser.add_argument("--feature-npz", action="append", required=True, help="Repeat as STRATEGY=/path/to/features.npz.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-real", type=int, default=10000)
    parser.add_argument("--max-query", type=int, default=10000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    real_features, real_paths = load_features(args.real_features, args.max_real, args.seed)
    rows = [
        evaluate_one(name, path, real_features, real_paths, args)
        for name, path in (parse_feature_spec(item) for item in args.feature_npz)
    ]
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    md = markdown_table(frame)
    args.output_md.write_text(md + "\n")
    print(md)
    print(f"\nwrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
