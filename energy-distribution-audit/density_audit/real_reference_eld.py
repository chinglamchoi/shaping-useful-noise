from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks
from density_audit.random_seed import resolve_seed


DISPLAY = {
    "imagenet_ref": "ImageNet reference",
    "imagenet_heldout": "ImageNet held-out",
    "coco_val": "COCO val2017",
    "places365_val": "Places365 val",
    "cifar10_test": "CIFAR-10 test",
    "cifar100_test": "CIFAR-100 test",
    "dtd_test": "DTD test",
    "oxford_pets_test": "Oxford-IIIT Pets test",
}


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"dataset", "logp_db_dim"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def load_score_dir(score_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for path in sorted(score_dir.glob("scores_*.csv")):
        frame = read_scores(path)
        names = sorted(frame["dataset"].astype(str).unique())
        if len(names) != 1:
            raise ValueError(f"{path} contains multiple datasets: {names}")
        tables[names[0]] = frame
    if not tables:
        raise FileNotFoundError(f"No scores_*.csv files found in {score_dir}")
    return tables


def subset(values: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if limit is None or values.size <= limit:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=limit, replace=False)
    return values[idx]


def summarize(
    tables: dict[str, pd.DataFrame],
    real_dataset: str,
    datasets: list[str] | None,
    limit_reference: int | None,
    limit_candidate: int | None,
    seed: int,
) -> pd.DataFrame:
    if real_dataset not in tables:
        raise ValueError(f"{real_dataset!r} not found. Available: {sorted(tables)}")

    real_energy = logp_to_energy(tables[real_dataset]["logp_db_dim"].to_numpy(np.float64))
    real_energy = subset(real_energy, limit_reference, seed)
    wanted = datasets if datasets is not None else sorted(tables)

    rows = []
    for dataset in wanted:
        if dataset not in tables:
            raise ValueError(f"{dataset!r} not found. Available: {sorted(tables)}")
        energy = logp_to_energy(tables[dataset]["logp_db_dim"].to_numpy(np.float64))
        energy = subset(energy, limit_candidate, seed + hash(dataset) % 1_000_000)
        ranks = empirical_percentile_ranks(real_energy, energy)
        q25, median, q75 = np.quantile(ranks, [0.25, 0.5, 0.75])
        rows.append(
            {
                "dataset": dataset,
                "display": DISPLAY.get(dataset, dataset),
                "n": int(energy.size),
                "raw_eld_db_dim": float(wasserstein_distance(real_energy, energy)),
                "peld": uniform_w1_from_ranks(ranks),
                "median_percentile": float(median),
                "iqr_percentile": float(q75 - q25),
                "percentile_q25": float(q25),
                "percentile_q75": float(q75),
                "mean_energy_db_dim": float(np.mean(energy)),
                "energy_shift_db_dim": float(np.mean(energy) - np.mean(real_energy)),
                "energy_spread_ratio": float(np.std(energy) / np.std(real_energy)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute real-to-real ELD under a shared ImageNet density reference.")
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--real-dataset", default="imagenet_ref")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit-reference", type=int, default=None)
    parser.add_argument("--limit-candidate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    tables = load_score_dir(args.score_dir)
    out = summarize(
        tables=tables,
        real_dataset=args.real_dataset,
        datasets=args.datasets,
        limit_reference=args.limit_reference,
        limit_candidate=args.limit_candidate,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    table = out[
        [
            "display",
            "raw_eld_db_dim",
            "peld",
            "median_percentile",
            "iqr_percentile",
            "n",
        ]
    ].rename(
        columns={
            "display": "Evaluation dataset (Q)",
            "raw_eld_db_dim": "Raw ELD to ImageNet",
            "peld": "pELD to ImageNet",
            "median_percentile": "Median percentile",
            "iqr_percentile": "IQR",
        }
    )
    text = table.to_markdown(index=False, floatfmt=".3f")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text)
    print(text, flush=True)
    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
