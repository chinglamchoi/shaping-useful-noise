from __future__ import annotations

import argparse
from pathlib import Path

from .image_data import find_images, iter_image_batches


DEFAULT_MODEL_NAME = "finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays"


def parse_dataset_specs(specs: list[str]) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Dataset spec must be name=/path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name:
            raise ValueError(f"Dataset name is empty in {spec!r}")
        datasets[name] = Path(path)
    return datasets


def score_dataset(name: str, root: Path, scorer, batch_size: int, limit: int | None):
    import pandas as pd
    from tqdm import tqdm

    files = find_images(root)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No images found for dataset {name!r} at {root}")

    rows = []
    iterator = iter_image_batches(
        files,
        batch_size=batch_size,
        image_size=scorer.image_size,
        grayscale=scorer.channels == 1,
    )
    for records, batch in tqdm(iterator, desc=f"score {name}", unit="batch"):
        logp = scorer.logp_db_per_dim(batch).detach().cpu().numpy()
        for record, value in zip(records, logp):
            rows.append(
                {
                    "dataset": name,
                    "index": record.index,
                    "path": str(record.path),
                    "logp_db_dim": float(value),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score image datasets with Guth et al.'s ImageNet64 density model.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets as name=/path/to/images.")
    parser.add_argument("--real-dataset", default=None, help="Dataset name to use for real-density percentile bands.")
    parser.add_argument("--guth-repo", type=Path, default=Path("external/DualScoreMatching"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--normalization-constant", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/guth_histogram"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cpu, cuda, or mps.")
    args = parser.parse_args()

    from .guth_model import load_guth_scorer
    from .plotting import plot_histograms
    from .summaries import add_real_percentiles, summarize_logp, write_summary_table

    datasets = parse_dataset_specs(args.datasets)
    scorer = load_guth_scorer(
        guth_repo=args.guth_repo,
        model_name=args.model_name,
        model_dir=args.model_dir,
        normalization_constant_path=args.normalization_constant,
        device=args.device,
    )
    if scorer.normalization_constant is None:
        print("Warning: no normalization constant loaded; scores are internally comparable but not normalized log probabilities.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_tables: dict[str, pd.DataFrame] = {}
    rows = []
    for name, path in datasets.items():
        table = score_dataset(name, path, scorer=scorer, batch_size=args.batch_size, limit=args.limit)
        table.to_csv(args.output_dir / f"scores_{name}.csv", index=False)
        score_tables[name] = table
        summary = {"dataset": name, **summarize_logp(table["logp_db_dim"].to_numpy())}
        rows.append(summary)

    real_name = args.real_dataset
    if real_name is None and "imagenet" in {name.lower() for name in datasets}:
        real_name = next(name for name in datasets if name.lower() == "imagenet")
    if real_name is not None:
        if real_name not in score_tables:
            raise ValueError(f"--real-dataset {real_name!r} is not one of {list(score_tables)}")
        real_values = score_tables[real_name]["logp_db_dim"].to_numpy()
        for row in rows:
            values = score_tables[row["dataset"]]["logp_db_dim"].to_numpy()
            add_real_percentiles(row, values, real_values)

    summary = write_summary_table(rows, args.output_dir)
    plot_histograms(score_tables, summary, args.output_dir)

    print(summary.to_string(index=False))
    print(f"\nWrote scores, summaries, and figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
