from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from density_audit.cross_scorer_eld import DATASET_ORDER, canonical_dataset, display_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dataset-to-score-CSV table from scores_*.csv files.")
    parser.add_argument("--score-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--absolute", action="store_true", help="Write absolute score CSV paths.")
    args = parser.parse_args()

    found: dict[str, Path] = {}
    for root in args.score_root:
        for path in sorted(root.glob("scores_*.csv")):
            dataset = canonical_dataset(path.stem.removeprefix("scores_"))
            if dataset not in found:
                found[dataset] = path

    rows = []
    for dataset in DATASET_ORDER:
        if dataset not in found:
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing scores_{dataset}.csv under {[str(p) for p in args.score_root]}")
        source = found[dataset].resolve() if args.absolute else found[dataset]
        rows.append({"dataset": dataset, "display": display_name(dataset), "source_scores": str(source)})

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
