from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .eld import compute_eld_rows, load_score_tables, regime_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute manuscript ELD/pELD tables from Guth histogram score CSVs."
    )
    parser.add_argument("--score-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--real-dataset", default="imagenet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-n", type=int, default=1000)
    args = parser.parse_args()

    tables = load_score_tables(args.score_dirs, keep_first_duplicate={args.real_dataset})
    rows = compute_eld_rows(tables, real_dataset=args.real_dataset)
    df = pd.DataFrame(rows)
    df["regime"] = df.apply(regime_label, axis=1)

    too_small = df[df["n"] < args.min_n]
    if not too_small.empty:
        names = ", ".join(f"{r.dataset} (n={r.n})" for r in too_small.itertuples())
        raise ValueError(f"Sample-count precondition failed: {names}; expected n >= {args.min_n}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("peld").to_csv(args.output_dir / "eld_table.csv", index=False)

    columns = [
        "dataset",
        "n",
        "peld",
        "raw_eld_db_dim",
        "energy_shift_db_dim",
        "energy_spread_ratio",
        "mean_energy_db_dim",
        "std_energy_db_dim",
        "regime",
    ]
    print(df[columns].sort_values("peld").to_string(index=False))
    print(f"\nWrote {args.output_dir / 'eld_table.csv'}")


if __name__ == "__main__":
    main()
