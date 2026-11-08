from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from density_audit.manifests import SMALL_SCALE_DATASETS, SMALL_SCALE_ALIASES


def read_val_acc(path: Path) -> float:
    if not path.exists():
        return float("nan")
    text = path.read_text(errors="replace")
    matches = re.findall(r"Accuracy:\s*([0-9.]+)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"val_acc\s+([0-9.]+)%", text)
    if matches:
        return float(matches[-1]) / 100.0
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Learning-to-See downstream results into one CSV.")
    parser.add_argument("--learning-repo", type=Path, default=Path("external/learning_with_noise"))
    parser.add_argument(
        "--linear-root",
        type=Path,
        default=None,
        help="Root containing <l2s-dataset>/val_acc.txt linear-eval outputs. Defaults to learning repo lincls path.",
    )
    parser.add_argument(
        "--finetune-root",
        type=Path,
        default=None,
        help="Root containing <l2s-dataset>/val_acc.txt full fine-tune outputs.",
    )
    parser.add_argument("--shift-csv", type=Path, default=None, help="CSV from eval_l2s_classifier.py with in_c/in_a rows.")
    parser.add_argument("--finetune-csv", type=Path, default=None, help="Optional CSV with dataset,in_ft, overriding --finetune-root.")
    parser.add_argument("--output", type=Path, default=Path("results/l2s_downstream.csv"))
    args = parser.parse_args()

    shift = pd.read_csv(args.shift_csv) if args.shift_csv and args.shift_csv.exists() else pd.DataFrame()
    ft = pd.read_csv(args.finetune_csv) if args.finetune_csv and args.finetune_csv.exists() else pd.DataFrame()
    linear_root = args.linear_root or (args.learning_repo / "lincls_imagenet" / "small_scale")
    finetune_root = args.finetune_root
    rows = []
    for name in SMALL_SCALE_DATASETS:
        alias = SMALL_SCALE_ALIASES[name]
        val_acc = read_val_acc(linear_root / name / "val_acc.txt")
        row = {
            "dataset": alias,
            "l2s_dataset": name,
            "in_lin": val_acc,
            "in_ft": np.nan,
            "in_c": np.nan,
            "in_a": np.nan,
        }
        if not shift.empty:
            for split, col in [("in_c", "in_c"), ("in_a", "in_a")]:
                match = shift[(shift["dataset"].astype(str) == alias) & (shift["split"].astype(str) == split)]
                if not match.empty:
                    row[col] = float(match.iloc[-1]["accuracy"])
        if not ft.empty:
            match = ft[ft["dataset"].astype(str) == alias]
            if not match.empty and "in_ft" in match.columns:
                row["in_ft"] = float(match.iloc[-1]["in_ft"])
        elif finetune_root is not None:
            row["in_ft"] = read_val_acc(finetune_root / name / "val_acc.txt")
        rows.append(row)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
