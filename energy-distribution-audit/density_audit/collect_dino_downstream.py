from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DOWNSTREAM_COLUMNS = ["in1k_lin", "in1k_ft", "in_c", "in_a"]


def read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    if "dataset" not in manifest.columns:
        raise ValueError(f"{path} missing dataset column")
    if "include" in manifest.columns:
        include = manifest["include"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
        manifest = manifest[include].copy()
    return manifest.reset_index(drop=True)


def parse_linear_accuracy(root: Path) -> float:
    if not root.exists():
        return float("nan")
    candidates = list(root.rglob("*.txt")) + list(root.rglob("*.log"))
    best = float("nan")
    patterns = [
        r"Acc@1\s+([0-9.]+)",
        r"\* Acc@1\s+([0-9.]+)",
        r"best_acc1[:=]\s*([0-9.]+)",
        r"Max accuracy:\s*([0-9.]+)",
    ]
    for path in candidates:
        text = path.read_text(errors="replace")
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                best = max(best if not np.isnan(best) else 0.0, max(float(m) for m in matches) / 100.0)
    return best


def load_optional_metric_csv(path: Path | None, column: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["dataset", column])
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        raise ValueError(f"{path} missing dataset column")
    if column not in frame.columns:
        # Accept split-style CSVs from future eval scripts.
        if "split" in frame.columns and "accuracy" in frame.columns:
            split = {"in_c": "in_c", "in_a": "in_a"}.get(column, column)
            frame = frame[frame["split"].astype(str) == split].copy()
            frame[column] = frame["accuracy"]
        else:
            raise ValueError(f"{path} missing {column} column")
    return frame[["dataset", column]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DINO/ViT downstream results into results/dino_downstream.csv.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/dino_sources.csv"))
    parser.add_argument("--linear-root", type=Path, default=Path("results/dino_downstream/linear_eval"))
    parser.add_argument("--finetune-csv", type=Path, default=None)
    parser.add_argument("--in-c-csv", type=Path, default=None)
    parser.add_argument("--in-a-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/dino_downstream.csv"))
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    rows = []
    for _, item in manifest.iterrows():
        dataset = str(item["dataset"])
        rows.append(
            {
                "dataset": dataset,
                "display": item.get("display", dataset),
                "in1k_lin": parse_linear_accuracy(args.linear_root / dataset),
                "in1k_ft": np.nan,
                "in_c": np.nan,
                "in_a": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    for path, column in [
        (args.finetune_csv, "in1k_ft"),
        (args.in_c_csv, "in_c"),
        (args.in_a_csv, "in_a"),
    ]:
        extra = load_optional_metric_csv(path, column)
        if not extra.empty:
            out = out.drop(columns=[column]).merge(extra, on="dataset", how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
