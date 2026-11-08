from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_score_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{requested!r} not found in {list(frame.columns)}")
        return requested
    for column in ["bpd", "bits_per_dim", "nll", "negative_log_likelihood", "raw_score", "score"]:
        if column in frame.columns:
            return column
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not infer DenseFlow score column from {list(frame.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DenseFlow per-image likelihood output into cross-scorer score CSV.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest from prepare_denseflow_scoring.py")
    parser.add_argument("--denseflow-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-column", default=None)
    parser.add_argument(
        "--join-column",
        default=None,
        help="Column shared with manifest. Defaults to image_id if present, else canonical_path/path.",
    )
    parser.add_argument(
        "--score-sign",
        choices=["lower_is_more_likely", "higher_is_more_likely"],
        default="lower_is_more_likely",
        help="DenseFlow bpd/NLL should use lower_is_more_likely.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    dense = pd.read_csv(args.denseflow_output)
    score_column = infer_score_column(dense, args.score_column)

    join = args.join_column
    if join is None:
        if "image_id" in dense.columns and "image_id" in manifest.columns:
            join = "image_id"
        elif "canonical_path" in dense.columns and "canonical_path" in manifest.columns:
            join = "canonical_path"
        elif "path" in dense.columns and "canonical_path" in manifest.columns:
            dense = dense.rename(columns={"path": "canonical_path"})
            join = "canonical_path"
        else:
            raise ValueError("Could not infer join column; pass --join-column.")

    joined = manifest.merge(dense[[join, score_column]], on=join, how="inner")
    if len(joined) != len(manifest):
        raise ValueError(f"Joined {len(joined)} rows but manifest has {len(manifest)} rows.")

    score = pd.to_numeric(joined[score_column], errors="raise")
    if args.score_sign == "higher_is_more_likely":
        score = -score

    out = pd.DataFrame(
        {
            "image_id": joined["image_id"].astype(str),
            "dataset": joined["dataset"].astype(str),
            "split": joined["split"].astype(str),
            "scorer": "denseflow",
            "raw_score": score.astype(float),
            "score_unit": score_column,
            "path": joined["canonical_path"].astype(str),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
