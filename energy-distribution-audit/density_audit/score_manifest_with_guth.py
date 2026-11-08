from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from density_audit.modern_metrics import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a modern-dataset manifest with the Guth ImageNet density model."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--guth-repo", type=Path, required=True)
    parser.add_argument("--real-image-dir", type=Path, required=True, help="ImageNet reference ImageFolder/image directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--normalization-constant", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    if "image_dir" not in manifest.columns:
        raise ValueError(f"{args.manifest} must contain an image_dir column for scoring.")
    missing = manifest["image_dir"].isna() | (manifest["image_dir"].astype(str).str.strip() == "")
    if missing.any():
        rows = manifest.loc[missing, ["dataset", "display"]]
        raise ValueError(f"Missing image_dir for manifest rows:\n{rows.to_string(index=False)}")

    expected = [args.output_dir / "scores_imagenet.csv"]
    expected.extend(args.output_dir / f"scores_{dataset}.csv" for dataset in manifest["dataset"].astype(str))
    if not args.force and all(path.exists() and path.stat().st_size > 0 for path in expected):
        print(f"Score files already exist in {args.output_dir}; pass --force to recompute.")
    else:
        datasets = [f"imagenet={args.real_image_dir}"]
        datasets.extend(
            f"{row.dataset}={row.image_dir}"
            for row in manifest[["dataset", "image_dir"]].itertuples(index=False)
        )
        command = [
            sys.executable,
            "-m",
            "density_audit.guth_histogram",
            "--guth-repo",
            str(args.guth_repo),
            "--real-dataset",
            "imagenet",
            "--datasets",
            *datasets,
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--output-dir",
            str(args.output_dir),
        ]
        if args.model_name:
            command.extend(["--model-name", args.model_name])
        if args.model_dir:
            command.extend(["--model-dir", str(args.model_dir)])
        if args.normalization_constant:
            command.extend(["--normalization-constant", str(args.normalization_constant)])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    scored = manifest.copy()
    scored["score_csv"] = [
        str(args.output_dir / f"scores_{dataset}.csv") for dataset in scored["dataset"].astype(str)
    ]
    if "feature_npz" not in scored.columns:
        scored["feature_npz"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = args.output_dir / "manifest_scored.csv"
    scored.to_csv(out_manifest, index=False)
    print(f"Wrote {out_manifest}")
    print(f"Reference scores: {args.output_dir / 'scores_imagenet.csv'}")


if __name__ == "__main__":
    main()
