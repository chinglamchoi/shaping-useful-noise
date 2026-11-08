from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from density_audit.cross_scorer_eld import DATASET_ORDER, canonical_dataset
from density_audit.random_seed import resolve_seed
from density_audit.surrogate_density_scores import PathResolver, parse_replacement


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    scores_csv: Path


def parse_table(path: Path) -> list[DatasetSpec]:
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns or "source_scores" not in frame.columns:
        raise ValueError(f"{path} must contain dataset and source_scores columns")
    specs = []
    for row in frame.itertuples(index=False):
        dataset = canonical_dataset(getattr(row, "dataset"))
        if dataset == "imagenet":
            continue
        specs.append(DatasetSpec(dataset=dataset, scores_csv=Path(getattr(row, "source_scores"))))
    return specs


def read_score_paths(
    scores_csv: Path,
    dataset: str,
    split: str,
    limit: int | None,
    seed: int,
    resolver: PathResolver,
) -> pd.DataFrame:
    frame = pd.read_csv(scores_csv)
    if "path" not in frame.columns:
        raise ValueError(f"{scores_csv} missing path column")
    out = pd.DataFrame()
    out["source_path"] = frame["path"].map(resolver.resolve)
    if "image_id" in frame.columns:
        out["image_id"] = frame["image_id"].astype(str)
    else:
        out["image_id"] = [f"{dataset}_{i:08d}" for i in range(len(frame))]
    out["dataset"] = dataset
    out["split"] = split
    if limit is not None and len(out) > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(out), size=limit, replace=False)
        idx.sort()
        out = out.iloc[idx].reset_index(drop=True)
    return out


def canonicalize(source: Path, dest: Path, size: int) -> None:
    if dest.exists():
        return
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG", compress_level=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed 64x64 PNG manifest for DenseFlow per-image likelihood scoring."
    )
    parser.add_argument("--real-scores", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit-real", type=int, default=1000)
    parser.add_argument("--limit-synthetic", type=int, default=500)
    parser.add_argument("--datasets", nargs="*", default=DATASET_ORDER)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--path-prefix", type=Path, default=None)
    parser.add_argument("--replace-prefix", action="append", default=[])
    parser.add_argument("--no-train-synthetic-fallback", action="store_true")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    resolver = PathResolver(
        path_prefix=args.path_prefix,
        replacements=tuple(parse_replacement(x) for x in args.replace_prefix),
        try_train_synthetic=not args.no_train_synthetic_fallback,
    )
    specs = {spec.dataset: spec for spec in parse_table(args.table)}
    wanted = [canonical_dataset(d) for d in args.datasets]
    pieces = [
        read_score_paths(args.real_scores, "imagenet", "cdf", args.limit_real, args.seed, resolver)
    ]
    for dataset in wanted:
        if dataset not in specs:
            raise ValueError(f"{dataset} not found in {args.table}")
        pieces.append(
            read_score_paths(
                specs[dataset].scores_csv,
                dataset,
                "evaluation",
                args.limit_synthetic,
                args.seed,
                resolver,
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    missing = [p for p in frame["source_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing image paths; first: {missing[0]}")

    canonical_paths = []
    for idx, row in enumerate(frame.itertuples(index=False)):
        dest = args.output_dir / row.dataset / f"{row.image_id}.png"
        canonicalize(Path(row.source_path), dest, args.image_size)
        canonical_paths.append(str(dest))
        if (idx + 1) % 1000 == 0 or idx + 1 == len(frame):
            print(f"canonicalized {idx + 1}/{len(frame)}", flush=True)

    out = frame.copy()
    out["canonical_path"] = canonical_paths
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.manifest, index=False)
    print(f"wrote {args.manifest}", flush=True)
    print(f"wrote canonical images under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
