from __future__ import annotations

import argparse
import json
import sys
import time
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
    limit: int | None,
    seed: int,
    resolver: PathResolver,
    split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(scores_csv)
    if "path" not in frame.columns:
        raise ValueError(f"{scores_csv} missing path column")
    out = pd.DataFrame()
    out["path"] = frame["path"].map(resolver.resolve)
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


def load_image(path: str, image_size: int) -> torch.Tensor:
    import torch

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def batches(frame: pd.DataFrame, batch_size: int, image_size: int):
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start : start + batch_size]
        images = torch.stack([load_image(p, image_size) for p in rows["path"]], dim=0)
        yield rows, images


def load_model_and_diffusion(
    improved_diffusion_repo: Path,
    checkpoint: Path,
    device: str,
    timestep_respacing: str,
):
    import torch

    sys.path.insert(0, str(improved_diffusion_repo.resolve()))
    from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            image_size=64,
            num_channels=128,
            num_res_blocks=3,
            learn_sigma=True,
            diffusion_steps=4000,
            noise_schedule="cosine",
            use_kl=True,
            timestep_respacing=timestep_respacing,
        )
    )
    model, diffusion = create_model_and_diffusion(**defaults)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing[:5]:
            print(f"missing examples: {missing[:5]}", flush=True)
        if unexpected[:5]:
            print(f"unexpected examples: {unexpected[:5]}", flush=True)
    model.to(device)
    model.eval()
    return model, diffusion, defaults


def score_frame(
    frame: pd.DataFrame,
    model,
    diffusion,
    batch_size: int,
    image_size: int,
    device: str,
) -> pd.DataFrame:
    import torch

    rows_out = []
    done = 0
    with torch.inference_mode():
        for rows, images in batches(frame, batch_size, image_size):
            images = images.to(device, non_blocking=True)
            out = diffusion.calc_bpd_loop(model, images, clip_denoised=True, model_kwargs={})
            total_bpd = out["total_bpd"].detach().cpu().numpy()
            for row, score in zip(rows.itertuples(index=False), total_bpd):
                rows_out.append(
                    {
                        "image_id": row.image_id,
                        "dataset": row.dataset,
                        "split": row.split,
                        "scorer": "improved_diffusion_vlb",
                        "raw_score": float(score),
                        "score_unit": "total_bpd",
                        "path": row.path,
                    }
                )
            done += len(rows)
            print(f"scored {done}/{len(frame)}", flush=True)
    return pd.DataFrame(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score images with OpenAI improved-diffusion ImageNet64 VLB total_bpd.")
    parser.add_argument("--improved-diffusion-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-scores", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--limit-real", type=int, default=1000)
    parser.add_argument("--limit-synthetic", type=int, default=500)
    parser.add_argument("--datasets", nargs="*", default=DATASET_ORDER)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--timestep-respacing",
        default="250",
        help="Use empty string for full 4000-step VLB; 250/500 are faster likelihood-surrogate pilots.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--path-prefix", type=Path, default=None)
    parser.add_argument("--replace-prefix", action="append", default=[])
    parser.add_argument("--no-train-synthetic-fallback", action="store_true")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    start = time.time()
    resolver = PathResolver(
        path_prefix=args.path_prefix,
        replacements=tuple(parse_replacement(x) for x in args.replace_prefix),
        try_train_synthetic=not args.no_train_synthetic_fallback,
    )

    specs = {spec.dataset: spec for spec in parse_table(args.table)}
    wanted = [canonical_dataset(d) for d in args.datasets]
    pieces = [
        read_score_paths(args.real_scores, "imagenet", args.limit_real, args.seed, resolver, split="cdf")
    ]
    for dataset in wanted:
        if dataset not in specs:
            raise ValueError(f"{dataset} not found in {args.table}")
        pieces.append(
            read_score_paths(
                specs[dataset].scores_csv,
                dataset,
                args.limit_synthetic,
                args.seed,
                resolver,
                split="evaluation",
            )
        )
    manifest = pd.concat(pieces, ignore_index=True)
    missing = [p for p in manifest["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths are missing; first: {missing[0]}")

    model, diffusion, model_config = load_model_and_diffusion(
        args.improved_diffusion_repo,
        args.checkpoint,
        args.device,
        args.timestep_respacing,
    )
    scored = score_frame(manifest, model, diffusion, args.batch_size, args.image_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "improved_diffusion_repo": str(args.improved_diffusion_repo),
        "timestep_respacing": args.timestep_respacing,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "limit_real": args.limit_real,
        "limit_synthetic": args.limit_synthetic,
        "datasets": wanted,
        "n_images": len(scored),
        "runtime_seconds": time.time() - start,
        "model_config": model_config,
    }
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    Path(metadata_path).write_text(json.dumps(metadata, indent=2, default=str))
    print(f"wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
