from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def _parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    rewrites = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected OLD=NEW for --rewrite-path, got {item!r}")
        old, new = item.split("=", 1)
        rewrites.append((old, new))
    return rewrites


def _rewrite_path(path: str, rewrites: list[tuple[str, str]]) -> str:
    out = path
    for old, new in rewrites:
        out = out.replace(old, new)
    return out


def _infer_dataset_root(path: Path) -> Path:
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "modern_synthetic" and i + 3 < len(parts) and parts[i + 2] == "train":
            return Path(*parts[: i + 3])
    return path.parent


def _build_stem_index(root: Path) -> dict[str, list[Path]]:
    from density_audit.image_data import find_images

    if not root.exists():
        return {}
    index: dict[str, list[Path]] = {}
    all_images = sorted(find_images(root))
    index["__all__"] = all_images
    for image_path in all_images:
        index.setdefault(image_path.stem, []).append(image_path)
    return index


def _ordinal_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _recover_missing_path(
    path: Path,
    index_cache: dict[Path, dict[str, list[Path]]],
    search_roots: list[Path],
) -> Path:
    if path.exists():
        return path

    candidates = []
    for suffix in IMAGE_SUFFIXES:
        candidates.append(path.with_suffix(suffix))
        candidates.append(path.parent / "synthetic" / f"{path.stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    ordinal = _ordinal_from_stem(path.stem)
    for root in [_infer_dataset_root(path), *search_roots]:
        if root not in index_cache:
            index_cache[root] = _build_stem_index(root)
        matches = index_cache[root].get(path.stem, [])
        if matches:
            return sorted(matches)[0]
        all_images = index_cache[root].get("__all__", [])
        if ordinal is not None and ordinal < len(all_images):
            return all_images[ordinal]
    return path


def _load_paths(
    scores_csv: Path | None,
    image_dir: Path | None,
    limit: int | None,
    rewrites: list[tuple[str, str]],
    recover_missing_paths: bool,
    search_roots: list[Path],
) -> tuple[str, list[Path]]:
    from density_audit.image_data import find_images

    if scores_csv is not None:
        scores = pd.read_csv(scores_csv)
        required = {"dataset", "path"}
        missing = required.difference(scores.columns)
        if missing:
            raise ValueError(f"{scores_csv} missing columns: {sorted(missing)}")
        names = sorted(scores["dataset"].unique())
        if len(names) != 1:
            raise ValueError(f"{scores_csv} contains multiple datasets: {names}")
        paths = [Path(_rewrite_path(p, rewrites)) for p in scores["path"].astype(str)]
        dataset = names[0]
    elif image_dir is not None:
        paths = find_images(image_dir)
        dataset = image_dir.name
    else:
        raise ValueError("Pass either --scores-csv or --image-dir")

    if limit is not None:
        paths = paths[:limit]
    if recover_missing_paths:
        index_cache: dict[Path, dict[str, list[Path]]] = {}
        paths = [_recover_missing_path(path, index_cache, search_roots) for path in paths]
    if not paths:
        raise ValueError("No images found")
    return dataset, paths


def _build_model(model_name: str, device: str):
    import torch
    from torch import nn
    from torchvision import models

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        transform = weights.transforms()
        model.fc = nn.Identity()
        feature_dim = 2048
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        transform = weights.transforms()
        model.fc = nn.Identity()
        feature_dim = 512
    else:
        raise ValueError(f"Unsupported --model {model_name!r}")

    model.eval().to(device)
    return model, transform, feature_dim


def _load_batch(paths: list[Path], transform):
    import torch

    tensors = []
    kept_paths = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensors.append(transform(image))
            kept_paths.append(str(path))
    return kept_paths, torch.stack(tensors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract torchvision features for scored image paths.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scores-csv", type=Path, help="scores_<dataset>.csv; features are extracted for its path column.")
    group.add_argument("--image-dir", type=Path, help="Image directory to scan recursively if no scores CSV is available.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz feature table.")
    parser.add_argument("--model", choices=["resnet50", "resnet18"], default="resnet50")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rewrite-path", action="append", default=[], help="Rewrite OLD=NEW in paths loaded from scores CSV.")
    parser.add_argument(
        "--recover-missing-paths",
        action="store_true",
        help="If a scored path is stale, search the inferred dataset train root for an image with the same stem.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Additional image root to search when recovering stale score paths.",
    )
    args = parser.parse_args()

    import torch

    dataset, paths = _load_paths(
        args.scores_csv,
        args.image_dir,
        args.limit,
        _parse_rewrites(args.rewrite_path),
        args.recover_missing_paths,
        args.search_root,
    )
    model, transform, feature_dim = _build_model(args.model, args.device)

    features = np.empty((len(paths), feature_dim), dtype=np.float32)
    out_paths: list[str] = []
    offset = 0
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc=f"features:{dataset}"):
            batch_paths = paths[start : start + args.batch_size]
            kept_paths, batch = _load_batch(batch_paths, transform)
            batch = batch.to(args.device, non_blocking=True)
            feat = model(batch).detach().cpu().numpy().astype(np.float32)
            end = offset + feat.shape[0]
            features[offset:end] = feat
            out_paths.extend(kept_paths)
            offset = end

    if offset != len(paths):
        features = features[:offset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset=np.array(dataset),
        model=np.array(args.model),
        paths=np.asarray(out_paths, dtype=object),
        features=features,
    )
    print(f"Wrote {args.output} with {features.shape[0]} rows and {features.shape[1]} features")


if __name__ == "__main__":
    main()
