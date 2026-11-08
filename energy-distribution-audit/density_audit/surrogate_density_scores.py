from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from density_audit.random_seed import resolve_seed


DATASET_ALIASES = {
    "ImageNet": "imagenet",
    "imagenet": "imagenet",
    "FractalDB": "fractaldb",
    "Perlin noise": "perlin_noise",
    "Gaussian noise": "gaussian_noise",
    "Diffusion, high CFG": "high_cfg_diffusion",
    "high-CFG diffusion": "high_cfg_diffusion",
    "StyleGAN3": "stylegan3",
    "BigGAN": "biggan",
    "Diffusion, mid CFG": "diffusion_mid_cfg",
    "diffusion (mid CFG)": "diffusion_mid_cfg",
    "Diffusion, low CFG": "diffusion_low_cfg",
    "diffusion (low CFG)": "diffusion_low_cfg",
    "Uniform mixture": "uniform_mixture",
    "uniform mixture": "uniform_mixture",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    scores_csv: Path


@dataclass(frozen=True)
class PathResolver:
    path_prefix: Path | None = None
    replacements: tuple[tuple[str, str], ...] = ()
    try_train_synthetic: bool = True

    def resolve(self, value: object) -> str:
        text = str(value)
        for old, new in self.replacements:
            if text == old or text.startswith(old.rstrip("/") + "/"):
                text = new.rstrip("/") + text[len(old.rstrip("/")) :]
                break
        path = Path(text)
        if not path.is_absolute() and self.path_prefix is not None:
            path = self.path_prefix / path
        if self.try_train_synthetic and not path.exists():
            parts = path.parts
            if "train" in parts:
                train_idx = parts.index("train")
                if train_idx + 1 < len(parts) and parts[train_idx + 1] != "synthetic":
                    candidate = Path(*parts[: train_idx + 1], "synthetic", *parts[train_idx + 1 :])
                    if candidate.exists():
                        path = candidate
        return str(path)


def canonical_dataset(name: object) -> str:
    text = str(name)
    return DATASET_ALIASES.get(text, text)


def read_score_paths(
    scores_csv: Path,
    dataset: str | None,
    limit: int | None,
    seed: int,
    resolver: PathResolver,
) -> pd.DataFrame:
    frame = pd.read_csv(scores_csv)
    if "path" not in frame.columns:
        raise ValueError(f"{scores_csv} missing path column")
    out = pd.DataFrame()
    out["path"] = frame["path"].map(resolver.resolve)
    if "image_id" in frame.columns:
        out["image_id"] = frame["image_id"].astype(str)
    else:
        out["image_id"] = out["path"]
    if dataset is not None:
        out["dataset"] = dataset
    elif "dataset" in frame.columns:
        names = sorted({canonical_dataset(x) for x in frame["dataset"].unique()})
        if len(names) != 1:
            raise ValueError(f"{scores_csv} contains multiple datasets: {names}")
        out["dataset"] = names[0]
    else:
        raise ValueError(f"Need dataset name for {scores_csv}")
    if limit is not None and len(out) > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(out), size=limit, replace=False)
        idx.sort()
        out = out.iloc[idx].reset_index(drop=True)
    return out


def parse_manifest(path: Path) -> list[DatasetSpec]:
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        raise ValueError(f"{path} missing dataset column")
    score_col = "source_scores" if "source_scores" in frame.columns else "score_csv"
    if score_col not in frame.columns:
        raise ValueError(f"{path} needs source_scores or score_csv column")
    specs = []
    for row in frame.itertuples(index=False):
        dataset = canonical_dataset(getattr(row, "dataset"))
        if dataset == "imagenet":
            continue
        score_csv = str(getattr(row, score_col))
        if score_csv and score_csv != "nan":
            specs.append(DatasetSpec(dataset=dataset, scores_csv=Path(score_csv)))
    return specs


def parse_replacement(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Replacement must look like old=new, got {text!r}")
    old, new = text.split("=", 1)
    return old, new


def zigzag_indices(size: int, keep: int) -> tuple[np.ndarray, np.ndarray]:
    coords = [(i, s - i) for s in range(2 * size - 1) for i in range(size) if 0 <= s - i < size]
    coords = coords[:keep]
    return np.asarray([i for i, _ in coords]), np.asarray([j for _, j in coords])


def dct_features(paths: list[str], image_size: int, keep: int) -> np.ndarray:
    rows, cols = zigzag_indices(image_size, keep)
    features = np.empty((len(paths), keep), dtype=np.float32)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        coeff = dctn(array, norm="ortho")
        feat = coeff[rows, cols].astype(np.float32)
        feat[0] = 0.0
        features[index] = feat
        if (index + 1) % 1000 == 0 or index + 1 == len(paths):
            print(f"dct features: {index + 1}/{len(paths)}", flush=True)
    return features


def resnet_features(paths: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    import torch
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        feature_dim = 512
    elif model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")
    model = create_feature_extractor(model.eval(), {"avgpool": "features"}).to(device)
    transform = weights.transforms()
    features = np.empty((len(paths), feature_dim), dtype=np.float32)
    kept = 0
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).to(device, non_blocking=True)
            feat = model(batch)["features"].flatten(1).cpu().numpy().astype(np.float32)
            features[kept : kept + len(feat)] = feat
            kept += len(feat)
            print(f"{model_name} features: {kept}/{len(paths)}", flush=True)
    return features


def diag_gaussian_energy(features: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    centered = features.astype(np.float64) - mean
    return 0.5 * np.sum((centered * centered) / var + np.log(var), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-image scores from simple ImageNet-fitted surrogate density models.")
    parser.add_argument("--mode", choices=["dct_diag", "dct_gmm", "resnet_diag"], required=True)
    parser.add_argument("--scorer-name", default=None)
    parser.add_argument("--real-scores", type=Path, required=True, help="ImageNet score CSV used only for paths.")
    parser.add_argument("--table", type=Path, required=True, help="CSV with dataset and source_scores/score_csv columns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-real", type=int, default=5000)
    parser.add_argument("--limit-synthetic", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--keep-dct", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=96)
    parser.add_argument("--gmm-components", type=int, default=16)
    parser.add_argument("--gmm-covariance", choices=["diag", "full"], default="diag")
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-4)
    parser.add_argument("--resnet", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--path-prefix",
        type=Path,
        default=None,
        help="Prefix for relative image paths stored in score CSVs, e.g. /scratch/.../noiseAudits.",
    )
    parser.add_argument(
        "--replace-prefix",
        action="append",
        default=[],
        help="Rewrite image paths before reading them. Use old=new; can be passed multiple times.",
    )
    parser.add_argument(
        "--no-train-synthetic-fallback",
        action="store_true",
        help="Disable fallback from train/file.png to train/synthetic/file.png when the former is missing.",
    )
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    resolver = PathResolver(
        path_prefix=args.path_prefix,
        replacements=tuple(parse_replacement(x) for x in args.replace_prefix),
        try_train_synthetic=not args.no_train_synthetic_fallback,
    )

    real = read_score_paths(args.real_scores, "imagenet", args.limit_real, args.seed, resolver)
    specs = parse_manifest(args.table)
    frames = [real]
    for spec in specs:
        frames.append(read_score_paths(spec.scores_csv, spec.dataset, args.limit_synthetic, args.seed, resolver))
    manifest = pd.concat(frames, ignore_index=True)

    if args.mode in {"dct_diag", "dct_gmm"}:
        scorer = args.scorer_name or "dct_diag"
        features = dct_features(manifest["path"].tolist(), args.image_size, args.keep_dct)
    else:
        scorer = args.scorer_name or f"{args.resnet}_diag"
        features = resnet_features(manifest["path"].tolist(), args.resnet, args.batch_size, args.device)

    real_mask = manifest["dataset"].to_numpy() == "imagenet"
    real_features = features[real_mask]
    if args.mode == "dct_gmm":
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        scorer = args.scorer_name or "dct_pca_gmm"
        pca_dim = min(args.pca_dim, real_features.shape[0] - 1, real_features.shape[1])
        transform = make_pipeline(
            StandardScaler(),
            PCA(n_components=pca_dim, whiten=True, random_state=args.seed),
        )
        real_z = transform.fit_transform(real_features)
        gmm = GaussianMixture(
            n_components=args.gmm_components,
            covariance_type=args.gmm_covariance,
            reg_covar=args.gmm_reg_covar,
            max_iter=300,
            n_init=2,
            random_state=args.seed,
            verbose=1,
        )
        print(
            f"fitting DCT-PCA-GMM: pca_dim={pca_dim} components={args.gmm_components} "
            f"covariance={args.gmm_covariance}",
            flush=True,
        )
        gmm.fit(real_z)
        energy = -gmm.score_samples(transform.transform(features))
    else:
        mean = real_features.astype(np.float64).mean(axis=0)
        var = real_features.astype(np.float64).var(axis=0)
        positive = var[var > 0]
        floor = float(args.ridge * np.median(positive)) if positive.size else 1e-8
        var = np.maximum(var, floor)
        energy = diag_gaussian_energy(features, mean, var)

    out = manifest[["image_id", "dataset"]].copy()
    out["split"] = "evaluation"
    out.loc[out["dataset"] == "imagenet", "split"] = "cdf"
    out["scorer"] = scorer
    out["raw_score"] = energy
    out["score_unit"] = f"{args.mode}_negative_log_density"
    out["path"] = manifest["path"].to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
