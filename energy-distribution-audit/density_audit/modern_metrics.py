from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks
from density_audit.random_seed import resolve_seed
from density_audit.l2s_metrics import (
    align_features_to_scores,
    compute_lpips_var_from_scores,
    fid_from_features,
    improved_recall,
    standardize_pca,
)


REQUIRED_MANIFEST_COLUMNS = {"regime", "dataset", "display", "score_csv"}
DOWNSTREAM_COLUMNS = ["in1k_lin", "in1k_ft", "in_c", "in_a"]


def _path_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path)


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"dataset", "path", "logp_db_dim"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["energy"] = logp_to_energy(frame["logp_db_dim"].to_numpy(np.float64))
    frame["path_key"] = frame["path"].astype(str).map(_path_key)
    return frame


def read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path} missing manifest columns: {sorted(missing)}")
    manifest = manifest.copy()
    if "include" in manifest.columns:
        include = manifest["include"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
        manifest = manifest[include].copy()
    for column in ["feature_npz", "image_dir", "notes"]:
        if column not in manifest.columns:
            manifest[column] = ""
    if manifest["dataset"].duplicated().any():
        dup = sorted(manifest.loc[manifest["dataset"].duplicated(), "dataset"].astype(str).unique())
        raise ValueError(f"Duplicate dataset ids in {path}: {dup}")
    return manifest


def resolve_feature_npz(row: pd.Series, feature_dir: Path, feature_model: str) -> Path:
    raw = str(row.get("feature_npz", "")).strip()
    if raw and raw.lower() != "nan":
        return Path(raw)
    return feature_dir / f"{row['dataset']}_{feature_model}.npz"


def extract_features_if_needed(
    score_csv: Path,
    feature_npz: Path,
    feature_model: str,
    device: str,
    batch_size: int,
    force: bool,
) -> None:
    if feature_npz.exists() and feature_npz.stat().st_size > 0 and not force:
        print(f"Features exist: {feature_npz}")
        return
    command = [
        sys.executable,
        "-m",
        "density_audit.extract_features",
        "--scores-csv",
        str(score_csv),
        "--output",
        str(feature_npz),
        "--model",
        feature_model,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    print("Extracting features:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def cache_lookup(cache_path: Path, key: dict[str, object]) -> float | None:
    if not cache_path.exists():
        return None
    frame = pd.read_csv(cache_path)
    if frame.empty:
        return None
    key_json = json.dumps(key, sort_keys=True)
    match = frame[frame["key_json"].astype(str) == key_json]
    if match.empty:
        return None
    return float(match.iloc[-1]["lpips_var"])


def cache_store(cache_path: Path, key: dict[str, object], value: float, implementation: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "key_json": json.dumps(key, sort_keys=True),
                "dataset": key.get("dataset"),
                "samples": key.get("samples"),
                "seed": key.get("seed"),
                "implementation": implementation,
                "lpips_var": value,
            }
        ]
    )
    if cache_path.exists():
        row.to_csv(cache_path, mode="a", header=False, index=False)
    else:
        row.to_csv(cache_path, index=False)


def load_downstream(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["dataset", *DOWNSTREAM_COLUMNS])
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        raise ValueError(f"{path} missing required downstream column 'dataset'")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DINO/ViT-S table data metrics for modern synthetic datasets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-scores", type=Path, required=True)
    parser.add_argument("--real-features", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("results/dino_table/features"))
    parser.add_argument("--feature-model", choices=["resnet50", "resnet18"], default="resnet50")
    parser.add_argument("--extract-missing-features", action="store_true")
    parser.add_argument("--force-feature-extract", action="store_true")
    parser.add_argument("--feature-device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_table"))
    parser.add_argument("--downstream-csv", type=Path, default=None)
    parser.add_argument("--max-feature-rows", type=int, default=20000)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--pca-fit-samples", type=int, default=50000)
    parser.add_argument("--recall-k", type=int, default=3)
    parser.add_argument("--compute-lpips-var", action="store_true")
    parser.add_argument("--lpips-samples", type=int, default=5000)
    parser.add_argument("--lpips-device", default="cuda")
    parser.add_argument("--lpips-batch-size", type=int, default=256)
    parser.add_argument("--lpips-cache", type=Path, default=None)
    parser.add_argument("--refresh-lpips-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)
    downstream = load_downstream(args.downstream_csv)
    real_scores = read_scores(args.real_scores)
    real_energy = real_scores["energy"].to_numpy(np.float64)

    if args.extract_missing_features and (
        not args.real_features.exists() or args.force_feature_extract
    ):
        extract_features_if_needed(
            args.real_scores,
            args.real_features,
            args.feature_model,
            args.feature_device,
            args.feature_batch_size,
            args.force_feature_extract,
        )
    if not args.real_features.exists():
        raise FileNotFoundError(f"Missing real feature NPZ: {args.real_features}")

    real_features = align_features_to_scores(
        real_scores,
        args.real_features,
        max_rows=args.max_feature_rows,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    for i, row_in in manifest.reset_index(drop=True).iterrows():
        dataset = str(row_in["dataset"])
        score_csv = Path(str(row_in["score_csv"]))
        feature_npz = resolve_feature_npz(row_in, args.feature_dir, args.feature_model)
        if args.extract_missing_features:
            extract_features_if_needed(
                score_csv,
                feature_npz,
                args.feature_model,
                args.feature_device,
                args.feature_batch_size,
                args.force_feature_extract,
            )
        if not score_csv.exists():
            raise FileNotFoundError(f"Missing score CSV for {dataset}: {score_csv}")
        if not feature_npz.exists():
            raise FileNotFoundError(
                f"Missing feature NPZ for {dataset}: {feature_npz}. "
                "Pass --extract-missing-features or fill feature_npz in the manifest."
            )

        scores = read_scores(score_csv)
        energy = scores["energy"].to_numpy(np.float64)
        eld = uniform_w1_from_ranks(empirical_percentile_ranks(real_energy, energy))

        synth_features = align_features_to_scores(
            scores,
            feature_npz,
            max_rows=args.max_feature_rows,
            seed=args.seed + i + 1,
        )
        real_pca, synth_pca = standardize_pca(
            real_features,
            synth_features,
            dim=args.pca_dim,
            seed=args.seed + i + 100,
            fit_samples=args.pca_fit_samples,
        )
        fid = fid_from_features(real_pca, synth_pca)
        recall = improved_recall(real_pca, synth_pca, k=args.recall_k)

        lpips_var = np.nan
        if args.compute_lpips_var:
            cache_path = args.lpips_cache or (args.output_dir / "lpips_var_cache.csv")
            lpips_seed = args.seed + i + 200
            cache_key = {
                "dataset": dataset,
                "score_csv": str(score_csv),
                "samples": int(args.lpips_samples),
                "seed": int(lpips_seed),
                "image_count": int(len(scores)),
                "metric": "crop-variation",
                "crop_size": 64,
            }
            cached = None if args.refresh_lpips_cache else cache_lookup(cache_path, cache_key)
            if cached is not None:
                lpips_var = cached
                print(f"{dataset}: using cached LPIPS-var {lpips_var:.6g}")
            else:
                print(f"{dataset}: computing LPIPS-var on {args.lpips_samples} samples")
                lpips_var, implementation = compute_lpips_var_from_scores(
                    scores,
                    samples=args.lpips_samples,
                    device=args.lpips_device,
                    seed=lpips_seed,
                    batch_size=args.lpips_batch_size,
                )
                cache_store(cache_path, cache_key, lpips_var, implementation)
                print(f"{dataset}: LPIPS-var {lpips_var:.6g} ({implementation})")

        out = {
            "regime": row_in["regime"],
            "dataset": dataset,
            "display": row_in["display"],
            "eld": eld,
            "fid": fid,
            "lpips_var": lpips_var,
            "recall": recall,
            "n_scores": int(len(scores)),
            "score_csv": str(score_csv),
            "feature_npz": str(feature_npz),
        }
        match = downstream[downstream["dataset"].astype(str) == dataset]
        if not match.empty:
            for column in DOWNSTREAM_COLUMNS:
                out[column] = match.iloc[-1].get(column, np.nan)
        rows.append(out)

    table = pd.DataFrame(rows)
    for column in DOWNSTREAM_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    out_csv = args.output_dir / "dino_metric_landscape.csv"
    table.to_csv(out_csv, index=False)

    latex = table.copy()
    for column in ["eld", "fid", "lpips_var", "recall", *DOWNSTREAM_COLUMNS]:
        latex[column] = latex[column].map(lambda x: "TBD" if pd.isna(x) else f"{float(x):.3g}")
    latex[
        ["regime", "display", "eld", "fid", "lpips_var", "recall", *DOWNSTREAM_COLUMNS]
    ].to_csv(args.output_dir / "dino_metric_landscape_latex_rows.csv", index=False)
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.output_dir / 'dino_metric_landscape_latex_rows.csv'}")


if __name__ == "__main__":
    main()
