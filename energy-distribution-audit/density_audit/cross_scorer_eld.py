from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wasserstein_distance

from density_audit.random_seed import resolve_seed


DATASET_ORDER = [
    "fractaldb",
    "perlin_noise",
    "gaussian_noise",
    "high_cfg_diffusion",
    "stylegan3",
    "biggan",
    "diffusion_mid_cfg",
    "diffusion_low_cfg",
    "uniform_mixture",
]

DISPLAY = {
    "fractaldb": "FractalDB",
    "perlin_noise": "Perlin noise",
    "gaussian_noise": "Gaussian noise",
    "high_cfg_diffusion": "Diffusion, high CFG",
    "stylegan3": "StyleGAN3",
    "biggan": "BigGAN",
    "diffusion_mid_cfg": "Diffusion, mid CFG",
    "diffusion_low_cfg": "Diffusion, low CFG",
    "uniform_mixture": "Uniform mixture",
    "imagenet": "ImageNet",
    "ImageNet": "ImageNet",
}

ALIASES = {
    "FractalDB": "fractaldb",
    "Perlin noise": "perlin_noise",
    "Gaussian noise": "gaussian_noise",
    "Diffusion, high CFG": "high_cfg_diffusion",
    "high-CFG diffusion": "high_cfg_diffusion",
    "high_cfg_diffusion": "high_cfg_diffusion",
    "StyleGAN3": "stylegan3",
    "BigGAN": "biggan",
    "Diffusion, mid CFG": "diffusion_mid_cfg",
    "diffusion (mid CFG)": "diffusion_mid_cfg",
    "Diffusion, low CFG": "diffusion_low_cfg",
    "diffusion (low CFG)": "diffusion_low_cfg",
    "Uniform mixture": "uniform_mixture",
    "uniform mixture": "uniform_mixture",
    "imagenet": "imagenet",
    "ImageNet": "imagenet",
}

DEFAULT_DOWNSTREAM = {
    "fractaldb": (23.91, 56.66, 18.70, 10.30),
    "perlin_noise": (21.53, 55.38, 16.79, 9.11),
    "gaussian_noise": (20.26, 52.26, 15.53, 5.01),
    "high_cfg_diffusion": (28.18, 59.34, 21.41, 10.84),
    "stylegan3": (24.71, 57.49, 18.34, 10.59),
    "biggan": (28.36, 58.97, 18.10, 10.35),
    "diffusion_mid_cfg": (29.17, 59.57, 21.44, 12.35),
    "diffusion_low_cfg": (28.67, 59.60, 21.38, 11.99),
    "uniform_mixture": (27.45, 57.61, 20.80, 11.07),
}

DOWNSTREAM_COLUMNS = ["in1k_linear", "in1k_ft", "in_c", "in_a"]


@dataclass(frozen=True)
class ScoreSpec:
    scorer: str
    path: Path
    score_column: str | None = None
    score_kind: str = "auto"


def canonical_dataset(name: object) -> str:
    text = str(name)
    return ALIASES.get(text, text)


def display_name(dataset: str) -> str:
    return DISPLAY.get(dataset, dataset)


def parse_score_spec(text: str) -> ScoreSpec:
    # Forms:
    #   scorer=/path/to/dir
    #   scorer=/path/to/file.csv:column
    #   scorer=/path/to/file.csv:column:logp
    if "=" not in text:
        raise ValueError(f"Score spec must look like scorer=path[:column[:kind]], got {text!r}")
    scorer, rest = text.split("=", 1)
    parts = rest.split(":")
    path = Path(parts[0])
    column = parts[1] if len(parts) >= 2 and parts[1] else None
    kind = parts[2] if len(parts) >= 3 and parts[2] else "auto"
    return ScoreSpec(scorer=scorer, path=path, score_column=column, score_kind=kind)


def choose_score_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Requested score column {requested!r} not in columns {list(frame.columns)}")
        return requested
    for candidate in ["raw_score", "energy", "nll", "bpd", "total_bpd", "score", "logp_db_dim"]:
        if candidate in frame.columns:
            return candidate
    numeric = [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and c not in {"epoch", "label", "target"}
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not infer score column from {list(frame.columns)}")


def infer_score_kind(column: str, requested: str) -> str:
    if requested != "auto":
        if requested not in {"energy", "lower", "logp", "higher"}:
            raise ValueError("--score kind must be one of auto, energy/lower, logp/higher")
        return requested
    if column in {"logp", "logp_db_dim", "log_likelihood", "ll"}:
        return "logp"
    return "energy"


def scores_to_energy(scores: np.ndarray, column: str, kind: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if kind in {"logp", "higher"}:
        values = -values
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"Found {bad} non-finite scores in {column}")
    return values


def read_one_csv(path: Path, spec: ScoreSpec, dataset_from_filename: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    column = choose_score_column(frame, spec.score_column)
    kind = infer_score_kind(column, spec.score_kind)

    out = pd.DataFrame()
    if "image_id" in frame.columns:
        out["image_id"] = frame["image_id"].astype(str)
    elif "path" in frame.columns:
        out["image_id"] = frame["path"].astype(str)
    else:
        out["image_id"] = [f"{path.stem}_{i:08d}" for i in range(len(frame))]

    if "dataset" in frame.columns:
        out["dataset"] = frame["dataset"].map(canonical_dataset)
    elif dataset_from_filename is not None:
        out["dataset"] = canonical_dataset(dataset_from_filename)
    else:
        raise ValueError(f"{path} needs a dataset column, or use scores_<dataset>.csv naming.")

    if "split" in frame.columns:
        out["split"] = frame["split"].astype(str)
    else:
        out["split"] = "evaluation"

    out["scorer"] = spec.scorer
    out["raw_score"] = pd.to_numeric(frame[column], errors="raise")
    out["energy"] = scores_to_energy(out["raw_score"].to_numpy(), column, kind)
    out["score_column"] = column
    out["score_kind"] = kind
    out["source_file"] = str(path)
    return out


def read_score_spec(spec: ScoreSpec) -> pd.DataFrame:
    if spec.path.is_dir():
        pieces = []
        for csv_path in sorted(spec.path.glob("scores_*.csv")):
            name = csv_path.stem.removeprefix("scores_")
            pieces.append(read_one_csv(csv_path, spec, dataset_from_filename=name))
        if not pieces:
            raise FileNotFoundError(f"No scores_*.csv files found in {spec.path}")
        return pd.concat(pieces, ignore_index=True)
    return read_one_csv(spec.path, spec)


def empirical_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    pct = (0.5 * (left + right) + 0.5) / len(reference)
    eps = 0.5 / len(reference)
    return np.clip(pct, eps, 1.0 - eps)


def uniform_w1(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    target = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return float(np.mean(np.abs(values - target)))


def peld_from_scores(reference_energy: np.ndarray, candidate_energy: np.ndarray) -> float:
    return uniform_w1(empirical_percentiles(reference_energy, candidate_energy))


def bootstrap_peld(
    reference_energy: np.ndarray,
    candidate_energy: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = peld_from_scores(reference_energy, candidate_energy)
    if n_bootstrap <= 0:
        return estimate, math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        ref = rng.choice(reference_energy, size=len(reference_energy), replace=True)
        cand = rng.choice(candidate_energy, size=len(candidate_energy), replace=True)
        values[idx] = peld_from_scores(ref, cand)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def load_downstream(path: Path | None) -> pd.DataFrame:
    if path is None:
        rows = []
        for dataset, values in DEFAULT_DOWNSTREAM.items():
            rows.append({"dataset": dataset, **dict(zip(DOWNSTREAM_COLUMNS, values))})
        return pd.DataFrame(rows)
    frame = pd.read_csv(path)
    rename = {
        "display": "display",
        "IN-1K Lin.": "in1k_linear",
        "IN-1K FT": "in1k_ft",
        "IN-C": "in_c",
        "IN-A": "in_a",
        "imagenet_linear": "in1k_linear",
        "imagenet_ft": "in1k_ft",
        "imagenet_c": "in_c",
        "imagenet_a": "in_a",
    }
    frame = frame.rename(columns=rename).copy()
    if "dataset" not in frame.columns and "display" in frame.columns:
        frame["dataset"] = frame["display"]
    frame["dataset"] = frame["dataset"].map(canonical_dataset)
    missing = set(DOWNSTREAM_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing downstream columns: {sorted(missing)}")
    return frame[["dataset", *DOWNSTREAM_COLUMNS]]


def subset_scores(frame: pd.DataFrame, dataset: str, split: str | None, limit: int | None, seed: int) -> np.ndarray:
    rows = frame[frame["dataset"] == dataset]
    if split is not None and "split" in rows.columns:
        split_rows = rows[rows["split"] == split]
        if len(split_rows):
            rows = split_rows
    if rows.empty:
        raise ValueError(f"No scores for dataset={dataset!r}, split={split!r}")
    values = rows["energy"].to_numpy(np.float64)
    if limit is not None and len(values) > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(values), size=limit, replace=False)
        values = values[idx]
    return values


def build_tables(
    scores: pd.DataFrame,
    downstream: pd.DataFrame,
    real_dataset: str,
    real_split: str | None,
    synthetic_split: str | None,
    limit_real: int | None,
    limit_synthetic: int | None,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    boot_rows = []
    validation_rows = []

    for scorer, scorer_frame in scores.groupby("scorer", sort=False):
        ref = subset_scores(scorer_frame, real_dataset, real_split, limit_real, seed)
        validation_rows.append(
            {
                "scorer": scorer,
                "dataset": real_dataset,
                "n": len(ref),
                "finite": int(np.isfinite(ref).sum()),
                "min": float(np.min(ref)),
                "max": float(np.max(ref)),
                "mean": float(np.mean(ref)),
                "median": float(np.median(ref)),
                "std": float(np.std(ref)),
            }
        )
        for dataset in DATASET_ORDER:
            cand = subset_scores(scorer_frame, dataset, synthetic_split, limit_synthetic, seed)
            estimate, lo, hi = bootstrap_peld(ref, cand, bootstrap, seed + hash((scorer, dataset)) % 1_000_000)
            rows.append(
                {
                    "dataset": dataset,
                    "display": display_name(dataset),
                    "scorer": scorer,
                    "peld": estimate,
                    "peld_ci_low": lo,
                    "peld_ci_high": hi,
                    "n_reference": len(ref),
                    "n_candidate": len(cand),
                }
            )
            boot_rows.append(
                {
                    "dataset": dataset,
                    "display": display_name(dataset),
                    "scorer": scorer,
                    "peld": estimate,
                    "peld_ci_low": lo,
                    "peld_ci_high": hi,
                }
            )
            validation_rows.append(
                {
                    "scorer": scorer,
                    "dataset": dataset,
                    "n": len(cand),
                    "finite": int(np.isfinite(cand).sum()),
                    "min": float(np.min(cand)),
                    "max": float(np.max(cand)),
                    "mean": float(np.mean(cand)),
                    "median": float(np.median(cand)),
                    "std": float(np.std(cand)),
                }
            )

    per_dataset = pd.DataFrame(rows)
    pivot = per_dataset.pivot(index="dataset", columns="scorer", values="peld").reindex(DATASET_ORDER)
    downstream = downstream.set_index("dataset").reindex(DATASET_ORDER)

    scorer_rows = []
    scorer_names = list(per_dataset["scorer"].drop_duplicates())
    guth_name = "guth" if "guth" in scorer_names else scorer_names[0]
    guth = pivot[guth_name].to_numpy(np.float64)
    for scorer in scorer_names:
        vals = pivot[scorer].to_numpy(np.float64)
        row = {
            "scorer": scorer,
            "rank_agreement_with_guth_spearman": 1.0 if scorer == guth_name else float(spearmanr(guth, vals).statistic),
            "rank_agreement_with_guth_kendall": 1.0 if scorer == guth_name else float(kendalltau(guth, vals).statistic),
        }
        for column in DOWNSTREAM_COLUMNS:
            target = downstream[column].to_numpy(np.float64)
            row[f"{column}_spearman"] = float(spearmanr(vals, target).statistic)
        scorer_rows.append(row)

    return per_dataset, pd.DataFrame(scorer_rows), pd.DataFrame(boot_rows), pd.DataFrame(validation_rows)


def format_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def markdown_tables(per_dataset: pd.DataFrame, summary: pd.DataFrame) -> str:
    p = per_dataset.pivot(index="dataset", columns="scorer", values="peld").reindex(DATASET_ORDER)
    p.insert(0, "Synthetic dataset", [display_name(d) for d in p.index])
    p = p.reset_index(drop=True)
    p.columns = ["Synthetic dataset", *[f"{c} pELD" for c in p.columns[1:]]]

    lines = []
    lines.append("### Per-dataset pELD")
    lines.append("")
    lines.append(p.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("### Ranking and downstream agreement")
    lines.append("")
    renamed = summary.rename(
        columns={
            "scorer": "Density scorer",
            "rank_agreement_with_guth_spearman": "Rank agreement with Guth rho",
            "in1k_linear_spearman": "IN-1K Lin. rho",
            "in1k_ft_spearman": "IN-1K FT rho",
            "in_c_spearman": "IN-C rho",
            "in_a_spearman": "IN-A rho",
        }
    )
    keep = [
        "Density scorer",
        "Rank agreement with Guth rho",
        "IN-1K Lin. rho",
        "IN-1K FT rho",
        "IN-C rho",
        "IN-A rho",
    ]
    lines.append(renamed[keep].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("pELD is computed with a separate empirical ImageNet reference CDF for each scorer; raw likelihood scales are not compared across model families.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-scorer pELD analysis for rebuttal tables.")
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Score source as scorer=path[:score_column[:kind]]. kind is energy/lower or logp/higher.",
    )
    parser.add_argument("--downstream-csv", type=Path, default=None)
    parser.add_argument("--real-dataset", default="imagenet")
    parser.add_argument("--real-split", default=None, help="Optional split to use for ImageNet CDF calibration.")
    parser.add_argument("--synthetic-split", default=None)
    parser.add_argument("--limit-real", type=int, default=None)
    parser.add_argument("--limit-synthetic", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cross_scorer_eld"))
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    start = time.time()
    specs = [parse_score_spec(text) for text in args.score]
    score_frames = [read_score_spec(spec) for spec in specs]
    scores = pd.concat(score_frames, ignore_index=True)
    scores["dataset"] = scores["dataset"].map(canonical_dataset)
    downstream = load_downstream(args.downstream_csv)

    per_dataset, summary, bootstrap, validation = build_tables(
        scores=scores,
        downstream=downstream,
        real_dataset=canonical_dataset(args.real_dataset),
        real_split=args.real_split,
        synthetic_split=args.synthetic_split,
        limit_real=args.limit_real,
        limit_synthetic=args.limit_synthetic,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(args.output_dir / "per_dataset_peld.csv", index=False)
    summary.to_csv(args.output_dir / "scorer_summary.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_intervals.csv", index=False)
    validation.to_csv(args.output_dir / "validation_score_summary.csv", index=False)
    (args.output_dir / "rebuttal_tables.md").write_text(markdown_tables(per_dataset, summary))
    metadata = {
        "score_specs": [spec.__dict__ | {"path": str(spec.path)} for spec in specs],
        "limit_real": args.limit_real,
        "limit_synthetic": args.limit_synthetic,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "runtime_seconds": time.time() - start,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir / 'rebuttal_tables.md'}", flush=True)


if __name__ == "__main__":
    main()
