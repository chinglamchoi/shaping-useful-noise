from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from density_audit.cross_scorer_eld import DEFAULT_DOWNSTREAM, DISPLAY, canonical_dataset


DEFAULT_METRICS = {
    "fractaldb": {"peld": 0.426, "fid": 1254, "lpips": 0.260, "recall": 0.018, "family": "noise_like"},
    "perlin_noise": {"peld": 0.427, "fid": 1922, "lpips": 0.208, "recall": 0.000, "family": "noise_like"},
    "gaussian_noise": {"peld": 0.500, "fid": 2145, "lpips": 0.237, "recall": 0.000, "family": "noise_like"},
    "high_cfg_diffusion": {"peld": 0.138, "fid": 481, "lpips": 0.342, "recall": 0.020, "family": "diffusion"},
    "stylegan3": {"peld": 0.309, "fid": 1148, "lpips": 0.305, "recall": 0.001, "family": "gan"},
    "biggan": {"peld": 0.112, "fid": 566, "lpips": 0.303, "recall": 0.101, "family": "gan"},
    "diffusion_mid_cfg": {"peld": 0.088, "fid": 459, "lpips": 0.338, "recall": 0.030, "family": "diffusion"},
    "diffusion_low_cfg": {"peld": 0.153, "fid": 1129, "lpips": 0.297, "recall": 0.022, "family": "diffusion"},
    "uniform_mixture": {"peld": 0.112, "fid": 616, "lpips": 0.294, "recall": 0.023, "family": "mixture"},
}

L2S_METRICS = {
    "dead_leaves_squares": {"display": "dead-leaves squares", "family": "dead_leaves", "peld": 0.393, "fid": 1480, "lpips": 0.316, "recall": 0.005, "in_lin": 38.60, "in_ft": 67.52, "in_c": 27.23, "in_a": 3.24},
    "dead_leaves_oriented": {"display": "dead-leaves oriented", "family": "dead_leaves", "peld": 0.319, "fid": 1390, "lpips": 0.299, "recall": 0.035, "in_lin": 40.90, "in_ft": 68.48, "in_c": 25.12, "in_a": 3.97},
    "dead_leaves_mixed": {"display": "dead-leaves mixed", "family": "dead_leaves", "peld": 0.310, "fid": 1310, "lpips": 0.301, "recall": 0.074, "in_lin": 40.84, "in_ft": 68.86, "in_c": 25.00, "in_a": 3.82},
    "dead_leaves_textures": {"display": "dead-leaves textures", "family": "dead_leaves", "peld": 0.261, "fid": 1340, "lpips": 0.241, "recall": 0.084, "in_lin": 41.00, "in_ft": 68.20, "in_c": 27.93, "in_a": 3.83},
    "spectrum": {"display": "spectrum", "family": "statistical", "peld": 0.314, "fid": 1560, "lpips": 0.159, "recall": 0.003, "in_lin": 30.70, "in_ft": 62.20, "in_c": 24.57, "in_a": 3.68},
    "wmm": {"display": "WMM", "family": "statistical", "peld": 0.172, "fid": 1520, "lpips": 0.152, "recall": 0.003, "in_lin": 34.08, "in_ft": 63.30, "in_c": 16.90, "in_a": 3.53},
    "spectrum_color": {"display": "spectrum+color", "family": "statistical", "peld": 0.251, "fid": 1290, "lpips": 0.258, "recall": 0.001, "in_lin": 37.88, "in_ft": 65.18, "in_c": 27.82, "in_a": 4.12},
    "spectrum_color_wmm": {"display": "spectrum+color+WMM", "family": "statistical", "peld": 0.049, "fid": 1210, "lpips": 0.228, "recall": 0.010, "in_lin": 38.84, "in_ft": 65.62, "in_c": 30.28, "in_a": 4.56},
    "stylegan_random": {"display": "StyleGAN random", "family": "stylegan", "peld": 0.479, "fid": 1420, "lpips": 0.279, "recall": 0.000, "in_lin": 35.30, "in_ft": 64.82, "in_c": 29.10, "in_a": 4.41},
    "stylegan_highfreq": {"display": "StyleGAN high-freq.", "family": "stylegan", "peld": 0.085, "fid": 1320, "lpips": 0.268, "recall": 0.002, "in_lin": 39.62, "in_ft": 66.21, "in_c": 29.84, "in_a": 4.38},
    "stylegan_sparse": {"display": "StyleGAN sparse", "family": "stylegan", "peld": 0.278, "fid": 1240, "lpips": 0.305, "recall": 0.001, "in_lin": 38.76, "in_ft": 66.16, "in_c": 27.23, "in_a": 3.24},
    "stylegan_oriented": {"display": "StyleGAN oriented", "family": "stylegan", "peld": 0.355, "fid": 903, "lpips": 0.272, "recall": 0.007, "in_lin": 39.08, "in_ft": 65.84, "in_c": 27.70, "in_a": 3.38},
    "feature_vis_random": {"display": "feature-vis random", "family": "feature_vis", "peld": 0.402, "fid": 1590, "lpips": 0.252, "recall": 0.000, "in_lin": 35.26, "in_ft": 66.22, "in_c": 22.89, "in_a": 2.65},
    "feature_vis_dead_leaves": {"display": "feature-vis dead leaves", "family": "feature_vis", "peld": 0.365, "fid": 1400, "lpips": 0.223, "recall": 0.001, "in_lin": 38.70, "in_ft": 68.00, "in_c": 32.04, "in_a": 4.71},
}

DOWNSTREAM_RENAME = {
    "IN-1K Lin.": "in1k_lin",
    "IN-1K FT": "in1k_ft",
    "IN Lin.": "in_lin",
    "IN FT": "in_ft",
    "IN Lin.%": "in_lin",
    "IN FT%": "in_ft",
    "IN-C": "in_c",
    "IN-A": "in_a",
    "in1k_linear": "in1k_lin",
    "in1k_ft": "in1k_ft",
    "in1k_lin": "in1k_lin",
    "in_lin": "in_lin",
    "in_ft": "in_ft",
    "imagenet_linear": "in1k_lin",
    "imagenet_finetune": "in1k_ft",
    "imagenet100_linear": "in_lin",
    "imagenet100_finetune": "in_ft",
    "in_c": "in_c",
    "in_a": "in_a",
}

DEFAULT_TARGETS = ["in1k_lin", "in1k_ft", "in_c", "in_a"]
TARGET_LABELS = {
    "in1k_lin": "IN-1K Lin.",
    "in1k_ft": "IN-1K FT",
    "in_lin": "IN Lin.",
    "in_ft": "IN FT",
    "in_c": "IN-C",
    "in_a": "IN-A",
}


def builtin_frame(preset: str = "dino") -> pd.DataFrame:
    if preset == "l2s":
        rows = []
        for dataset, values in L2S_METRICS.items():
            rows.append({"dataset": dataset, **values, "eld": values["peld"]})
        return pd.DataFrame(rows)
    rows = []
    for dataset, metrics in DEFAULT_METRICS.items():
        lin, ft, inc, ina = DEFAULT_DOWNSTREAM[dataset]
        rows.append(
            {
                "dataset": dataset,
                "display": DISPLAY[dataset],
                **metrics,
                "eld": metrics["peld"],
                "in1k_lin": lin,
                "in1k_ft": ft,
                "in_c": inc,
                "in_a": ina,
            }
        )
    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns=DOWNSTREAM_RENAME)
    if "dataset" not in frame.columns:
        if "display" not in frame.columns:
            raise ValueError(f"{path} must contain dataset or display")
        frame["dataset"] = frame["display"]
    frame["dataset"] = frame["dataset"].map(canonical_dataset)
    if "display" not in frame.columns:
        frame["display"] = frame["dataset"].map(lambda x: DISPLAY.get(x, x))
    if "peld" not in frame.columns and "eld" in frame.columns:
        frame["peld"] = frame["eld"]
    if "eld" not in frame.columns and "peld" in frame.columns:
        frame["eld"] = frame["peld"]
    if "family" not in frame.columns:
        frame["family"] = frame["dataset"].map(lambda x: DEFAULT_METRICS.get(x, {}).get("family", "unknown"))
    metric_required = {"eld", "peld", "fid", "lpips", "recall"}
    missing = metric_required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 3:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x[ok], y[ok]).statistic)
    return float(spearmanr(x[ok], y[ok]).statistic)


def metric_correlations(frame: pd.DataFrame, metrics: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        row = {"Metric": metric.upper() if metric in {"eld", "fid"} else metric}
        for target in targets:
            label = TARGET_LABELS.get(target, target)
            row[f"{label} Spearman"] = corr(frame[metric], frame[target], "spearman")
            if target == targets[0]:
                row[f"{label} Pearson"] = corr(frame[metric], frame[target], "pearson")
        rows.append(row)
    return pd.DataFrame(rows)


def subset_correlations(frame: pd.DataFrame, metrics: list[str], target: str) -> pd.DataFrame:
    subsets = [("All datasets", frame)]
    families = sorted(str(x) for x in frame["family"].dropna().unique())
    if "noise_like" in families:
        subsets.append(("Excluding noise-like sources", frame[frame["family"] != "noise_like"]))
    if "noise_like" in families and "mixture" in families:
        subsets.append(
            (
                "Excluding noise-like and uniform mixture",
                frame[(frame["family"] != "noise_like") & (frame["family"] != "mixture")],
            )
        )
    for family in families:
        if family == "unknown":
            continue
        subsets.append((f"Leave out {family} family", frame[frame["family"].astype(str) != family]))
    rows = []
    for name, sub in subsets:
        row = {"Evaluation subset": name, "n": len(sub)}
        for metric in metrics:
            if metric in sub.columns:
                row[f"{metric} rho"] = corr(sub[metric], sub[target], "spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rebuttal correlations for DINO metrics and restricted subsets.")
    parser.add_argument("--table", type=Path, default=None, help="Optional CSV with dataset metrics and downstream columns.")
    parser.add_argument("--preset", choices=["dino", "l2s"], default="dino", help="Built-in table to use when --table is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rebuttal_metric_correlations"))
    parser.add_argument("--target", default="in1k_lin", help="Target column for restricted-subset correlations.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target columns for the matched-correlation table. Defaults to available DINO targets, or L2S targets if present.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["eld", "peld", "fid", "lpips", "precision", "recall"],
        help="Metrics to include if present.",
    )
    args = parser.parse_args()

    frame = builtin_frame(args.preset) if args.table is None else read_table(args.table)
    metrics = [m for m in args.metrics if m in frame.columns]
    missing = [m for m in args.metrics if m not in frame.columns]
    if missing:
        print(f"Skipping missing metrics: {', '.join(missing)}", flush=True)

    if args.targets is None:
        candidate_targets = DEFAULT_TARGETS if "in1k_lin" in frame.columns else ["in_lin", "in_ft", "in_c", "in_a"]
        targets = [target for target in candidate_targets if target in frame.columns]
    else:
        targets = args.targets
    missing_targets = [target for target in targets + [args.target] if target not in frame.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {sorted(set(missing_targets))}")

    matched = metric_correlations(frame, metrics, targets)
    subsets = subset_correlations(frame, [m for m in ["eld", "peld", "fid", "recall"] if m in frame.columns], args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "analysis_input.csv", index=False)
    matched.to_csv(args.output_dir / "matched_metric_correlations.csv", index=False)
    subsets.to_csv(args.output_dir / "restricted_subset_correlations.csv", index=False)

    lines = [
        "### Matched Metric Correlations",
        "",
        matched.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"### Restricted Subset Correlations Against {TARGET_LABELS.get(args.target, args.target)}",
        "",
        subsets.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (args.output_dir / "rebuttal_metric_correlations.md").write_text("\n".join(lines))
    print((args.output_dir / "rebuttal_metric_correlations.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
