from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from density_audit.rebuttal_metric_correlations import TARGET_LABELS, builtin_frame, read_table


DEFAULT_TARGETS = ["in1k_lin", "in1k_ft", "in_c", "in_a"]


PREDICTOR_LABELS = {
    "fid": "FID",
    "eld": "ELD",
    "peld": "pELD",
}


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return values * 0.0, mean, 1.0
    return (values - mean) / std, mean, std


def design_matrix(frame: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    columns = [np.ones(len(frame), dtype=np.float64)]
    for predictor in predictors:
        z, _, _ = standardize(frame[predictor].to_numpy(np.float64))
        columns.append(z)
    return np.column_stack(columns)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], target: str) -> np.ndarray:
    means = {}
    stds = {}
    train_columns = [np.ones(len(train), dtype=np.float64)]
    test_columns = [np.ones(len(test), dtype=np.float64)]
    for predictor in predictors:
        x_train = train[predictor].to_numpy(np.float64)
        z_train, mean, std = standardize(x_train)
        means[predictor] = mean
        stds[predictor] = std
        train_columns.append(z_train)
        test_columns.append((test[predictor].to_numpy(np.float64) - mean) / std)

    x = np.column_stack(train_columns)
    y = train[target].to_numpy(np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.column_stack(test_columns) @ coef


def loocv(frame: pd.DataFrame, predictors: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    y_true = frame[target].to_numpy(np.float64)
    y_pred = np.empty(len(frame), dtype=np.float64)
    for idx in range(len(frame)):
        train = frame.drop(frame.index[idx])
        test = frame.iloc[[idx]]
        y_pred[idx] = fit_predict(train, test, predictors, target)[0]
    return y_true, y_pred


def residuals_after(frame: pd.DataFrame, predictor: str, target: str) -> np.ndarray:
    x = design_matrix(frame, [predictor])
    y = frame[target].to_numpy(np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coef


def partial_spearman(frame: pd.DataFrame, x_name: str, y_name: str, control_name: str) -> float:
    rx = rankdata(frame[x_name].to_numpy(np.float64))
    ry = rankdata(frame[y_name].to_numpy(np.float64))
    rc = rankdata(frame[control_name].to_numpy(np.float64))

    def residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
        mat = np.column_stack([np.ones(len(control)), control])
        coef, *_ = np.linalg.lstsq(mat, values, rcond=None)
        return values - mat @ coef

    x_resid = residualize(rx, rc)
    y_resid = residualize(ry, rc)
    return float(pearsonr(x_resid, y_resid).statistic)


def pairwise_order_stats(frame: pd.DataFrame, metric: str, target: str) -> dict[str, object]:
    correct = 0
    incorrect = 0
    ties = 0
    mistakes = []
    values = frame[metric].to_numpy(np.float64)
    targets = frame[target].to_numpy(np.float64)
    names = frame["display"].astype(str).tolist()
    for i, j in combinations(range(len(frame)), 2):
        dy = np.sign(targets[i] - targets[j])
        if dy == 0:
            ties += 1
            continue
        # Lower ELD/FID/pELD should imply higher downstream performance.
        dm = -np.sign(values[i] - values[j])
        if dm == 0:
            ties += 1
        elif dm == dy:
            correct += 1
        else:
            incorrect += 1
            better = names[i] if dy > 0 else names[j]
            worse = names[j] if dy > 0 else names[i]
            metric_prefers = names[i] if dm > 0 else names[j]
            mistakes.append(f"{better} > {worse}; {PREDICTOR_LABELS.get(metric, metric)} prefers {metric_prefers}")
    total = correct + incorrect
    return {
        "correct_pairs": correct,
        "incorrect_pairs": incorrect,
        "tied_pairs": ties,
        "pair_accuracy": correct / total if total else np.nan,
        "mistakes": mistakes,
    }


def pairwise_complementarity(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    fid = pairwise_order_stats(frame, "fid", target)
    eld = pairwise_order_stats(frame, "eld", target)

    rows = []
    values_fid = frame["fid"].to_numpy(np.float64)
    values_eld = frame["eld"].to_numpy(np.float64)
    targets = frame[target].to_numpy(np.float64)
    names = frame["display"].astype(str).tolist()
    for i, j in combinations(range(len(frame)), 2):
        dy = np.sign(targets[i] - targets[j])
        if dy == 0:
            continue
        fid_order = -np.sign(values_fid[i] - values_fid[j])
        eld_order = -np.sign(values_eld[i] - values_eld[j])
        fid_correct = fid_order == dy
        eld_correct = eld_order == dy
        if eld_correct and not fid_correct:
            rows.append(
                {
                    "dataset_better": names[i] if dy > 0 else names[j],
                    "dataset_worse": names[j] if dy > 0 else names[i],
                    "target_gap": abs(float(targets[i] - targets[j])),
                    "fid_values": f"{values_fid[i]:.3g} vs {values_fid[j]:.3g}",
                    "eld_values": f"{values_eld[i]:.3g} vs {values_eld[j]:.3g}",
                }
            )
    out = pd.DataFrame(rows).sort_values("target_gap", ascending=False) if rows else pd.DataFrame(rows)
    out.attrs["fid_pair_accuracy"] = fid["pair_accuracy"]
    out.attrs["eld_pair_accuracy"] = eld["pair_accuracy"]
    return out


def signed_metric_rank_correlation(frame: pd.DataFrame, metric: str, target: str) -> float:
    # ELD, pELD, and FID are lower-is-better, so negate them before comparing
    # with higher-is-better downstream performance.
    return float(spearmanr(-frame[metric].to_numpy(np.float64), frame[target].to_numpy(np.float64)).statistic)


def model_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    specs = [
        ("FID", ["fid"]),
        ("ELD", ["eld"]),
        ("FID + ELD", ["fid", "eld"]),
    ]
    rows = []
    for label, predictors in specs:
        y_true, y_pred = loocv(frame, predictors, target)
        pair_stats = pairwise_order_stats(frame, predictors[0], target) if len(predictors) == 1 else None
        rows.append(
            {
                "Predictors": label,
                "n": len(frame),
                "LOOCV MAE ↓": float(np.mean(np.abs(y_true - y_pred))),
                "LOOCV median AE ↓": float(np.median(np.abs(y_true - y_pred))),
                "LOOCV RMSE ↓": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "Predicted rank ρ ↑": float(spearmanr(y_pred, y_true).statistic),
                "Metric rank ρ ↑": signed_metric_rank_correlation(frame, predictors[0], target)
                if len(predictors) == 1
                else np.nan,
                "Pairwise order acc. ↑": pair_stats["pair_accuracy"] if pair_stats is not None else np.nan,
                "Correct pairs": pair_stats["correct_pairs"] if pair_stats is not None else np.nan,
                "Incorrect pairs": pair_stats["incorrect_pairs"] if pair_stats is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def improvement_table(models: pd.DataFrame) -> pd.DataFrame:
    indexed = models.set_index("Predictors")
    rows = []
    for first, second in [("ELD", "FID"), ("FID + ELD", "FID"), ("FID + ELD", "ELD")]:
        if first not in indexed.index or second not in indexed.index:
            continue
        rows.append(
            {
                "Comparison": f"{first} vs {second}",
                "Δ MAE ↓": float(indexed.loc[second, "LOOCV MAE ↓"] - indexed.loc[first, "LOOCV MAE ↓"]),
                "Δ RMSE ↓": float(indexed.loc[second, "LOOCV RMSE ↓"] - indexed.loc[first, "LOOCV RMSE ↓"]),
                "Δ predicted rank ρ ↑": float(indexed.loc[first, "Predicted rank ρ ↑"] - indexed.loc[second, "Predicted rank ρ ↑"]),
            }
        )
    return pd.DataFrame(rows)


def available_targets(frame: pd.DataFrame) -> list[str]:
    if "in1k_lin" in frame.columns or "in1k_ft" in frame.columns:
        return [target for target in DEFAULT_TARGETS if target in frame.columns]
    return [target for target in ["in_lin", "in_ft", "in_c", "in_a"] if target in frame.columns]


def all_target_model_table(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    pieces = []
    for target in targets:
        table = model_table(frame, target)
        table.insert(0, "Target", TARGET_LABELS.get(target, target))
        pieces.append(table)
    return pd.concat(pieces, ignore_index=True)


def subset_model_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    subsets = [("All", frame)]
    for family in sorted(str(x) for x in frame["family"].dropna().unique()):
        if family == "unknown":
            continue
        subsets.append((f"without {family}", frame[frame["family"].astype(str) != family]))

    rows = []
    for subset_name, sub in subsets:
        if len(sub) < 4:
            continue
        table = model_table(sub.reset_index(drop=True), target).set_index("Predictors")
        for predictor in ["FID", "ELD"]:
            if predictor not in table.index:
                continue
            rows.append(
                {
                    "Subset": subset_name,
                    "Predictor": predictor,
                    "n": int(table.loc[predictor, "n"]),
                    "LOOCV MAE ↓": float(table.loc[predictor, "LOOCV MAE ↓"]),
                    "Predicted rank ρ ↑": float(table.loc[predictor, "Predicted rank ρ ↑"]),
                    "Metric rank ρ ↑": float(table.loc[predictor, "Metric rank ρ ↑"]),
                    "Pairwise order acc. ↑": float(table.loc[predictor, "Pairwise order acc. ↑"]),
                }
            )
    return pd.DataFrame(rows)


def diagnostics_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    fid_resid = residuals_after(frame, "fid", target)
    eld_resid = residuals_after(frame, "eld", target)
    return pd.DataFrame(
        [
            {
                "Diagnostic": "partial Spearman: ELD vs target | FID",
                "value": partial_spearman(frame, "eld", target, "fid"),
            },
            {
                "Diagnostic": "residual Spearman: ELD vs residual(target ~ FID)",
                "value": float(spearmanr(frame["eld"], fid_resid).statistic),
            },
            {
                "Diagnostic": "residual Spearman: FID vs residual(target ~ ELD)",
                "value": float(spearmanr(frame["fid"], eld_resid).statistic),
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-n complementarity tests for ELD beyond FID.")
    parser.add_argument("--table", type=Path, default=None, help="Optional CSV with dataset, eld/peld, fid, downstream columns.")
    parser.add_argument("--preset", choices=["dino", "l2s"], default="dino", help="Built-in table to use when --table is omitted.")
    parser.add_argument("--target", default="in1k_lin")
    parser.add_argument("--all-targets", action="store_true", help="Also emit the LOOCV table for every available target.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rebuttal_complementarity"))
    parser.add_argument("--top-pairs", type=int, default=12)
    args = parser.parse_args()

    frame = builtin_frame(args.preset) if args.table is None else read_table(args.table)
    if "eld" not in frame.columns and "peld" in frame.columns:
        frame["eld"] = frame["peld"]
    required = {"display", "eld", "fid", args.target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["eld", "fid", args.target]).reset_index(drop=True)

    models = model_table(frame, args.target)
    improvements = improvement_table(models)
    all_targets = all_target_model_table(frame, available_targets(frame)) if args.all_targets else pd.DataFrame()
    subsets = subset_model_table(frame, args.target)
    diagnostics = diagnostics_table(frame, args.target)
    pairs = pairwise_complementarity(frame, args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "analysis_input.csv", index=False)
    models.to_csv(args.output_dir / "loocv_models.csv", index=False)
    improvements.to_csv(args.output_dir / "loocv_improvements.csv", index=False)
    if not all_targets.empty:
        all_targets.to_csv(args.output_dir / "loocv_models_all_targets.csv", index=False)
    subsets.to_csv(args.output_dir / "loocv_subset_models.csv", index=False)
    diagnostics.to_csv(args.output_dir / "partial_and_residual_correlations.csv", index=False)
    pairs.to_csv(args.output_dir / "eld_correct_fid_wrong_pairs.csv", index=False)

    lines = [
        f"### Complementarity for {TARGET_LABELS.get(args.target, args.target)}",
        "",
        models.to_markdown(index=False, floatfmt=".2f"),
        "",
        "### LOOCV Improvements",
        "",
        improvements.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    if not all_targets.empty:
        lines.extend(
            [
                "### Target-Wise LOOCV",
                "",
                all_targets.to_markdown(index=False, floatfmt=".2f"),
                "",
            ]
        )
    if not subsets.empty:
        lines.extend(
            [
                f"### Restricted-Subset LOOCV Against {TARGET_LABELS.get(args.target, args.target)}",
                "",
                subsets.to_markdown(index=False, floatfmt=".2f"),
                "",
            ]
        )
    lines.extend(
        [
            "### Partial and Residual Tests",
            "",
            diagnostics.to_markdown(index=False, floatfmt=".2f"),
            "",
            "### Pairs ELD Orders Correctly and FID Does Not",
            "",
        ]
    )
    if pairs.empty:
        lines.append("No such pairs for this target.")
    else:
        lines.append(pairs.head(args.top_pairs).to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append(
        "Use the combined model only if its LOOCV error improves over both single-predictor baselines."
    )
    report = "\n".join(lines)
    (args.output_dir / "rebuttal_complementarity.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
