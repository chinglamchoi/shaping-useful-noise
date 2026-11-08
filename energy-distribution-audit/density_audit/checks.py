from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .manifests import BARADAD_SMALL_SCALE_14, MANUSCRIPT_SMALL_SCALE_SCORE_DATASETS, guth_train_experiment_name


def check_score_run(result_dir: Path, min_n: int, expected: list[str]) -> list[str]:
    errors: list[str] = []
    result_dir = result_dir.expanduser()
    if not result_dir.is_dir():
        return [f"missing result dir: {result_dir}"]
    for dataset in expected:
        path = result_dir / f"scores_{dataset}.csv"
        if not path.exists():
            errors.append(f"missing score file: {path}")
            continue
        try:
            df = pd.read_csv(path, usecols=["dataset", "logp_db_dim"])
        except Exception as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if len(df) < min_n:
            errors.append(f"{path}: n={len(df)} < {min_n}")
        names = sorted(df["dataset"].unique())
        if names != [dataset]:
            errors.append(f"{path}: expected dataset column {dataset!r}, got {names}")
        if df["logp_db_dim"].isna().any():
            errors.append(f"{path}: NaN logp values")
    for required in ["summary.csv", "logp_histograms_overlay.png", "logp_histograms_overlay.pdf"]:
        if not (result_dir / required).exists():
            errors.append(f"missing artifact: {result_dir / required}")
    return errors


def check_training_run(model_root: Path, color: bool, expected: list[str]) -> list[str]:
    errors: list[str] = []
    model_root = model_root.expanduser()
    if not model_root.is_dir():
        return [f"missing model root: {model_root}"]
    for dataset in expected:
        exp = model_root / guth_train_experiment_name(dataset, color=color)
        if not exp.is_dir():
            errors.append(f"missing experiment dir: {exp}")
            continue
        for filename in ["args.json", "log.txt", "model.pth.tar"]:
            if not (exp / filename).exists():
                errors.append(f"missing {filename}: {exp}")
        log_path = exp / "log.txt"
        if log_path.exists():
            text = log_path.read_text(errors="replace")[-20000:]
            if "Training loss is NaN" in text:
                errors.append(f"NaN failure in log: {log_path}")
            if "OutOfMemoryError" in text:
                errors.append(f"OOM failure in log: {log_path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/post condition checks for manuscript experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("scores", help="Check histogram/ELD score outputs.")
    score.add_argument("--result-dir", type=Path, required=True)
    score.add_argument("--min-n", type=int, default=50_000)
    score.add_argument("--expected", nargs="*", default=MANUSCRIPT_SMALL_SCALE_SCORE_DATASETS)

    train = sub.add_parser("training", help="Check fitted Guth density model outputs.")
    train.add_argument("--model-root", type=Path, required=True)
    train.add_argument("--color", action="store_true")
    train.add_argument("--expected", nargs="*", default=BARADAD_SMALL_SCALE_14)

    args = parser.parse_args()
    if args.command == "scores":
        errors = check_score_run(args.result_dir, min_n=args.min_n, expected=args.expected)
    else:
        errors = check_training_run(args.model_root, color=args.color, expected=args.expected)

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
