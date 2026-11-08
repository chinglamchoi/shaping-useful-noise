from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._core import PELD
from ._io import load_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peld",
        description="Compute percentile and raw energy-level distribution distances.",
    )
    parser.add_argument("reference", type=Path, help="Reference scores: NPY, NPZ, CSV, TSV, or text.")
    parser.add_argument("candidate", type=Path, help="Candidate scores in a supported format.")
    parser.add_argument("--reference-column", default=None, help="CSV column or NPZ key for reference scores.")
    parser.add_argument("--candidate-column", default=None, help="CSV column or NPZ key for candidate scores.")
    parser.add_argument("--score-kind", choices=["energy", "logp"], default="energy")
    parser.add_argument("--metric", choices=["percentile", "raw", "both"], default="percentile")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--save-reference", type=Path, default=None, help="Save the fitted reference CDF as NPZ.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference = load_scores(args.reference, column=args.reference_column, key=args.reference_column)
    candidate = load_scores(args.candidate, column=args.candidate_column, key=args.candidate_column)
    metric = PELD(reference, score_kind=args.score_kind)
    if args.save_reference is not None:
        metric.save(args.save_reference)

    result = metric.both(candidate)
    payload = result.as_dict()
    if args.metric == "percentile":
        payload = {"percentile_eld": result.percentile}
    elif args.metric == "raw":
        payload = {"raw_eld": result.raw}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        labels = {"percentile_eld": "pELD", "raw_eld": "raw ELD"}
        for key, value in payload.items():
            label = labels.get(key, key.replace("_", " "))
            print(f"{label}: {value}")


if __name__ == "__main__":
    main()
