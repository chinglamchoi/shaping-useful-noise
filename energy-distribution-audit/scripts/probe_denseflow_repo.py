from __future__ import annotations

import argparse
import ast
from pathlib import Path


KEYWORDS = [
    "log_prob",
    "loglik",
    "likelihood",
    "bits",
    "bpd",
    "kbs",
    "DataLoader",
    "test_loader",
    "val_loader",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DenseFlow eval_loglik.py for per-image scoring patch points.")
    parser.add_argument("--denseflow-repo", type=Path, required=True)
    args = parser.parse_args()

    eval_path = args.denseflow_repo / "experiments" / "image" / "eval_loglik.py"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    candidates = [eval_path]
    candidates.extend(sorted((args.denseflow_repo / "denseflow").rglob("*.py")))

    for path in candidates:
        source = path.read_text()
        interesting = (
            path == eval_path
            or "dataset_iwbo_bpd" in source
            or "dataset_elbo_bpd" in source
            or any(k in source for k in ["loglik", "bpd", "bits_per_dim"])
        )
        if not interesting:
            continue

        print(f"\nfile: {path}")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            print(f"  could not parse AST: {exc}")
            continue

        print("functions/classes:")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (
                    node.name in {"dataset_iwbo_bpd", "dataset_elbo_bpd"}
                    or path == eval_path
                    or any(k in node.name for k in ["bpd", "log", "lik"])
                ):
                    print(f"  line {node.lineno:4d}: {type(node).__name__} {node.name}")

        print("keyword lines:")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if any(k in line for k in KEYWORDS):
                print(f"{lineno:4d}: {line}")

    print(
        "\nWhat to look for:\n"
        "  1. the tensor containing per-image log likelihood or bpd before it is averaged;\n"
        "  2. the test/eval DataLoader loop;\n"
        "  3. whether batches include paths or only images/labels.\n"
        "Patch that loop to write one row per input image with image_id/canonical_path,bpd.\n"
    )


if __name__ == "__main__":
    main()
