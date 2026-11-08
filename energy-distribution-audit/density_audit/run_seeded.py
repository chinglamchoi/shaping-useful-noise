from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from density_audit.random_seed import resolve_seed, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Python script after explicitly seeding common RNGs.")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args, passthrough = parser.parse_known_args()

    seed = resolve_seed(args.seed)
    seed_everything(seed)
    sys.argv = [str(args.script), *passthrough]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
