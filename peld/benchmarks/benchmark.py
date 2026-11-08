from __future__ import annotations

from time import perf_counter

import numpy as np

import peld


def main() -> None:
    rng = np.random.default_rng()
    reference = rng.normal(size=50_000)
    candidates = [rng.normal(loc=shift, size=50_000) for shift in np.linspace(0, 1, 10)]

    start = perf_counter()
    metric = peld.PELD(reference)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    values = [metric(candidate) for candidate in candidates]
    compare_seconds = perf_counter() - start

    print(f"reference fit: {fit_seconds:.4f}s")
    print(f"10 x 50k comparisons: {compare_seconds:.4f}s")
    print("pELD:", ", ".join(f"{value:.4f}" for value in values))


if __name__ == "__main__":
    main()
