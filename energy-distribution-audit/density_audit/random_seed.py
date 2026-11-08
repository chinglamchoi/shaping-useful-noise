from __future__ import annotations

import secrets
import random

import numpy as np


def resolve_seed(seed: int | None) -> int:
    """Return an explicit seed, generating one when the caller omits it."""
    resolved = int(seed) if seed is not None else secrets.randbelow(2**31)
    print(f"seed={resolved}", flush=True)
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    print(secrets.randbelow(2**31))


if __name__ == "__main__":
    main()
