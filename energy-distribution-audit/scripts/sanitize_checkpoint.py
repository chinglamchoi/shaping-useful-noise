#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


SENSITIVE_KEY = re.compile(
    r"(?:^|[_.-])(?:api[_-]?key|password|passwd|secret|credential|access[_-]?token|auth[_-]?token|account|qos|partition)(?:$|[_.-])",
    re.I,
)
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
REMOTE = re.compile(r"[A-Z0-9._-]+@[A-Z0-9._-]+:", re.I)


def redact_string(value: str) -> str:
    value = EMAIL.sub("<redacted-email>", value)
    value = REMOTE.sub("<redacted-remote>:", value)
    if value.startswith("/"):
        return f"<redacted-path>/{Path(value).name}"
    return value


def sanitize(value, key: str = ""):
    if SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Path):
        return Path("<redacted-path>") / value.name
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, dict):
        return {k: sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize(item) for item in value)
    return value


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a PyTorch checkpoint while redacting path and credential metadata.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = sanitize(torch_load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
