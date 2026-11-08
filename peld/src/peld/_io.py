from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._core import _as_numpy


def _choose_name(names: tuple[str, ...], requested: str | None, path: Path) -> str:
    if requested is not None:
        if requested not in names:
            raise ValueError(f"Column/key {requested!r} not found in {path}; available: {list(names)}")
        return requested
    preferred = [name for name in ("energy", "logp", "log_probability", "score", "scores") if name in names]
    if len(preferred) == 1:
        return preferred[0]
    if len(names) == 1:
        return names[0]
    raise ValueError(f"{path} contains multiple arrays/columns; choose one explicitly from {list(names)}")


def load_scores(path: str | Path, *, column: str | None = None, key: str | None = None) -> NDArray[np.float64]:
    """Load one-dimensional scores from NPY, NPZ, CSV, TSV, or text."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return _as_numpy(np.load(source, allow_pickle=False))
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            selected = _choose_name(tuple(archive.files), key or column, source)
            return _as_numpy(archive[selected])
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        table = np.genfromtxt(source, delimiter=delimiter, names=True, encoding="utf-8")
        names = table.dtype.names
        if not names:
            return _as_numpy(np.loadtxt(source, delimiter=delimiter))
        selected = _choose_name(names, column, source)
        return _as_numpy(table[selected])
    return _as_numpy(np.loadtxt(source))
