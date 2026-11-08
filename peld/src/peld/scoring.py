from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._core import ELDResult, PELD, ScoreKind, _as_numpy


def _default_batch_input(batch: Any) -> Any:
    if isinstance(batch, dict):
        for key in ("image", "images", "pixel_values", "input"):
            if key in batch:
                return batch[key]
        raise ValueError("Dictionary batches need image, images, pixel_values, or input.")
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Encountered an empty batch.")
        return batch[0]
    return batch


def collect_scores(
    scorer: Callable[[Any], Any],
    batches: Iterable[Any],
    *,
    get_input: Callable[[Any], Any] | None = None,
) -> NDArray[np.float64]:
    """Collect one scalar density score per image from an iterable of batches.

    This helper has no PyTorch dependency. A scorer may return NumPy arrays,
    pandas values, or CPU/GPU tensors supporting ``detach().cpu().numpy()``.
    """
    select = _default_batch_input if get_input is None else get_input
    pieces = []
    for batch in batches:
        scores = _as_numpy(scorer(select(batch)))
        pieces.append(scores)
    if not pieces:
        raise ValueError("The batch iterable produced no scores.")
    return np.concatenate(pieces)


def dataset_eld(
    scorer: Callable[[Any], Any],
    reference_batches: Iterable[Any],
    candidate_batches: Iterable[Any],
    *,
    score_kind: ScoreKind = "energy",
    get_input: Callable[[Any], Any] | None = None,
) -> ELDResult:
    """Score two batched image datasets and return raw ELD and pELD."""
    reference = collect_scores(scorer, reference_batches, get_input=get_input)
    candidate = collect_scores(scorer, candidate_batches, get_input=get_input)
    return PELD(reference, score_kind=score_kind).both(candidate)
