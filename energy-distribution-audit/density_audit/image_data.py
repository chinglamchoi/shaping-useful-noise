from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    index: int
    path: Path


def find_images(root: Path) -> list[Path]:
    """Return image files recursively, in deterministic order."""
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Expected an image file or directory, got {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path, image_size: int, grayscale: bool) -> "torch.Tensor":
    """Load an image as a CHW float tensor in [0, 1]."""
    import torch

    mode = "L" if grayscale else "RGB"
    with Image.open(path) as im:
        im = im.convert(mode)
        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    if grayscale:
        arr = arr[None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def iter_image_batches(
    files: Sequence[Path],
    batch_size: int,
    image_size: int,
    grayscale: bool,
) -> Iterator[tuple[list[ImageRecord], torch.Tensor]]:
    import torch

    records: list[ImageRecord] = []
    tensors: list[torch.Tensor] = []
    for i, path in enumerate(files):
        records.append(ImageRecord(index=i, path=path))
        tensors.append(load_image(path, image_size=image_size, grayscale=grayscale))
        if len(tensors) == batch_size:
            yield records, torch.stack(tensors, dim=0)
            records, tensors = [], []
    if tensors:
        yield records, torch.stack(tensors, dim=0)
