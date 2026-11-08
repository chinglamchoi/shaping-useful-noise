from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from density_audit.image_data import find_images
from density_audit.random_seed import resolve_seed


def save_uint8_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB").save(path)


def gaussian_image(rng: np.random.Generator, size: int, std: float) -> np.ndarray:
    image = rng.normal(loc=127.5, scale=255.0 * std, size=(size, size, 3))
    return np.clip(image, 0, 255)


def _smooth_noise(rng: np.random.Generator, grid_size: int, channels: int) -> np.ndarray:
    # One-cell border lets bilinear interpolation wrap cleanly at high x/y.
    return rng.random((grid_size + 1, grid_size + 1, channels), dtype=np.float32)


def _resize_bilinear(grid: np.ndarray, size: int) -> np.ndarray:
    grid_h, grid_w, channels = grid.shape
    h = grid_h - 1
    w = grid_w - 1
    ys = np.linspace(0, h, size, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, w, size, endpoint=False, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = grid[y0[:, None], x0[None, :]] * (1 - wx) + grid[y0[:, None], x1[None, :]] * wx
    bottom = grid[y1[:, None], x0[None, :]] * (1 - wx) + grid[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def perlin_like_image(
    rng: np.random.Generator,
    size: int,
    octaves: int,
    persistence: float,
    color: bool,
) -> np.ndarray:
    channels = 3 if color else 1
    field = np.zeros((size, size, channels), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    for octave in range(octaves):
        grid_size = 2 ** (octave + 1)
        field += amplitude * _resize_bilinear(_smooth_noise(rng, grid_size, channels), size)
        total += amplitude
        amplitude *= persistence
    field /= max(total, 1e-8)
    # Random contrast and phase make the set less one-note while preserving the
    # intended smooth low-frequency noise profile.
    contrast = rng.uniform(0.65, 1.35)
    bias = rng.uniform(-0.15, 0.15, size=(1, 1, channels))
    field = np.clip((field - 0.5) * contrast + 0.5 + bias, 0.0, 1.0)
    if not color:
        field = np.repeat(field, 3, axis=2)
    return field * 255.0


def generate_gaussian(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out = args.output_dir / "train"
    for i in range(args.n):
        image = gaussian_image(rng, args.size, args.std)
        save_uint8_image(image, out / f"{i:08d}.png")
        if (i + 1) % args.log_every == 0:
            print(f"gaussian: wrote {i + 1}/{args.n}", flush=True)


def generate_perlin(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out = args.output_dir / "train"
    for i in range(args.n):
        image = perlin_like_image(
            rng,
            size=args.size,
            octaves=args.octaves,
            persistence=args.persistence,
            color=not args.grayscale,
        )
        save_uint8_image(image, out / f"{i:08d}.png")
        if (i + 1) % args.log_every == 0:
            print(f"perlin: wrote {i + 1}/{args.n}", flush=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")
    return mode


def generate_uniform_mixture(args: argparse.Namespace) -> None:
    if not args.components:
        raise ValueError("Pass --components name=path [name=path ...]")
    rng = np.random.default_rng(args.seed)
    components: list[tuple[str, list[Path]]] = []
    for spec in args.components:
        if "=" not in spec:
            raise ValueError(f"Expected component spec name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        paths = find_images(Path(path))
        if not paths:
            raise ValueError(f"No images found for component {name}: {path}")
        components.append((name, paths))

    out = args.output_dir / "train"
    rows = []
    for i in range(args.n):
        comp_idx = i % len(components)
        name, paths = components[comp_idx]
        src = paths[int(rng.integers(0, len(paths)))]
        suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".png"
        dst = out / f"{i:08d}_{name}{suffix}"
        try:
            mode = _link_or_copy(src, dst, args.link_mode)
        except OSError:
            mode = _link_or_copy(src, dst, "copy")
        rows.append({"path": str(dst), "source": str(src), "component": name, "mode": mode})
        if (i + 1) % args.log_every == 0:
            print(f"uniform mixture: wrote {i + 1}/{args.n}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mixture_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "component", "mode"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple modern-table synthetic datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--n", type=int, default=50000)
    common.add_argument("--size", type=int, default=64)
    common.add_argument("--seed", type=int, default=None, help="Run seed; generated and printed when omitted.")
    common.add_argument("--log-every", type=int, default=5000)

    gaussian = sub.add_parser("gaussian", parents=[common])
    gaussian.add_argument("--std", type=float, default=0.25, help="Pixel std as a fraction of 255.")

    perlin = sub.add_parser("perlin", parents=[common])
    perlin.add_argument("--octaves", type=int, default=6)
    perlin.add_argument("--persistence", type=float, default=0.55)
    perlin.add_argument("--grayscale", action="store_true")

    mixture = sub.add_parser("uniform-mixture", parents=[common])
    mixture.add_argument("--components", nargs="+", required=True, help="Component specs: name=path")
    mixture.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    mixture.set_defaults(size=0)

    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    if args.command == "gaussian":
        generate_gaussian(args)
    elif args.command == "perlin":
        generate_perlin(args)
    elif args.command == "uniform-mixture":
        generate_uniform_mixture(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
