from __future__ import annotations

import argparse
import hashlib
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from density_audit.random_seed import resolve_seed


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DINO_DEFAULT = [
    ("FractalDB", "data/modern_synthetic/fractaldb/train/synthetic"),
    ("Perlin noise", "data/modern_synthetic/perlin_noise/train/synthetic"),
    ("Gaussian noise", "data/modern_synthetic/gaussian_noise/train/synthetic"),
    ("high-CFG diffusion", "data/modern_synthetic/high_cfg_diffusion/train/synthetic"),
    ("StyleGAN3", "data/modern_synthetic/stylegan3/train/synthetic"),
    ("BigGAN", "data/modern_synthetic/biggan/train/synthetic"),
    ("diffusion mid CFG", "data/modern_synthetic/diffusion_mid_cfg/train/synthetic"),
    ("diffusion low CFG", "data/modern_synthetic/diffusion_low_cfg/train/synthetic"),
    ("uniform mixture", "data/modern_synthetic/uniform_mixture/train/synthetic"),
]

L2S_DEFAULT = [
    ("dead leaves squares", "external/learning_with_noise/images/small_scale/dead_leaves-squares"),
    ("dead leaves oriented", "external/learning_with_noise/images/small_scale/dead_leaves-oriented"),
    ("dead leaves textures", "external/learning_with_noise/images/small_scale/dead_leaves-textures"),
    ("spectrum", "external/learning_with_noise/images/small_scale/stat-spectrum"),
    ("spectrum+color+WMM", "external/learning_with_noise/images/small_scale/stat-spectrum_color_wmm"),
    ("StyleGAN high-freq.", "external/learning_with_noise/images/small_scale/stylegan-highfreq"),
    ("feature-vis random", "external/learning_with_noise/images/small_scale/feature_vis-random"),
    ("feature-vis dead leaves", "external/learning_with_noise/images/small_scale/feature_vis-dead_leaves"),
]


def stable_sample(paths: list[Path], k: int, seed: int, key: str) -> list[Path]:
    scored = []
    for p in paths:
        h = hashlib.sha1(f"{seed}:{key}:{p}".encode()).hexdigest()
        scored.append((h, p))
    return [p for _, p in sorted(scored)[: min(k, len(scored))]]


def find_images(root: Path, max_scan: int | None = None) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            out.append(p)
            if max_scan is not None and len(out) >= max_scan:
                break
    return sorted(out)


def read_dino_rows(manifest: Path | None) -> list[tuple[str, Path]]:
    if manifest is None or not manifest.exists():
        return [(name, Path(path)) for name, path in DINO_DEFAULT]
    df = pd.read_csv(manifest)
    if "include" in df.columns:
        keep = df["include"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
        df = df[keep]
    rows = []
    for _, row in df.iterrows():
        display = str(row.get("display", row.get("dataset", "")))
        image_dir = str(row.get("image_dir", "")).strip()
        if image_dir:
            rows.append((display, Path(image_dir)))
    return rows


def resolve_l2s_rows(root: Path) -> list[tuple[str, Path]]:
    rows = []
    for name, rel in L2S_DEFAULT:
        p = Path(rel)
        if p.exists():
            rows.append((name, p))
            continue
        candidates = [
            root / p.name,
            Path("external/learning_with_noise/data/small_scale") / p.name,
            Path("external/learning_with_noise/images/small_scale") / p.name,
        ]
        rows.append((name, next((c for c in candidates if c.exists()), candidates[0])))
    return rows


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def center_crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def load_tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = center_crop_square(im)
        return im.resize((size, size), Image.Resampling.LANCZOS)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fnt) -> None:
    max_chars = max(9, width // 10)
    wrap_text = text.replace("+", "+ ")
    lines = textwrap.wrap(wrap_text, width=max_chars, break_long_words=False, break_on_hyphens=False)
    lines = [line.replace("+ ", "+") for line in lines]
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    if len(lines) == 2 and len(lines[1]) > max_chars + 2:
        lines[1] = lines[1][: max_chars + 1] + "…"
    line_h = int(fnt.size * 1.08) if hasattr(fnt, "size") else 15
    y = xy[1] - (len(lines) - 1) * line_h // 2
    for i, line in enumerate(lines):
        draw.text((xy[0], y + i * line_h), line, fill=(20, 20, 20), font=fnt)


def make_grid(
    rows: list[tuple[str, Path]],
    output: Path,
    samples_per_dataset: int,
    tile: int,
    gap: int,
    label_w: int,
    seed: int,
    max_scan: int | None,
    title: str,
) -> None:
    label_font = font(18, bold=True)
    title_font = font(28, bold=True)
    small_font = font(13)
    valid_rows: list[tuple[str, Path, list[Path]]] = []
    for label, root in rows:
        paths = find_images(root, max_scan=max_scan)
        chosen = stable_sample(paths, samples_per_dataset, seed, label)
        if chosen:
            valid_rows.append((label, root, chosen))

    if not valid_rows:
        raise RuntimeError(
            "No images found for any requested dataset. For Learning-to-See, try "
            "--l2s-root external/learning_with_noise/data/small_scale."
        )

    title_h = 34
    row_h = tile + gap
    w = label_w + samples_per_dataset * tile + (samples_per_dataset - 1) * gap
    h = title_h + len(valid_rows) * row_h + gap
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 2), title, fill=(0, 0, 0), font=title_font)

    y = title_h
    for label, root, paths in valid_rows:
        draw_label(draw, (gap, y + tile // 2 - 2), label, label_w - gap, label_font)
        x = label_w
        for p in paths:
            try:
                tile_im = load_tile(p, tile)
            except Exception:
                continue
            canvas.paste(tile_im, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(230, 230, 230), width=1)
            x += tile + gap
        y += row_h

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    pdf_path = output.with_suffix(".pdf")
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"wrote {output}")
    print(f"wrote {pdf_path}")
    print(f"rows: {len(valid_rows)}")
    for label, root, paths in valid_rows:
        print(f"{label:24s} {len(paths):2d} samples from {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create appendix contact sheets of synthetic dataset samples.")
    parser.add_argument("--which", choices=["dino", "l2s", "both"], default="both")
    parser.add_argument("--dino-manifest", type=Path, default=Path("configs/dino_sources.csv"))
    parser.add_argument("--l2s-root", type=Path, default=Path("external/learning_with_noise/images/small_scale"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--tile", type=int, default=120)
    parser.add_argument("--gap", type=int, default=7)
    parser.add_argument("--label-width", type=int, default=178)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    if args.which in {"dino", "both"}:
        make_grid(
            read_dino_rows(args.dino_manifest),
            args.output_dir / "appendix_dino_synthetic_samples.png",
            args.samples,
            args.tile,
            args.gap,
            args.label_width,
            args.seed,
            args.max_scan,
            "DINO/ViT-S synthetic pretraining sources",
        )
    if args.which in {"l2s", "both"}:
        make_grid(
            resolve_l2s_rows(args.l2s_root),
            args.output_dir / "appendix_l2s_synthetic_samples.png",
            args.samples,
            args.tile,
            args.gap,
            args.label_width,
            args.seed,
            args.max_scan,
            "Learning-to-See synthetic pretraining sources",
        )


if __name__ == "__main__":
    main()
