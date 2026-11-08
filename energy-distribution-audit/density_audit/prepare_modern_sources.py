from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from density_audit.generate_modern_datasets import generate_gaussian, generate_perlin, generate_uniform_mixture
from density_audit.image_data import find_images
from density_audit.random_seed import resolve_seed


IMAGENET_PROMPT_TEMPLATES = [
    "a photo of a {label}",
    "a close-up photo of a {label}",
    "a natural image of a {label}",
    "a cropped photo of a {label}",
]


@dataclass(frozen=True)
class ManifestRow:
    regime: str
    dataset: str
    display: str
    image_dir: Path
    include: int
    notes: str = ""


def normalize_label(label: str) -> str:
    return label.split(",")[0].replace("_", " ").strip().lower()


def imagenet_labels() -> list[str]:
    try:
        from torchvision.models import ResNet50_Weights

        labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
        return [normalize_label(x) for x in labels]
    except Exception:
        # Short fallback only for environments where torchvision metadata is
        # unavailable. Users can pass --prompts-file for a full fixed prompt set.
        return [
            "goldfish",
            "tabby cat",
            "golden retriever",
            "school bus",
            "espresso",
            "volcano",
            "baseball",
            "airliner",
            "monarch butterfly",
            "mountain bike",
        ]


def build_prompts(n: int, seed: int, prompts_file: Path | None) -> list[str]:
    if prompts_file is not None:
        prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_file}")
    else:
        labels = imagenet_labels()
        prompts = [template.format(label=label) for label in labels for template in IMAGENET_PROMPT_TEMPLATES]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prompts), size=n, replace=True)
    return [prompts[int(i)] for i in idx]


def next_needed_indices(out_dir: Path, n: int, force: bool) -> list[int]:
    train = out_dir / "train"
    if force and train.exists():
        for path in train.glob("*.png"):
            path.unlink()
    existing = {int(path.stem.split("_")[0]) for path in train.glob("*.png") if path.stem.split("_")[0].isdigit()}
    return [i for i in range(n) if i not in existing]


def save_tensor_image(tensor, path: Path) -> None:
    import torch

    image = tensor.detach().cpu().float()
    if image.ndim == 4:
        raise ValueError("Expected one image tensor, got batch")
    # BigGAN returns [-1, 1], diffusers returns PIL elsewhere.
    if float(image.min()) < -0.05:
        image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def export_fractaldb(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install Hugging Face datasets: pip install datasets") from exc

    needed = next_needed_indices(args.output_dir, args.n, args.force)
    if not needed:
        print(f"FractalDB exists: {args.output_dir / 'train'}")
        return
    ds = load_dataset(args.hf_dataset, split=args.split, streaming=args.streaming)
    train = args.output_dir / "train"
    train.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for i, row in enumerate(ds):
        if i >= args.n:
            break
        if i not in needed:
            continue
        image = row.get(args.image_column)
        if image is None:
            raise ValueError(f"Column {args.image_column!r} not found in row keys {list(row)}")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
        image.save(train / f"{i:08d}.png")
        wrote += 1
        if wrote % args.log_every == 0:
            print(f"fractaldb: wrote {wrote} new images", flush=True)
    print(f"fractaldb: wrote {wrote} new images to {train}")


def sample_biggan(args: argparse.Namespace) -> None:
    try:
        import torch
        from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
    except Exception as exc:
        raise RuntimeError("Install BigGAN dependency: pip install pytorch-pretrained-biggan") from exc

    needed = next_needed_indices(args.output_dir, args.n, args.force)
    if not needed:
        print(f"BigGAN exists: {args.output_dir / 'train'}")
        return

    device = torch.device(args.device)
    model = BigGAN.from_pretrained(args.model).to(device).eval()
    rng = np.random.default_rng(args.seed)
    train = args.output_dir / "train"
    train.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(needed), args.batch_size):
        batch_indices = needed[start : start + args.batch_size]
        classes = rng.integers(0, 1000, size=len(batch_indices)).tolist()
        noise = truncated_noise_sample(
            truncation=args.truncation,
            batch_size=len(batch_indices),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        class_vec = one_hot_from_int(classes, batch_size=len(batch_indices))
        noise_t = torch.from_numpy(noise).to(device)
        class_t = torch.from_numpy(class_vec).to(device)
        with torch.inference_mode():
            images = model(noise_t, class_t, args.truncation)
        for idx, image in zip(batch_indices, images):
            save_tensor_image(image, train / f"{idx:08d}.png")
        print(f"biggan: wrote {min(start + len(batch_indices), len(needed))}/{len(needed)} new images", flush=True)


def sample_diffusion(args: argparse.Namespace) -> None:
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as exc:
        raise RuntimeError("Install diffusers stack: pip install diffusers transformers accelerate safetensors") from exc

    needed = next_needed_indices(args.output_dir, args.n, args.force)
    if not needed:
        print(f"Diffusion set exists: {args.output_dir / 'train'}")
        return

    dtype = torch.float16 if args.fp16 and str(args.device).startswith("cuda") else torch.float32
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        token=token,
    )
    pipe = pipe.to(args.device)
    if args.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    prompts = build_prompts(args.n, args.seed, args.prompts_file)
    train = args.output_dir / "train"
    train.mkdir(parents=True, exist_ok=True)

    generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
    for start in range(0, len(needed), args.batch_size):
        batch_indices = needed[start : start + args.batch_size]
        batch_prompts = [prompts[i] for i in batch_indices]
        generators = [
            torch.Generator(device=generator_device).manual_seed(args.seed + i)
            for i in batch_indices
        ]
        result = pipe(
            batch_prompts,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generators,
        )
        for idx, image in zip(batch_indices, result.images):
            image.convert("RGB").save(train / f"{idx:08d}.png")
        print(f"{args.name}: wrote {min(start + len(batch_indices), len(needed))}/{len(needed)} new images", flush=True)


def sample_stylegan3(args: argparse.Namespace) -> None:
    needed = next_needed_indices(args.output_dir, args.n, args.force)
    if not needed:
        print(f"StyleGAN3 exists: {args.output_dir / 'train'}")
        return
    repo = args.stylegan3_repo.resolve()
    gen_script = repo / "gen_images.py"
    if not gen_script.exists():
        raise FileNotFoundError(
            f"Missing {gen_script}. Clone https://github.com/NVlabs/stylegan3 or set --stylegan3-repo."
        )
    if args.disable_custom_ops:
        # StyleGAN3's CUDA extensions often fail against newer PyTorch/CUDA
        # stacks. The reference PyTorch implementations are slower but robust
        # and preserve the generator's output definition.
        for rel in [
            "torch_utils/ops/bias_act.py",
            "torch_utils/ops/upfirdn2d.py",
            "torch_utils/ops/filtered_lrelu.py",
        ]:
            path = repo / rel
            if not path.exists():
                continue
            text = path.read_text()
            patched = text.replace("impl='cuda'", "impl='ref'").replace('impl="cuda"', 'impl="ref"')
            if patched != text:
                path.write_text(patched)
                print(f"stylegan3: disabled custom CUDA op default in {path}", flush=True)
    train = (args.output_dir / "train").resolve()
    train.mkdir(parents=True, exist_ok=True)

    def seed_arg_for(indices: list[int]) -> str:
        seeds = [args.seed + i for i in indices]
        if not seeds:
            return ""
        runs: list[str] = []
        start = prev = seeds[0]
        for seed in seeds[1:]:
            if seed == prev + 1:
                prev = seed
                continue
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = seed
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        return ",".join(runs)

    # StyleGAN3 supports seed ranges. Chunk to avoid enormous command lines.
    for start in range(0, len(needed), args.chunk_size):
        chunk = needed[start : start + args.chunk_size]
        seed_arg = seed_arg_for(chunk)
        command = [
            sys.executable,
            "gen_images.py",
            "--outdir",
            str(train),
            "--trunc",
            str(args.truncation),
            "--seeds",
            seed_arg,
            "--network",
            args.network,
        ]
        print("stylegan3:", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
    # Normalize filenames to our manifest convention if StyleGAN3 wrote seed*.png.
    for i in needed:
        src_candidates = [
            train / f"seed{args.seed + i:04d}.png",
            train / f"seed{args.seed + i:06d}.png",
            train / f"seed{args.seed + i}.png",
        ]
        dst = train / f"{i:08d}.png"
        if dst.exists():
            continue
        for src in src_candidates:
            if src.exists():
                src.rename(dst)
                break


def generate_simple(args: argparse.Namespace) -> None:
    class Obj:
        pass

    common = Obj()
    common.n = args.n
    common.size = args.size
    common.seed = args.seed
    common.log_every = args.log_every
    common.output_dir = args.output_dir / "gaussian_noise"
    common.std = args.gaussian_std
    generate_gaussian(common)

    common.output_dir = args.output_dir / "perlin_noise"
    common.seed = args.seed + 1
    common.octaves = args.perlin_octaves
    common.persistence = args.perlin_persistence
    common.grayscale = False
    generate_perlin(common)


def make_manifest(args: argparse.Namespace) -> None:
    root = args.root
    rows = [
        ManifestRow("Noise-like", "fractaldb", "FractalDB", root / "fractaldb", 1),
        ManifestRow("Noise-like", "perlin_noise", "Perlin noise", root / "perlin_noise", 1),
        ManifestRow("Noise-like", "gaussian_noise", "Gaussian noise", root / "gaussian_noise", 1),
        ManifestRow("Over-concentrated", "high_cfg_diffusion", "high-CFG diffusion", root / "high_cfg_diffusion", 1),
        ManifestRow("Over-concentrated", "stablerep_high", "StableRep (high)", root / "stablerep_high", 1),
        ManifestRow("Broad / shifted", "stylegan3", "StyleGAN3", root / "stylegan3", 1),
        ManifestRow("Broad / shifted", "biggan", "BigGAN", root / "biggan", 1),
        ManifestRow("Broad / shifted", "ldm_mid_steps", "LDM (mid steps)", root / "ldm_mid_steps", 1),
        ManifestRow("Broad / shifted", "diffusion_mid_cfg", "diffusion (mid CFG)", root / "diffusion_mid_cfg", 1),
        ManifestRow("Broad / shifted", "diffusion_low_cfg", "diffusion (low CFG)", root / "diffusion_low_cfg", 1),
        ManifestRow("Curated mixtures", "stablerep", "StableRep", root / "stablerep", 1),
        ManifestRow("Curated mixtures", "uniform_mixture", "uniform mixture", root / "uniform_mixture", 1),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["regime", "dataset", "display", "image_dir", "score_csv", "feature_npz", "include", "notes"],
        )
        writer.writeheader()
        for row in rows:
            image_dir = row.image_dir / "train"
            include = int(image_dir.exists() and bool(find_images(image_dir)))
            writer.writerow(
                {
                    "regime": row.regime,
                    "dataset": row.dataset,
                    "display": row.display,
                    "image_dir": str(image_dir),
                    "score_csv": "",
                    "feature_npz": "",
                    "include": include if args.auto_include else row.include,
                    "notes": row.notes,
                }
            )
    print(f"Wrote {args.output}")


def make_uniform(args: argparse.Namespace) -> None:
    class Obj:
        pass

    obj = Obj()
    obj.output_dir = args.output_dir
    obj.n = args.n
    obj.seed = args.seed
    obj.log_every = args.log_every
    obj.components = args.components
    obj.link_mode = args.link_mode
    generate_uniform_mixture(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare modern synthetic sources for the DINO metric table.")
    sub = parser.add_subparsers(dest="command", required=True)

    simple = sub.add_parser("simple", help="Generate Gaussian and Perlin datasets.")
    simple.add_argument("--output-dir", type=Path, default=Path("data/modern_synthetic"))
    simple.add_argument("--n", type=int, default=50000)
    simple.add_argument("--size", type=int, default=64)
    simple.add_argument("--seed", type=int, default=None)
    simple.add_argument("--gaussian-std", type=float, default=0.25)
    simple.add_argument("--perlin-octaves", type=int, default=6)
    simple.add_argument("--perlin-persistence", type=float, default=0.55)
    simple.add_argument("--log-every", type=int, default=5000)

    fractal = sub.add_parser("fractaldb", help="Export FractalDB-1k from Hugging Face datasets.")
    fractal.add_argument("--output-dir", type=Path, default=Path("data/modern_synthetic/fractaldb"))
    fractal.add_argument("--hf-dataset", default="p1atdev/FractalDB-1k")
    fractal.add_argument("--split", default="train")
    fractal.add_argument("--image-column", default="image")
    fractal.add_argument("--streaming", action="store_true")
    fractal.add_argument("--n", type=int, default=50000)
    fractal.add_argument("--size", type=int, default=64)
    fractal.add_argument("--seed", type=int, default=None)
    fractal.add_argument("--force", action="store_true")
    fractal.add_argument("--log-every", type=int, default=5000)

    biggan = sub.add_parser("biggan", help="Sample Hugging Face PyTorch BigGAN.")
    biggan.add_argument("--output-dir", type=Path, default=Path("data/modern_synthetic/biggan"))
    biggan.add_argument("--model", default="biggan-deep-256")
    biggan.add_argument("--n", type=int, default=50000)
    biggan.add_argument("--batch-size", type=int, default=64)
    biggan.add_argument("--truncation", type=float, default=0.4)
    biggan.add_argument("--device", default="cuda")
    biggan.add_argument("--seed", type=int, default=None)
    biggan.add_argument("--force", action="store_true")

    diffusion = sub.add_parser("diffusion", help="Sample Stable Diffusion / LDM variants with diffusers.")
    diffusion.add_argument("--name", required=True)
    diffusion.add_argument("--output-dir", type=Path, required=True)
    diffusion.add_argument("--model", default="stabilityai/stable-diffusion-2-base")
    diffusion.add_argument("--n", type=int, default=50000)
    diffusion.add_argument("--size", type=int, default=512)
    diffusion.add_argument("--batch-size", type=int, default=16)
    diffusion.add_argument("--steps", type=int, default=30)
    diffusion.add_argument("--cfg", type=float, default=7.5)
    diffusion.add_argument("--device", default="cuda")
    diffusion.add_argument("--seed", type=int, default=None)
    diffusion.add_argument("--prompts-file", type=Path, default=None)
    diffusion.add_argument("--hf-token", default=None)
    diffusion.add_argument("--fp16", action="store_true")
    diffusion.add_argument("--enable-xformers", action="store_true")
    diffusion.add_argument("--force", action="store_true")

    stylegan3 = sub.add_parser("stylegan3", help="Sample official NVLabs StyleGAN3 through gen_images.py.")
    stylegan3.add_argument("--output-dir", type=Path, default=Path("data/modern_synthetic/stylegan3"))
    stylegan3.add_argument("--stylegan3-repo", type=Path, default=Path("external/stylegan3"))
    stylegan3.add_argument(
        "--network",
        default="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl",
    )
    stylegan3.add_argument("--n", type=int, default=50000)
    stylegan3.add_argument("--truncation", type=float, default=1.0)
    stylegan3.add_argument("--seed", type=int, default=None)
    stylegan3.add_argument("--chunk-size", type=int, default=1000)
    stylegan3.add_argument("--disable-custom-ops", action="store_true")
    stylegan3.add_argument("--force", action="store_true")

    uniform = sub.add_parser("uniform-mixture", help="Create an equal symlink/copy mixture from existing components.")
    uniform.add_argument("--output-dir", type=Path, default=Path("data/modern_synthetic/uniform_mixture"))
    uniform.add_argument("--components", nargs="+", required=True)
    uniform.add_argument("--n", type=int, default=50000)
    uniform.add_argument("--seed", type=int, default=None)
    uniform.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    uniform.add_argument("--log-every", type=int, default=5000)

    manifest = sub.add_parser("manifest", help="Write a manifest from prepared source folders.")
    manifest.add_argument("--root", type=Path, default=Path("data/modern_synthetic"))
    manifest.add_argument("--output", type=Path, default=Path("configs/dino_sources.csv"))
    manifest.add_argument("--auto-include", action="store_true")

    args = parser.parse_args()
    if hasattr(args, "seed"):
        args.seed = resolve_seed(args.seed)
    if args.command == "simple":
        generate_simple(args)
    elif args.command == "fractaldb":
        export_fractaldb(args)
    elif args.command == "biggan":
        sample_biggan(args)
    elif args.command == "diffusion":
        sample_diffusion(args)
    elif args.command == "stylegan3":
        sample_stylegan3(args)
    elif args.command == "uniform-mixture":
        make_uniform(args)
    elif args.command == "manifest":
        make_manifest(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
