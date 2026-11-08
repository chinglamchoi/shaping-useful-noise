from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Hugging Face image dataset split to PNG files.")
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. benjamin-paine/imagenet-1k-64x64")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    for i, row in enumerate(tqdm(ds, desc=f"export {args.dataset}:{args.split}")):
        if args.limit is not None and i >= args.limit:
            break
        image = row[args.image_key]
        label = row.get(args.label_key, "unknown")
        label_dir = args.output_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        image.save(label_dir / f"{i:08d}.png")

    print(f"Wrote images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
