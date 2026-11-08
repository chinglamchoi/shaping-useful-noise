from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision

from density_audit.eval_l2s_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RemappedImageFolder,
    build_model,
    evaluate,
    load_class_map,
)


def discover_imagefolder_roots(root: Path, layout: str) -> list[tuple[str, Path]]:
    """Return named ImageFolder roots for ImageNet-A or ImageNet-C style data."""
    if layout == "single":
        return [(root.name, root)]
    if layout == "imagenet-c":
        roots: list[tuple[str, Path]] = []
        for corruption in sorted(p for p in root.iterdir() if p.is_dir()):
            for severity in sorted(p for p in corruption.iterdir() if p.is_dir()):
                roots.append((f"{corruption.name}/{severity.name}", severity))
        if not roots:
            raise ValueError(f"No ImageNet-C severity folders found under {root}")
        return roots
    if layout == "children":
        roots = [(p.name, p) for p in sorted(root.iterdir()) if p.is_dir()]
        if not roots:
            raise ValueError(f"No child ImageFolder roots found under {root}")
        return roots
    raise ValueError(f"Unknown layout: {layout}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one Learning-to-See classifier over one or more ImageNet shift ImageFolder roots."
    )
    parser.add_argument("--learning-repo", type=Path, default=Path("external/learning_with_noise"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--class-map-from", type=Path, required=True, help="Usually /path/to/imagenet100/val.")
    parser.add_argument("--layout", choices=["single", "imagenet-c", "children"], default="single")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-name", required=True, choices=["in_c", "in_a", "custom"])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-root-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    encoder, classifier, layer_index, _ = build_model(
        args.learning_repo,
        args.checkpoint,
        device,
        feat_dim=args.feat_dim,
        layer_index_arg=args.layer_index,
    )
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(70),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    class_map = load_class_map(args.class_map_from)
    roots = discover_imagefolder_roots(args.data, args.layout)

    per_root_rows = []
    weighted_correct = 0.0
    weighted_total = 0
    for root_name, root in roots:
        dataset = RemappedImageFolder(root, class_map, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        acc, correct, total = evaluate(encoder, classifier, loader, device, layer_index)
        per_root_rows.append(
            {
                "dataset": args.dataset,
                "split": args.split_name,
                "root": root_name,
                "accuracy": acc,
                "n": total,
                "checkpoint": str(args.checkpoint),
                "data": str(root),
            }
        )
        weighted_correct += correct
        weighted_total += total
        print(f"{args.dataset} {args.split_name} {root_name}: {acc:.4f} ({correct}/{total})", flush=True)

    # ImageNet-C roots are usually equal sized; using sample-weighted accuracy
    # also remains correct if a class subset or missing files make roots uneven.
    mean_acc = weighted_correct / weighted_total if weighted_total else float("nan")
    macro_acc = float(np.mean([row["accuracy"] for row in per_root_rows])) if per_root_rows else float("nan")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output_csv.exists()
    with args.output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "split", "accuracy", "macro_accuracy", "n", "checkpoint", "data", "layout"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": args.dataset,
                "split": args.split_name,
                "accuracy": mean_acc,
                "macro_accuracy": macro_acc,
                "n": weighted_total,
                "checkpoint": str(args.checkpoint),
                "data": str(args.data),
                "layout": args.layout,
            }
        )
    if args.per_root_csv is not None:
        args.per_root_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = args.per_root_csv.exists()
        with args.per_root_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_root_rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(per_root_rows)
    print(f"{args.dataset} {args.split_name}: {mean_acc:.4f} weighted, {macro_acc:.4f} macro")


if __name__ == "__main__":
    main()
