from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def torch_load(path: Path, device: torch.device | str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_wnid_map(imagenet_root: Path) -> dict[str, int]:
    for split in ("train", "val"):
        root = imagenet_root / split
        if root.is_dir():
            names = sorted(path.name for path in root.iterdir() if path.is_dir())
            if names:
                return {name: index for index, name in enumerate(names)}
    raise FileNotFoundError(f"No ImageNet class folders found under {imagenet_root}")


class ShiftFolder(Dataset):
    def __init__(self, root: Path, wnid_to_idx: dict[str, int], transform) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = next((wnid_to_idx[parent.name] for parent in path.parents if parent.name in wnid_to_idx), None)
            if label is not None:
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class FineTuneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int = 1000) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


def build_model(repo: Path, arch: str, patch_size: int, checkpoint: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(repo.expanduser().resolve()))
    import vision_transformer as vits

    backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    model = FineTuneClassifier(backbone, backbone.embed_dim).to(device)
    saved = torch_load(checkpoint, device)
    state = saved.get("model", saved)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {checkpoint}: missing={missing[:10]} unexpected={unexpected[:10]}")
    model.eval()
    return model


def has_class_dirs(root: Path, wnids: set[str]) -> bool:
    return root.is_dir() and any(path.is_dir() and path.name in wnids for path in root.iterdir())


def evaluation_roots(root: Path, wnids: set[str], shift_name: str) -> list[tuple[str, Path]]:
    if has_class_dirs(root, wnids):
        return [(shift_name, root)]
    children = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    direct = [path for path in children if has_class_dirs(path, wnids)]
    if len(direct) == 1 and shift_name != "in_c":
        return [(shift_name, direct[0])]
    roots = []
    for corruption in children:
        for severity in sorted(path for path in corruption.iterdir() if path.is_dir()):
            if has_class_dirs(severity, wnids):
                roots.append((f"{corruption.name}/{severity.name}", severity))
    if not roots:
        raise ValueError(f"No ImageFolder-compatible evaluation roots found under {root}")
    return roots


def included_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "include" not in frame:
        return frame
    values = frame["include"].fillna(True)
    if values.dtype == bool:
        return frame[values]
    include = values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return frame[include]


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[int, int]:
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += int((model(images).argmax(1) == labels).sum())
        total += int(labels.numel())
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DINO classifiers on ImageNet-C, ImageNet-A, or clean ImageNet.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--shift-name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imagenet1k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-root-output", type=Path, default=None)
    parser.add_argument("--dino-repo", type=Path, default=Path("external/dino"))
    parser.add_argument("--arch", default="vit_small")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    wnid_to_idx = load_wnid_map(args.imagenet1k)
    roots = evaluation_roots(args.data, set(wnid_to_idx), args.shift_name)
    transform = transforms.Compose(
        [
            transforms.Resize(int(round(args.image_size / 0.875)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    manifest = included_rows(pd.read_csv(args.manifest))
    device = torch.device(args.device)
    aggregate_rows, root_rows = [], []
    for dataset in manifest["dataset"].astype(str):
        checkpoint = args.checkpoint_root / dataset / "finetuned_classifier.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = build_model(args.dino_repo, args.arch, args.patch_size, checkpoint, device)
        root_acc = []
        total_correct = total_count = 0
        for root_name, root in roots:
            data = ShiftFolder(root, wnid_to_idx, transform)
            loader = DataLoader(
                data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            correct, count = evaluate(model, loader, device)
            accuracy = correct / count if count else float("nan")
            root_acc.append(accuracy)
            total_correct += correct
            total_count += count
            root_rows.append({"dataset": dataset, "split": args.shift_name, "root": root_name, "accuracy": accuracy, "n": count})
            print(f"{dataset} {args.shift_name} {root_name}: {accuracy:.4f} ({count})", flush=True)
        aggregate_rows.append(
            {
                "dataset": dataset,
                "split": args.shift_name,
                "accuracy": total_correct / total_count if total_count else float("nan"),
                "macro_accuracy": float(np.nanmean(root_acc)) if root_acc else float("nan"),
                "n": total_count,
                "checkpoint": str(checkpoint),
                "data": str(args.data),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregate_rows).to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    if args.per_root_output:
        args.per_root_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(root_rows).to_csv(args.per_root_output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
