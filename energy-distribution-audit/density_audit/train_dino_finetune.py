from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from density_audit.random_seed import resolve_seed, seed_everything


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def torch_load(path: Path, device: torch.device | str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def import_dino(repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / "vision_transformer.py").exists():
        raise FileNotFoundError(f"Expected facebookresearch/dino at {repo}")
    sys.path.insert(0, str(repo))
    import vision_transformer as vits

    return vits


def clean_backbone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "teacher.", "student.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if not key.startswith("head."):
            cleaned[key] = value
    return cleaned


class FineTuneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


def build_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> FineTuneClassifier:
    vits = import_dino(args.dino_repo)
    backbone = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    checkpoint = torch_load(args.checkpoint, "cpu")
    state = checkpoint.get(args.checkpoint_key, checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Could not find a state dict in {args.checkpoint}")
    missing, unexpected = backbone.load_state_dict(clean_backbone_state(state), strict=False)
    missing = [key for key in missing if not key.startswith("head.")]
    unexpected = [key for key in unexpected if not key.startswith("head.")]
    if missing or unexpected:
        raise RuntimeError(f"Backbone load mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return FineTuneClassifier(backbone, backbone.embed_dim, num_classes).to(device)


def build_loaders(root: Path, image_size: int, batch_size: int, workers: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size / 0.875)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train = datasets.ImageFolder(root / "train", transform=train_transform)
    val = datasets.ImageFolder(root / "val", transform=val_transform)
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("ImageNet train and val folders use different class mappings")
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False, **common),
        train.class_to_idx,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += int((model(images).argmax(1) == labels).sum())
        total += int(labels.numel())
    return correct / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a DINO ViT checkpoint on ImageNet-1K.")
    parser.add_argument("--dino-repo", type=Path, default=Path("external/dino"))
    parser.add_argument("--arch", default="vit_small")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default="teacher")
    parser.add_argument("--imagenet1k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.seed = resolve_seed(args.seed)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, class_to_idx = build_loaders(
        args.imagenet1k, args.image_size, args.batch_size, args.num_workers
    )
    model = build_model(args, len(class_to_idx), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "checkpoint_last.pth"
    best_path = args.output_dir / "finetuned_classifier.pth"
    start_epoch, best_acc = 0, float("-inf")
    if args.resume and last_path.exists():
        saved = torch_load(last_path, device)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best_acc = float(saved.get("best_val_acc", best_acc))

    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    log_path = args.output_dir / "log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = sample_count = 0
        started = time.time()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = F.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss) * labels.numel()
            sample_count += labels.numel()
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        print(json.dumps(row), flush=True)
        with log_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "best_val_acc": max(best_acc, val_acc),
            "class_to_idx": class_to_idx,
            "args": vars(args),
        }
        torch.save(payload, last_path)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({key: payload[key] for key in ("model", "epoch", "val_acc", "class_to_idx", "args")}, best_path)

    print(f"best_val_acc={best_acc:.6f}")
    print(f"wrote {best_path}")


if __name__ == "__main__":
    main()
