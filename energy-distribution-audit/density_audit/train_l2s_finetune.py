from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from density_audit.random_seed import resolve_seed, seed_everything


IMAGENET_MEAN = (0.44087801806139126, 0.42790631331699347, 0.3867879370752931)
IMAGENET_STD = (0.26826768628079806, 0.2610450402318512, 0.26866836876860795)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def build_loaders(data: Path, batch_size: int, num_workers: int):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(70),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(data / "train", transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(data / "val", transform=val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("ImageNet train/val class_to_idx mismatch; use the same class folders in both splits.")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.class_to_idx


def evaluate(encoder, classifier, loader, device: torch.device, layer_index: int) -> float:
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(encoder(images, layer_index=layer_index).flatten(1))
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Learning-to-See SmallAlexNet encoder on ImageNet100 at 64x64."
    )
    parser.add_argument("--learning-repo", type=Path, default=Path("external/learning_with_noise"))
    parser.add_argument("--encoder", type=Path, required=True, help="Path to encoders/small_scale/<dataset>/encoder.pth.")
    parser.add_argument("--imagenet100", type=Path, required=True, help="ImageNet100 root with train/ and val/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset alias or Learning-to-See dataset name for metadata.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-epochs", default="60,80")
    parser.add_argument("--lr-decay-rate", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    seed_everything(args.seed)

    sys.path.insert(0, str(args.learning_repo.resolve()))
    from align_uniform.encoder import SmallAlexNet

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, class_to_idx = build_loaders(args.imagenet100, args.batch_size, args.num_workers)
    num_classes = len(class_to_idx)

    encoder = SmallAlexNet(feat_dim=args.feat_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    with torch.no_grad():
        sample, _ = train_loader.dataset[0]
        feature_dim = encoder(sample.unsqueeze(0).to(device), layer_index=args.layer_index).flatten(1).shape[1]
    classifier = torch.nn.Linear(feature_dim, num_classes).to(device)

    ckpt_path = args.output_dir / "finetuned_classifier.pth"
    start_epoch = 0
    best_acc = float("-inf")
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        classifier.load_state_dict(checkpoint["classifier"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_acc = float(checkpoint.get("best_val_acc", best_acc))

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    milestones = [int(x) for x in args.lr_decay_epochs.split(",") if x]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=args.lr_decay_rate,
    )
    for _ in range(start_epoch):
        scheduler.step()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    log_path = args.output_dir / "log_finetune.txt"
    val_acc = float("nan")
    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(encoder(images, layer_index=args.layer_index).flatten(1))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        scheduler.step()
        train_loss = total_loss / total
        train_acc = total_correct / total
        val_acc = evaluate(encoder, classifier, val_loader, device, args.layer_index)
        elapsed = time.time() - t0
        line = (
            f"epoch={epoch} train_loss={train_loss:.6g} train_acc={train_acc:.6g} "
            f"val_acc={val_acc:.6g} lr={scheduler.get_last_lr()[0]:.6g} sec={elapsed:.1f}"
        )
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "layer_index": args.layer_index,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "best_val_acc": best_acc,
                    "class_to_idx": class_to_idx,
                    "checkpoint_type": "l2s_finetune",
                },
                ckpt_path,
            )
            (args.output_dir / "val_acc.txt").write_text(f"Accuracy: {best_acc}\n")

    print(f"Wrote {ckpt_path}")
    print(f"Best val accuracy: {best_acc:.6g}")


if __name__ == "__main__":
    main()
