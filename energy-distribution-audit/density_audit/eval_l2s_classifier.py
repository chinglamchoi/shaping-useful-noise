from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torchvision


IMAGENET_MEAN = (0.44087801806139126, 0.42790631331699347, 0.3867879370752931)
IMAGENET_STD = (0.26826768628079806, 0.2610450402318512, 0.26866836876860795)


class RemappedImageFolder(torchvision.datasets.ImageFolder):
    """ImageFolder whose WordNet-folder labels are remapped to a reference order."""

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform, strict: bool = False):
        super().__init__(root, transform=transform)
        missing = sorted(set(self.class_to_idx).difference(class_to_idx))
        if missing and strict:
            raise ValueError(
                f"{root} contains classes not present in the reference class map: {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )
        keep_samples = []
        keep_targets = []
        for path, local_target in self.samples:
            class_name = self.classes[local_target]
            if class_name not in class_to_idx:
                continue
            target = int(class_to_idx[class_name])
            keep_samples.append((path, target))
            keep_targets.append(target)
        if not keep_samples:
            raise ValueError(f"No classes in {root} overlap the reference class map.")
        self.samples = keep_samples
        self.targets = keep_targets


def load_class_map(path: Path) -> dict[str, int]:
    reference = torchvision.datasets.ImageFolder(path)
    return dict(reference.class_to_idx)


def build_model(learning_repo: Path, checkpoint_path: Path, device: torch.device, feat_dim: int, layer_index_arg: int | None):
    sys.path.insert(0, str(learning_repo.resolve()))
    from align_uniform.encoder import SmallAlexNet

    checkpoint = torch.load(checkpoint_path, map_location=device)
    layer_index = layer_index_arg if layer_index_arg is not None else int(checkpoint.get("layer_index", 5))
    classifier_state = checkpoint["classifier"]
    num_classes = int(classifier_state["weight"].shape[0])

    encoder = SmallAlexNet(feat_dim=feat_dim).to(device).eval()
    encoder.load_state_dict(checkpoint["encoder"])

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        feature_dim = encoder(dummy, layer_index=layer_index).flatten(1).shape[1]
    classifier = torch.nn.Linear(feature_dim, num_classes).to(device).eval()
    classifier.load_state_dict(classifier_state)
    return encoder, classifier, layer_index, num_classes


def evaluate(encoder, classifier, loader, device: torch.device, layer_index: int) -> tuple[float, int, int]:
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = encoder(images, layer_index=layer_index).flatten(1)
            pred = classifier(feats).argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    acc = correct / total if total else float("nan")
    return acc, correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Learning-to-See linear classifier on an ImageFolder split.")
    parser.add_argument("--learning-repo", type=Path, default=Path("external/learning_with_noise"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="encoder_classifier.pth from linear_eval_imagenet100.py")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with class subdirectories.")
    parser.add_argument(
        "--class-map-from",
        type=Path,
        default=None,
        help="Reference ImageFolder split, usually ImageNet100 val, whose class_to_idx defines classifier label order.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset alias for output CSV.")
    parser.add_argument("--split-name", required=True, choices=["in_c", "in_a", "in_val", "custom"])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--strict-classes", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    encoder, classifier, layer_index, num_classes = build_model(
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
    if args.class_map_from is None:
        dataset = torchvision.datasets.ImageFolder(args.data, transform=transform)
        if len(dataset.classes) != num_classes:
            raise ValueError(
                f"{args.data} has {len(dataset.classes)} classes but classifier has {num_classes}. "
                "Pass --class-map-from /path/to/imagenet100/val for ImageNet-A/C subsets."
            )
    else:
        class_map = load_class_map(args.class_map_from)
        if len(class_map) != num_classes:
            raise ValueError(
                f"Reference class map has {len(class_map)} classes but classifier has {num_classes}: {args.class_map_from}"
            )
        dataset = RemappedImageFolder(args.data, class_map, transform=transform, strict=args.strict_classes)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    acc, correct, total = evaluate(encoder, classifier, loader, device, layer_index)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output_csv.exists()
    with args.output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "accuracy", "n", "checkpoint", "data"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": args.dataset,
                "split": args.split_name,
                "accuracy": acc,
                "n": total,
                "checkpoint": str(args.checkpoint),
                "data": str(args.data),
            }
        )
    print(f"{args.dataset} {args.split_name}: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
