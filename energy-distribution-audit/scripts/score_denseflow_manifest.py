from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from density_audit.random_seed import resolve_seed


def infer_data_shape(args, model_dir: Path) -> tuple[int, int, int]:
    dataset = str(getattr(args, "dataset", "")).lower()
    hint = f"{dataset} {model_dir}".lower()
    if "imagenet64" in hint or "celeba" in hint:
        return (3, 64, 64)
    if "imagenet32" in hint or "cifar" in hint:
        return (3, 32, 32)
    raise ValueError(
        "Could not infer DenseFlow data shape. Pass --data-shape C H W, "
        "for example --data-shape 3 64 64 for ImageNet64."
    )


class ManifestImages(Dataset):
    def __init__(self, manifest: Path, limit: int | None = None):
        frame = pd.read_csv(manifest)
        if "canonical_path" not in frame.columns:
            raise ValueError(f"{manifest} must contain a canonical_path column.")
        if "image_id" not in frame.columns:
            frame["image_id"] = [f"image_{i:08d}" for i in range(len(frame))]
        if "dataset" not in frame.columns:
            frame["dataset"] = "unknown"
        if limit is not None:
            frame = frame.iloc[:limit].reset_index(drop=True)
        self.frame = frame
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row["canonical_path"])
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            x = self.to_tensor(image)
        return x, str(row["image_id"]), str(row["dataset"]), str(path)


def iwbo(model, x: torch.Tensor, k: int) -> torch.Tensor:
    x_stack = torch.cat([x for _ in range(k)], dim=0)
    ll_stack = model.log_prob(x_stack)
    ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
    return torch.logsumexp(ll, dim=0) - math.log(k)


def iwbo_batched(model, x: torch.Tensor, k: int, kbs: int) -> torch.Tensor:
    if k % kbs != 0:
        raise ValueError(f"k={k} must be divisible by kbs={kbs}.")
    pieces = []
    for _ in range(k // kbs):
        x_stack = torch.cat([x for _ in range(kbs)], dim=0)
        ll_stack = model.log_prob(x_stack)
        pieces.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
    ll = torch.cat(pieces, dim=0)
    return torch.logsumexp(ll, dim=0) - math.log(k)


def per_image_bpd(model, x: torch.Tensor, k: int | None, kbs: int | None) -> torch.Tensor:
    if k is None:
        logp = model.log_prob(x)
    elif kbs is None:
        logp = iwbo(model, x, k)
    else:
        logp = iwbo_batched(model, x, k, kbs)
    dims = x[0].numel()
    return -logp / (math.log(2) * dims)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a prepared image manifest with DenseFlow and write per-image BPD."
    )
    parser.add_argument("--denseflow-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="DenseFlow model dir with args.pickle and check/checkpoint.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=None, help="Use IWBO with k samples. Omit for ELBO/log_prob.")
    parser.add_argument("--kbs", type=int, default=None, help="IWBO sub-batch size.")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-shape", type=int, nargs=3, default=None, metavar=("C", "H", "W"))
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    torch.manual_seed(args.seed)

    denseflow_repo = args.denseflow_repo.resolve()
    sys.path.insert(0, str(denseflow_repo))
    sys.path.insert(0, str(denseflow_repo / "experiments" / "image"))

    from model.model_flow import get_model
    from denseflow.distributions import DataParallelDistribution

    with (args.model / "args.pickle").open("rb") as f:
        model_args = pickle.load(f)

    data_shape = tuple(args.data_shape) if args.data_shape is not None else infer_data_shape(model_args, args.model)
    model = get_model(model_args, data_shape=data_shape)
    if getattr(model_args, "parallel", None) == "dp":
        model = DataParallelDistribution(model)

    checkpoint = torch.load(args.model / "check" / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print(
        f"Loaded DenseFlow checkpoint epoch {checkpoint.get('current_epoch', 'unknown')} "
        f"from {args.model}",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device).eval()
    if args.double:
        model = model.double()

    dataset = ManifestImages(args.manifest, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "dataset", "canonical_path", "bpd"],
        )
        writer.writeheader()
        seen = 0
        with torch.no_grad():
            for batch_idx, (x, image_ids, datasets, paths) in enumerate(loader, start=1):
                if args.double:
                    x = x.double()
                x = x.to(device, non_blocking=True)
                bpd = per_image_bpd(model, x, k=args.k, kbs=args.kbs).detach().cpu().tolist()
                for image_id, dataset_name, path, value in zip(image_ids, datasets, paths, bpd):
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "dataset": dataset_name,
                            "canonical_path": path,
                            "bpd": float(value),
                        }
                    )
                seen += len(bpd)
                if batch_idx % 10 == 0 or seen == len(dataset):
                    print(f"scored {seen}/{len(dataset)}", flush=True)

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
