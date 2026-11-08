from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def save_image(image: Image.Image, path: Path, image_format: str | None = None) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
    image.convert("RGB").save(path, format=image_format, **kwargs)
    return True


def save_array_job(job: tuple[np.ndarray, int, str]) -> bool:
    array, label, path_text = job
    return save_image(Image.fromarray(array), Path(path_text), image_format="PNG")


def save_path_job(job: tuple[str, int, str]) -> bool:
    source_text, _label, path_text = job
    with Image.open(source_text) as image:
        return save_image(image, Path(path_text))


def run_jobs(jobs: list[tuple], worker_fn, workers: int, desc: str) -> None:
    if workers <= 1:
        done = 0
        for job in jobs:
            worker_fn(job)
            done += 1
            if done % 1000 == 0 or done == len(jobs):
                print(f"{desc}: exported {done}/{len(jobs)}", flush=True)
        return
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(worker_fn, jobs, chunksize=64):
            done += 1
            if done % 1000 == 0 or done == len(jobs):
                print(f"{desc}: exported {done}/{len(jobs)}", flush=True)


def export_cifar(root: Path, output_root: Path, dataset_name: str, train: bool, limit: int | None, workers: int) -> None:
    from torchvision.datasets import CIFAR10, CIFAR100

    cls = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    dataset = cls(root=str(root), train=train, download=True)
    split = "train" if train else "test"
    out_dir = output_root / dataset_name / split
    n = len(dataset) if limit is None else min(limit, len(dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = dataset.targets[:n]
    jobs = [
        (dataset.data[idx], int(labels[idx]), str(out_dir / f"{idx:06d}_class{int(labels[idx]):03d}.png"))
        for idx in range(n)
        if not (out_dir / f"{idx:06d}_class{int(labels[idx]):03d}.png").exists()
    ]
    print(f"{dataset_name} {split}: {n - len(jobs)}/{n} already exist; writing {len(jobs)}", flush=True)
    run_jobs(jobs, save_array_job, workers, f"{dataset_name} {split}")


def export_dtd(root: Path, output_root: Path, split: str, partition: int, limit: int | None, workers: int) -> None:
    from torchvision.datasets import DTD

    dataset = DTD(root=str(root), split=split, partition=partition, download=True)
    out_dir = output_root / "dtd" / f"{split}{partition}"
    n = len(dataset) if limit is None else min(limit, len(dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for idx in range(n):
        source, label = dataset._image_files[idx], int(dataset._labels[idx])
        dest = out_dir / f"{idx:06d}_class{label:03d}.jpg"
        if not dest.exists():
            jobs.append((str(source), label, str(dest)))
    print(f"dtd {split}{partition}: {n - len(jobs)}/{n} already exist; writing {len(jobs)}", flush=True)
    run_jobs(jobs, save_path_job, workers, f"dtd {split}{partition}")


def export_pets(root: Path, output_root: Path, split: str, limit: int | None, workers: int) -> None:
    from torchvision.datasets import OxfordIIITPet

    dataset = OxfordIIITPet(root=str(root), split=split, target_types="category", download=True)
    out_dir = output_root / "oxford_pets" / split
    n = len(dataset) if limit is None else min(limit, len(dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for idx in range(n):
        source, label = dataset._images[idx], int(dataset._labels[idx])
        dest = out_dir / f"{idx:06d}_class{label:03d}.jpg"
        if not dest.exists():
            jobs.append((str(source), label, str(dest)))
    print(f"oxford_pets {split}: {n - len(jobs)}/{n} already exist; writing {len(jobs)}", flush=True)
    run_jobs(jobs, save_path_job, workers, f"oxford_pets {split}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/export small torchvision real-image datasets as image folders for Guth scoring."
    )
    parser.add_argument("--torchvision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100"],
        choices=["cifar10", "cifar100", "dtd", "oxford_pets"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cifar-train", action="store_true")
    parser.add_argument("--dtd-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dtd-partition", type=int, default=1)
    parser.add_argument("--pets-split", default="test", choices=["trainval", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.torchvision_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        if dataset_name in {"cifar10", "cifar100"}:
            export_cifar(args.torchvision_root, args.output_root, dataset_name, args.cifar_train, args.limit, args.workers)
        elif dataset_name == "dtd":
            export_dtd(args.torchvision_root, args.output_root, args.dtd_split, args.dtd_partition, args.limit, args.workers)
        elif dataset_name == "oxford_pets":
            export_pets(args.torchvision_root, args.output_root, args.pets_split, args.limit, args.workers)


if __name__ == "__main__":
    main()
