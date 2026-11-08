from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from density_audit.random_seed import resolve_seed


def check_dino_repo(repo: Path) -> None:
    missing = [name for name in ["main_dino.py", "eval_linear.py"] if not (repo / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{repo} does not look like facebookresearch/dino; missing {missing}. "
            "Clone with: git clone https://github.com/facebookresearch/dino external/dino"
        )


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("Running:")
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def parse_unknown_as_passthrough(args: list[str]) -> list[str]:
    # Allows wrappers such as --dino-arg value --another-flag to pass through
    # without our parser needing to chase the DINO CLI surface.
    return args


def train(args: argparse.Namespace, passthrough: list[str]) -> None:
    check_dino_repo(args.dino_repo)
    data_path = args.data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_wrapper_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    command = [
        sys.executable,
        "main_dino.py",
        "--arch",
        args.arch,
        "--data_path",
        str(data_path),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--epochs",
        str(args.epochs),
        "--batch_size_per_gpu",
        str(args.batch_size_per_gpu),
        "--num_workers",
        str(args.num_workers),
        "--saveckp_freq",
        str(args.saveckp_freq),
        "--seed",
        str(args.seed),
    ]
    if args.teacher_temp is not None:
        command.extend(["--teacher_temp", str(args.teacher_temp)])
    if args.warmup_teacher_temp is not None:
        command.extend(["--warmup_teacher_temp", str(args.warmup_teacher_temp)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.min_lr is not None:
        command.extend(["--min_lr", str(args.min_lr)])
    if args.weight_decay is not None:
        command.extend(["--weight_decay", str(args.weight_decay)])
    if args.weight_decay_end is not None:
        command.extend(["--weight_decay_end", str(args.weight_decay_end)])
    command.extend(passthrough)
    run(command, cwd=args.dino_repo, dry_run=args.dry_run)


def linear_eval(args: argparse.Namespace, passthrough: list[str]) -> None:
    check_dino_repo(args.dino_repo)
    if not args.imagenet1k.exists():
        raise FileNotFoundError(args.imagenet1k)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.output_dir / "checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing DINO checkpoint: {checkpoint}")
    args.linear_output_dir.mkdir(parents=True, exist_ok=True)
    (args.linear_output_dir / "linear_wrapper_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    command = [
        sys.executable,
        "eval_linear.py",
        "--arch",
        args.arch,
        "--data_path",
        str(args.imagenet1k.resolve()),
        "--pretrained_weights",
        str(checkpoint.resolve()),
        "--output_dir",
        str(args.linear_output_dir.resolve()),
        "--epochs",
        str(args.epochs),
        "--batch_size_per_gpu",
        str(args.batch_size_per_gpu),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    command.extend(passthrough)
    run(command, cwd=args.dino_repo, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper around facebookresearch/dino for modern synthetic datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dino-repo", type=Path, default=Path("external/dino"))
    common.add_argument("--arch", default="vit_small")
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--dry-run", action="store_true")

    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--data-path", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    train_parser.add_argument("--num-workers", type=int, default=10)
    train_parser.add_argument("--saveckp-freq", type=int, default=20)
    train_parser.add_argument("--teacher-temp", type=float, default=None)
    train_parser.add_argument("--warmup-teacher-temp", type=float, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--min-lr", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)
    train_parser.add_argument("--weight-decay-end", type=float, default=None)

    linear_parser = sub.add_parser("linear-eval", parents=[common])
    linear_parser.add_argument("--imagenet1k", type=Path, required=True)
    linear_parser.add_argument("--output-dir", type=Path, required=True, help="DINO pretraining output dir.")
    linear_parser.add_argument("--linear-output-dir", type=Path, required=True)
    linear_parser.add_argument("--checkpoint", type=Path, default=None)
    linear_parser.add_argument("--epochs", type=int, default=100)
    linear_parser.add_argument("--batch-size-per-gpu", type=int, default=128)
    linear_parser.add_argument("--num-workers", type=int, default=10)
    linear_parser.add_argument("--lr", type=float, default=None)

    args, unknown = parser.parse_known_args()
    args.seed = resolve_seed(args.seed)
    passthrough = parse_unknown_as_passthrough(unknown)
    if args.command == "train":
        train(args, passthrough)
    elif args.command == "linear-eval":
        linear_eval(args, passthrough)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
