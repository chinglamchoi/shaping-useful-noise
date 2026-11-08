from __future__ import annotations

import ast
import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL_NAME = "finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays"


@dataclass
class GuthEnergyScorer:
    model: Any
    model_input: Any
    log_tensor_cls: Any
    dimension: int
    channels: int
    image_size: int
    device: torch.device
    normalization_constant: torch.Tensor | None

    @torch.no_grad()
    def logp_db_per_dim(self, x: torch.Tensor, noise_variance: float = 0.0) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        inp = self.model_input(x, noise_level=noise_variance)
        out = self.model(inp, compute_scores=False, create_graph=False)
        return self.log_tensor_cls(out.energy, self.dimension).to(
            base="dBs", sign="logp", per_dimension=True
        )


@contextmanager
def guth_imports(guth_repo: Path):
    repo = guth_repo.expanduser().resolve()
    if not (repo / "main.py").exists():
        raise FileNotFoundError(f"{repo} does not look like DualScoreMatching")
    old_path = list(sys.path)
    sys.path.insert(0, str(repo))
    try:
        yield repo
    finally:
        sys.path[:] = old_path


def _literal_or_eval(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        # Guth's args are trusted local experiment configs and occasionally use
        # constructor-like strings, e.g. WhiteGaussianSampler().
        return eval(value)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_normalization_constant(repo: Path, explicit_path: Path | None) -> torch.Tensor | None:
    path = explicit_path or repo / "outputs" / "energies_test.pt"
    if not path.exists():
        return None
    loaded = _torch_load(path)
    if isinstance(loaded, tuple) and len(loaded) >= 1:
        constant = loaded[0]
    else:
        constant = loaded
    if constant is None:
        return None
    return torch.as_tensor(constant, dtype=torch.float32)


def _install_guth_pickle_aliases() -> None:
    """Expose Guth checkpoint classes where old pickles may expect them."""
    modules = {
        "data": importlib.import_module("data"),
        "main": importlib.import_module("main"),
        "noise": importlib.import_module("noise"),
        "networks.model": importlib.import_module("networks.model"),
    }
    names = [
        "PerformanceInfo",
        "NoiseLevel",
        "DenoisingError",
        "LogTensor",
        "DatasetInfo",
    ]
    targets = [sys.modules.get("__main__")]
    caller = sys.modules.get("density_audit.guth_histogram")
    if caller is not None:
        targets.append(caller)
    for target in targets:
        if target is None:
            continue
        for name in names:
            for module in modules.values():
                if hasattr(module, name):
                    setattr(target, name, getattr(module, name))
                    break


def load_guth_scorer(
    guth_repo: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: Path | None = None,
    normalization_constant_path: Path | None = None,
    device: str | None = None,
) -> GuthEnergyScorer:
    """Load a Guth DualScoreMatching energy model for clean-image logp scoring."""
    with guth_imports(guth_repo) as repo:
        from data import DatasetInfo
        from main import TrainingContext
        from networks import DnCNN, GradResNet, UNet
        from networks.model import EnergyModel, LogTensor, Reparameterization, model_input
        from noise import IdentityCovariance

        model_dir = (model_dir or repo / "models" / model_name).expanduser().resolve()
        args_path = model_dir / "args.json"
        ckpt_path = model_dir / "model.pth.tar"
        if not args_path.exists():
            raise FileNotFoundError(args_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        args = json.loads(args_path.read_text())
        args["dir"] = str(model_dir.parent)
        args["name"] = model_dir.name

        checkpoint_state = None
        try:
            ctx = TrainingContext(
                [],
                **args,
                step="last",
                key_remap=None,
                seed=None,
                dataloaders=False,
                writer=False,
            )
            model = ctx.model
            if hasattr(model.network, "module"):
                model.network = model.network.module
            dataset_info = ctx.dataset_info
            resolved_device = torch.device(device or ctx.device)
        except Exception:
            _install_guth_pickle_aliases()
            grayscale = bool(args.get("grayscale", True))
            channels = 1 if grayscale else 3
            dataset_info = DatasetInfo(
                spatial_size=64,
                num_channels=channels,
                mean=0.5,
                variance=1 / 12,
            )
            network_cls = {"DnCNN": DnCNN, "GradResNet": GradResNet, "UNet": UNet}[args["network"]]
            network = network_cls(
                dataset_info=dataset_info,
                **_literal_or_eval(args.get("network_kwargs", "dict()")),
            )
            network = Reparameterization(
                network,
                dataset_info=dataset_info,
                **_literal_or_eval(args.get("reparam_kwargs", "dict()")),
            )
            model = EnergyModel(
                network=network,
                dataset_info=dataset_info,
                noise_covariance=IdentityCovariance(),
                **_literal_or_eval(args.get("model_kwargs", "dict()")),
            )
            state = _torch_load(ckpt_path)
            checkpoint_state = state
            target = model.network.module if hasattr(model.network, "module") else model.network
            target.load_state_dict(state["state_dict"])
            resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = model.to(resolved_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        constant = None
        if normalization_constant_path is not None:
            constant = _load_normalization_constant(repo, normalization_constant_path)
        elif model_name == DEFAULT_MODEL_NAME:
            constant = _load_normalization_constant(repo, None)
        if constant is None and checkpoint_state is not None:
            for key in ("test_perf", "train_perf"):
                perf = checkpoint_state.get(key)
                if perf is not None and getattr(perf, "log_normalization_constant", None) is not None:
                    constant = torch.as_tensor(perf.log_normalization_constant, dtype=torch.float32)
                    break
        if constant is not None:
            constant = constant.to(resolved_device)
            model.log_normalization_constant = constant

        return GuthEnergyScorer(
            model=model,
            model_input=model_input,
            log_tensor_cls=LogTensor,
            dimension=dataset_info.dimension,
            channels=dataset_info.num_channels,
            image_size=dataset_info.spatial_size,
            device=resolved_device,
            normalization_constant=constant,
        )


def gumbel_mle_torch(values: np.ndarray, n_iters: int = 1000, lr: float = 0.01) -> dict[str, float]:
    """Fit a Gumbel distribution by maximum likelihood, initialized by moments."""
    data = torch.as_tensor(values, dtype=torch.float64)
    scale = torch.sqrt(data.std(unbiased=False) ** 2 * 6 / np.pi**2)
    loc = data.mean() - scale * np.euler_gamma
    loc.requires_grad = True
    scale_raw = torch.log(torch.expm1(scale)).detach().requires_grad_(True)
    opt = torch.optim.Adam([loc, scale_raw], lr=lr)
    for _ in range(n_iters):
        positive_scale = torch.nn.functional.softplus(scale_raw) + 1e-8
        z = (data - loc) / positive_scale
        loss = torch.mean(z + torch.exp(-z)) + torch.log(positive_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
    scale_final = torch.nn.functional.softplus(scale_raw).item()
    loc_final = loc.item()
    z = (data - loc_final) / scale_final
    nll = (torch.mean(z + torch.exp(-z)) + np.log(scale_final)).item()
    return {"location": loc_final, "scale": scale_final, "mean_nll": nll}
