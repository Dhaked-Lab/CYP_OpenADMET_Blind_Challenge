"""Centralised seeding. Original seed values are kept in config (seeds.*).
Python/NumPy are always seeded. PyTorch is seeded only for steps that declare requires=("torch",), and a PyTorch
failure there is raised (DependencyError), never swallowed."""
import os, random
import numpy as np
from .deps import import_framework


def seed_python_numpy(cfg: dict) -> dict:
    s = cfg["seeds"]
    random.seed(s["python"]); np.random.seed(s["python"])
    os.environ.setdefault("PYTHONHASHSEED", str(s["python"]))
    return {"python": s["python"]}


def seed_torch(cfg: dict, torch_deterministic: bool | None = None) -> dict:
    torch = import_framework("torch", "seeding for a PyTorch-based step")
    s = cfg["seeds"]
    torch.manual_seed(s["torch"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s["torch"])
    det = cfg["embeddings"]["torch_deterministic"] if torch_deterministic is None else torch_deterministic
    if det:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {"torch": s["torch"], "torch_deterministic": bool(det)}


def seed_everything(cfg: dict, frameworks=(), torch_deterministic: bool | None = None) -> dict:
    """frameworks: names of heavy frameworks the CURRENT step needs (from StepSpec.requires)."""
    info = seed_python_numpy(cfg)
    if "torch" in frameworks:
        info.update(seed_torch(cfg, torch_deterministic))
    return info
