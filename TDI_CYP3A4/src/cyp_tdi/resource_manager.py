"""CPU/GPU settings and checkpoint provenance validation."""
from __future__ import annotations
import os
from pathlib import Path
from .io_utils import sha256_file

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class ResourceError(RuntimeError):
    pass


def apply_thread_env(res: dict) -> int | None:
    """Must run BEFORE numpy/torch are imported to be fully effective."""
    n = (res.get("cpu") or {}).get("threads")
    if n is None:
        return None
    n = int(n)
    if n < 1:
        raise ResourceError("cpu.threads must be >= 1")
    for k in THREAD_ENV:
        os.environ[k] = str(n)
    return n


def cpu_threads(res: dict) -> int | None:
    n = (res.get("cpu") or {}).get("threads")
    pj = res.get("parallel_jobs")
    vals = [int(v) for v in (n, pj if pj and pj > 1 else None) if v]
    return min(vals) if vals else None


def lightgbm_n_jobs(res: dict) -> int:
    n = cpu_threads(res)
    return -1 if n is None else n


def resolve_device(res: dict) -> str:
    """Return 'cpu' or 'cuda:<i>'. Never claims GPU if it is unavailable."""
    g = res.get("gpu") or {}
    mode = str(g.get("enabled", "auto")).lower()
    if mode == "false":
        return "cpu"
    from .deps import import_framework, DependencyError
    try:
        torch = import_framework("torch", "device selection")
        avail = torch.cuda.is_available()
        if avail and int(g.get("device_index", 0)) >= torch.cuda.device_count():
            raise ResourceError(f"gpu.device_index {g.get('device_index')} >= device_count {torch.cuda.device_count()}")
    except DependencyError as e:
        if e.kind == "broken":
            raise                                   # an installed-but-broken PyTorch is never treated as 'no GPU'
        avail = False
    if avail:
        return f"cuda:{int(g.get('device_index', 0))}"
    if mode == "true" and g.get("required", False):
        raise ResourceError("GPU requested (gpu.enabled=true, required=true) but CUDA is unavailable")
    return "cpu"


def apply_cli_overrides(res: dict, cpu_threads=None, gpu=None, gpu_device=None) -> dict:
    res = {**res, "cpu": dict(res.get("cpu") or {}), "gpu": dict(res.get("gpu") or {})}
    if cpu_threads is not None:
        res["cpu"]["threads"] = cpu_threads
    if gpu is not None:
        res["gpu"]["enabled"] = "true" if gpu else "false"
    if gpu_device is not None:
        res["gpu"]["device_index"] = gpu_device
    return res


def checkpoint_spec(res: dict, name: str, root: Path) -> tuple[Path, str | None]:
    spec = (res.get("checkpoints") or {}).get(name)
    if not spec:
        raise ResourceError(f"checkpoint {name!r} not declared in resources.yaml")
    p = Path(spec["path"])
    if not p.is_absolute():
        p = Path(root) / p
    return p, spec.get("sha256")


def validate_checkpoint(path: Path, expected_sha256: str | None, name: str = "checkpoint") -> str:
    path = Path(path)
    if not path.exists():
        raise ResourceError(f"missing {name}: {path} (see README 'Checkpoints' for filename, source and retrieval)")
    if path.stat().st_size == 0:
        raise ResourceError(f"{name} is empty: {path}")
    h = sha256_file(path)
    if expected_sha256 and h != expected_sha256:
        raise ResourceError(f"{name} sha256 mismatch for {path}: expected {expected_sha256}, got {h}. "
                            "Refusing to load an unverified checkpoint.")
    return h
