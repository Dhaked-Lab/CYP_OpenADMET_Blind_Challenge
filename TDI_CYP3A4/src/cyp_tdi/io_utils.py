import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_strings(items) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode()); h.update(b"\n")
    return h.hexdigest()


def atomic_write_text(path, text: str, newline=None):
    """newline=None: platform line endings (default); newline="": write `text` byte-for-byte."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as fh:
            fh.write(text); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(obj, path):
    def default(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o))
    atomic_write_text(path, json.dumps(obj, indent=2, default=default))


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npy(path, arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr); os.replace(tmp, path)


def library_versions() -> dict:
    """Versions from distribution metadata ONLY (no heavy imports: reading a version must never initialise
    PyTorch/Chemprop/CUDA). None = distribution not installed."""
    import platform, sys
    from .deps import distribution_version
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "lightgbm", "torch", "chemprop", "tabicl", "yaml"):
        out[m] = distribution_version(m)
    return out
