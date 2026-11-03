"""Per-step state with atomic writes; corrupted/missing state => step treated as not run."""
from __future__ import annotations
import time
from pathlib import Path
from .io_utils import read_json, write_json, sha256_file

COMPLETE, FAILED, RUNNING = "complete", "failed", "running"


class StateManager:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "state"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hash_cache: dict = {}

    def path(self, sid):
        return self.dir / f"{sid}.json"

    def load(self, sid) -> dict | None:
        p = self.path(sid)
        if not p.exists():
            return None
        try:
            d = read_json(p)
            if not isinstance(d, dict) or d.get("step") != sid or "status" not in d:
                raise ValueError("bad state schema")
            return d
        except Exception:
            try:
                p.replace(p.with_suffix(".corrupt"))
            except OSError:
                pass
            return None

    def save(self, sid, **fields):
        d = {"step": sid, **fields}
        write_json(d, self.path(sid))

    def file_hash(self, path: Path) -> str:
        path = Path(path)
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        h = self._hash_cache.get(key)
        if h is None:
            h = sha256_file(path)
            self._hash_cache[key] = h
        return h

    def hash_outputs(self, step_dir: Path, outputs) -> dict:
        res = {}
        for name in outputs:
            p = Path(step_dir) / name
            if not p.exists():
                raise FileNotFoundError(f"declared output missing: {p}")
            res[name] = {"sha256": self.file_hash(p), "bytes": p.stat().st_size}
        return res

    def outputs_intact(self, step_dir: Path, recorded: dict) -> bool:
        for name, meta in (recorded or {}).items():
            p = Path(step_dir) / name
            if not p.exists() or p.stat().st_size != meta["bytes"] or self.file_hash(p) != meta["sha256"]:
                return False
        return bool(recorded)

    def invalidate(self, sid):
        p = self.path(sid)
        if p.exists():
            p.unlink()

    @staticmethod
    def now():
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")
