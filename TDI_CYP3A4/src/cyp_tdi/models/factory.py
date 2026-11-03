"""Model construction. Kept thin so tests can mock the heavy libraries."""
from __future__ import annotations
import functools
from pathlib import Path
from ..resource_manager import lightgbm_n_jobs, validate_checkpoint, ResourceError


def lgbm_params(cfg, res):
    p = dict(cfg["lightgbm"]["params"])
    p["random_state"] = cfg["seeds"]["lightgbm"]
    p["n_jobs"] = lightgbm_n_jobs(res)
    if cfg["lightgbm"]["deterministic"]:
        p["deterministic"] = True
        p["force_row_wise"] = True
    return p


def make_lgbm(cfg, res):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**lgbm_params(cfg, res))


@functools.lru_cache(maxsize=4)
def _verified_tabicl_checkpoint(hf_repo, version, sha256) -> Path:
    """Locate the pinned TabICL checkpoint in the local Hugging Face cache (never downloads) and verify its sha256."""
    from huggingface_hub import hf_hub_download
    try:
        path = Path(hf_hub_download(repo_id=hf_repo, filename=version, local_files_only=True))
    except Exception as e:
        raise ResourceError(f"TabICL checkpoint {version!r} ({hf_repo}) is not in the local Hugging Face cache ({type(e).__name__}). "
                            "The pipeline never downloads it during a run; fetch it once as described in README 'TabICL checkpoint'.") from e
    validate_checkpoint(path, sha256, f"TabICL checkpoint {version}")
    return path


def tabicl_fit_predict(cfg, device, Xtr, ytr, Xte_list):
    """[JB] TabICLClassifier(n_estimators=4, random_state=42); returns list of P(class 1).
    The pretrained checkpoint is pinned (tabicl.checkpoint: version + sha256) instead of TabICL's version-dependent default,
    loaded from the verified local file, and auto-download is disabled. GPU execution may not be bitwise reproducible."""
    from tabicl import TabICLClassifier
    ck = cfg["tabicl"]["checkpoint"]
    path = _verified_tabicl_checkpoint(ck["hf_repo"], ck["version"], ck["sha256"])
    clf = TabICLClassifier(n_estimators=cfg["tabicl"]["n_estimators"], random_state=cfg["seeds"]["tabicl"], device=device,
                           checkpoint_version=ck["version"], model_path=path, allow_auto_download=False)
    clf.fit(Xtr, ytr)
    return [clf.predict_proba(X)[:, 1] for X in Xte_list]


def save_booster(booster, path):
    """Write the model text with Python I/O. LightGBM's own file I/O receives the path as UTF-8 bytes and, on Windows,
    cannot open paths with non-ASCII characters (e.g. an emoji project folder): 'is not available for writes'.
    Writes model_to_string() verbatim (same defaults as save_model); the model text equals save_model's except that the
    native Windows writer emits a few CRLF line endings."""
    from ..io_utils import atomic_write_text
    atomic_write_text(path, booster.model_to_string(), newline="")


def load_booster(path):
    """Read with Python I/O for the same reason as save_booster (native load fails: 'Could not open ...')."""
    from lightgbm import Booster
    return Booster(model_str=Path(path).read_text(encoding="utf-8"))
