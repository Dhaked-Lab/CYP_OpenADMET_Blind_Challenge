"""Configuration loading, profile application, validation and fingerprinting."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_PROFILES = ("corrected", "baseline")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(p: Path) -> dict:
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    if not isinstance(d, dict):
        raise ConfigError(f"config file is not a mapping: {p}")
    return d


def validate_config(cfg: dict) -> None:
    def need(path, typ=None):
        cur = cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise ConfigError(f"missing config key: {path}")
            cur = cur[part]
        if typ and not isinstance(cur, typ):
            raise ConfigError(f"config key {path} must be {typ}, got {type(cur).__name__}")
        return cur
    if need("profile") not in VALID_PROFILES:
        raise ConfigError(f"profile must be one of {VALID_PROFILES}")
    if need("pca.fold_scope") not in ("fold_train_only", "global_legacy"):
        raise ConfigError("pca.fold_scope must be fold_train_only|global_legacy")
    if need("selection.count_convention") not in ("exact_k", "legacy_k_plus_1"):
        raise ConfigError("selection.count_convention must be exact_k|legacy_k_plus_1")
    if need("selection.tie_break") != "lowest_index":
        raise ConfigError("selection.tie_break: only 'lowest_index' is implemented")
    if need("nn_features.mode") not in ("baseline", "drop_count", "exclude_same_scaffold"):
        raise ConfigError("nn_features.mode must be baseline|drop_count|exclude_same_scaffold")
    if need("cv.argsort_kind") not in ("legacy", "stable"):
        raise ConfigError("cv.argsort_kind must be legacy|stable")
    if need("cv.fold_strategy") not in ("original_all_isoforms", "cyp_only"):
        raise ConfigError("cv.fold_strategy must be original_all_isoforms|cyp_only")
    if cfg["cv"]["fold_strategy"] == "original_all_isoforms":
        need("cv.fold_universe.path", str); need("cv.fold_universe.sha256", str)
    k = need("cv.k", int)
    if k < 2:
        raise ConfigError("cv.k must be >= 2")
    members, order = need("ensemble.members", list), need("ensemble.member_order", list)
    if not set(members) <= set(order):
        raise ConfigError("ensemble.members must be a subset of ensemble.member_order")
    if len(set(members)) != len(members):
        raise ConfigError("ensemble.members contains duplicates")
    if need("ensemble.greedy_rounds", int) < 1:
        raise ConfigError("ensemble.greedy_rounds must be >= 1")
    f = need("ensemble.fracs")
    if not (0 < f["start"] < f["stop"] <= 1 and f["step"] > 0):
        raise ConfigError("ensemble.fracs must satisfy 0 < start < stop <= 1, step > 0")
    fo = need("threshold.fraction_override")
    if fo is not None and not (0.0 < float(fo) < 1.0):
        raise ConfigError("threshold.fraction_override must be null or in (0,1)")
    for s in ("folds", "lightgbm", "pca", "tabicl", "threshold_mc"):
        if not isinstance(need(f"seeds.{s}"), int):
            raise ConfigError(f"seeds.{s} must be an integer")
    if need("pca.n_components", int) < 1:
        raise ConfigError("pca.n_components must be >= 1")


def load_config(config_path=None, profile=None, overrides=None, root: Path = PROJECT_ROOT) -> dict:
    cfg = _load_yaml(Path(config_path) if config_path else root / "config" / "pipeline.yaml")
    prof = profile or cfg.get("profile", "corrected")
    if prof not in VALID_PROFILES:
        raise ConfigError(f"unknown profile {prof!r}; choose from {VALID_PROFILES}")
    profiles = cfg.pop("profiles", {}) or {}
    if prof != "corrected":
        if prof not in profiles:
            raise ConfigError(f"profile {prof!r} not defined in config")
        cfg = deep_merge(cfg, profiles[prof])
    cfg["profile"] = prof
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    cfg["_root"] = str(root)
    return cfg


def load_isoform_config(isoform: str, root: Path = PROJECT_ROOT) -> dict:
    from . import SUPPORTED_ISOFORMS
    if isoform not in SUPPORTED_ISOFORMS:
        raise ConfigError(f"unsupported isoform {isoform!r}; supported: {list(SUPPORTED_ISOFORMS)}")
    d = _load_yaml(root / "config" / "isoforms" / f"{isoform}.yaml")
    if d.get("isoform") != isoform:
        raise ConfigError(f"isoform config mismatch: file declares {d.get('isoform')!r}, expected {isoform!r}")
    return d


def load_resources(path=None, root: Path = PROJECT_ROOT) -> dict:
    return _load_yaml(Path(path) if path else root / "config" / "resources.yaml")


def canonical_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def get_path(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def config_subset(cfg: dict, keys) -> dict:
    return {k: get_path(cfg, k) for k in sorted(keys)}


def fracs(cfg: dict) -> np.ndarray:
    f = cfg["ensemble"]["fracs"]
    return np.round(np.arange(f["start"], f["stop"], f["step"]), 2)
