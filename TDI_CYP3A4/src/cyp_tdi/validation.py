"""Fail-fast validation helpers (isoform, schemas, features, weights, submission)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    pass


_LABEL_RE = re.compile(r"^(CYP[0-9A-Za-z]+)_is_TDI$", re.I)
_TRUE = {"true", "1", "1.0", "yes", "t"}
_FALSE = {"false", "0", "0.0", "no", "f"}


def assert_isoform(requested: str, supported) -> None:
    if requested not in supported:
        raise ValidationError(f"unsupported isoform {requested!r}; supported: {list(supported)}")


def detect_foreign_isoform(columns, isoform: str) -> list[str]:
    """Return label-like columns that name a DIFFERENT isoform."""
    bad = []
    for c in columns:
        m = _LABEL_RE.match(str(c).strip())
        if m and m.group(1).upper() != isoform.upper():
            bad.append(str(c))
    return bad


def parse_label(v) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in _TRUE: return 1
    if s in _FALSE: return 0
    raise ValidationError(f"unrecognised label value {v!r}")


def validate_training_columns(columns, iso_cfg: dict, file_label: str):
    cols = [str(c).strip() for c in columns]
    foreign = detect_foreign_isoform(cols, iso_cfg["isoform"])
    if foreign:
        raise ValidationError(f"{file_label}: columns {foreign} belong to another isoform; "
                              f"this pipeline is {iso_cfg['isoform']} only")
    for need in (iso_cfg["smiles_column"], iso_cfg["id_column"]):
        if need not in cols:
            raise ValidationError(f"{file_label}: missing required column {need!r}; found {cols}")
    if iso_cfg["label_column"] not in cols:
        raise ValidationError(f"{file_label}: missing label column {iso_cfg['label_column']!r}; found {cols}")


def validate_blind_columns(columns, iso_cfg: dict, label: str = "blind file"):
    cols = [str(c).strip() for c in columns]
    foreign = detect_foreign_isoform(cols, iso_cfg["isoform"])
    if foreign:
        raise ValidationError(f"{label}: contains label columns of another isoform: {foreign}")
    if iso_cfg["label_column"] in cols:
        raise ValidationError(f"{label}: unexpected label column {iso_cfg['label_column']!r} (blind data must be unlabeled)")
    for need in (iso_cfg["smiles_column"], iso_cfg["id_column"]):
        if need not in cols:
            raise ValidationError(f"{label}: missing required column {need!r}; found {cols}")
    extra = [c for c in cols if c not in (iso_cfg["smiles_column"], iso_cfg["id_column"]) and not c.startswith("Unnamed")]
    if extra:
        raise ValidationError(f"{label}: unexpected extra columns {extra}")


def validate_blind_frame(df: pd.DataFrame, iso_cfg: dict) -> None:
    validate_blind_columns(df.columns, iso_cfg)
    idc, smc = iso_cfg["id_column"], iso_cfg["smiles_column"]
    if len(df) == 0:
        raise ValidationError("blind file has no rows")
    if df[idc].isna().any() or (df[idc].astype(str).str.strip() == "").any():
        raise ValidationError("blind file has missing compound identifiers")
    if df[idc].duplicated().any():
        d = df.loc[df[idc].duplicated(keep=False), idc].unique()[:5].tolist()
        raise ValidationError(f"blind file has duplicate compound identifiers, e.g. {d}")
    if df[smc].isna().any() or (df[smc].astype(str).str.strip() == "").any():
        raise ValidationError("blind file has missing SMILES")


def validate_feature_matrix(X: np.ndarray, expected_cols: int, label="X"):
    if X.ndim != 2 or X.shape[1] != expected_cols:
        raise ValidationError(f"{label}: expected 2-D with {expected_cols} columns, got shape {X.shape}")
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).any(axis=1).sum())
        raise ValidationError(f"{label}: {bad} row(s) contain non-finite values")


def validate_descriptor_names(actual, reference, allow_mismatch=False):
    actual, reference = list(actual), list(reference)
    if actual == reference:
        return
    only_a = [n for n in actual if n not in set(reference)]
    only_r = [n for n in reference if n not in set(actual)]
    msg = (f"RDKit descriptor list differs from the JB cache reference: {len(actual)} vs {len(reference)} names; "
           f"only-in-installed={only_a[:5]}, only-in-reference={only_r[:5]}, "
           f"same-set-different-order={set(actual) == set(reference)}. Install the RDKit version in requirements.txt.")
    if allow_mismatch:
        return msg
    raise ValidationError(msg)


def validate_weights(w: dict, members, tol=1e-9):
    if set(w) != set(members):
        raise ValidationError(f"ensemble weights keys {sorted(w)} != members {sorted(members)}")
    vals = np.array([w[m] for m in members], dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValidationError("ensemble weights must be finite and non-negative")
    if abs(vals.sum() - 1.0) > 1e-6 + tol:
        raise ValidationError(f"ensemble weights sum to {vals.sum():.9f}, expected 1")


def validate_scores(scores: np.ndarray, n_expected: int, label="scores"):
    scores = np.asarray(scores)
    if scores.shape != (n_expected,):
        raise ValidationError(f"{label}: expected shape ({n_expected},), got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValidationError(f"{label}: {int((~np.isfinite(scores)).sum())} missing/non-finite prediction(s)")


def validate_submission(sub: pd.DataFrame, blind: pd.DataFrame, iso_cfg: dict, expected_positives: int):
    cols = list(iso_cfg["submission"]["columns"])
    if list(sub.columns) != cols:
        raise ValidationError(f"submission columns {list(sub.columns)} != required {cols}")
    idc, smc, lab = iso_cfg["id_column"], iso_cfg["smiles_column"], iso_cfg["label_column"]
    if len(sub) != len(blind):
        raise ValidationError(f"submission has {len(sub)} rows, blind input has {len(blind)}")
    if sub[idc].duplicated().any():
        raise ValidationError("submission has duplicate compound identifiers")
    if not (sub[idc].astype(str).values == blind[idc].astype(str).values).all():
        raise ValidationError("submission compound IDs are reordered or mismatched relative to the blind file")
    if not (sub[smc].astype(str).values == blind[smc].astype(str).values).all():
        raise ValidationError("submission SMILES do not match the blind file row-for-row")
    if sub[lab].isna().any() or sub[lab].dtype != bool:
        raise ValidationError(f"submission column {lab} must be boolean with no missing values")
    n_pos = int(sub[lab].sum())
    if n_pos != int(expected_positives):
        raise ValidationError(f"submission has {n_pos} positives, expected exactly {expected_positives}")
