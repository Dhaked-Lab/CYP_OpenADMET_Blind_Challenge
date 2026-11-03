"""Input discovery: separate positive/negative files or one combined per-isoform CSV."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .validation import (ValidationError, validate_training_columns, validate_blind_columns,
                         parse_label)


class InputError(RuntimeError):
    pass


def discover_inputs(data_dir: Path, iso_cfg: dict) -> dict:
    data_dir = Path(data_dir)
    f = iso_cfg["files"]
    if not data_dir.is_dir():
        raise InputError(f"data directory not found: {data_dir}")
    pos, neg, comb, blind = (data_dir / f[k] for k in ("positive", "negative", "combined", "blind"))
    if pos.exists() and neg.exists():
        mode = "positive_negative"
        train = [pos, neg]
    elif comb.exists():
        mode, train = "combined", [comb]
    else:
        raise InputError(f"no training data for {iso_cfg['isoform']} in {data_dir}: expected "
                         f"{f['positive']} + {f['negative']} or {f['combined']}")
    if not blind.exists():
        raise InputError(f"blind input not found: {blind}")
    return {"mode": mode, "training": train, "blind": blind}


def read_raw_csv(path: Path):
    """Original read(): all-string, utf-8-sig, whitespace-only cells -> NaN, fully empty rows dropped."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    n_raw = len(df)
    blank = df.apply(lambda c: c.str.fullmatch(r"\s*"))
    df = df.mask(blank)
    n_empty = int(df.isna().all(axis=1).sum())
    return df.dropna(how="all").reset_index(drop=True), n_raw, n_empty


def load_training_frames(inputs: dict, iso_cfg: dict):
    """Return list of (name, role_label_or_None, df, n_raw, n_empty). Role label comes from the file
    role (original behaviour) and is CROSS-CHECKED against the label column (JB-10)."""
    lab = iso_cfg["label_column"]
    out = []
    for p in inputs["training"]:
        df, n_raw, n_empty = read_raw_csv(p)
        validate_training_columns(df.columns, iso_cfg, p.name)
        if inputs["mode"] == "positive_negative":
            role = 1 if p.name == iso_cfg["files"]["positive"] else 0
            vals = df[lab].map(parse_label)
            if vals.isna().any():
                raise ValidationError(f"{p.name}: missing values in label column {lab}")
            if not (vals == role).all():
                raise ValidationError(f"{p.name}: label column {lab} disagrees with file role "
                                      f"({'positive' if role else 'negative'}) in {int((vals != role).sum())} row(s) "
                                      "- swapped or mixed files?")
        else:
            role = None
        out.append((p.name, role, df, n_raw, n_empty))
    return out


def load_blind(inputs: dict, iso_cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(inputs["blind"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    blank = df.apply(lambda c: c.str.fullmatch(r"\s*"))
    df = df.mask(blank).dropna(how="all").reset_index(drop=True)
    validate_blind_columns(df.columns, iso_cfg, inputs["blind"].name)
    return df[[iso_cfg["id_column"], iso_cfg["smiles_column"]]]
