"""Per-isoform data loading.

REPRODUCTION DECISION (see README, "Divergences from the reference designs"):
Jermy's own code loads ONE combined table (both isoforms, one row per
molecule, NaN where untested). This project loads each isoform separately
(the shipped pipeline runs CYP2D6) with its own duplicate/validity checks.
This is a deliberate divergence from Jermy's combined-table design.

Each isoform's raw data arrives as two files (Positive / Negative), each
containing a small number of real rows followed by a very large block of
fully-blank rows (a spreadsheet-export artifact). This module drops the
blank rows, checks for duplicate/conflicting molecules, validates SMILES with
RDKit, and returns one clean per-isoform DataFrame with columns
[Molecule_Name, SMILES, is_TDI].
"""

from __future__ import annotations

import pandas as pd

from .paths import RAW_FILES, TEST_BLINDED_PATH


def _read_raw_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_isoform_raw(isoform: str) -> dict:
    """Load the Positive/Negative CSVs for one isoform, drop fully-blank
    trailing rows, and report exact counts. Does NOT deduplicate or validate
    SMILES yet -- that happens in load_isoform_clean(), which is the
    authoritative entry point for downstream steps.
    """
    pos_path = RAW_FILES[isoform]["positive"]
    neg_path = RAW_FILES[isoform]["negative"]
    label_col = f"{isoform}_is_TDI"

    report = {"isoform": isoform}
    frames = []
    for tag, path in (("positive", pos_path), ("negative", neg_path)):
        raw = _read_raw_csv(path)
        total_rows = len(raw)
        blank_mask = raw["Molecule_Name"].isna() & raw["SMILES"].isna() & raw[label_col].isna()
        blank_rows = int(blank_mask.sum())
        clean = raw.loc[~blank_mask].copy()
        report[f"{tag}_total_rows"] = total_rows
        report[f"{tag}_blank_rows"] = blank_rows
        report[f"{tag}_real_rows"] = len(clean)
        clean["is_TDI"] = clean[label_col].astype(bool)
        clean = clean[["Molecule_Name", "SMILES", "is_TDI"]]
        frames.append(clean)

    combined = pd.concat(frames, ignore_index=True)
    report["combined_real_rows"] = len(combined)
    return {"df": combined, "report": report}


def load_isoform_clean(isoform: str) -> dict:
    """Authoritative loader: drop blanks, check for duplicate/conflicting
    molecules, validate SMILES with RDKit. Returns a dict with the clean
    DataFrame and a report dict (both are also written to disk by
    01_data_audit.py / 02_features.py).
    """
    from rdkit import Chem  # local import: RDKit may not be installed until step 00 passes

    raw = load_isoform_raw(isoform)
    df, report = raw["df"], raw["report"]

    # Duplicate / label-conflict check (by Molecule_Name)
    dup_name = df["Molecule_Name"].duplicated(keep=False)
    n_dup_names = int(df.loc[dup_name, "Molecule_Name"].nunique())
    conflicts = 0
    if n_dup_names:
        for name, grp in df.loc[dup_name].groupby("Molecule_Name"):
            if grp["is_TDI"].nunique() > 1:
                conflicts += 1
    report["duplicate_molecule_names"] = n_dup_names
    report["label_conflicts"] = conflicts

    # Drop exact duplicate rows (same name, same SMILES, same label)
    before = len(df)
    df = df.drop_duplicates(subset=["Molecule_Name", "SMILES", "is_TDI"]).reset_index(drop=True)
    report["exact_duplicate_rows_dropped"] = before - len(df)

    # RDKit validity check
    valid_mask = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None if isinstance(s, str) else False)
    report["invalid_smiles_rows"] = int((~valid_mask).sum())
    df = df.loc[valid_mask].reset_index(drop=True)

    report["final_clean_rows"] = len(df)
    report["final_positive"] = int(df["is_TDI"].sum())
    report["final_negative"] = int((~df["is_TDI"]).sum())
    report["final_pct_positive"] = round(100 * df["is_TDI"].mean(), 2) if len(df) else 0.0

    return {"df": df, "report": report}


def load_test_blinded() -> pd.DataFrame | None:
    """Returns the blind test set if TEST-BLINDED.csv exists at
    data/raw/TEST-BLINDED.csv, else None. Only used by the blind-prediction
    stages (step 09 and the blind stage of step 11)."""
    if not TEST_BLINDED_PATH.exists():
        return None
    df = _read_raw_csv(TEST_BLINDED_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df
