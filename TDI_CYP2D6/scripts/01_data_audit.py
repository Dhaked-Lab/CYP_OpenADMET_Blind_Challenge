"""CYP2D6 STEP 01 -- Data audit + cleaning/standardization.

Uses the project's authoritative data-loading architecture from
cyp_tdi_hybrid.data.

Workflow:
1. Load raw CYP2D6 positive/negative files through load_isoform_raw().
2. Audit missing SMILES, RDKit validity, duplicates, and label conflicts.
3. Load the authoritative cleaned dataset through load_isoform_clean().
4. Preserve RAW SMILES; no canonicalisation, salt stripping,
   tautomerisation, or other chemical standardisation is applied.

Important:
- Only CYP2D6 Positive/Negative files are processed here.
- TEST-BLINDED.csv is NOT read in Step 01.
- data.py is the authoritative source for raw/clean data loading.

Outputs:
    outputs/CYP2D6/01_data_audit_report.json
    outputs/CYP2D6/01_clean_cyp2d6.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyp_tdi_hybrid.data import load_isoform_raw, load_isoform_clean
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir

from rdkit import Chem, RDLogger


ISO = "CYP2D6"


# ---------------------------------------------------------------------
# RDKit helper
# ---------------------------------------------------------------------

def mol_of(smiles):
    """Return an RDKit molecule or None for an invalid/missing SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def audit_dataset(
    df: pd.DataFrame,
    raw_report: dict,
    log: StepLogger,
) -> dict:
    """Audit the raw CYP2D6 dataset.

    The dataframe comes directly from load_isoform_raw(), therefore its
    columns are:

        Molecule_Name
        SMILES
        is_TDI

    This function performs audit checks only. It does not modify the
    authoritative cleaned dataset.
    """

    audit = dict(raw_report)

    # -------------------------------------------------------------
    # Basic counts
    # -------------------------------------------------------------

    audit["raw_combined_rows"] = int(len(df))

    audit["missing_molecule_name"] = int(
        df["Molecule_Name"].isna().sum()
    )

    audit["missing_smiles"] = int(
        df["SMILES"].isna().sum()
    )

    audit["missing_label"] = int(
        df["is_TDI"].isna().sum()
    )

    audit["n_positive"] = int(
        (df["is_TDI"] == True).sum()
    )

    audit["n_negative"] = int(
        (df["is_TDI"] == False).sum()
    )

    audit["positive_rate"] = (
        audit["n_positive"] / len(df)
        if len(df)
        else 0.0
    )

    # -------------------------------------------------------------
    # Work only on rows having SMILES for chemical validation
    # -------------------------------------------------------------

    smi_df = df.dropna(subset=["SMILES"]).copy()

    # -------------------------------------------------------------
    # RDKit validation
    # -------------------------------------------------------------

    RDLogger.DisableLog("rdApp.*")

    mols = [
        mol_of(smiles)
        for smiles in smi_df["SMILES"]
    ]

    valid_mask = [
        mol is not None
        for mol in mols
    ]

    smi_df["valid"] = valid_mask

    audit["invalid_smiles"] = int(
        (~smi_df["valid"]).sum()
    )

    audit["valid_smiles"] = int(
        smi_df["valid"].sum()
    )

    # -------------------------------------------------------------
    # Canonical SMILES
    # -------------------------------------------------------------

    smi_df["canon"] = [
        Chem.MolToSmiles(mol)
        if mol is not None
        else None
        for mol in mols
    ]

    # -------------------------------------------------------------
    # Connectivity InChIKey
    # -------------------------------------------------------------

    def connectivity_key(mol):
        if mol is None:
            return None

        try:
            return Chem.MolToInchiKey(mol).split("-")[0]
        except Exception:
            return None

    smi_df["ik_conn"] = [
        connectivity_key(mol)
        for mol in mols
    ]

    # -------------------------------------------------------------
    # Duplicate checks
    # -------------------------------------------------------------

    audit["duplicate_raw_smiles"] = int(
        smi_df["SMILES"].duplicated().sum()
    )

    audit["unique_raw_smiles"] = int(
        smi_df["SMILES"].nunique()
    )

    audit["duplicate_canonical"] = int(
        smi_df["canon"]
        .dropna()
        .duplicated()
        .sum()
    )

    audit["unique_canonical"] = int(
        smi_df["canon"]
        .dropna()
        .nunique()
    )

    audit["duplicate_inchikey_connectivity"] = int(
        smi_df["ik_conn"]
        .dropna()
        .duplicated()
        .sum()
    )

    audit["unique_inchikey_connectivity"] = int(
        smi_df["ik_conn"]
        .dropna()
        .nunique()
    )

    audit["duplicate_molecule_names"] = int(
        df["Molecule_Name"]
        .dropna()
        .duplicated()
        .sum()
    )

    audit["unique_molecule_names"] = int(
        df["Molecule_Name"]
        .dropna()
        .nunique()
    )

    # -------------------------------------------------------------
    # Label conflicts
    #
    # Check whether the same molecule representation occurs with
    # both positive and negative labels.
    # -------------------------------------------------------------

    for key in ("SMILES", "canon", "ik_conn"):

        tmp = smi_df.dropna(subset=[key])

        grouped = (
            tmp.groupby(key)["is_TDI"]
            .nunique()
        )

        audit[f"label_conflicts_by_{key}"] = int(
            (grouped > 1).sum()
        )

    # -------------------------------------------------------------
    # Log summary
    # -------------------------------------------------------------

    log.log(
        f"[{ISO}] Raw rows: {audit['raw_combined_rows']}"
    )

    log.log(
        f"[{ISO}] Positive: {audit['n_positive']}"
    )

    log.log(
        f"[{ISO}] Negative: {audit['n_negative']}"
    )

    log.log(
        f"[{ISO}] Missing SMILES: {audit['missing_smiles']}"
    )

    log.log(
        f"[{ISO}] Invalid SMILES: {audit['invalid_smiles']}"
    )

    log.log(
        f"[{ISO}] Duplicate raw SMILES: "
        f"{audit['duplicate_raw_smiles']}"
    )

    log.log(
        f"[{ISO}] Label conflicts by raw SMILES: "
        f"{audit['label_conflicts_by_SMILES']}"
    )

    return audit


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    ensure_dirs()

    log = StepLogger(
        f"{ISO}_01_data_audit"
    )

    out_dir = isoform_dir(ISO)

    # ================================================================
    # 1. LOAD RAW CYP2D6 DATA
    #
    # IMPORTANT:
    # This uses data.py -> RAW_FILES.
    # No data_in/ path is constructed here.
    #
    # TEST-BLINDED.csv is NOT touched.
    # ================================================================

    raw_result = load_isoform_raw(ISO)

    raw_df = raw_result["df"]
    raw_report = raw_result["report"]

    log.log(
        f"[{ISO}] Loaded raw CYP2D6 positive/negative data."
    )

    log.log(
        f"[{ISO}] Combined raw rows: {len(raw_df)}"
    )

    # ================================================================
    # 2. AUDIT RAW DATA
    # ================================================================

    audit = audit_dataset(
        raw_df.copy(),
        raw_report,
        log,
    )

    audit_path = (
        out_dir /
        "01_data_audit_report.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log.log(
        f"Wrote audit report: {audit_path}"
    )

    # ================================================================
    # 3. LOAD AUTHORITATIVE CLEAN DATASET
    #
    # data.py performs:
    # - blank-row removal
    # - duplicate/conflict checks
    # - exact duplicate removal
    # - RDKit SMILES validity filtering
    #
    # ================================================================

    clean_result = load_isoform_clean(ISO)

    clean_df = clean_result["df"]
    clean_report = clean_result["report"]

    # ================================================================
    # 4. SAVE CLEAN DATASET
    #
    # data.py uses the canonical project column:
    #     Molecule_Name
    #     SMILES
    #     is_TDI
    #
    # ================================================================

    clean_path = (
        out_dir /
        "01_clean_cyp2d6.csv"
    )

    clean_df.to_csv(
        clean_path,
        index=False,
    )

    log.log(
        f"Wrote clean dataset: {clean_path}"
    )

    # ================================================================
    # 5. LOG FINAL CLEAN COUNTS
    # ================================================================

    log.log(
        f"[{ISO}] Final clean dataset: "
        f"n={len(clean_df)}, "
        f"positive={int(clean_df['is_TDI'].sum())}, "
        f"negative={int((~clean_df['is_TDI']).sum())}"
    )

    log.log(
        f"[{ISO}] Cleaning report: "
        f"{json.dumps(clean_report, default=str)}"
    )

    log.done()


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
