"""CYP2D6 STEP 02 -- Classical features: ECFP4 (2048-bit Morgan) + RDKit2D
descriptors. Matches TeamPozeSCAF report Section 1.1/1.2. Structural alerts
are added separately in step 03 (kept as its own reviewable step since they
are a RECONSTRUCTION, not documented fact -- see config/structural_alerts.yaml).

Requires: outputs/CYP2D6/01_clean_cyp2d6.csv (step 01)
Writes:   outputs/CYP2D6/02_features_ecfp4_rdkit2d.csv
          outputs/CYP2D6/02_features_summary.txt

Run:  python scripts/02_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from cyp_tdi_hybrid.features import featurize_classical
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir

ISO = "CYP2D6"


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_02_features")
    out_dir = isoform_dir(ISO)
    data_path = out_dir / "01_clean_cyp2d6.csv"
    log.require_file(data_path, hint="Run scripts/01_data_audit.py first.")

    df = pd.read_csv(data_path)
    log.log(f"[{ISO}] STEP 02: Featurizing {len(df)} molecules (ECFP4 2048-bit + RDKit2D)")

    feats = featurize_classical(df["SMILES"].tolist())
    n_ecfp = sum(c.startswith("ecfp4_") for c in feats.columns)
    n_desc = feats.shape[1] - n_ecfp
    missing = feats.isna().sum().sum()

    out = pd.concat([df[["Molecule_Name", "SMILES", "is_TDI"]], feats], axis=1)
    out_path = out_dir / "02_features_ecfp4_rdkit2d.csv"
    out.to_csv(out_path, index=False)

    summary = (
        f"CYP2D6 classical features\n"
        f"molecules: {len(df)}\n"
        f"ecfp4_bits: {n_ecfp}\n"
        f"rdkit2d_descriptors: {n_desc}  "
        f"(TeamPozeSCAF report states 217; actual count measured in THIS RDKit "
        f"installation, may differ slightly by RDKit version)\n"
        f"total_classical_feature_dim: {feats.shape[1]}\n"
        f"missing_values_total: {int(missing)}\n"
    )
    (out_dir / "02_features_summary.txt").write_text(summary)
    log.log(summary)
    log.log(f"  Wrote {out_path}")
    log.log(f"  Wrote {out_dir / '02_features_summary.txt'}")
    if missing:
        log.log("  NOTE: RDKit2D descriptors can contain NaN for some molecules/descriptors "
                 "(e.g. 3D-only descriptors on molecules without conformers). Step 05 imputes "
                 "these with the training-fold median before fitting tree models.")
    log.done()


if __name__ == "__main__":
    main()
