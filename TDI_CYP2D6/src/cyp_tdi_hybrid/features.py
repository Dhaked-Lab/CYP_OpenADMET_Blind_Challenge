"""Featurization: ECFP4 (2048-bit Morgan) + RDKit2D descriptors.

REUSED VERBATIM from the Jermy TDI reproduction ZIP
(`JERMY_TDI_REPRODUCTION/src/cyp_tdi_repro/features.py`), which itself is a
verbatim copy of Jermy's own
`openadmet_scripts-main/CYP_Challenge/src/cyp_submission/features.py`.
No parameters changed (radius=2, fpSize=2048, full RDKit descriptor set via
`Descriptors.CalcMolDescriptors`). This matches the TeamPozeSCAF methodology
report's Section 1.1/1.2 exactly (ECFP4 2048-bit + RDKit 2D descriptors,
reported count 217 -- the actual installed-environment count is measured and
reported by 02_features.py rather than hard-coded, since RDKit descriptor
counts can vary slightly by RDKit version).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors, rdFingerprintGenerator

_ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(smiles_list: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _ECFP4_GEN.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def rdkit2d(smiles_list: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append(Descriptors.CalcMolDescriptors(mol) if mol is not None else {})
    return pd.DataFrame(rows)


def featurize_classical(smiles_list: list[str]) -> pd.DataFrame:
    """ECFP4 (2048 cols, prefixed ecfp4_) + RDKit2D descriptors (prefixed as-is)."""
    fp = ecfp4(smiles_list)
    fp_df = pd.DataFrame(fp, columns=[f"ecfp4_{i}" for i in range(fp.shape[1])])
    desc_df = rdkit2d(smiles_list).reset_index(drop=True)
    return pd.concat([fp_df, desc_df], axis=1)
