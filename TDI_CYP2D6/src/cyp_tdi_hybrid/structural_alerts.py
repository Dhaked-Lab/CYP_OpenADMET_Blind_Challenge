"""RECONSTRUCTED_TEAMPOZESCAF_ALERTS -- 13 mechanism-based CYP-TDI structural
alerts, reconstructed from the medicinal-chemistry literature because the
TeamPozeSCAF report's original 13 SMARTS definitions were not present in any
provided material (see config/structural_alerts.yaml header and
docs/STRUCTURAL_ALERTS_RECONSTRUCTION.md for the full search trail and
rationale). Do NOT treat this module's output as a recovery of TeamPozeSCAF's
actual original features -- it is a documented, labeled replacement.
"""

from __future__ import annotations

import pandas as pd
import yaml
from rdkit import Chem

from .paths import CONFIG_DIR

_ALERTS_YAML = CONFIG_DIR / "structural_alerts.yaml"


def load_alert_defs() -> list[dict]:
    cfg = yaml.safe_load(_ALERTS_YAML.read_text())
    alerts = cfg["alerts"]
    if len(alerts) != 13:
        raise ValueError(f"Expected exactly 13 structural alerts, found {len(alerts)}")
    return alerts


def _compiled_patterns() -> list[tuple[str, "Chem.Mol"]]:
    compiled = []
    for a in load_alert_defs():
        patt = Chem.MolFromSmarts(a["smarts"])
        if patt is None:
            raise ValueError(f"Alert {a['name']!r} has an unparseable SMARTS: {a['smarts']!r}")
        compiled.append((a["name"], patt))
    return compiled


def structural_alert_features(smiles_list: list[str]) -> pd.DataFrame:
    """Returns a DataFrame with 13 binary columns, `alert_<NAME>`, one row per
    input SMILES, in the same order as config/structural_alerts.yaml."""
    patterns = _compiled_patterns()
    cols = {f"alert_{name}": [] for name, _ in patterns}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        for name, patt in patterns:
            hit = int(mol is not None and mol.HasSubstructMatch(patt))
            cols[f"alert_{name}"].append(hit)
    return pd.DataFrame(cols)


def self_test() -> dict:
    """Sanity-checks every SMARTS pattern parses and reports a hit-rate on a
    small set of known example molecules (used by 03_structural_alerts.py)."""
    examples = {
        "furan": "c1ccoc1",
        "thiophene_mol": "c1ccc(s1)-c1ccccc1",
        "propargylamine": "C#CCN",
        "mdp_example_piperonal": "O=Cc1ccc2OCOc2c1",
        "phenylhydrazine": "c1ccccc1NN",
        "aniline": "Nc1ccccc1",
        "thioacetamide": "CC(N)=S",
        "nifedipine_core": "C1=CC(C)NC(C)=C1",
        "cyclopropylamine": "NC1CC1",
        "nitrobenzene": "[O-][N+](=O)c1ccccc1",
        "thiazolidinedione": "O=C1CSC(=O)N1",
        "acetaldehyde_hydrazone": "CC=NN",
        "propene": "C=CC",
        "benzene_no_hits": "c1ccccc1",
    }
    patterns = _compiled_patterns()
    results = {}
    for label, smi in examples.items():
        mol = Chem.MolFromSmiles(smi)
        results[label] = [name for name, patt in patterns if mol is not None and mol.HasSubstructMatch(patt)]
    return results
