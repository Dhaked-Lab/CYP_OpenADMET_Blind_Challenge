"""RDKit featurization ([JB] featurize.py): descriptors (non-finite -> 0) + Morgan r2 count (2048, sqrt)
+ Morgan r3 count (1024, sqrt) + MACCS(167). RDKit is imported lazily."""
from __future__ import annotations
import numpy as np
from .validation import ValidationError, validate_descriptor_names


def mol_of(smi):
    """[JB] featurize.mol_of: strict parse, then lenient fallback sanitisation."""
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is not None:
                try:
                    m.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                                     ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                except Exception:
                    return None
        return m
    except Exception:
        return None


def scaffold_keys(smiles_list):
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    keys = []
    for smi in smiles_list:
        m = mol_of(smi)
        try:
            k = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m is not None else ""
        except Exception:
            k = ""
        keys.append(k)
    return keys


def descriptor_names():
    from rdkit.Chem import Descriptors
    return [n for n, _ in Descriptors._descList]


def feature_layout(cfg_features, n_descriptors):
    """Column layout as plain integers (testable without RDKit)."""
    f = cfg_features
    sizes = [("desc", n_descriptors), ("m2", f["morgan2_bits"]), ("m3", f["morgan3_bits"]), ("mk", f["maccs_bits"])]
    out, start = {}, 0
    for name, n in sizes:
        out[name] = (start, start + n); start += n
    out["total"] = start
    return out


def feature_names(cfg_features, desc_names=None):
    f = cfg_features
    d = desc_names if desc_names is not None else descriptor_names()
    return (list(d) + [f"m2_{i}" for i in range(f["morgan2_bits"])] + [f"m3_{i}" for i in range(f["morgan3_bits"])]
            + [f"mk_{i}" for i in range(f["maccs_bits"])])


def check_descriptor_definition(cfg_features, reference_names, logger=None):
    msg = validate_descriptor_names(descriptor_names(), reference_names, cfg_features["allow_descriptor_mismatch"])
    if msg and logger:
        logger.warning(msg)
    return msg


def featurize(smiles, cfg_features, ids=None):
    """Return (X float32, ok_mask). Invalid SMILES give NaN rows and ok_mask False (caller decides;
    pipeline steps raise listing the offending identifiers - JB-05)."""
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys
    from rdkit.Chem import rdFingerprintGenerator as G
    RDLogger.DisableLog("rdApp.*")
    f = cfg_features
    g2 = G.GetMorganGenerator(radius=f["morgan2_radius"], fpSize=f["morgan2_bits"], countSimulation=False)
    g3 = G.GetMorganGenerator(radius=f["morgan3_radius"], fpSize=f["morgan3_bits"])
    nd = len(Descriptors._descList)
    width = nd + f["morgan2_bits"] + f["morgan3_bits"] + f["maccs_bits"]
    X = np.full((len(smiles), width), np.nan, dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        m = mol_of(smi)
        if m is None:
            continue
        d = []
        for _, fn in Descriptors._descList:
            try:
                v = fn(m); v = v if np.isfinite(v) else 0.0
            except Exception:
                v = 0.0
            d.append(float(v))
        a2 = np.sqrt(g2.GetCountFingerprintAsNumPy(m).astype(np.float32))
        a3 = np.sqrt(g3.GetCountFingerprintAsNumPy(m).astype(np.float32))
        mk = np.zeros(f["maccs_bits"], dtype=np.float32)
        mk[list(MACCSkeys.GenMACCSKeys(m).GetOnBits())] = 1.0
        X[i] = np.hstack([np.array(d, dtype=np.float64), a2, a3, mk]).astype(np.float32)
        ok[i] = True
    return X, ok


def require_all_valid(ok, ids, label):
    if not ok.all():
        bad = [str(ids[i]) for i in np.where(~ok)[0]]
        raise ValidationError(f"{label}: {len(bad)} unparseable SMILES (ids: {bad[:10]})")


def m2_binary(X, names):
    idx = [i for i, n in enumerate(names) if n.startswith("m2_")]
    return (X[:, idx] > 0).astype(np.float32)
