"""data_audit, data_clean, features, similarity, scaffold_cv (original scripts 01-05)."""
from __future__ import annotations
import collections
from pathlib import Path
import numpy as np
import pandas as pd
from ..registry import step
from ..io_utils import write_json, save_npy, read_json
from ..discovery import discover_inputs, load_training_frames, load_blind
from ..validation import validate_blind_frame, ValidationError
from ..core.nn_features import tanimoto_binary, nn_pos_features
from ..core.folds import make_folds, assign_none_buckets, assert_scaffold_purity, universe_folds
from . import _common as C


def _detect_cols(df, iso_cfg):
    return iso_cfg["smiles_column"], iso_cfg["label_column"], iso_cfg["id_column"]


def _frames(ctx):
    inp = discover_inputs(ctx.data_dir, ctx.iso_cfg)
    out = []
    for name, role, df, n_raw, n_empty in load_training_frames(inp, ctx.iso_cfg):
        smi, lab, idc = _detect_cols(df, ctx.iso_cfg)
        if role is None:
            from ..validation import parse_label
            y = df[lab].map(parse_label)
            if y.isna().any():
                raise ValidationError(f"{name}: missing labels")
            y = y.astype(int).values
        else:
            y = np.full(len(df), role)
        out.append((name, df, y, n_raw, n_empty, smi, idc))
    return out


# ------------------------------------------------------------------------------------ data_audit
@step("data_audit", "Audit raw inputs: schema, isoform, labels, SMILES validity, duplicates (original 01)",
      version="1.0", config_keys=("features",),
      outputs=("audit_report.json",), inputs_fn=C.raw_input_files,
      inputs_doc="data/CYP3A4_Positive.csv + CYP3A4_Negative.csv (or CYP3A4.csv) + TEST-BLINDED.csv",
      outputs_doc="audit_report.json")
def data_audit(ctx):
    from rdkit import Chem, RDLogger
    from ..featurization import mol_of
    RDLogger.DisableLog("rdApp.*")
    iso = ctx.iso_cfg
    rep, frames = {"isoform": iso["isoform"]}, []
    for name, df, y, n_raw, n_empty, smi, idc in _frames(ctx):
        rep[name] = dict(raw_rows=n_raw, empty_rows=n_empty, kept_rows=len(df), smiles_col=smi, id_col=idc,
                         missing_smiles=int(df[smi].isna().sum()),
                         label_values=dict(collections.Counter(df[iso["label_column"]].astype(str))))
        frames.append(pd.DataFrame({"Molecule_Name": df[idc], "SMILES": df[smi], "TDI_Label": y}))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["SMILES"]).reset_index(drop=True)
    mols = [mol_of(s) for s in d["SMILES"]]
    d["valid"] = [m is not None for m in mols]
    d["canon"] = [Chem.MolToSmiles(m) if m is not None else None for m in mols]

    def ik(m):
        try:
            return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None
        except Exception:
            return None
    d["ik_conn"] = [ik(m) for m in mols]
    rep["invalid_smiles"] = int((~d["valid"]).sum())
    rep["duplicate_raw_smiles"] = int(d["SMILES"].duplicated().sum())
    rep["duplicate_canonical"] = int(d["canon"].dropna().duplicated().sum())
    rep["duplicate_inchikey_connectivity"] = int(d["ik_conn"].dropna().duplicated().sum())
    rep["duplicate_ids"] = int(d["Molecule_Name"].duplicated().sum())
    for key in ("SMILES", "canon", "ik_conn"):
        g = d.dropna(subset=[key]).groupby(key)["TDI_Label"].nunique()
        rep[f"label_conflicts_by_{key}"] = int((g > 1).sum())
    rep["n_positive"] = int((d.TDI_Label == 1).sum()); rep["n_negative"] = int((d.TDI_Label == 0).sum())
    rep["positive_rate"] = rep["n_positive"] / len(d)
    ref = iso.get("reference_counts") or {}
    warn = []
    for k in ("n_positive", "n_negative"):
        if k in ref and rep[k] != ref[k]:
            warn.append(f"{k}={rep[k]} differs from reference {ref[k]} (informational)")
    blind = load_blind(discover_inputs(ctx.data_dir, iso), iso)
    validate_blind_frame(blind, iso)
    rep["blind"] = {"rows": len(blind), "invalid_smiles": int(sum(mol_of(s) is None for s in blind["SMILES"]))}
    rep["warnings"] = warn
    for w in warn:
        ctx.logger.warning(w)
    write_json(rep, ctx.step_dir("data_audit") / "audit_report.json")


# ------------------------------------------------------------------------------------ data_clean
@step("data_clean", "Build the clean training table: drop unparseable SMILES / label conflicts / exact duplicates (original 02)",
      depends_on=("data_audit",), version="1.0", outputs=("clean_data.csv", "clean_report.json"),
      inputs_fn=C.train_input_files, inputs_doc="raw training CSVs", outputs_doc="clean_data.csv (Molecule_Name,SMILES,TDI_Label)")
def data_clean(ctx):
    from ..featurization import mol_of
    frames = []
    for name, df, y, n_raw, n_empty, smi, idc in _frames(ctx):
        frames.append(pd.DataFrame({"Molecule_Name": df[idc], "SMILES": df[smi], "TDI_Label": y}))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["SMILES"])
    ok = np.array([mol_of(s) is not None for s in d["SMILES"]])
    n_bad = int((~ok).sum()); d = d[ok]
    nun = d.groupby("SMILES")["TDI_Label"].transform("nunique")
    n_conf = int((nun > 1).sum()); d = d[nun == 1]
    n0 = len(d); d = d.drop_duplicates("SMILES").reset_index(drop=True)
    d.to_csv(ctx.step_dir("data_clean") / "clean_data.csv", index=False)
    write_json({"dropped_unparseable": n_bad, "dropped_label_conflict_rows": n_conf, "removed_exact_duplicates": n0 - len(d),
                "n": len(d), "n_positive": int(d.TDI_Label.sum()), "n_negative": int((d.TDI_Label == 0).sum())},
               ctx.step_dir("data_clean") / "clean_report.json")
    ctx.logger.info(f"clean n={len(d)} pos={int(d.TDI_Label.sum())}")


# ------------------------------------------------------------------------------------ features
@step("features", "Molecular features: RDKit descriptors + Morgan r2/r3 sqrt counts + MACCS (original 03)",
      depends_on=("data_clean",), version="1.0", config_keys=("features",),
      outputs=("X_train.npy", "feature_names.json", "features_meta.json"),
      outputs_doc="X_train.npy (n x 3456 for the reference RDKit), feature_names.json")
def features(ctx):
    from .. import featurization as F
    from ..validation import validate_feature_matrix
    ref = read_json(ctx.assets_dir / "reference" / "jb_descriptor_names.json")
    msg = F.check_descriptor_definition(ctx.cfg["features"], ref, ctx.logger)
    d, y = C.load_clean(ctx)
    X, ok = F.featurize(d["SMILES"].tolist(), ctx.cfg["features"])
    F.require_all_valid(ok, d["Molecule_Name"].tolist(), "training set")
    names = F.feature_names(ctx.cfg["features"])
    validate_feature_matrix(X, len(names), "X_train")
    sd = ctx.step_dir("features")
    save_npy(sd / "X_train.npy", X); write_json(names, sd / "feature_names.json")
    write_json({"shape": list(X.shape), "n_descriptors": len(names) - 2048 - 1024 - 167,
                "descriptor_check": msg or "identical to JB cache (names and order)"}, sd / "features_meta.json")


# ------------------------------------------------------------------------------------ similarity
def leakage_check(S, y, topk, thr, n_sample=200, seed=0, tol=1e-6):
    """Verify nn_pos_features(self_pos=[i]) removes molecule i's own 1.0 similarity (original 04 check)."""
    pos_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(pos_idx, size=min(n_sample, len(pos_idx)), replace=False))
    failed = []
    for i in map(int, sample):
        row = S[i:i + 1]
        f_ex = nn_pos_features(row, pos_idx, self_pos=[i], topk=topk, thr=thr)
        f_no = nn_pos_features(row, pos_idx, self_pos=None, topk=topk, thr=thr)
        others = pos_idx[pos_idx != i]
        n_other_ones = int((S[i, others] > 1 - tol).sum())
        checks = [abs(S[i, i] - 1.0) < 1e-4, f_no[0, 0] >= 1 - tol, abs(float(f_ex[0, 0]) - float(S[i, others].max())) < 1e-5,
                  n_other_ones > 0 or f_ex[0, 0] < 1 - tol, int(f_no[0, 2]) - int(f_ex[0, 2]) == 1,
                  f_ex[0, 1] <= f_no[0, 1] + 1e-6]
        if not all(checks):
            failed.append(i)
    return {"n_tested": int(len(sample)), "n_failed": len(failed), "failed_indices": failed[:20],
            "status": "PASS" if (len(sample) > 0 and not failed) else "FAIL"}


@step("similarity", "Train-train Tanimoto matrix on binarised Morgan-r2 + self-exclusion leakage self-check (original 04)",
      depends_on=("features",), version="1.0", config_keys=("nn_features",),
      outputs=("sim_train_train.npy", "leakage_check.json"), outputs_doc="n x n float32 similarity; PASS/FAIL leakage check")
def similarity(ctx):
    from ..featurization import m2_binary
    d, y = C.load_clean(ctx); X, names = C.load_X(ctx)
    B = m2_binary(X, names)
    S = tanimoto_binary(B, B).astype(np.float32)
    if not np.allclose(np.diag(S), 1.0, atol=1e-4):
        raise ValidationError("similarity diagonal is not 1.0 - fingerprint problem")
    nn = ctx.cfg["nn_features"]
    res = leakage_check(S, y, nn["topk"], nn["sim_threshold"])
    sd = ctx.step_dir("similarity")
    write_json({"n": len(d), "n_positive": int(y.sum()), **res}, sd / "leakage_check.json")
    if res["status"] != "PASS":
        raise ValidationError(f"nearest-positive self-exclusion check FAILED: {res}")
    save_npy(sd / "sim_train_train.npy", S)


# ------------------------------------------------------------------------------------ scaffold_cv
def fold_universe_path(cv, root):
    p = Path(cv["fold_universe"]["path"])
    return p if p.is_absolute() else Path(root) / p


def fold_universe_inputs(ctx):
    """Fingerprinted input of scaffold_cv: the fold-universe file, only when the strategy uses it."""
    cv = ctx.cfg["cv"]
    return [fold_universe_path(cv, ctx.root)] if cv["fold_strategy"] == "original_all_isoforms" else []


def load_fold_universe(cv, root):
    """[JB] cache/X_train.parquet SMILES after drop_duplicates (all isoforms, original row order), sha256-checked."""
    from ..io_utils import sha256_file
    p = fold_universe_path(cv, root)
    if not p.exists():
        raise ValidationError(f"fold universe file missing: {p}")
    got = sha256_file(p)
    if got != cv["fold_universe"]["sha256"]:
        raise ValidationError(f"fold universe sha256 mismatch for {p}: expected {cv['fold_universe']['sha256']}, got {got}")
    return pd.read_csv(p, keep_default_na=False)["SMILES"].tolist(), got


def compute_folds(smiles, cv, seed, root):
    """Return (folds, groups, info) aligned with `smiles`.
    cyp_only: scaffold folds over the given rows (behaviour of package version 1.0).
    original_all_isoforms: [JB] folds over the whole original training table, then the given rows are selected."""
    from ..featurization import scaffold_keys
    if cv["fold_strategy"] == "cyp_only":
        groups = assign_none_buckets(scaffold_keys(smiles), cv["none_bucket"])
        return make_folds(groups, k=cv["k"], seed=seed, argsort_kind=cv["argsort_kind"]), groups, {"fold_strategy": "cyp_only"}
    universe, sha = load_fold_universe(cv, root)
    ugroups = assign_none_buckets(scaffold_keys(universe), cv["none_bucket"])
    folds, groups, fu = universe_folds(universe, ugroups, smiles, k=cv["k"], seed=seed, argsort_kind=cv["argsort_kind"])
    assert_scaffold_purity(ugroups, fu)
    return folds, groups, {"fold_strategy": "original_all_isoforms", "fold_universe_sha256": sha, "n_universe": len(universe),
                           "n_universe_scaffold_groups": len(set(ugroups)), "universe_fold_sizes": np.bincount(fu, minlength=cv["k"]).tolist()}


@step("scaffold_cv", "Murcko-scaffold greedy-balanced K-fold assignment, seed 7, over cv.fold_strategy's row universe (original 05)",
      depends_on=("data_clean",), version="1.0", config_keys=("cv", "seeds.folds"),
      outputs=("folds.npy", "scaffold_groups.csv", "fold_report.json"), inputs_fn=fold_universe_inputs)
def scaffold_cv(ctx):
    d, y = C.load_clean(ctx)
    cv = ctx.cfg["cv"]
    folds, groups, info = compute_folds(d["SMILES"].tolist(), cv, ctx.cfg["seeds"]["folds"], ctx.root)
    assert_scaffold_purity(groups, folds)
    sd = ctx.step_dir("scaffold_cv")
    save_npy(sd / "folds.npy", folds)
    pd.DataFrame({"group": groups}).to_csv(sd / "scaffold_groups.csv", index=False)
    write_json({"k": cv["k"], "seed": ctx.cfg["seeds"]["folds"], "argsort_kind": cv["argsort_kind"], **info,
                "n_scaffold_groups": len(set(groups)),
                "per_fold": [{"fold": f, "n": int((folds == f).sum()), "pos": int(y[folds == f].sum()),
                              "pos_rate": float(y[folds == f].mean())} for f in range(cv["k"])]}, sd / "fold_report.json")
    ctx.logger.info(f"folds ({info['fold_strategy']}): sizes {np.bincount(folds, minlength=cv['k']).tolist()}")
