"""CYP2D6 STEP 08 -- Final-model training check + reproducibility report.

Does NOT train anything (the final models were already fit on all labeled
data inside step 05, immediately after the OOF loop, matching Jermy's own
design where there is no separate "final training" script).
This step verifies the final-model artifact (RF + ExtraTrees + LightGBM +
XGBoost), checks that the controlled XGBoost-addition experiment left CV,
features, existing model parameters and the threshold grid untouched, and
writes the full reproducibility report (package versions + per-isoform
data/feature/fold counts).

Requires: outputs/CYP2D6/models/final_models.pkl (step 05)
Writes:   outputs/CYP2D6/10_reproduction_report.txt

Run:  python scripts/08_final_training_check.py
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir

ISO = "CYP2D6"

# ---- Baseline invariants for the controlled model-addition experiment ----
EXPECTED_N_FEATURES = 2278   # base-pipeline feature count; must not change
EXPECTED_N_FOLDS = 5         # Murcko Scaffold-CV (train on 4 folds, validate on 1)
EXPECTED_MEMBERS = ["RF", "ExtraTrees", "LightGBM", "XGBoost"]
EXPECTED_PARAMS = {
    "RF": dict(n_estimators=600, max_depth=8, random_state=42, n_jobs=-1),
    "ExtraTrees": dict(n_estimators=600, max_depth=10, random_state=42, n_jobs=-1),
    "LightGBM": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                      colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1),
    "XGBoost": dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1,
                     reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                     tree_method="hist", random_state=42, n_jobs=-1),
}
EXPECTED_GRID = dict(grid_min=0.10, grid_max=0.90, grid_step=0.02)


def _versions() -> dict:
    out = {}
    import sys as _sys
    out["python"] = _sys.version
    for mod in ("rdkit", "sklearn", "lightgbm", "xgboost", "pandas", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"NOT INSTALLED ({e})"
    return out


def _param_mismatches(model, expected: dict) -> list[str]:
    actual = model.get_params()
    bad = []
    for k, v in expected.items():
        a = actual.get(k, "<missing>")
        same = math.isclose(a, v, rel_tol=1e-12) if isinstance(v, float) and isinstance(a, (int, float)) else a == v
        if not same:
            bad.append(f"{k}: expected {v!r}, found {a!r}")
    return bad


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_08_final_training_check")
    out_dir = isoform_dir(ISO)
    models_path = out_dir / "models" / "final_models.pkl"
    log.require_file(models_path, hint="Run scripts/05_train_models_oof.py first.")
    log.log(f"[{ISO}] STEP 08: final model artifact present at {models_path} -- OK, no retraining needed.")

    data_report = json.loads((out_dir / "01_data_audit_report.json").read_text())
    feat_summary = (out_dir / "02_features_summary.txt").read_text()
    fold_df = pd.read_csv(out_dir / "04_scaffold_folds.csv")
    params = json.loads((out_dir / "09_model_parameters.json").read_text())

    # ------------------------------------------------------------------
    # Verification of the controlled XGBoost-addition experiment
    # ------------------------------------------------------------------
    passed: list[str] = []
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        (passed if ok else failures).append(f"{name}" + (f" -- {detail}" if detail and not ok else ""))

    with open(models_path, "rb") as f:
        bundle = pickle.load(f)
    models = bundle["models"]
    feature_names = list(bundle["feature_names"])

    # 1. Bundle contents: exactly RF + ExtraTrees + LightGBM + XGBoost
    check("final bundle contains exactly RF + ExtraTrees + LightGBM + XGBoost",
          sorted(models) == sorted(EXPECTED_MEMBERS), f"found {sorted(models)}")
    check("bundle ensemble_members == 4-model list",
          list(bundle.get("ensemble_members", [])) == EXPECTED_MEMBERS, f"found {bundle.get('ensemble_members')}")

    # 2. Parameters: XGBoost exact config + existing models unchanged
    for name, expected in EXPECTED_PARAMS.items():
        if name in models:
            bad = _param_mismatches(models[name], expected)
            check(f"{name} parameters match required configuration", not bad, "; ".join(bad))
    check("09_model_parameters.json lists XGBoost with the exact config",
          params.get("XGBoost") == EXPECTED_PARAMS["XGBoost"], f"found {params.get('XGBoost')}")

    # 3. Feature count + no feature-generation change
    check(f"bundle feature count == {EXPECTED_N_FEATURES}", len(feature_names) == EXPECTED_N_FEATURES,
          f"found {len(feature_names)}")
    check("imputer.n_features_in_ matches", getattr(bundle["imputer"], "n_features_in_", -1) == len(feature_names))
    for name, mdl in models.items():
        check(f"{name}.n_features_in_ matches", getattr(mdl, "n_features_in_", -1) == len(feature_names))
    feat_cols_raw = list(pd.read_csv(out_dir / "02_features_ecfp4_rdkit2d.csv", nrows=0).columns)
    alert_cols_raw = list(pd.read_csv(out_dir / "03_structural_alerts.csv", nrows=0).columns)
    rebuilt = ([c for c in feat_cols_raw if c.startswith("ecfp4_") or c not in ("Molecule_Name", "SMILES", "is_TDI")]
               + [c for c in alert_cols_raw if c.startswith("alert_")])
    check("feature names/order identical to step 02/03 outputs (no feature-generation change)",
          rebuilt == feature_names)
    check("model parameters JSON n_features matches", params.get("n_features") == len(feature_names))

    # 4. Murcko Scaffold-CV still the CV used (same fold file, same row order, same fold count)
    mol_names = pd.read_csv(out_dir / "02_features_ecfp4_rdkit2d.csv", usecols=["Molecule_Name"])["Molecule_Name"]
    check(f"fold file has {EXPECTED_N_FOLDS} folds (04_scaffold_folds.csv)",
          fold_df["fold"].nunique() == EXPECTED_N_FOLDS, f"found {fold_df['fold'].nunique()}")
    check("fold file row order matches feature file", (mol_names.to_numpy() == fold_df["Molecule_Name"].to_numpy()).all())

    # 5. OOF file: all four models produced OOF predictions on the same folds; ensemble is the /4 mean
    oof = pd.read_csv(out_dir / "05_oof_predictions.csv")
    oof_cols = [f"{n}_oof_prob" for n in EXPECTED_MEMBERS]
    check("OOF file has all four per-model columns", all(c in oof.columns for c in oof_cols))
    if all(c in oof.columns for c in oof_cols):
        check("every molecule has an OOF prediction from all four models", bool(oof[oof_cols].notna().all().all()))
        check("ensemble_oof_prob == (RF + ET + LGBM + XGB) / 4",
              bool(np.allclose(oof[oof_cols].mean(axis=1), oof["ensemble_oof_prob"])))

    # 6. Threshold file: grid unchanged, ensemble is the four-model one
    thr = json.loads((out_dir / "07_best_threshold.json").read_text())
    for k, v in EXPECTED_GRID.items():
        check(f"threshold {k} == {v}", math.isclose(thr.get(k, float('nan')), v, rel_tol=1e-9), f"found {thr.get(k)}")
    check("threshold file was produced from the 4-model ensemble",
          list(thr.get("ensemble_members", [])) == EXPECTED_MEMBERS)

    # 7. Informational (non-fatal): are scaffolds disjoint across folds? (plain Murcko, no chirality)
    scaffold_note = "not run"
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        smi = pd.read_csv(out_dir / "02_features_ecfp4_rdkit2d.csv", usecols=["SMILES"])["SMILES"]
        scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)
                if Chem.MolFromSmiles(s) is not None else "" for s in smi]
        n_split = int((pd.DataFrame({"s": scaf, "f": fold_df["fold"].to_numpy()}).groupby("s")["f"].nunique() > 1).sum())
        scaffold_note = (f"{n_split} plain-Murcko scaffolds span >1 fold"
                         + (" (scaffold-disjoint)" if n_split == 0 else
                            " (INFO only: step 04 may use a different scaffold variant)"))
    except Exception as e:  # noqa: BLE001
        scaffold_note = f"skipped ({e})"

    for line in passed:
        log.log(f"  [PASS] {line}")
    for line in failures:
        log.log(f"  [FAIL] {line}")
    log.log(f"  [INFO] scaffold-vs-fold check: {scaffold_note}")

    lines = [
        f"=== {ISO} reproduction report ===", "",
        "--- Package versions ---",
    ]
    for k, v in _versions().items():
        lines.append(f"  {k}: {v}")
    lines += ["", "--- Data ---",
              f"  molecules: {data_report['combined_real_rows']}",
              f"  positive: {data_report['n_positive']}",
              f"  negative: {data_report['n_negative']}",
              f"  pct_positive: {data_report['positive_rate'] * 100:.2f}",
              "", "--- Features ---", "  " + feat_summary.replace("\n", "\n  "),
              "--- Murcko Scaffold folds ---",
              f"  n_folds: {fold_df['fold'].nunique()}",
              f"  fold_sizes: {fold_df['fold'].value_counts().sort_index().to_dict()}",
              "", "--- Model configuration ---",
              json.dumps(params, indent=2),
              "", "--- XGBoost-addition verification ---"]
    lines += [f"  [PASS] {p_}" for p_ in passed] + [f"  [FAIL] {f_}" for f_ in failures]
    lines += [f"  [INFO] scaffold-vs-fold check: {scaffold_note}", "",
              f"Architecture: TeamPozeSCAF CYP2D6 (RF + ExtraTrees + LightGBM, "
              f"probability-averaged) with the JERMY hybrid LightGBM configuration "
              f"substituted for TeamPozeSCAF's reported LightGBM hyperparameters, "
              f"PLUS XGBoost as a fourth equal-weight member: (RF + ET + LGBM + XGB) / 4.",
              "This is a HYBRID RECONSTRUCTION, not a claim of exact TeamPozeSCAF reproduction.",
              ]
    report = "\n".join(lines) + "\n"
    report_path = out_dir / "10_reproduction_report.txt"
    report_path.write_text(report)
    log.log(report)
    log.log(f"  Wrote {report_path}")
    log.done()
    if failures:
        sys.exit(f"[{ISO}] STEP 08: {len(failures)} verification check(s) FAILED -- see report.")


if __name__ == "__main__":
    main()
