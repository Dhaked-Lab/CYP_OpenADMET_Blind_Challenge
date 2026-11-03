"""CYP2D6 STEP 05 -- Train the four-model ensemble (Random Forest, Extra
Trees, LightGBM, XGBoost) with Murcko-scaffold-fold out-of-fold (OOF) prediction, then
refit each model on the full labeled dataset for later blind-test prediction.

Model configurations (TeamPozeSCAF report Section 3.1 + this project's
intentional Jermy LightGBM substitution -- see README "Model configuration"):
  Random Forest : n_estimators=600, max_depth=8,  random_state=42
  Extra Trees   : n_estimators=600, max_depth=10, random_state=42
  LightGBM      : n_estimators=500, learning_rate=0.05, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8,
                   class_weight="balanced", random_state=42
                   (this is the JERMY hybrid config, intentionally replacing
                   TeamPozeSCAF's reported n_estimators=300/depth=4/15 leaves
                   LightGBM -- see README)
  XGBoost       : n_estimators=500, max_depth=5, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                   gamma=0.1, reg_alpha=0.1, reg_lambda=2.0,
                   objective="binary:logistic", eval_metric="logloss",
                   tree_method="hist", random_state=42, n_jobs=-1
                   (CONTROLLED MODEL-ADDITION EXPERIMENT: 4th model, same folds)

Ensemble: equal-weight mean of the four models' probabilities
          (RF + ExtraTrees + LightGBM + XGBoost) / 4.
          The legacy 3-model mean is also saved as ensemble3_oof_prob so old
          vs new can be compared.

Data-leakage protections:
  - Each fold's model is trained ONLY on the other 4 folds' molecules.
  - RDKit2D descriptor NaNs are imputed with the TRAINING fold's median only
    (a fresh imputer is fit per fold; the held-out fold never contributes to
    its own imputation values).
  - Final (all-data) models are separate objects, fit only after every OOF
    fold is complete; they are not used to generate OOF predictions.

Requires: outputs/CYP2D6/02_features_ecfp4_rdkit2d.csv (step 02)
          outputs/CYP2D6/03_structural_alerts.csv (step 03)
          outputs/CYP2D6/04_scaffold_folds.csv (step 04)
Writes:   outputs/CYP2D6/05_oof_predictions.csv  (per-model OOF probabilities)
          outputs/CYP2D6/09_model_parameters.json
          outputs/CYP2D6/models/ (pickled final models, for step 09 blind prediction)

Run:  python scripts/05_train_models_oof.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

ISO = "CYP2D6"

RF_PARAMS = dict(n_estimators=600, max_depth=8, random_state=42, n_jobs=-1)
ET_PARAMS = dict(n_estimators=600, max_depth=10, random_state=42, n_jobs=-1)
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                    colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1)
XGB_PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1,
                   reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                   tree_method="hist", random_state=42, n_jobs=-1)

LEGACY_MEMBERS = ["RF", "ExtraTrees", "LightGBM"]
ENSEMBLE_MEMBERS = ["RF", "ExtraTrees", "LightGBM", "XGBoost"]


def _feature_matrix(feat_df: pd.DataFrame, alert_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat_cols = [c for c in feat_df.columns if c.startswith("ecfp4_") or c not in
                 ("Molecule_Name", "SMILES", "is_TDI")]
    alert_cols = [c for c in alert_df.columns if c.startswith("alert_")]
    X = pd.concat([feat_df[feat_cols].reset_index(drop=True),
                    alert_df[alert_cols].reset_index(drop=True)], axis=1)
    return X, list(X.columns)


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_05_train_models_oof")
    out_dir = isoform_dir(ISO)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    feat_path = out_dir / "02_features_ecfp4_rdkit2d.csv"
    alert_path = out_dir / "03_structural_alerts.csv"
    fold_path = out_dir / "04_scaffold_folds.csv"
    log.require_file(feat_path, hint="Run scripts/02_features.py first.")
    log.require_file(alert_path, hint="Run scripts/03_structural_alerts.py first.")
    log.require_file(fold_path, hint="Run scripts/04_scaffold_folds.py first.")

    feat_df = pd.read_csv(feat_path)
    alert_df = pd.read_csv(alert_path)
    fold_df = pd.read_csv(fold_path)
    assert (feat_df["Molecule_Name"] == alert_df["Molecule_Name"]).all(), "row order mismatch: features vs alerts"
    assert (feat_df["Molecule_Name"] == fold_df["Molecule_Name"]).all(), "row order mismatch: features vs folds"

    X_raw, feature_names = _feature_matrix(feat_df, alert_df)
    y = feat_df["is_TDI"].astype(bool).to_numpy()
    y_int = y.astype(int)  # XGBoost only: explicit 0/1 labels (RF/ET/LGBM still receive the bool y)
    folds = fold_df["fold"].to_numpy()
    n_folds = int(folds.max()) + 1

    log.log(f"[{ISO}] STEP 05: training RF/ExtraTrees/LightGBM/XGBoost, {n_folds}-fold OOF, "
            f"n_molecules={len(y)}, n_features={X_raw.shape[1]}")

    oof = {name: np.full(len(y), np.nan) for name in ENSEMBLE_MEMBERS}

    for f in range(n_folds):
        tr_mask, va_mask = folds != f, folds == f
        log.log(f"  fold {f}: train={int(tr_mask.sum())} val={int(va_mask.sum())}")

        imputer = SimpleImputer(strategy="median").fit(X_raw.loc[tr_mask])
        X_tr = imputer.transform(X_raw.loc[tr_mask])
        X_va = imputer.transform(X_raw.loc[va_mask])
        y_tr = y[tr_mask]

        rf = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr)
        oof["RF"][va_mask] = rf.predict_proba(X_va)[:, 1]

        et = ExtraTreesClassifier(**ET_PARAMS).fit(X_tr, y_tr)
        oof["ExtraTrees"][va_mask] = et.predict_proba(X_va)[:, 1]

        lgbm = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr)
        oof["LightGBM"][va_mask] = lgbm.predict_proba(X_va)[:, 1]

        # XGBoost: same imputed fold matrices, same folds -- no new CV split
        xgb = XGBClassifier(**XGB_PARAMS).fit(X_tr, y_int[tr_mask])
        oof["XGBoost"][va_mask] = xgb.predict_proba(X_va)[:, 1]

    # Final models refit on ALL labeled data (used only for blind prediction, step 09)
    log.log("  Refitting final models on the full labeled dataset...")
    final_imputer = SimpleImputer(strategy="median").fit(X_raw)
    X_all = final_imputer.transform(X_raw)
    final_models = {
        "RF": RandomForestClassifier(**RF_PARAMS).fit(X_all, y),
        "ExtraTrees": ExtraTreesClassifier(**ET_PARAMS).fit(X_all, y),
        "LightGBM": LGBMClassifier(**LGBM_PARAMS).fit(X_all, y),
        "XGBoost": XGBClassifier(**XGB_PARAMS).fit(X_all, y_int),
    }
    with open(models_dir / "final_models.pkl", "wb") as f_:
        pickle.dump({"models": final_models, "imputer": final_imputer,
                     "feature_names": feature_names,
                     "ensemble_members": ENSEMBLE_MEMBERS,
                     "xgboost_params": XGB_PARAMS}, f_)

    oof_df = feat_df[["Molecule_Name", "SMILES", "is_TDI"]].copy()
    for name, arr in oof.items():
        oof_df[f"{name}_oof_prob"] = arr
    # Legacy 3-model mean kept for old-vs-new comparison; NOT used for thresholding.
    oof_df["ensemble3_oof_prob"] = oof_df[[f"{n}_oof_prob" for n in LEGACY_MEMBERS]].mean(axis=1)
    # New equal-weight 4-model mean = (RF + ET + LGBM + XGB) / 4 (consumed by steps 06/07)
    oof_df["ensemble_oof_prob"] = oof_df[[f"{n}_oof_prob" for n in ENSEMBLE_MEMBERS]].mean(axis=1)
    oof_path = out_dir / "05_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    params_path = out_dir / "09_model_parameters.json"
    with open(params_path, "w") as f_:
        json.dump({"RandomForest": RF_PARAMS, "ExtraTrees": ET_PARAMS,
                    "LightGBM_JERMY_HYBRID": LGBM_PARAMS,
                    "XGBoost": XGB_PARAMS,
                    "ensemble_members": ENSEMBLE_MEMBERS,
                    "ensemble_rule": "equal-weight mean of 4 models: (RF + ExtraTrees + LightGBM + XGBoost) / 4",
                    "note": "LightGBM uses the intentional Jermy hybrid config, "
                            "NOT TeamPozeSCAF's reported n_estimators=300/depth=4/15leaves.",
                    "n_features": X_raw.shape[1], "feature_names_head": feature_names[:5] + ["..."]},
                   f_, indent=2)

    log.log(f"  Wrote {oof_path}")
    log.log(f"  Wrote {params_path}")
    log.log(f"  Wrote {models_dir / 'final_models.pkl'}")
    for name in oof:
        n_oof = int(np.isfinite(oof[name]).sum())
        log.log(f"  {name}: n_oof_predicted={n_oof}/{len(y)}")
    log.done()


if __name__ == "__main__":
    main()
