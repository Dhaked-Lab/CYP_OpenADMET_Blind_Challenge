"""CYP2D6 STEP 09 -- Blind-test prediction.

ONLY run this after reviewing steps 01-08. Loads data/raw/TEST-BLINDED.csv
(if present), featurizes it with the SAME code as training (step 02/03),
imputes missing descriptor values using the imputer FIT ON TRAINING DATA
ONLY (from step 05's final_models.pkl -- never refit on test data), applies
the four final (all-training-data) models (RF, ExtraTrees, LightGBM,
XGBoost), averages their probabilities as (RF + ET + LGBM + XGB) / 4, and
applies the FROZEN threshold from step 06. This script never reads or
computes ground-truth labels for the blind set -- it only ever writes
predictions.

Requires: data/raw/TEST-BLINDED.csv
          outputs/CYP2D6/models/final_models.pkl (step 05)
          outputs/CYP2D6/07_best_threshold.json (step 06)
Writes:   outputs/CYP2D6/08_final_test_predictions.csv

Run:  python scripts/09_blind_prediction.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from cyp_tdi_hybrid.data import load_test_blinded
from cyp_tdi_hybrid.features import featurize_classical
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir
from cyp_tdi_hybrid.structural_alerts import structural_alert_features

ISO = "CYP2D6"
ENSEMBLE_MEMBERS = ["RF", "ExtraTrees", "LightGBM", "XGBoost"]


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_09_blind_prediction")
    out_dir = isoform_dir(ISO)

    test_df = load_test_blinded()
    if test_df is None:
        log.log(f"[{ISO}] STEP 09: data/raw/TEST-BLINDED.csv not found yet -- nothing to predict. "
                f"Place it there and re-run this script when you're ready for blind prediction.")
        log.done()
        return

    models_path = out_dir / "models" / "final_models.pkl"
    thr_path = out_dir / "07_best_threshold.json"
    log.require_file(models_path, hint="Run scripts/05_train_models_oof.py first.")
    log.require_file(thr_path, hint="Run scripts/06_threshold_optimization.py first.")

    with open(models_path, "rb") as f:
        bundle = pickle.load(f)
    imputer = bundle["imputer"]
    models = bundle["models"]
    feature_names = bundle["feature_names"]
    best_t = json.loads(thr_path.read_text())["best_threshold"]

    missing = [n for n in ENSEMBLE_MEMBERS if n not in models]
    if missing:
        raise SystemExit(f"[{ISO}] STEP 09: final_models.pkl is missing {missing} -- re-run step 05 (XGBoost version).")

    log.log(f"[{ISO}] STEP 09: predicting {len(test_df)} blind-test molecules "
            f"at FROZEN threshold={best_t:.2f}. Ground-truth labels are never read or inferred.")

    feats = featurize_classical(test_df["SMILES"].tolist())
    alerts = structural_alert_features(test_df["SMILES"].tolist())
    ecfp_cols = [c for c in feats.columns if c.startswith("ecfp4_")]
    desc_cols = [c for c in feats.columns if not c.startswith("ecfp4_")]
    alert_cols = [c for c in alerts.columns if c.startswith("alert_")]
    X = pd.concat([feats[ecfp_cols + desc_cols].reset_index(drop=True),
                    alerts[alert_cols].reset_index(drop=True)], axis=1)
    X = X.reindex(columns=feature_names)  # exact train-time column order
    X_imputed = imputer.transform(X)  # imputer fit on TRAINING data only (step 05)

    probs = {}
    for name in ENSEMBLE_MEMBERS:
        probs[name] = models[name].predict_proba(X_imputed)[:, 1]
    # Equal-weight four-model ensemble: (RF + ET + LGBM + XGB) / 4
    ensemble_prob = (probs["RF"] + probs["ExtraTrees"] + probs["LightGBM"] + probs["XGBoost"]) / 4
    pred = ensemble_prob >= best_t

    out = test_df[["Molecule_Name", "SMILES"]].copy() if "Molecule_Name" in test_df.columns else pd.DataFrame({"SMILES": test_df["SMILES"]})
    for name, arr in probs.items():
        out[f"{name}_prob"] = arr
    out["ensemble_prob"] = ensemble_prob
    out["threshold_used"] = best_t
    out[f"{ISO}_is_TDI_predicted"] = pred

    out_path = out_dir / "08_final_test_predictions.csv"
    out.to_csv(out_path, index=False)
    log.log(f"  Predicted positive: {int(pred.sum())}/{len(pred)}")
    log.log(f"  Wrote {out_path}")
    log.done()


if __name__ == "__main__":
    main()
