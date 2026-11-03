"""CYP2D6 STEP 06 -- OOF threshold optimization.

Uses ONLY outputs/CYP2D6/05_oof_predictions.csv: the four per-model OOF
probability columns (RF, ExtraTrees, LightGBM, XGBoost), combined here into the
equal-weight four-model ensemble (RF + ET + LGBM + XGB) / 4, and the true
is_TDI labels. A single ensemble-level threshold is optimized (no
model-specific thresholds).
TEST-BLINDED.csv is never read by this script -- there is no code path here
that could touch it.

Grid: 0.10 -> 0.90, step 0.02. TeamPozeSCAF's report does not specify a
threshold grid for CYP2D6, so this range is a REPRODUCTION DECISION,
documented as such below and in the README. UNCHANGED from the 3-model
baseline.

Requires: outputs/CYP2D6/05_oof_predictions.csv (step 05)
Writes:   outputs/CYP2D6/06_threshold_scan.csv
          outputs/CYP2D6/07_best_threshold.json

Run:  python scripts/06_threshold_optimization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.metrics import compute_classification_metrics
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir

ISO = "CYP2D6"
GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.02), 4)
ENSEMBLE_MEMBERS = ["RF", "ExtraTrees", "LightGBM", "XGBoost"]


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_06_threshold_optimization")
    out_dir = isoform_dir(ISO)
    oof_path = out_dir / "05_oof_predictions.csv"
    log.require_file(oof_path, hint="Run scripts/05_train_models_oof.py first.")

    df = pd.read_csv(oof_path)
    member_cols = [f"{n}_oof_prob" for n in ENSEMBLE_MEMBERS]
    missing = [c for c in member_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{ISO}] STEP 06: missing OOF columns {missing} -- re-run step 05 (XGBoost version).")

    # New four-model equal-weight ensemble: (RF + ET + LGBM + XGB) / 4
    ens4 = df[member_cols].sum(axis=1, min_count=len(member_cols)) / len(member_cols)
    assert np.allclose(ens4.fillna(-1), df["ensemble_oof_prob"].fillna(-1)), \
        "ensemble_oof_prob in step 05 output is not the 4-model mean"

    mask = ens4.notna()
    y_true = df.loc[mask, "is_TDI"].astype(bool).to_numpy()
    p = ens4[mask].to_numpy()
    log.log(f"[{ISO}] STEP 06: threshold search on {mask.sum()} OOF predictions "
            f"(4-model ensemble: RF+ExtraTrees+LightGBM+XGBoost, mean/4), "
            f"REPRODUCTION DECISION grid=[0.10, 0.90] step 0.02 (TeamPozeSCAF report "
            f"does not specify a CYP2D6 threshold grid)")

    rows = []
    for t in GRID:
        y_pred = p >= t
        m = compute_classification_metrics(y_true, y_pred)
        rows.append({"threshold": float(t), **m})
    scan_df = pd.DataFrame(rows)
    scan_path = out_dir / "06_threshold_scan.csv"
    scan_df.to_csv(scan_path, index=False)

    best_row = scan_df.loc[scan_df["MCC"].idxmax()]
    best = {"best_threshold": float(best_row["threshold"]),
            "MCC": float(best_row["MCC"]), "Accuracy": float(best_row["Accuracy"]),
            "Precision": float(best_row["Precision"]), "Recall": float(best_row["Recall"]),
            "F1": float(best_row["F1"]), "n_oof": int(mask.sum()),
            "grid_min": float(GRID.min()), "grid_max": float(GRID.max()), "grid_step": 0.02,
            "ensemble_members": ENSEMBLE_MEMBERS}
    best_path = out_dir / "07_best_threshold.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    log.log(f"  Best threshold: {best['best_threshold']:.2f}  MCC={best['MCC']:.4f}  "
            f"Acc={best['Accuracy']:.4f}  Prec={best['Precision']:.4f}  Rec={best['Recall']:.4f}  "
            f"F1={best['F1']:.4f}")
    log.log(f"  Wrote {scan_path}")
    log.log(f"  Wrote {best_path}")
    log.log("  Threshold FROZEN as of this point -- not to be changed after seeing blind-test predictions.")
    log.done()


if __name__ == "__main__":
    main()
