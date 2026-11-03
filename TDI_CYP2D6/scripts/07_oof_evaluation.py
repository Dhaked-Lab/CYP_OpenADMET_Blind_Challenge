"""CYP2D6 STEP 07 -- Final OOF evaluation report.

Reports OOF metrics (MCC, Accuracy, Precision, Recall, F1) for each of the
four individual models (RF, Extra Trees, LightGBM, XGBoost) AND the new
four-model mean-probability ensemble, both at the optimized threshold
(step 06) and at threshold=0.50 for transparency. The legacy 3-model ensemble
(RF + ET + LGBM) is preserved as an extra row for old-vs-new comparison.

Requires: outputs/CYP2D6/05_oof_predictions.csv (step 05)
          outputs/CYP2D6/07_best_threshold.json (step 06)
Writes:   outputs/CYP2D6/08_oof_metrics.csv

Run:  python scripts/07_oof_evaluation.py
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
GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.02), 4)  # identical to step 06 (used only for the legacy reference row)


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_07_oof_evaluation")
    out_dir = isoform_dir(ISO)
    oof_path = out_dir / "05_oof_predictions.csv"
    thr_path = out_dir / "07_best_threshold.json"
    log.require_file(oof_path, hint="Run scripts/05_train_models_oof.py first.")
    log.require_file(thr_path, hint="Run scripts/06_threshold_optimization.py first.")

    df = pd.read_csv(oof_path)
    best = json.loads(thr_path.read_text())
    best_t = best["best_threshold"]
    log.log(f"[{ISO}] STEP 07: OOF evaluation at optimized threshold={best_t:.2f} and at 0.50")

    # Legacy 3-model ensemble (RF + ET + LGBM)/3, rebuilt from the saved per-model OOF columns.
    df["ensemble3_oof_prob"] = df[["RF_oof_prob", "ExtraTrees_oof_prob", "LightGBM_oof_prob"]].mean(axis=1)

    prob_cols = {"RF": "RF_oof_prob", "ExtraTrees": "ExtraTrees_oof_prob",
                 "LightGBM": "LightGBM_oof_prob", "XGBoost": "XGBoost_oof_prob",
                 "Ensemble3(mean, legacy)": "ensemble3_oof_prob",
                 "Ensemble4(mean)": "ensemble_oof_prob"}
    rows = []
    for model_name, col in prob_cols.items():
        mask = df[col].notna()
        y_true = df.loc[mask, "is_TDI"].astype(bool).to_numpy()
        p = df.loc[mask, col].to_numpy()
        thresholds = [("optimized", best_t), ("0.50", 0.5)]
        if col == "ensemble3_oof_prob":
            # Reference only: the legacy ensemble's own best-MCC threshold on the SAME grid.
            # Never saved/frozen; the frozen threshold is the step-06 4-model one.
            scan = [(float(t), compute_classification_metrics(y_true, p >= t)["MCC"]) for t in GRID]
            thresholds.append(("own_optimal_reference", max(scan, key=lambda s: s[1])[0]))
        for thr_name, t in thresholds:
            m = compute_classification_metrics(y_true, p >= t)
            rows.append({"model": model_name, "threshold_type": thr_name, "threshold": t,
                         "n_oof": int(mask.sum()), **m})
    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "08_oof_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    log.log(metrics_df.to_string(index=False))
    log.log(f"  Wrote {metrics_path}")
    log.done()


if __name__ == "__main__":
    main()
