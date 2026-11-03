# Workflow reference

Inputs and outputs of every script. All paths are relative to the repository
root; `outputs/CYP2D6/` is abbreviated `out/`. Every step also appends to
`logs/CYP2D6_<step>.log`.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 00 | `00_environment_check.py` | — | (stdout only) |
| 01 | `01_data_audit.py` | `data/raw/CYP2D6_Positive.csv`, `data/raw/CYP2D6_Negative.csv` | `out/01_data_audit_report.json`, `out/01_clean_cyp2d6.csv` |
| 02 | `02_features.py` | `out/01_clean_cyp2d6.csv` | `out/02_features_ecfp4_rdkit2d.csv`, `out/02_features_summary.txt` |
| 03 | `03_structural_alerts.py` | `out/01_clean_cyp2d6.csv`, `config/structural_alerts.yaml` | `out/03_structural_alerts.csv`, `out/03_structural_alerts_summary.txt` |
| 04 | `04_scaffold_folds.py` | `out/01_clean_cyp2d6.csv` | `out/04_scaffold_folds.csv`, `out/04_scaffold_folds_summary.txt` |
| 05 | `05_train_models_oof.py` | `out/02_…`, `out/03_…`, `out/04_…` | `out/05_oof_predictions.csv`, `out/09_model_parameters.json`, `out/models/final_models.pkl` |
| 06 | `06_threshold_optimization.py` | `out/05_oof_predictions.csv` | `out/06_threshold_scan.csv`, `out/07_best_threshold.json` |
| 07 | `07_oof_evaluation.py` | `out/05_oof_predictions.csv`, `out/07_best_threshold.json` | `out/08_oof_metrics.csv` |
| 08 | `08_final_training_check.py` | `out/models/final_models.pkl` and the outputs of steps 01 – 06 | `out/10_reproduction_report.txt` (exit code ≠ 0 if a check fails) |
| 09 | `09_blind_prediction.py` | `data/raw/TEST-BLINDED.csv`, `out/models/final_models.pkl`, `out/07_best_threshold.json` | `out/08_final_test_predictions.csv` |
| 10 | `10_xgb_complementarity_analysis.py` | auto-detected OOF table and threshold (overridable by CLI) | `out/10_xgb_complementarity_analysis/` |
| 11 | `11_hyperparameter_tuning.py` | `out/01_…` – `out/05_…`, `out/08_final_test_predictions.csv` (baseline); `data/raw/TEST-BLINDED.csv` for the blind stage | `out/hp_tuning/` |

(The numeric prefixes of the output files are historical and do not always
match the step number that writes them, e.g. step 05 writes
`09_model_parameters.json`.)

## Invariants enforced by the code

- Step 04 aborts if any Murcko scaffold group appears in more than one fold.
- Steps 05, 06 and 08 require `ensemble_oof_prob` to equal the mean of the four
  per-model OOF probabilities.
- Step 08 verifies: bundle contains exactly RF, ExtraTrees, LightGBM, XGBoost;
  model parameters equal the expected configuration; 2,278 features in the
  training order; 5 folds; the 0.10 – 0.90 / 0.02 threshold grid.
- Step 09 reindexes blind-set features to the training column order and uses
  the training-only imputer; it never reads labels.

## Pipeline diagram

See [`architecture.txt`](architecture.txt).
