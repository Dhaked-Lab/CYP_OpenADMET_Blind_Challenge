# CYP2D6 TDI Classification — Murcko-Scaffold-CV Ensemble (RF + ExtraTrees + LightGBM + XGBoost)

A reproducible machine-learning pipeline for classifying **CYP2D6
time-dependent inhibition (TDI)** from molecular structure. Molecules are
represented by ECFP4 fingerprints, RDKit 2D descriptors and 13 binary
structural-alert features; four tree-ensemble models are evaluated with 5-fold
**Murcko-scaffold cross-validation**, averaged with equal weights, and
thresholded at the out-of-fold MCC optimum before blind-set prediction.

> **Status of the method.** The pipeline is a *hybrid reconstruction* of the
> architecture described in the TeamPozeSCAF methodology report, with the
> LightGBM hyper-parameters of the "Jermy" reproduction. It is **not** a claim
> of exact reproduction of either source. Notably, the 13 structural alerts are
> a documented reconstruction (see [Structural alerts](#structural-alerts)).

---

## Table of contents

1. [Overview](#1-overview)
2. [Scientific background and methodology](#2-scientific-background-and-methodology)
3. [Pipeline architecture](#3-pipeline-architecture)
4. [Workflow, step by step](#4-workflow-step-by-step)
5. [Feature extraction and molecular representation](#5-feature-extraction-and-molecular-representation)
6. [Machine-learning algorithms](#6-machine-learning-algorithms)
7. [Training and validation methodology](#7-training-and-validation-methodology)
8. [Repository structure](#8-repository-structure)
9. [Installation and environment](#9-installation-and-environment)
10. [Input data](#10-input-data)
11. [Configuration](#11-configuration)
12. [Running the pipeline](#12-running-the-pipeline)
13. [Expected outputs](#13-expected-outputs)
14. [Dependencies](#14-dependencies)
15. [Tests](#15-tests)
16. [Citation and acknowledgements](#16-citation-and-acknowledgements)
17. [License](#17-license)

---

## 1. Overview

**Task.** Binary classification of whether a compound is a CYP2D6
time-dependent inhibitor (`CYP2D6_is_TDI`), given its SMILES.

**Objectives.**

- Build a leakage-controlled out-of-fold (OOF) estimate of generalisation to
  *new scaffolds* using Murcko-scaffold-grouped cross-validation.
- Combine four complementary tree ensembles into one equal-weight ensemble.
- Select a single ensemble decision threshold from OOF predictions only and
  freeze it before any blind-set prediction.
- Predict a blinded test set without ever reading its labels.

## 2. Scientific background and methodology

Time-dependent CYP inhibition arises when a compound (or a reactive
metabolite) inactivates the enzyme during incubation, often through
mechanism-based activation. The pipeline therefore combines

- general structure descriptors (ECFP4 fingerprints, RDKit 2D descriptors), and
- binary substructure flags for motifs commonly discussed as mechanism-based
  CYP-inactivator pharmacophores (furan/thiophene rings, terminal alkynes and
  alkenes, methylenedioxyphenyl, hydrazines/hydrazones, and others).

Generalisation is assessed with **scaffold-grouped** folds so that molecules
sharing a Murcko scaffold are never split between training and validation,
which is a stricter test than random k-fold.

Design decisions that originate from the source methodologies:

| Component | Setting | Origin |
|---|---|---|
| Representation | ECFP4 2048-bit + RDKit2D + 13 alerts | TeamPozeSCAF report (alerts reconstructed here) |
| RF / ExtraTrees | 600 trees, depth 8 / 10 | TeamPozeSCAF report, CYP2D6 architecture |
| LightGBM | 500 trees, lr 0.05, 31 leaves, subsample 0.8, colsample 0.8, balanced class weight | "Jermy" LightGBM configuration (replaces the LightGBM settings of the TeamPozeSCAF report) |
| XGBoost | see [Section 6](#6-machine-learning-algorithms) | Fourth ensemble member added in this project |
| Threshold grid | 0.10 – 0.90, step 0.02 | Reproduction decision (the TeamPozeSCAF report specifies no CYP2D6 grid) |
| Metrics | MCC, Accuracy, Precision, Recall, F1 | Vendored challenge configuration in `reference/` |

## 3. Pipeline architecture

```
 data/raw/CYP2D6_Positive.csv + CYP2D6_Negative.csv
                      │
        01  data audit + cleaning ──────────────► 01_clean_cyp2d6.csv
                      │
        02  ECFP4 (2048) + RDKit2D ─────────────► 02_features_ecfp4_rdkit2d.csv
        03  13 structural alerts ───────────────► 03_structural_alerts.csv
        04  Murcko scaffold folds (5, seed 7) ──► 04_scaffold_folds.csv
                      │
        05  RF / ET / LightGBM / XGBoost
            5-fold OOF + final refit ───────────► 05_oof_predictions.csv, models/final_models.pkl
                      │       equal-weight mean of 4 probabilities
        06  threshold search (max OOF MCC) ─────► 07_best_threshold.json  (frozen)
        07  OOF evaluation ─────────────────────► 08_oof_metrics.csv
        08  final-model / consistency check ────► 10_reproduction_report.txt
                      │
        09  blind prediction (frozen threshold) ► 08_final_test_predictions.csv
                      ▲
          data/raw/TEST-BLINDED.csv (no labels)

 Optional analyses (read existing outputs):
        10  XGBoost complementarity analysis
        11  nested, leakage-safe hyper-parameter tuning
```

The full ASCII diagram is in [`docs/architecture.txt`](docs/architecture.txt);
per-step inputs and outputs are in [`docs/workflow.md`](docs/workflow.md).

## 4. Workflow, step by step

| Step | Script | What it does |
|---|---|---|
| 00 | `scripts/00_environment_check.py` | Verifies that all required packages import; prints versions. |
| 01 | `scripts/01_data_audit.py` | Loads the Positive/Negative CSVs, drops fully blank rows, audits missing/invalid/duplicate SMILES and label conflicts, writes the cleaned table. Raw SMILES are preserved (no standardisation). |
| 02 | `scripts/02_features.py` | Computes ECFP4 (2048 bits) and all RDKit2D descriptors. |
| 03 | `scripts/03_structural_alerts.py` | Computes the 13 binary structural-alert features from `config/structural_alerts.yaml` and runs a SMARTS self-test. |
| 04 | `scripts/04_scaffold_folds.py` | Assigns molecules to 5 Murcko-scaffold folds (seed 7) and verifies that no scaffold spans more than one fold (aborts otherwise). |
| 05 | `scripts/05_train_models_oof.py` | Trains RF, ExtraTrees, LightGBM and XGBoost per fold to obtain OOF probabilities, then refits all four on the full labelled set. |
| 06 | `scripts/06_threshold_optimization.py` | Scans thresholds on the 4-model OOF mean probability and freezes the MCC-optimal one. |
| 07 | `scripts/07_oof_evaluation.py` | Reports OOF metrics per model and for the ensembles at the frozen and at the 0.50 threshold. |
| 08 | `scripts/08_final_training_check.py` | Trains nothing; verifies the final-model bundle, feature count/order, parameters, folds, OOF ensemble definition and threshold grid, and writes a reproducibility report. Exits non-zero if a check fails. |
| 09 | `scripts/09_blind_prediction.py` | Featurises `TEST-BLINDED.csv` identically, applies the training-only imputer, the four final models and the frozen threshold. Skips gracefully if the file is absent. |
| 10 | `scripts/10_xgb_complementarity_analysis.py` | Analysis only: does XGBoost add information beyond RF+ET+LightGBM? No training, no changes to folds, features or thresholds. |
| 11 | `scripts/11_hyperparameter_tuning.py` | Additive experiment: nested, leakage-safe random-search tuning of the four models against the fixed baseline. Writes only to `outputs/CYP2D6/hp_tuning/`. |

## 5. Feature extraction and molecular representation

Features are concatenated in this order (2,278 columns in the development run):

| Block | Columns | Implementation |
|---|---|---|
| ECFP4 | 2048 bits (`ecfp4_0 … ecfp4_2047`) | RDKit Morgan generator, radius 2, `fpSize=2048` (`src/cyp_tdi_hybrid/features.py`) |
| RDKit2D | 217 in the development run | `Descriptors.CalcMolDescriptors` (all RDKit 2D descriptors; the count can vary with the RDKit version) |
| Structural alerts | 13 binary columns (`alert_<NAME>`) | `config/structural_alerts.yaml` via `src/cyp_tdi_hybrid/structural_alerts.py` |

Missing descriptor values are imputed with the **training-fold median** (a new
`SimpleImputer` is fitted for every fold; the final imputer is fitted on all
labelled data and reused for the blind set).

### Structural alerts

The TeamPozeSCAF report describes 13 mechanism-based structural alerts but does
not publish their definitions. `config/structural_alerts.yaml` therefore holds
**13 independently chosen replacement alerts** (SMARTS and rationale for each),
labelled `RECONSTRUCTED_TEAMPOZESCAF_ALERTS` in code and logs. Model results
using them must not be presented as reproducing the original numbers. See
[`docs/STRUCTURAL_ALERTS_RECONSTRUCTION.md`](docs/STRUCTURAL_ALERTS_RECONSTRUCTION.md).
`structural_alert_features()` raises an error if the file does not contain
exactly 13 alerts, to guard the feature dimensionality.

## 6. Machine-learning algorithms

All models are scikit-learn-compatible classifiers; the definitions below are
the constants in `scripts/05_train_models_oof.py` (verified by step 08).

| Model | Parameters |
|---|---|
| Random Forest | `n_estimators=600, max_depth=8, random_state=42, n_jobs=-1` |
| Extra Trees | `n_estimators=600, max_depth=10, random_state=42, n_jobs=-1` |
| LightGBM | `n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1` |
| XGBoost | `n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1` |

**Ensemble:** `(RF + ExtraTrees + LightGBM + XGBoost) / 4` on predicted
probabilities. The legacy three-model mean is also stored (`ensemble3_oof_prob`)
for comparison only; it is not used for thresholding.

## 7. Training and validation methodology

- **Cross-validation:** 5-fold Murcko-scaffold CV. Scaffolds are computed
  without chirality; molecules with an empty scaffold are split into
  deterministic pseudo-groups of 40; whole groups are assigned greedily
  (largest first) to the currently smallest fold with seed-7 tie-breaking.
- **Out-of-fold prediction:** each molecule is predicted by models trained on
  the other four folds only.
- **Threshold:** one ensemble-level threshold, chosen by maximum OOF MCC on the
  grid 0.10 – 0.90 (step 0.02), frozen before blind prediction.
- **Metrics:** MCC, Accuracy, Precision, Recall, F1 (vendored scikit-learn
  metric definitions in `reference/evaluation/config.py`).
- **Leakage controls:** per-fold imputers fitted on training folds only; final
  models are separate objects fitted after the OOF loop; threshold selection
  reads only OOF predictions and training labels; `TEST-BLINDED.csv` is read
  only by the blind-prediction stages and its labels are never read.

## 8. Repository structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── config/
│   └── structural_alerts.yaml          # 13 reconstructed structural alerts (SMARTS)
├── scripts/                            # numbered pipeline steps (run from the repo root)
│   ├── 00_environment_check.py
│   ├── 01_data_audit.py … 09_blind_prediction.py
│   ├── 10_xgb_complementarity_analysis.py   # optional analysis
│   └── 11_hyperparameter_tuning.py          # optional experiment
├── src/cyp_tdi_hybrid/                 # shared library code
│   ├── data.py  features.py  structural_alerts.py  splits.py
│   ├── metrics.py  paths.py  logging_utils.py
├── reference/                          # vendored challenge metric configuration
│   └── evaluation/{config.py, custom_scoring_functions.py}
├── data/
│   ├── README.md                       # expected inputs (data itself is not committed)
│   └── raw/                            # place input CSVs here (git-ignored)
├── docs/
│   ├── workflow.md
│   ├── architecture.txt
│   └── STRUCTURAL_ALERTS_RECONSTRUCTION.md
└── tests/                              # lightweight unit tests
```

`outputs/` and `logs/` are created by the scripts and are git-ignored.

## 9. Installation and environment

Python 3.12 was used during development; other Python 3.10+ versions have not
been verified. No GPU is required.

**pip**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate cyp2d6-tdi-sccv
```

Verify the installation:

```bash
python scripts/00_environment_check.py
```

For exact reproduction of the feature dimensionality, use the library versions
listed at the top of `requirements.txt` (the RDKit2D descriptor count depends
on the RDKit release).

## 10. Input data

No data is distributed with this repository. Place the following in
`data/raw/` (details and column definitions in [`data/README.md`](data/README.md)):

| File | Columns |
|---|---|
| `CYP2D6_Positive.csv` | `Molecule_Name`, `SMILES`, `CYP2D6_is_TDI` |
| `CYP2D6_Negative.csv` | `Molecule_Name`, `SMILES`, `CYP2D6_is_TDI` |
| `TEST-BLINDED.csv` (only for step 09 / step 11 blind stage) | `SMILES` (+ optional `Molecule_Name`) |

## 11. Configuration

There is no global configuration file; settings are constants at the top of the
scripts, and all paths are derived from the repository root
(`src/cyp_tdi_hybrid/paths.py`), so scripts can be launched from any directory.

| What | Where |
|---|---|
| Structural-alert SMARTS | `config/structural_alerts.yaml` (must contain exactly 13 entries) |
| Model hyper-parameters | `RF_PARAMS`, `ET_PARAMS`, `LGBM_PARAMS`, `XGB_PARAMS` in `scripts/05_train_models_oof.py` |
| Fold settings | `N_FOLDS`, `SEED`, `EMPTY_GROUP_SIZE` in `scripts/04_scaffold_folds.py` |
| Threshold grid | `GRID` in `scripts/06_threshold_optimization.py` |
| Input file names | `RAW_FILES`, `TEST_BLINDED_PATH` in `src/cyp_tdi_hybrid/paths.py` |

Step 08 compares the trained models against the expected parameters, feature
count (2,278), fold count (5) and threshold grid; if you intentionally change
any of these, update the `EXPECTED_*` constants in that script as well.

## 12. Running the pipeline

All commands are run from the repository root.

**Complete pipeline (steps 00 – 09)**

```bash
python scripts/00_environment_check.py
python scripts/01_data_audit.py
python scripts/02_features.py
python scripts/03_structural_alerts.py
python scripts/04_scaffold_folds.py
python scripts/05_train_models_oof.py
python scripts/06_threshold_optimization.py
python scripts/07_oof_evaluation.py
python scripts/08_final_training_check.py
python scripts/09_blind_prediction.py
```

or, stopping at the first failing step:

```bash
set -e
for step in scripts/0[0-9]_*.py; do python "$step"; done
```

Each script checks that its required inputs exist, prints a summary, and exits
with a non-zero status on a fatal problem. Reviewing each step's output before
running the next one is recommended. Steps 01 – 08 do not need
`TEST-BLINDED.csv`; step 09 reports that there is nothing to predict if it is
missing.

**Individual stages** — every numbered script can be run on its own once the
outputs of the steps it depends on exist (the dependency is stated in its
docstring and enforced by an input-file check).

**Optional analyses**

```bash
# XGBoost complementarity analysis (reads existing OOF results; trains nothing)
python scripts/10_xgb_complementarity_analysis.py
# optional overrides:
#   --oof PATH --threshold 0.xx --old-threshold 0.xx --threshold-file PATH
#   --train-csv PATH --outdir DIR --isoform CYP2D6 --project-root DIR --n-boot 500

# Nested hyper-parameter tuning (writes to outputs/CYP2D6/hp_tuning/)
python scripts/11_hyperparameter_tuning.py --n-iter 30     # full run
python scripts/11_hyperparameter_tuning.py --n-iter 2      # smoke test
python scripts/11_hyperparameter_tuning.py --stage report  # rebuild report only
# stages: nested, final, evaluate, blind, report (default: all)
```

Note: the `blind` stage of script 11 (included in `--stage all`) requires
`data/raw/TEST-BLINDED.csv` with 750 rows and the baseline
`08_final_test_predictions.csv` from step 09; its sanity checks encode the
dimensions of the development dataset (1,497 molecules, 324 positives,
2,278 features).

## 13. Expected outputs

Everything is written under `outputs/CYP2D6/` (and logs under `logs/`); none of
it is committed.

| Step | Files |
|---|---|
| 01 | `01_data_audit_report.json`, `01_clean_cyp2d6.csv` |
| 02 | `02_features_ecfp4_rdkit2d.csv`, `02_features_summary.txt` |
| 03 | `03_structural_alerts.csv`, `03_structural_alerts_summary.txt` |
| 04 | `04_scaffold_folds.csv`, `04_scaffold_folds_summary.txt` |
| 05 | `05_oof_predictions.csv` (per-model and ensemble OOF probabilities), `09_model_parameters.json`, `models/final_models.pkl` |
| 06 | `06_threshold_scan.csv`, `07_best_threshold.json` |
| 07 | `08_oof_metrics.csv` |
| 08 | `10_reproduction_report.txt` (package versions, data/feature/fold counts, verification checks) |
| 09 | `08_final_test_predictions.csv` (per-model probabilities, ensemble probability, threshold used, predicted label) |
| 10 | `10_xgb_complementarity_analysis/` (comparison tables and a markdown report) |
| 11 | `hp_tuning/` (tuning results, tuned OOF/blind predictions, `HYPERPARAMETER_TUNING_REPORT.md`) |

`models/final_models.pkl` is a pickle: only load files you produced yourself.

## 14. Dependencies

Third-party packages imported by the code: `pandas`, `numpy`, `scikit-learn`,
`scipy`, `lightgbm`, `xgboost`, `rdkit`, `PyYAML` (see `requirements.txt` and
`environment.yml`). Standard library modules only otherwise.

## 15. Tests

`tests/` contains lightweight unit tests that need neither the dataset nor any
model training. Tests that require RDKit, LightGBM or XGBoost are skipped
automatically when those packages are not installed.

```bash
python -m unittest discover -s tests -v
```

(`pytest tests` also works if pytest is installed.) The tests do not replace a
full pipeline run on real data.

## 16. Citation and acknowledgements

No journal article or DOI is associated with this repository; a `CITATION.cff`
file should be added by the authors once bibliographic information is final.

- the **OpenADMET CYP challenge** evaluation code vendored in `reference/`
  (source: <https://github.com/OpenADMET/CYP-Challenge-Tutorial>, commit
  `858ae63`, 2026-09-01; see `reference/VENDORED_FROM.txt`).

Method references: Bemis & Murcko, *J. Med. Chem.* 1996, 39, 2887 (Murcko
scaffolds); Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742 (extended-
connectivity fingerprints); Breiman, *Mach. Learn.* 2001, 45, 5 (random
forests); Geurts et al., *Mach. Learn.* 2006, 63, 3 (extremely randomised
trees); Ke et al., *NeurIPS* 2017 (LightGBM); Chen & Guestrin, *KDD* 2016
(XGBoost); the RDKit open-source cheminformatics toolkit (<https://www.rdkit.org>).

## 17. License

No license has been specified for this repository yet. Until the owners add
one, all rights are reserved by default. The code under `reference/` is
vendored from the OpenADMET repository and remains subject to that
repository's license terms.
