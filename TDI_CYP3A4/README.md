# CYP3A4 TDI Pipeline

A modular, resumable pipeline for classifying **time-dependent inhibition (TDI) of CYP3A4** from molecular structure. features, 5-fold
Murcko-scaffold cross-validation with seed 7, four ensemble members, nested greedy ensemble, rank-fraction selection) and corrects only
defects that were confirmed against the original source (see [`docs/corrections_log.md`](docs/corrections_log.md)).

> **Status.** Only the lightweight tests listed under [Tests](#tests) have been run. No scientific step (features, embeddings,
> training, prediction) has been executed with RDKit, LightGBM, PyTorch, chemprop or TabICL in the environment used to prepare this
> repository, so those steps are untested at runtime. No performance figure is claimed.

## Contents
1. [Overview](#overview) 2. [Methodology](#methodology) 3. [Pipeline steps](#pipeline-steps) 4. [Repository layout](#repository-layout)
5. [Installation](#installation) 6. [Inputs](#inputs) 7. [Configuration](#configuration) 8. [Running](#running)
9. [Outputs](#outputs) 10. [Checkpoints](#checkpoints) 11. [Troubleshooting](#troubleshooting) 12. [Tests](#tests)
13. [Known limitations](#known-limitations) 14. [Acknowledgements and citation](#acknowledgements-and-citation)

## Overview
Time-dependent CYP inhibition is a drug-safety liability: a compound that irreversibly or quasi-irreversibly inactivates CYP3A4 can
cause drug-drug interactions. Given labelled training compounds (SMILES + `CYP3A4_is_TDI`) and an unlabelled blind set, the pipeline
produces a score for every blind compound and a final binary TDI call obtained by selecting the top-ranked fraction of compounds.

## Methodology
**Molecular representation** (step `features`, 3,456 columns)
* 217 RDKit 2D descriptors (name list and order pinned in `assets/reference/jb_descriptor_names.json`; the run stops if the installed RDKit differs)
* Morgan fingerprint radius 2, 2,048 bits, and radius 3, 1,024 bits (square-root of counts)
* MACCS keys (167 bits)
* Nearest-positive similarity features (`core/nn_features.py`): Tanimoto similarity on binarised Morgan r2 to training-fold positives only, with self exclusion
* Frozen pretrained embeddings: CheMeleon (2,048-d) and ChemProp-medium (600-d)

**Ensemble members** (all evaluated by out-of-fold (OOF) prediction)
| Member | Model |
|---|---|
| `base` | LightGBM on descriptors, fingerprints and nearest-positive features |
| `emb` | LightGBM on the base features plus the CheMeleon embedding |
| `tab` | TabICL on a 256-component whitened PCA of the CheMeleon embedding |
| `tabcp` | TabICL on a 256-component whitened PCA of the ChemProp-medium embedding |

LightGBM parameters (`config/pipeline.yaml`): 1,200 trees, learning rate 0.03, 63 leaves, `min_child_samples` 15, subsample 0.8,
`colsample_bytree` 0.5, `reg_lambda` 2.0, seed 42. TabICL: 4 estimators, seed 42, pinned checkpoint.

**Validation.** 5-fold Murcko-scaffold cross-validation, seed 7. By default (`cv.fold_strategy: original_all_isoforms`) folds are built over the
original 6,145-compound table covering both isoforms (`assets/reference/jb_fold_universe_smiles.csv`, sha256-checked) and the CYP3A4 rows are
then selected, as in the original; `cyp_only` builds folds over the CYP3A4 rows only.

**Ensembling and decision rule.** Member scores are z-scored and combined with nested per-fold greedy weights (12 rounds, deterministic tie-break).
The final call labels the top-ranked fraction of blind compounds positive; the fraction is chosen by expected MCC over several prevalence
hypotheses (`threshold_selection`, fractions 0.05-0.61) and applied with an exact count.

**Corrections relative to the original** (profile `corrected`, default): PCA fitted inside each training fold only; exact-count rank selection
(the original selected one extra compound); deterministic LightGBM; fail-fast input and isoform validation; pinned TabICL checkpoint.
`--profile baseline` restores the original behaviour for these points. Details and evidence: [`docs/corrections_log.md`](docs/corrections_log.md).

## Pipeline steps
Dependencies between steps are resolved automatically. Original script numbers are in brackets.

| Step | Depends on | Purpose |
|---|---|---|
| `data_audit` [01] | - | schema, isoform, labels, SMILES, duplicates, blind file |
| `data_clean` [02] | data_audit | drop unparseable / conflicting / duplicate SMILES |
| `features` [03] | data_clean | RDKit descriptors + Morgan r2/r3 + MACCS |
| `similarity` [04] | features | Tanimoto matrix + self-exclusion leakage check |
| `scaffold_cv` [05] | data_clean | Murcko scaffold folds, K=5, seed 7 |
| `embeddings` | data_clean | CheMeleon + ChemProp-medium embeddings (cached) |
| `oof_base` [06] | features, similarity, scaffold_cv | LightGBM member |
| `oof_emb` [07a] | + embeddings | embedding + LightGBM member |
| `oof_tab` [07b] | scaffold_cv, embeddings | CheMeleon PCA256 + TabICL (fold-contained PCA) |
| `oof_tabcp` [08] | scaffold_cv, embeddings | ChemProp PCA256 + TabICL (fold-contained PCA) |
| `ensemble_selection` [09] | the four OOF steps | nested greedy weights |
| `threshold_selection` [10] | ensemble_selection | rank fraction |
| `model_training` [11] | features, similarity, embeddings | final LightGBM members + full-train PCAs |
| `blind_features` [12a] | data_clean, features | validate and featurize blind set |
| `blind_embeddings` [12b] | blind_features | blind embeddings |
| `prediction` [12c] | model_training, blind_*, ensemble_selection | member scores + frozen blend (TabICL refit on full training set) |
| `submission` [12d] | prediction, threshold_selection | exact-count rank selection, validation, files |

See [`docs/workflow.md`](docs/workflow.md) for the data flow.

## Repository layout
```
CYP3A4-TDI-Pipeline/
├── run_pipeline.py          # entry point
├── config/
│   ├── pipeline.yaml        # methodology settings, seeds, profiles
│   ├── resources.yaml       # CPU/GPU limits, checkpoint paths and sha256
│   └── isoforms/CYP3A4.yaml # column names, file names, submission format
├── src/cyp_tdi/             # package: steps/, core/, models/, executor, registry, validation, ...
├── scripts/                 # check_environment.py, compare_runs.py
├── tests/                   # unit and mocked-pipeline tests
├── assets/
│   ├── reference/           # descriptor-name list, fold-universe SMILES (tracked)
│   └── checkpoints/         # pretrained weights (external, see README there)
├── data/                    # input tables (external, see README there)
├── docs/                    # workflow.md, corrections_log.md, tested_environment.md
└── outputs/                 # run results (git-ignored)
```

## Installation
Python 3.10 or newer is declared; 3.12 is the tested version.

1. **Create the environment** (RDKit from conda-forge):
   ```bash
   conda env create -f environment.yml
   conda activate cyp-tdi
   ```
2. **Install PyTorch once**, with the build matching your OS/CUDA, using the selector at https://pytorch.org/get-started/locally/.
   It is deliberately not in `requirements.txt` so that pip never replaces a working CUDA build. The reported Windows environment used:
   ```bash
   python -m pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
   ```
   (Windows additionally needs the Microsoft Visual C++ 2015-2022 x64 runtime.) Use the `cpu` index for CPU-only machines.
3. **Install the remaining dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   python -m pip show torch     # must still report the build from step 2
   ```
   If pip replaced torch, repeat step 2. `python -m pip install tabicl==2.2.0 --no-deps` plus `einops huggingface-hub` avoids any torch change.
4. **Add the external inputs:** data tables ([`data/README.md`](data/README.md)), checkpoints ([`assets/checkpoints/README.md`](assets/checkpoints/README.md)) and the TabICL checkpoint (below).
5. **Verify the environment:**
   ```bash
   python scripts/check_environment.py
   python -m pytest -q
   ```

Reported environment (Windows 11, Python 3.12.14, torch 2.13.0+cu130, chemprop 2.3.1, lightgbm 4.7.0, RDKit 2026.3.6): [`docs/tested_environment.md`](docs/tested_environment.md).

**Dependency boundary.** Only steps that require PyTorch (`embeddings`, `blind_embeddings`, `oof_tab`, `oof_tabcp`, `prediction`) import it.
`--list-steps`, `--status`, `--dry-run` and version reporting never do. `run_pipeline.py` imports torch first when steps are about to run
(avoids Windows DLL shadowing); disable with `CYP_TDI_NO_TORCH_PRELOAD=1`.

## Inputs
Three files in `data/` (or `--data-dir`): `CYP3A4_Positive.csv`, `CYP3A4_Negative.csv` (`Molecule_Name,SMILES,CYP3A4_is_TDI`) and
`TEST-BLINDED.csv` (`Molecule_Name,SMILES`); a combined `CYP3A4.csv` is also accepted. Formats and checks: [`data/README.md`](data/README.md).

## Configuration
* `config/pipeline.yaml` - paths, seeds, CV, features, LightGBM, embeddings, PCA, TabICL, ensemble, threshold, selection, and the `baseline` profile overrides.
* `config/resources.yaml` - `cpu.threads`, `gpu.enabled auto|true|false`, `gpu.device_index`, `gpu.required`, checkpoint paths and sha256.
* `config/isoforms/CYP3A4.yaml` - file and column names, submission columns.

Command-line overrides: `--data-dir`, `--output-dir`, `--config`, `--resources`, `--cpu-threads N`, `--gpu/--no-gpu`, `--gpu-device I`, `--profile {corrected,baseline}`.
Optional experiments (default off, never part of the baseline): `nn_features.mode: drop_count | exclude_same_scaffold`, `embeddings.round_decimals`,
`cv.argsort_kind: stable`, `threshold.fraction_override`. Use a separate `--output-dir` for them.

## Running
```bash
python run_pipeline.py --list-steps
python run_pipeline.py --status --isoform CYP3A4
python run_pipeline.py --isoform CYP3A4 --run-all                    # complete pipeline
python run_pipeline.py --isoform CYP3A4 --run-all --dry-run          # print the plan only
python run_pipeline.py --isoform CYP3A4 --step data_audit            # a single step (dependencies resolved)
python run_pipeline.py --isoform CYP3A4 --steps data_audit,features  # several steps
python run_pipeline.py --isoform CYP3A4 --run-all --force            # rerun everything
python run_pipeline.py --isoform CYP3A4 --run-all --profile baseline # original behaviour
python scripts/compare_runs.py outputs/CYP3A4.baseline outputs/CYP3A4
```
Completed steps are reused only if their fingerprint (implementation version, config subset, input files, upstream output hashes) still matches
and their outputs are intact. A failed step keeps its valid upstream results; rerun to resume. Do not expect bitwise determinism: GPU TabICL/torch
operations cannot guarantee it.

## Outputs
Written to `outputs/CYP3A4/<step>/` (or `outputs/CYP3A4.baseline/<step>/`); nothing is included in the repository.
* Per-step artifacts: audit and cleaning reports, feature matrices, similarity matrix, fold assignments and `fold_report.json`, OOF scores per member, ensemble weights, threshold report, trained members and PCAs, blind features/embeddings, member scores.
* `state/run_summary.json`: weights, selected fraction, positive count, library versions, config fingerprint. Logs are written to `logs/`.
* Final files in `submission/`: `submission_CYP3A4_is_TDI.csv` (`Molecule_Name,SMILES,CYP3A4_is_TDI`) and `blind_predictions_CYP3A4.csv`.
* Embeddings are cached in `cache/` with provenance checks, so they are extracted once and shared by both profiles.

## Checkpoints
Two pretrained encoders in `assets/checkpoints/` (external; sources and sha256 in [`assets/checkpoints/README.md`](assets/checkpoints/README.md)):
`chemeleon_mp.pt` (CheMeleon; loaded with `weights_only=True`) and `chemprop_medium.pt`. A missing or modified checkpoint stops the run; there are no fallbacks.

### TabICL checkpoint
Pinned in `config/pipeline.yaml` (`tabicl.checkpoint`): `jingang/TabICL`, `tabicl-classifier-v2-20260212.ckpt`, sha256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`, with `tabicl==2.2.0`. The pipeline loads it only from the local Hugging Face
cache after a sha256 check and never downloads during a run. Fetch it once:
```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('jingang/TabICL', 'tabicl-classifier-v2-20260212.ckpt'))"
```
GPU is used only for embedding extraction and TabICL; `--no-gpu` runs on CPU.

## Troubleshooting
**`WinError 127` / `0xc0000139` on `torch\lib\shm.dll` (Windows).** Decide by evidence:
* `python -c "import torch; print(torch.cuda.is_available())"` fails in a fresh process: the install is the problem (wheel, driver, VC++ runtime, PATH). Remove every torch copy and reinstall one build; do not mix pip and conda builds.
* It works in a fresh process but fails inside a longer session: an import-order conflict. Run `python scripts/check_environment.py --order-probes` (needs `psutil`); the first row that flips from OK to FAIL names the library.

## Tests
```bash
python -m pytest -q          # or: python -m unittest discover -s tests
```
Tests that need RDKit, LightGBM, torch, the data files or the checkpoints are skipped when those are unavailable; skipped means **blocked, not passed**.
Mocked pipeline tests use synthetic arrays and stub models and validate orchestration only. Always run tests as `python -m pytest ...`.

## Known limitations
* Second-order dependence of ensemble weights and fraction on the pooled OOF predictions (JB-08) is assessed but not corrected; reported nested OOF MCC should be read as mildly optimistic.
* The prevalence prior behind the rank fraction is the original flat reconstruction; the fraction is not validated on blind labels.
* Run-to-run variation can be reduced but not removed .
* The descriptor count depends on RDKit; the check fails fast instead of adapting. Fold ids can differ under other numpy/RDKit versions: check `scaffold_cv/fold_report.json` (expected sizes 687/727/724/721/725).
* Not covered: scripts 13-21 of the original (post-hoc analysis) and external-data variants.

## Acknowledgements and citation
This pipeline re-implements the original CYP3A4 workflow by Jackson Burns and uses third-party pretrained models and libraries:
* **CheMeleon**: Burns, J., Zalte, A., Green, W. *Descriptor-based Foundation Models for Molecular Property Prediction*, arXiv:2506.15792; model weights: Zenodo, DOI 10.5281/zenodo.15460715.
* **ChemProp-medium** checkpoint from `github.com
* **chemprop**, **TabICL**, **LightGBM**, **RDKit**, **scikit-learn**: please cite according to their own documentation.

No license or `CITATION.cff` is included yet; see the publication checklist delivered with this repository.
