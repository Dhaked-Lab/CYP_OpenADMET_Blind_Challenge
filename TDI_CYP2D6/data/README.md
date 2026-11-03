# Data

This repository does **not** ship any dataset. The training and blind-test
tables are external inputs that must be placed in `data/raw/` before running
the pipeline. Files in `data/raw/` are git-ignored (except `.gitkeep`).

## Required files

| File | Used by | Description |
|---|---|---|
| `data/raw/CYP2D6_Positive.csv` | step 01 | CYP2D6 time-dependent-inhibition (TDI) **positive** molecules |
| `data/raw/CYP2D6_Negative.csv` | step 01 | CYP2D6 TDI **negative** molecules |
| `data/raw/TEST-BLINDED.csv` | steps 09 and 11 (`--stage blind` / `all`) | Blind test molecules (structures only, no labels). Optional: every earlier step runs without it |

## Expected format

All files are comma-separated with a header row. Column names are stripped of
surrounding whitespace on load.

`CYP2D6_Positive.csv` and `CYP2D6_Negative.csv`:

| Column | Type | Notes |
|---|---|---|
| `Molecule_Name` | string | Compound identifier |
| `SMILES` | string | Raw SMILES. No standardisation (salt stripping, canonicalisation, tautomer handling) is applied by the pipeline |
| `CYP2D6_is_TDI` | boolean (`True` / `False`) | TDI label |

Rows in which `Molecule_Name`, `SMILES` and `CYP2D6_is_TDI` are all empty
(e.g. trailing blank rows from a spreadsheet export) are dropped
automatically in step 01.

`TEST-BLINDED.csv`:

| Column | Type | Notes |
|---|---|---|
| `Molecule_Name` | string | Compound identifier (optional; carried into the output if present) |
| `SMILES` | string | Raw SMILES |

Any further columns in the blind file are ignored. Blind labels are never read.

## Origin of the data

The compound identifiers used during development (`OCNT-xxxxxxx`) and the
evaluation code vendored in `reference/` follow the schema of the OpenADMET
CYP challenge. Confirm the provenance and terms of use of your copy of the
data before redistributing it; the files are intentionally not committed here.

## Generated data

Everything the pipeline derives from these files (cleaned table, feature
matrices, folds, out-of-fold predictions, trained models, predictions) is
written to `outputs/CYP2D6/` and `logs/`, both git-ignored.
