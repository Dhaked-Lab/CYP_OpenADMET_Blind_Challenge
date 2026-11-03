# Input data

The training and blind-test tables are external inputs and are **not distributed with this repository**. Obtain them from the challenge
organisers (check their terms of use) and place them in this directory, or point the pipeline elsewhere with `--data-dir`.

| File | Columns | Role |
|---|---|---|
| `CYP3A4_Positive.csv` | `Molecule_Name,SMILES,CYP3A4_is_TDI` | training compounds labelled TDI-positive |
| `CYP3A4_Negative.csv` | `Molecule_Name,SMILES,CYP3A4_is_TDI` | training compounds labelled TDI-negative |
| `TEST-BLINDED.csv` | `Molecule_Name,SMILES` | blind compounds to score (no labels) |

A single combined `CYP3A4.csv` containing the label column is also accepted instead of the positive/negative pair
(file names are set in `config/isoforms/CYP3A4.yaml`).

Notes
* The original positive/negative files contain roughly one million blank padding rows; they are dropped automatically.
* Label values are parsed from `TRUE`/`FALSE`. The file role must agree with the label column, the label column must be `CYP3A4_is_TDI`,
  and any other `CYP*_is_TDI` column is rejected. Duplicate or missing blind IDs and labels in the blind file stop the run.
* Reference counts used for informational warnings only: 764 positives, 2,820 negatives, 750 blind compounds.

SHA-256 of the files the pipeline was prepared with (for checking you have the same inputs):

```
72f075ffe9716eb9f253442296915fe616639fa9b9ee0131e43ff5b296f9ac94  CYP3A4_Negative.csv
a0d40bab1531f2f5d2632f68b47d703a0480325e9496059fe55e9f66596f99a1  CYP3A4_Positive.csv
a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9  TEST-BLINDED.csv
```
