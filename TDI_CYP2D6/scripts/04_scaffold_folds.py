"""CYP2D6 STEP 04 -- 5-fold Murcko scaffold cross-validation split.

Murcko scaffold-grouped CV: whole scaffold groups are assigned to folds.

Settings:
    - Murcko scaffold, no chirality
    - Empty-scaffold molecules are bucketed into pseudo-groups of 40
    - 5 folds
    - seed = 7
    - greedy load-balanced assignment of scaffold groups

Requires:
    outputs/CYP2D6/01_clean_cyp2d6.csv

Writes:
    outputs/CYP2D6/04_scaffold_folds.csv
    outputs/CYP2D6/04_scaffold_folds_summary.txt

Run:
    python scripts/04_scaffold_folds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir


ISO = "CYP2D6"
N_FOLDS = 5
SEED = 7
EMPTY_GROUP_SIZE = 40


def scaffold_key(smiles: str) -> str:
    """Return a Murcko scaffold key with chirality ignored.

    Molecules with no Murcko scaffold are returned as an empty key and are
    handled separately by scaffold_groups().
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES encountered: {smiles!r}")

    # MurckoScaffold.GetScaffoldForMol() does not retain stereochemistry.
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def scaffold_groups(smiles_list: list[str]) -> list[str]:
    """Assign every molecule to a Murcko scaffold group.

    Empty-scaffold molecules are split into deterministic pseudo-groups of
    EMPTY_GROUP_SIZE, matching the supplied reference workflow.
    """
    groups = [scaffold_key(s) for s in smiles_list]

    empty_idx = [i for i, g in enumerate(groups) if not g]

    # Empty scaffolds cannot all form one giant group. Bucket them into
    # deterministic pseudo-groups of 40.
    for bucket, start in enumerate(range(0, len(empty_idx), EMPTY_GROUP_SIZE)):
        for i in empty_idx[start:start + EMPTY_GROUP_SIZE]:
            groups[i] = f"__EMPTY_SCAFFOLD_{bucket:06d}"

    return groups


def make_folds(groups: list[str], k: int = 5, seed: int = 7) -> np.ndarray:
    """Greedily assign whole scaffold groups to k load-balanced folds.

    Groups are never split. A seeded permutation is used to make
    tie-breaking deterministic; larger groups are assigned first to
    the currently smallest fold.
    """

    group_to_indices: dict[str, list[int]] = {}

    for idx, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(idx)

    group_items = list(group_to_indices.items())

    # Seeded deterministic tie-breaking.
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(group_items))

    # Attach the seeded tie rank BEFORE sorting.
    ranked_groups = [
        (group, indices, int(tie_order[i]))
        for i, (group, indices) in enumerate(group_items)
    ]

    # Largest scaffold groups first.
    # Seeded rank breaks equal-size ties.
    ranked_groups.sort(
        key=lambda item: (-len(item[1]), item[2])
    )

    folds = np.full(len(groups), -1, dtype=int)
    fold_sizes = np.zeros(k, dtype=int)

    for _, indices, _ in ranked_groups:
        target = int(np.argmin(fold_sizes))

        folds[indices] = target
        fold_sizes[target] += len(indices)

    if np.any(folds < 0):
        raise RuntimeError(
            "Some molecules were not assigned to a fold."
        )

    return folds


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_04_scaffold_folds")

    out_dir = isoform_dir(ISO)
    data_path = out_dir / "01_clean_cyp2d6.csv"
    log.require_file(
        data_path,
        hint="Run scripts/01_data_audit.py first.",
    )

    df = pd.read_csv(data_path)

    required = {"Molecule_Name", "SMILES", "is_TDI"}
    missing = required - set(df.columns)
    if missing:
        log.fatal(
            f"Missing required columns in {data_path}: {sorted(missing)}"
        )
        return

    log.log(
        f"[{ISO}] STEP 04: Murcko scaffold {N_FOLDS}-fold split "
        f"on {len(df)} molecules (no chirality, seed={SEED})"
    )

    groups = scaffold_groups(df["SMILES"].tolist())
    folds = make_folds(groups, k=N_FOLDS, seed=SEED)

    out = df[["Molecule_Name", "SMILES", "is_TDI"]].copy()
    out["fold"] = folds

    out_path = out_dir / "04_scaffold_folds.csv"
    out.to_csv(out_path, index=False)

    lines = [
        f"CYP2D6 Murcko scaffold folds "
        f"(n_folds={N_FOLDS}, seed={SEED})",
        "",
        f"n_molecules={len(out)}",
        f"n_scaffold_groups={len(set(groups))}",
        "",
    ]

    for f in range(N_FOLDS):
        sub = out[out["fold"] == f]
        pos = int(sub["is_TDI"].sum())
        neg = len(sub) - pos
        rate = pos / len(sub) if len(sub) else 0.0
        lines.append(
            f"  fold {f}: n={len(sub):4d}  "
            f"positive={pos:4d}  negative={neg:4d}  "
            f"pos_rate={rate:.3%}"
        )

    # Scaffold-purity check: no Murcko scaffold may span multiple folds.
    group_series = pd.Series(groups, name="scaffold")
    scaffold_fold_counts = (
        pd.DataFrame({"scaffold": group_series, "fold": folds})
        .groupby("scaffold")["fold"]
        .nunique()
    )
    leaking_scaffolds = int((scaffold_fold_counts > 1).sum())

    lines.append("")
    lines.append(
        f"Scaffolds appearing in >1 fold (should be 0): "
        f"{leaking_scaffolds}"
    )

    summary = "\n".join(lines) + "\n"
    summary_path = out_dir / "04_scaffold_folds_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    log.log(summary)
    log.log(f"  Wrote {out_path}")
    log.log(f"  Wrote {summary_path}")

    if leaking_scaffolds:
        log.fatal(
            "A Murcko scaffold appears in more than one fold. "
            "Do not proceed."
        )
        return

    log.done()


if __name__ == "__main__":
    main()
