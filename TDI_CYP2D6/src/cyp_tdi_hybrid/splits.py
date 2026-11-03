"""Murcko-scaffold grouping utilities.

``scaffold_groups`` assigns every molecule to a Murcko scaffold group
(chirality ignored; molecules with an empty scaffold are bucketed into
deterministic pseudo-groups of 40). It is used by
``scripts/11_hyperparameter_tuning.py`` for its scaffold-leakage sanity check.

The fold assignment used by the pipeline itself is implemented in
``scripts/04_scaffold_folds.py`` (5 folds, seed 7, greedy load-balanced
assignment of whole scaffold groups).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SCAFFOLD_EMPTY_BUCKET_SIZE = 40


def _murcko_scaffold_key(smiles: str) -> str:
    """Return a chirality-insensitive Murcko scaffold key.

    Empty/no-scaffold molecules return an empty key and are handled separately
    by scaffold_groups().
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # Murcko scaffolds are represented without stereochemistry.
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def scaffold_groups(
    smiles_list: list[str],
    empty_bucket_size: int = SCAFFOLD_EMPTY_BUCKET_SIZE,
) -> np.ndarray:
    """Assign each molecule to a Murcko scaffold group.

    Molecules sharing the same non-empty Murcko scaffold receive the same
    group ID and therefore can never be separated across CV folds.

    Molecules with an empty Murcko scaffold are divided into deterministic
    pseudo-groups of `empty_bucket_size`, following the empty-scaffold
    handling of the scaffold-CV reference workflow.
    """
    n = len(smiles_list)
    groups = np.empty(n, dtype=object)

    scaffold_to_group = {}
    next_group = 0
    empty_indices = []

    for i, smiles in enumerate(smiles_list):
        key = _murcko_scaffold_key(smiles)

        if key == "":
            empty_indices.append(i)
            continue

        if key not in scaffold_to_group:
            scaffold_to_group[key] = f"scaffold_{next_group}"
            next_group += 1

        groups[i] = scaffold_to_group[key]

    # Empty-scaffold molecules cannot all be one group because that could
    # create one extremely large group. Split them into deterministic
    # pseudo-groups of 40, following the scaffold-CV reference workflow.
    for start in range(0, len(empty_indices), empty_bucket_size):
        bucket = empty_indices[start:start + empty_bucket_size]
        group_name = f"empty_{start // empty_bucket_size}"
        for i in bucket:
            groups[i] = group_name

    return groups
