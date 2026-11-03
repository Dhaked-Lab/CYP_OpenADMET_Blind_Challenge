"""Tests that need RDKit (skipped automatically if it is not installed)."""

from __future__ import annotations

import sys
import unittest

import numpy as np

from helpers import SRC, have, load_script

sys.path.insert(0, str(SRC))

HAVE_RDKIT = have("rdkit")


@unittest.skipUnless(HAVE_RDKIT, "RDKit not installed")
class TestScaffoldFolds(unittest.TestCase):
    SMILES = (["c1ccccc1C", "c1ccccc1CC", "c1ccccc1O"] * 8          # benzene scaffold
              + ["c1ccncc1C", "c1ccncc1N"] * 6                       # pyridine scaffold
              + ["C1CCNCC1C", "C1CCNCC1O", "C1CCNCC1N"] * 4          # piperidine scaffold
              + ["CCO", "CCN", "CCC", "CC(=O)O", "CCCl"] * 10)       # empty scaffold

    @classmethod
    def setUpClass(cls):
        cls.s04 = load_script("04_scaffold_folds.py")

    def test_groups_are_never_split_and_assignment_is_deterministic(self):
        groups = self.s04.scaffold_groups(self.SMILES)
        folds = self.s04.make_folds(groups, k=5, seed=7)
        self.assertTrue((folds >= 0).all() and (folds < 5).all())
        by_group: dict[str, set[int]] = {}
        for g, f in zip(groups, folds):
            by_group.setdefault(g, set()).add(int(f))
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        np.testing.assert_array_equal(folds, self.s04.make_folds(groups, k=5, seed=7))

    def test_empty_scaffold_molecules_are_bucketed_by_40(self):
        groups = self.s04.scaffold_groups(self.SMILES)
        empty = [g for g in groups if g.startswith("__EMPTY_SCAFFOLD_")]
        self.assertEqual(len(empty), 50)
        counts = sorted(np.unique(empty, return_counts=True)[1].tolist())
        self.assertEqual(counts, [10, 40])


@unittest.skipUnless(HAVE_RDKIT, "RDKit not installed")
class TestFeaturesAndAlerts(unittest.TestCase):
    def test_ecfp4_shape_and_bits(self):
        from cyp_tdi_hybrid.features import ecfp4
        fp = ecfp4(["c1ccccc1", "CCO"])
        self.assertEqual(fp.shape, (2, 2048))
        self.assertTrue(set(np.unique(fp)) <= {0, 1})
        self.assertGreater(int(fp.sum()), 0)

    def test_structural_alert_matrix(self):
        from cyp_tdi_hybrid.structural_alerts import structural_alert_features
        df = structural_alert_features(["c1ccoc1", "c1ccccc1", "C#CCN"])
        self.assertEqual(df.shape, (3, 13))
        self.assertEqual(int(df.loc[0, "alert_FURAN_RING"]), 1)
        self.assertEqual(int(df.loc[1].sum()), 0)
        self.assertEqual(int(df.loc[2, "alert_TERMINAL_ALKYNE"]), 1)

    def test_scaffold_groups_helper_matches_step04_partition(self):
        from cyp_tdi_hybrid.splits import scaffold_groups
        s04 = load_script("04_scaffold_folds.py")
        smiles = TestScaffoldFolds.SMILES
        a = scaffold_groups(smiles)
        b = s04.scaffold_groups(smiles)
        mapping: dict = {}
        for x, y in zip(a, b):
            self.assertEqual(mapping.setdefault(x, y), y)
        self.assertEqual(len(set(a)), len(set(b)))


if __name__ == "__main__":
    unittest.main()
