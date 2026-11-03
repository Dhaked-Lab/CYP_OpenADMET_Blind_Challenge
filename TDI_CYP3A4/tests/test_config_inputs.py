import unittest, copy
from tests.helpers import tmpdir, cleanup
import pandas as pd
from cyp_tdi.config import (load_config, load_isoform_config, ConfigError, fingerprint, fracs, validate_config)
from cyp_tdi.discovery import discover_inputs, load_training_frames, load_blind, InputError
from cyp_tdi.validation import (ValidationError, validate_blind_frame, validate_weights,
                                validate_submission, validate_descriptor_names, validate_scores, validate_feature_matrix)
import numpy as np


class ConfigTests(unittest.TestCase):
    def test_default_and_baseline_profiles(self):
        c, b = load_config(), load_config(profile="baseline")
        self.assertEqual(c["profile"], "corrected"); self.assertEqual(b["profile"], "baseline")
        self.assertEqual(c["pca"]["fold_scope"], "fold_train_only"); self.assertEqual(b["pca"]["fold_scope"], "global_legacy")
        self.assertEqual(c["selection"]["count_convention"], "exact_k"); self.assertEqual(b["selection"]["count_convention"], "legacy_k_plus_1")
        self.assertFalse(b["lightgbm"]["deterministic"])

    def test_original_values_preserved(self):
        c = load_config()
        self.assertEqual((c["seeds"]["folds"], c["seeds"]["lightgbm"], c["seeds"]["pca"], c["seeds"]["tabicl"]), (7, 42, 0, 42))
        lp = c["lightgbm"]["params"]
        self.assertEqual((lp["n_estimators"], lp["num_leaves"], lp["learning_rate"], lp["colsample_bytree"]), (1200, 63, 0.03, 0.5))
        self.assertEqual((c["pca"]["n_components"], c["pca"]["whiten"], c["tabicl"]["n_estimators"]), (256, True, 4))
        self.assertEqual(c["ensemble"]["greedy_rounds"], 12)
        f = fracs(c); self.assertEqual((len(f), float(f[0]), float(f[-1])), (56, 0.05, 0.6))

    def test_invalid_configs_rejected(self):
        c = load_config()
        for path, val in (("pca.fold_scope", "bogus"), ("selection.count_convention", "x"), ("cv.k", 1),
                          ("threshold.fraction_override", 1.5), ("nn_features.mode", "zzz")):
            bad = copy.deepcopy(c); cur = bad
            *p, last = path.split("."); [cur := cur[x] for x in p]; cur[last] = val
            with self.assertRaises(ConfigError, msg=path):
                validate_config(bad)
        with self.assertRaises(ConfigError):
            load_config(profile="nonsense")

    def test_fingerprint_stable_and_sensitive(self):
        c = load_config(); self.assertEqual(fingerprint(c), fingerprint(copy.deepcopy(c)))
        d = copy.deepcopy(c); d["seeds"]["folds"] = 8; self.assertNotEqual(fingerprint(c), fingerprint(d))

    def test_isoform_config(self):
        iso = load_isoform_config("CYP3A4"); self.assertEqual(iso["label_column"], "CYP3A4_is_TDI")
        with self.assertRaises(ConfigError):
            load_isoform_config("CYP2D6")


def _write(d, name, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(d / name, index=False)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(); self.iso = load_isoform_config("CYP3A4")
        c = ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"]
        _write(self.d, "CYP3A4_Positive.csv", [["a", "CCO", "TRUE"], ["b", "CCN", "TRUE"]], c)
        _write(self.d, "CYP3A4_Negative.csv", [["c", "CCC", "FALSE"]], c)
        _write(self.d, "TEST-BLINDED.csv", [["x", "CCCl"], ["y", "CCBr"]], ["Molecule_Name", "SMILES"])

    def tearDown(self): cleanup(self.d)

    def test_discovery_pos_neg_and_labels_checked(self):
        inp = discover_inputs(self.d, self.iso); self.assertEqual(inp["mode"], "positive_negative")
        fr = load_training_frames(inp, self.iso); self.assertEqual([f[1] for f in fr], [1, 0])

    def test_swapped_files_detected(self):  # JB-10
        c = ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"]
        _write(self.d, "CYP3A4_Negative.csv", [["c", "CCC", "TRUE"]], c)
        with self.assertRaises(ValidationError):
            load_training_frames(discover_inputs(self.d, self.iso), self.iso)

    def test_foreign_isoform_rejected(self):
        c = ["Molecule_Name", "SMILES", "CYP2D6_is_TDI"]
        _write(self.d, "CYP3A4_Positive.csv", [["a", "CCO", "TRUE"]], c)
        with self.assertRaises(ValidationError) as cm:
            load_training_frames(discover_inputs(self.d, self.iso), self.iso)
        self.assertIn("another isoform", str(cm.exception))
        _write(self.d, "TEST-BLINDED.csv", [["x", "CC", "TRUE"]], ["Molecule_Name", "SMILES", "CYP2D6_is_TDI"])
        with self.assertRaises(ValidationError):
            load_blind(discover_inputs(self.d, self.iso), self.iso)

    def test_combined_mode_and_missing_inputs(self):
        (self.d / "CYP3A4_Positive.csv").unlink(); (self.d / "CYP3A4_Negative.csv").unlink()
        with self.assertRaises(InputError):
            discover_inputs(self.d, self.iso)
        _write(self.d, "CYP3A4.csv", [["a", "CCO", "TRUE"], ["c", "CCC", "FALSE"]], ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"])
        inp = discover_inputs(self.d, self.iso); self.assertEqual(inp["mode"], "combined")
        self.assertEqual(len(load_training_frames(inp, self.iso)[0][2]), 2)
        (self.d / "TEST-BLINDED.csv").unlink()
        with self.assertRaises(InputError):
            discover_inputs(self.d, self.iso)

    def test_blind_validation(self):
        ok = pd.DataFrame({"Molecule_Name": ["x", "y"], "SMILES": ["C", "CC"]}); validate_blind_frame(ok, self.iso)
        for bad in (pd.DataFrame({"Molecule_Name": ["x", "x"], "SMILES": ["C", "CC"]}),
                    pd.DataFrame({"Molecule_Name": ["x", "y"], "SMILES": ["C", None]}),
                    pd.DataFrame({"Molecule_Name": ["x", "y"]}),
                    pd.DataFrame({"Molecule_Name": ["x"], "SMILES": ["C"], "CYP3A4_is_TDI": [True]}),
                    pd.DataFrame({"Molecule_Name": ["x"], "SMILES": ["C"], "weird": [1]})):
            with self.assertRaises(ValidationError):
                validate_blind_frame(bad, self.iso)

    def test_padding_rows_dropped(self):
        with open(self.d / "CYP3A4_Positive.csv", "a") as fh:
            fh.write(",,\n" * 1000)
        fr = load_training_frames(discover_inputs(self.d, self.iso), self.iso)
        self.assertEqual((len(fr[0][2]), fr[0][4] >= 1000), (2, True))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.iso = load_isoform_config("CYP3A4")
        self.blind = pd.DataFrame({"Molecule_Name": list("abcd"), "SMILES": ["C", "CC", "CCC", "CCCC"]})

    def _sub(self, labels):
        return pd.DataFrame({"Molecule_Name": self.blind.Molecule_Name, "SMILES": self.blind.SMILES, "CYP3A4_is_TDI": np.array(labels, dtype=bool)})

    def test_valid_and_failures(self):
        validate_submission(self._sub([1, 1, 0, 0]), self.blind, self.iso, 2)
        with self.assertRaises(ValidationError): validate_submission(self._sub([1, 1, 0, 0]), self.blind, self.iso, 3)
        s = self._sub([1, 1, 0, 0]); s2 = s.rename(columns={"CYP3A4_is_TDI": "label"})
        with self.assertRaises(ValidationError): validate_submission(s2, self.blind, self.iso, 2)
        s3 = s.iloc[[1, 0, 2, 3]].reset_index(drop=True)
        with self.assertRaises(ValidationError): validate_submission(s3, self.blind, self.iso, 2)
        with self.assertRaises(ValidationError): validate_submission(s.iloc[:3], self.blind, self.iso, 2)
        s4 = s.copy(); s4.loc[1, "Molecule_Name"] = "a"
        with self.assertRaises(ValidationError): validate_submission(s4, self.blind, self.iso, 2)
        s5 = s.copy(); s5["CYP3A4_is_TDI"] = s5["CYP3A4_is_TDI"].astype(int)
        with self.assertRaises(ValidationError): validate_submission(s5, self.blind, self.iso, 2)

    def test_weights_scores_matrix_descriptors(self):
        validate_weights({"a": 0.5, "b": 0.5}, ["a", "b"])
        for w in ({"a": 0.5}, {"a": 0.7, "b": 0.7}, {"a": -0.1, "b": 1.1}, {"a": 0.5, "b": 0.5, "c": 0}):
            with self.assertRaises(ValidationError): validate_weights(w, ["a", "b"])
        with self.assertRaises(ValidationError): validate_scores(np.array([1.0, np.nan]), 2)
        with self.assertRaises(ValidationError): validate_scores(np.array([1.0]), 2)
        with self.assertRaises(ValidationError): validate_feature_matrix(np.array([[1.0, np.inf]]), 2)
        with self.assertRaises(ValidationError): validate_feature_matrix(np.ones((2, 3)), 2)
        validate_descriptor_names(["a", "b"], ["a", "b"])
        with self.assertRaises(ValidationError): validate_descriptor_names(["b", "a"], ["a", "b"])
        self.assertIsInstance(validate_descriptor_names(["a"], ["a", "b"], allow_mismatch=True), str)


if __name__ == "__main__":
    unittest.main()
