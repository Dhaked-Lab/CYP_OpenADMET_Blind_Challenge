"""Static consistency checks. Need only the standard library, PyYAML and pandas-free code."""

from __future__ import annotations

import ast
import re
import unittest

from helpers import ROOT, SCRIPTS, SRC, literal_kwargs, module_level_assignments

PY_FILES = sorted(p for p in ROOT.rglob("*.py") if ".git" not in p.parts)


class TestSourceFiles(unittest.TestCase):
    def test_all_python_files_parse(self):
        for path in PY_FILES:
            with self.subTest(file=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"))

    def test_no_machine_specific_paths(self):
        pattern = re.compile(r"(/home/\w+|/Users/\w+|[A-Za-z]:\\\\Users|/mnt/)")
        for path in PY_FILES + [ROOT / "README.md", ROOT / "config" / "structural_alerts.yaml"]:
            with self.subTest(file=str(path.relative_to(ROOT))):
                if path.parent.name == "tests":
                    continue  # this very file contains the pattern
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    def test_scripts_resolve_the_src_directory(self):
        """Steps 01-09 insert <repo>/src on sys.path; it must point at the package."""
        for path in sorted(SCRIPTS.glob("0[1-9]_*.py")):
            with self.subTest(script=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn('Path(__file__).resolve().parents[1] / "src"', text)
        self.assertTrue((SRC / "cyp_tdi_hybrid" / "__init__.py").is_file())

    def test_readme_commands_reference_existing_scripts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        referenced = set(re.findall(r"python (scripts/[\w./-]+\.py)", readme))
        self.assertTrue(referenced)
        for rel in referenced:
            with self.subTest(command=rel):
                self.assertTrue((ROOT / rel).is_file(), rel)

    def test_no_references_to_removed_modules(self):
        removed = ("caruana", "chemprop_transfer", "tabular_cv", "cyp3a4", "scripts/cyp2d6/")
        for path in PY_FILES:
            if path.parent.name == "tests" or "reference" in path.parts:
                continue  # tests mention the words; reference/ is vendored unchanged
            text = path.read_text(encoding="utf-8").lower()
            for word in removed:
                with self.subTest(file=path.name, word=word):
                    self.assertNotIn(word, text)


class TestConfigurationConsistency(unittest.TestCase):
    """The model definitions are repeated in steps 05 / 08 (and read by step 11)."""

    @classmethod
    def setUpClass(cls):
        cls.s05 = module_level_assignments(SCRIPTS / "05_train_models_oof.py")
        cls.s06 = module_level_assignments(SCRIPTS / "06_threshold_optimization.py")
        cls.s07 = module_level_assignments(SCRIPTS / "07_oof_evaluation.py")
        cls.s08 = module_level_assignments(SCRIPTS / "08_final_training_check.py")
        cls.s09 = module_level_assignments(SCRIPTS / "09_blind_prediction.py")
        cls.s11 = module_level_assignments(SCRIPTS / "11_hyperparameter_tuning.py")

    def test_model_parameters_match_step08_expectations(self):
        expected = literal_kwargs(self.s08["EXPECTED_PARAMS"])
        pairs = {"RF": "RF_PARAMS", "ExtraTrees": "ET_PARAMS",
                 "LightGBM": "LGBM_PARAMS", "XGBoost": "XGB_PARAMS"}
        for model, const in pairs.items():
            with self.subTest(model=model):
                self.assertEqual(literal_kwargs(self.s05[const]), expected[model])

    def test_ensemble_members_agree_across_steps(self):
        members = ast.literal_eval(self.s05["ENSEMBLE_MEMBERS"])
        self.assertEqual(members, ["RF", "ExtraTrees", "LightGBM", "XGBoost"])
        self.assertEqual(ast.literal_eval(self.s06["ENSEMBLE_MEMBERS"]), members)
        self.assertEqual(ast.literal_eval(self.s08["EXPECTED_MEMBERS"]), members)
        self.assertEqual(ast.literal_eval(self.s09["ENSEMBLE_MEMBERS"]), members)
        self.assertEqual(ast.literal_eval(self.s11["MODELS"]), members)

    def test_threshold_grid_is_identical_in_steps_06_07_11(self):
        grids = {ast.dump(m["GRID"]) for m in (self.s06, self.s07, self.s11)}
        self.assertEqual(len(grids), 1)

    def test_fold_settings(self):
        s04 = module_level_assignments(SCRIPTS / "04_scaffold_folds.py")
        self.assertEqual(ast.literal_eval(s04["N_FOLDS"]), 5)
        self.assertEqual(ast.literal_eval(s04["SEED"]), 7)
        self.assertEqual(ast.literal_eval(s04["EMPTY_GROUP_SIZE"]), 40)
        self.assertEqual(ast.literal_eval(self.s08["EXPECTED_N_FOLDS"]), 5)


class TestStructuralAlertConfig(unittest.TestCase):
    def test_alert_file_has_13_unique_named_smarts(self):
        yaml = __import__("yaml")
        cfg = yaml.safe_load((ROOT / "config" / "structural_alerts.yaml").read_text(encoding="utf-8"))
        alerts = cfg["alerts"]
        self.assertEqual(len(alerts), 13)
        self.assertEqual(len({a["name"] for a in alerts}), 13)
        for a in alerts:
            self.assertTrue(a["smarts"].strip())
            self.assertTrue(a["rationale"].strip())


if __name__ == "__main__":
    unittest.main()
