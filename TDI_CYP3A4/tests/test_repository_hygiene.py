import ast, re, unittest
from tests.helpers import ROOT


class RepositoryHygieneTests(unittest.TestCase):
    def test_syntax_and_imports(self):
        for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "run_pipeline.py"] + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
            ast.parse(p.read_text(encoding="utf-8"), str(p))
        import importlib
        for m in ("cyp_tdi.config", "cyp_tdi.cli", "cyp_tdi.steps", "cyp_tdi.executor", "cyp_tdi.core.ensemble"):
            importlib.import_module(m)         # lazily-imported heavy libs must not be needed at import time

    def test_no_placeholder_modules(self):
        for p in (ROOT / "src" / "cyp_tdi").rglob("*.py"):
            if p.name == "__init__.py": continue
            self.assertGreater(len(p.read_text().strip().splitlines()), 5, p)

    def test_no_machine_specific_paths_in_source(self):
        for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "config").rglob("*.yaml")):
            self.assertIsNone(re.search(r"/home/\w+|C:\\\\Users|/Users/\w+", p.read_text()), p)

    def test_no_generated_results_in_tree(self):
        for p in (ROOT / "outputs").rglob("*"):
            if p.is_file() and p.name != ".gitkeep":
                self.skipTest("outputs/ has local run results (ignored by .gitignore)")


class NoDeadModules(unittest.TestCase):
    def test_every_module_is_referenced(self):
        """Regression: an abandoned draft of modules had been shipped unused. Every module under src/ must be imported somewhere."""
        src = ROOT / "src" / "cyp_tdi"; mods = {}
        for p in src.rglob("*.py"):
            if p.name == "__init__.py": continue
            mods[p.stem] = p
        text = {p: p.read_text() for p in list(src.rglob("*.py")) + list((ROOT / "tests").glob("*.py")) + [ROOT / "run_pipeline.py"]}
        for stem, p in mods.items():
            pat = re.compile(rf"(from\s+(\.+|cyp_tdi\.?)[\w.]*\s+import\s+[^\n]*\b{stem}\b|from\s+(\.+|cyp_tdi\.)[\w.]*\b{stem}\b\s+import|import\s+cyp_tdi\.[\w.]*\b{stem}\b|from \.\s+import\s+[^\n]*\b{stem}\b)")
            used = any(pat.search(t) for q, t in text.items() if q != p)
            self.assertTrue(used, f"{p.relative_to(ROOT)} is not imported by anything (dead module?)")

    def test_no_stale_config_or_reference_files(self):
        self.assertEqual(sorted(x.name for x in (ROOT / "config").glob("*.yaml")), ["pipeline.yaml", "resources.yaml"])
        self.assertEqual(sorted(x.name for x in (ROOT / "assets" / "reference").iterdir()),
                         ["jb_descriptor_names.json", "jb_fold_universe_smiles.csv"])
