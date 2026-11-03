"""Regression tests for the Windows torch failure class (WinError 127 on torch\\lib\\shm.dll)."""
import contextlib, sys, unittest
from types import SimpleNamespace
from unittest import mock
import importlib.abc
import numpy as np
from tests.helpers import tmpdir, cleanup
from tests.test_registry_cli_state import mini
from cyp_tdi.config import load_config
from cyp_tdi.executor import Executor, StepFailed
from cyp_tdi.registry import step
from cyp_tdi.deps import import_framework, DependencyError, distribution_version, is_installed
from cyp_tdi.io_utils import library_versions
from cyp_tdi.seeds import seed_everything, seed_python_numpy
from cyp_tdi import resource_manager as RM
from cyp_tdi import embeddings as EMB

WINERR = '[WinError 127] The specified procedure could not be found. Error loading "torch\\lib\\shm.dll" or one of its dependencies.'
HEAVY = ("torch", "chemprop", "lightning", "tabicl")


class _Finder(importlib.abc.MetaPathFinder):
    """Simulates an installed-but-broken torch; records every attempt to import a heavy framework."""
    def __init__(self, broken=("torch",)): self.broken, self.attempts = broken, []
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in HEAVY:
            self.attempts.append(name)
        if root in self.broken:
            raise OSError(WINERR)
        return None


@contextlib.contextmanager
def broken_frameworks(broken=("torch",)):
    f = _Finder(broken); saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in HEAVY}
    for k in saved: del sys.modules[k]
    sys.meta_path.insert(0, f)
    try:
        yield f
    finally:
        sys.meta_path.remove(f)
        for k in [k for k in sys.modules if k.split(".")[0] in HEAVY]: del sys.modules[k]
        sys.modules.update(saved)


class LightweightWithoutTorch(unittest.TestCase):
    def setUp(self): self.t = tmpdir()
    def tearDown(self): cleanup(self.t)

    def test_lightweight_steps_run_when_torch_is_broken(self):          # root cause of 4 of the 6 failures
        reg, ctx, calls, _ = mini(self.t)
        with broken_frameworks() as f:
            Executor(reg, ctx).run(["C"]); lib = library_versions()
        self.assertEqual(calls, ["A", "B", "C"]); self.assertEqual(f.attempts, [])      # torch never even attempted
        self.assertIn("numpy", lib)

    def test_torch_step_fails_clearly_before_function_runs(self):
        reg, ctx, calls, inp = mini(self.t)
        Executor(reg, ctx).run(["A"]); ran = []
        step("T", "needs torch", depends_on=("A",), outputs=("t.txt",), requires=("torch",), registry=reg)(lambda c: ran.append(1))
        with broken_frameworks():
            with self.assertRaises(StepFailed) as cm: Executor(reg, ctx).run(["T"])
        self.assertIn("WinError 127", str(cm.exception.__cause__)); self.assertIn("check_environment.py", str(cm.exception.__cause__))
        self.assertEqual(ran, [])                                                         # never reached the step function
        st = ctx.state.load("T"); self.assertEqual((st["status"], st["phase"], st["step_function_invoked"]), ("failed", "setup", False))
        self.assertEqual(Executor(reg, ctx).status_of("A")[0], "complete")               # upstream untouched

    def test_torch_step_runs_when_torch_available(self):
        reg, ctx, calls, _ = mini(self.t); seen = []
        step("T", "needs torch", outputs=("t.txt",), requires=("torch",), registry=reg)(lambda c: (seen.append(c.device), (c.step_dir("T") / "t.txt").write_text("x")))
        fake = SimpleNamespace(manual_seed=lambda s: seen.append(("seed", s)), cuda=SimpleNamespace(is_available=lambda: False),
                               use_deterministic_algorithms=lambda *a, **k: None, backends=SimpleNamespace(cudnn=SimpleNamespace()))
        with mock.patch.dict(sys.modules, {"torch": fake}):
            ctx.cfg = load_config(); Executor(reg, ctx).run(["T"])
        self.assertEqual(seen, [("seed", 0), "cpu"])                                      # seeded with configured value, then run


class SeedingTests(unittest.TestCase):
    def test_python_numpy_seeding_without_any_framework(self):
        cfg = load_config()
        with broken_frameworks() as f:
            seed_everything(cfg); a = (np.random.rand(), __import__("random").random())
            seed_python_numpy(cfg); b = (np.random.rand(), __import__("random").random())
        self.assertEqual(a, b); self.assertEqual(f.attempts, [])

    def test_torch_seeding_failure_is_not_swallowed(self):
        with broken_frameworks():
            with self.assertRaises(DependencyError) as cm: seed_everything(load_config(), frameworks=("torch",))
        self.assertEqual(cm.exception.kind, "broken")


class DependencyReporting(unittest.TestCase):
    def test_versions_from_metadata_not_imports(self):
        with broken_frameworks(("torch", "chemprop")) as f:
            v = library_versions()
        self.assertEqual(f.attempts, [])
        for k in ("numpy", "pandas", "sklearn", "yaml", "torch", "chemprop", "tabicl", "lightgbm", "rdkit"): self.assertIn(k, v)
        self.assertIsNotNone(v["numpy"]); self.assertIsNotNone(v["sklearn"]); self.assertIsNotNone(v["yaml"])

    def test_distribution_name_mapping_and_missing(self):
        seen = []
        with mock.patch("cyp_tdi.deps.metadata.version", lambda d: seen.append(d) or "1.0"):
            distribution_version("sklearn"); distribution_version("yaml"); distribution_version("numpy")
        self.assertEqual(seen, ["scikit-learn", "PyYAML", "numpy"])
        self.assertIsNone(distribution_version("definitely_not_a_real_distribution_xyz"))

    def test_missing_dependency_messages(self):
        with self.assertRaises(DependencyError) as cm: import_framework("definitely_not_installed_pkg_xyz", "unit test")
        self.assertEqual(cm.exception.kind, "missing"); self.assertIn("pip install", str(cm.exception))
        with mock.patch.dict(sys.modules, {"torch": None}):           # 'import torch' -> ModuleNotFoundError
            with self.assertRaises(DependencyError) as cm: import_framework("torch", "embeddings")
        self.assertIn("pytorch.org", str(cm.exception)); self.assertFalse(is_installed("definitely_not_installed_pkg_xyz"))

    def test_optional_missing_torch_means_cpu_but_broken_torch_raises(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(RM.resolve_device({"gpu": {"enabled": "auto"}}), "cpu")
            with self.assertRaises(RM.ResourceError): RM.resolve_device({"gpu": {"enabled": "true", "required": True}})
        with broken_frameworks():
            with self.assertRaises(DependencyError): RM.resolve_device({"gpu": {"enabled": "auto"}})
            self.assertEqual(RM.resolve_device({"gpu": {"enabled": "false"}}), "cpu")      # explicit CPU never touches torch

    def test_embedding_functions_report_broken_torch_actionably(self):
        d = tmpdir()
        try:
            ck = d / "stand_in_checkpoint.pt"; ck.write_bytes(b"synthetic stand-in; torch import fails before it is read"); sha = RM.sha256_file(ck)
            with broken_frameworks():
                with self.assertRaises(DependencyError) as cm: EMB.chemprop_embed(["C"], ck, sha, "cpu", 8)
            self.assertIn("WinError 127", str(cm.exception))
        finally:
            cleanup(d)


class PreloadPolicy(unittest.TestCase):
    def test_wants_preload(self):
        from cyp_tdi.deps import wants_torch_preload as w
        self.assertTrue(w(['--isoform', 'CYP3A4', '--run-all'], {})); self.assertTrue(w(['--steps', 'a,b'], {})); self.assertTrue(w(['--step=x'], {}))
        for argv in (['--status'], ['--list-steps'], ['--help'], ['--run-all', '--dry-run'], []): self.assertFalse(w(argv, {}), argv)
        self.assertFalse(w(['--run-all'], {'CYP_TDI_NO_TORCH_PRELOAD': '1'}))

    def test_preload_is_noop_without_torch_and_records_failure_without_raising(self):
        from cyp_tdi import deps
        with mock.patch.dict(sys.modules, {'torch': None}), mock.patch.object(deps, 'is_installed', lambda m: False):
            self.assertIsNone(deps.preload_torch())
        with broken_frameworks(), mock.patch.object(deps, 'is_installed', lambda m: True):
            err = deps.preload_torch()
            self.assertIn('WinError 127', err); self.assertEqual(deps.PRELOAD_ERROR, err)
            with self.assertRaises(DependencyError): import_framework('torch')     # a torch step still fails loudly
        deps.PRELOAD_ERROR = None


class CheckpointSafety(unittest.TestCase):
    def test_safe_load_never_falls_back_to_unsafe(self):
        calls = []
        def load(path, map_location=None, weights_only=None):
            calls.append(weights_only)
            if weights_only: raise RuntimeError("unsupported global")
            return {"state_dict": {}}
        torch = SimpleNamespace(load=load)
        with self.assertRaises(EMB.EmbeddingError) as cm: EMB._safe_torch_load(torch, "x.pt", "cpu", True)
        self.assertEqual(calls, [True]); self.assertIn("weights_only=True", str(cm.exception))
        EMB._safe_torch_load(torch, "x.pt", "cpu", False); self.assertEqual(calls, [True, False])   # only on explicit opt-out

    def test_checkpoint_is_hash_verified_before_any_torch_import(self):
        with broken_frameworks() as f:
            with self.assertRaises(RM.ResourceError): EMB.chemeleon_embed(["C"], tmpdir() / "missing.pt", "0" * 64, "cpu")
        self.assertEqual(f.attempts, [])


if __name__ == "__main__":
    unittest.main()
