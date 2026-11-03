"""REAL PyTorch integration tests, each in a FRESH Python process.

Why subprocesses: on Windows the same torch install imported fine in a fresh process (and when this one test ran alone) but failed with
WinError 127 on torch\\lib\\shm.dll inside a long pytest session after numpy/scikit-learn had done native work. Running each real-torch test
in its own interpreter makes the result independent of test order. They are NOT mocks: they import the real torch, seed it, and load the
real checkpoint. Skipped (blocked) only when the package is not installed; an installed-but-broken torch FAILS with the child's output."""
import os, subprocess, sys, unittest
from tests.helpers import ROOT, have


def run_isolated(code, timeout=600):
    env = dict(os.environ); env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["CYP_TDI_ROOT"] = str(ROOT)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=timeout)


def assert_ok(tc, r):
    tc.assertEqual(r.returncode, 0, f"child failed (exit {r.returncode})\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr[-4000:]}")


@unittest.skipUnless(have("torch"), "BLOCKED: torch not installed")
class RealTorch(unittest.TestCase):
    def test_real_torch_seeding(self):
        r = run_isolated("""
from cyp_tdi.config import load_config
from cyp_tdi.seeds import seed_everything
from cyp_tdi.deps import import_framework
cfg = load_config(); torch = import_framework('torch')
seed_everything(cfg, frameworks=('torch',)); a = torch.rand(3)
seed_everything(cfg, frameworks=('torch',)); b = torch.rand(3)
assert bool((a == b).all()), (a, b)
print('OK seeding reproducible; torch', torch.__version__)
""")
        assert_ok(self, r); self.assertIn("OK seeding", r.stdout)

    def test_torch_after_numeric_libraries_with_preload(self):
        """Production order: run_pipeline.py preloads torch, THEN numpy/scipy/sklearn (and rdkit/lightgbm if installed) do native work,
        THEN a torch step runs. If this fails while test_real_torch_seeding passes, the import-order hypothesis is wrong."""
        extra = "".join(f"import {m}\n" for m in ("rdkit.Chem", "lightgbm") if have(m.split('.')[0]))
        r = run_isolated(f"""
from cyp_tdi.deps import preload_torch
err = preload_torch(); assert err is None, err
import numpy as np, scipy.linalg
from sklearn.decomposition import PCA
{extra}
X = np.random.default_rng(0).normal(size=(200, 30)); PCA(5).fit(X); np.linalg.svd(X); (X @ X.T).sum(); scipy.linalg.svd(X)
from cyp_tdi.deps import import_framework
torch = import_framework('torch'); t = torch.randn(8, 8); assert float((t @ t).sum()) == float((t @ t).sum())
print('OK torch usable after numeric libraries')
""")
        assert_ok(self, r); self.assertIn("OK torch usable", r.stdout)


@unittest.skipUnless(have("torch") and have("tabicl"), "BLOCKED: torch/tabicl not installed")
class RealTabICL(unittest.TestCase):
    def test_factory_runs_real_tabicl_offline_with_pinned_checkpoint(self):
        """API plumbing only (small synthetic table, CPU): the real TabICL 2.2.0 accepts the pinned-checkpoint arguments,
        loads the verified file without network access and returns P(class 1) per row. Not evidence of model quality."""
        r = run_isolated("""
import os; os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from cyp_tdi.config import load_config
from cyp_tdi.models import factory
cfg = load_config(); rng = np.random.default_rng(0)
X = rng.normal(size=(120, 16)).astype(np.float32); y = (X[:, 0] > 0).astype(int)
p1, p2 = factory.tabicl_fit_predict(cfg, 'cpu', X[:100], y[:100], [X[100:], X[:5]])
assert p1.shape == (20,) and p2.shape == (5,) and np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all(), (p1.shape, p2.shape)
print('OK real TabICL via factory, pinned', cfg['tabicl']['checkpoint']['version'])
""")
        if r.returncode != 0 and "not in the local Hugging Face cache" in r.stderr:
            self.skipTest("BLOCKED: pinned TabICL checkpoint not in the local Hugging Face cache")
        assert_ok(self, r); self.assertIn("OK real TabICL via factory", r.stdout)


@unittest.skipUnless(have("torch"), "BLOCKED: torch not installed")
@unittest.skipUnless((ROOT / "assets" / "checkpoints" / "chemeleon_mp.pt").exists() and (ROOT / "assets" / "checkpoints" / "chemprop_medium.pt").exists(),
                     "BLOCKED: checkpoints not present in assets/checkpoints (see assets/checkpoints/README.md)")
class RealCheMeleon(unittest.TestCase):
    def test_chemeleon_safe_load(self):
        r = run_isolated("""
from cyp_tdi.config import load_resources
from cyp_tdi import resource_manager as RM, embeddings as EMB
from cyp_tdi.deps import import_framework
from pathlib import Path; import os
root = Path(os.environ['CYP_TDI_ROOT'])
path, sha = RM.checkpoint_spec(load_resources(), 'chemeleon_mp', root); RM.validate_checkpoint(path, sha, 'chemeleon')
torch = import_framework('torch'); st = EMB._safe_torch_load(torch, path, 'cpu', True)
assert 'state_dict' in st and 'hyper_parameters' in st, list(st)
print('OK weights_only=True load;', len(st['state_dict']), 'tensors')
""")
        assert_ok(self, r); self.assertIn("OK weights_only", r.stdout)

    @unittest.skipUnless(have("chemprop"), "BLOCKED: chemprop not installed")
    def test_chemeleon_embedding_smoke_cpu(self):
        r = run_isolated("""
import numpy as np, os
from pathlib import Path
from cyp_tdi.config import load_resources
from cyp_tdi import resource_manager as RM, embeddings as EMB
root = Path(os.environ['CYP_TDI_ROOT'])
path, sha = RM.checkpoint_spec(load_resources(), 'chemeleon_mp', root)
E = EMB.chemeleon_embed(['CCO', 'c1ccccc1O'], path, sha, 'cpu', 2, True)
assert E.shape == (2, 2048) and np.isfinite(E).all() and not np.allclose(E[0], E[1]), E.shape
print('OK chemeleon embeddings', E.shape)
""")
        assert_ok(self, r); self.assertIn("OK chemeleon embeddings", r.stdout)

    @unittest.skipUnless(have("chemprop"), "BLOCKED: chemprop not installed")
    def test_chemprop_medium_embedding_smoke_cpu(self):
        r = run_isolated("""
import numpy as np, os
from pathlib import Path
from cyp_tdi.config import load_resources
from cyp_tdi import resource_manager as RM, embeddings as EMB
root = Path(os.environ['CYP_TDI_ROOT'])
path, sha = RM.checkpoint_spec(load_resources(), 'chemprop_medium', root)
E = EMB.chemprop_embed(['CCO', 'c1ccccc1O'], path, sha, 'cpu', 2)
assert E.shape[0] == 2 and np.isfinite(E).all(), E.shape
print('OK chemprop embeddings', E.shape)
""")
        assert_ok(self, r); self.assertIn("OK chemprop embeddings", r.stdout)


if __name__ == "__main__":
    unittest.main()
