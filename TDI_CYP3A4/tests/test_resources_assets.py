import os, unittest
from unittest import mock
import numpy as np
from tests.helpers import ROOT, tmpdir, cleanup, have
from cyp_tdi.config import load_config, load_resources
from cyp_tdi import resource_manager as RM
from cyp_tdi.models import factory
from cyp_tdi.seeds import seed_everything
from cyp_tdi import embeddings as EMB
import logging


class ResourceTests(unittest.TestCase):
    def test_thread_env_and_lightgbm_jobs(self):
        keep = {k: os.environ.get(k) for k in RM.THREAD_ENV}
        try:
            self.assertEqual(RM.apply_thread_env({"cpu": {"threads": 3}}), 3)
            self.assertTrue(all(os.environ[k] == "3" for k in RM.THREAD_ENV))
            with self.assertRaises(RM.ResourceError): RM.apply_thread_env({"cpu": {"threads": 0}})
            self.assertIsNone(RM.apply_thread_env({"cpu": {"threads": None}}))
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual(RM.lightgbm_n_jobs({"cpu": {"threads": None}}), -1)
        self.assertEqual(RM.lightgbm_n_jobs({"cpu": {"threads": 4}}), 4)

    def test_device_resolution(self):
        self.assertEqual(RM.resolve_device({"gpu": {"enabled": "false"}}), "cpu")
        res = RM.apply_cli_overrides(load_resources(), cpu_threads=2, gpu=False, gpu_device=1)
        self.assertEqual((res["cpu"]["threads"], res["gpu"]["enabled"], res["gpu"]["device_index"]), (2, "false", 1))
        if not have("torch"):
            self.assertEqual(RM.resolve_device({"gpu": {"enabled": "auto"}}), "cpu")
            with self.assertRaises(RM.ResourceError): RM.resolve_device({"gpu": {"enabled": "true", "required": True}})
            self.assertEqual(RM.resolve_device({"gpu": {"enabled": "true", "required": False}}), "cpu")

    def test_local_checkpoints_match_pinned_hashes(self):
        res = load_resources()
        if not all(RM.checkpoint_spec(res, n, ROOT)[0].exists() for n in ("chemeleon_mp", "chemprop_medium")):
            self.skipTest("BLOCKED: checkpoints not present in assets/checkpoints (see assets/checkpoints/README.md)")
        for name in ("chemeleon_mp", "chemprop_medium"):
            path, sha = RM.checkpoint_spec(res, name, ROOT)
            self.assertTrue(path.exists(), path); self.assertEqual(RM.validate_checkpoint(path, sha, name), sha)

    def test_checkpoint_validation_errors(self):
        d = tmpdir()
        try:
            with self.assertRaises(RM.ResourceError) as cm: RM.validate_checkpoint(d / "nope.pt", None, "x")
            self.assertIn("missing", str(cm.exception))
            (d / "e.pt").write_bytes(b"")
            with self.assertRaises(RM.ResourceError): RM.validate_checkpoint(d / "e.pt", None)
            (d / "f.pt").write_bytes(b"abc")
            with self.assertRaises(RM.ResourceError): RM.validate_checkpoint(d / "f.pt", "0" * 64)
            with self.assertRaises(RM.ResourceError): RM.checkpoint_spec({"checkpoints": {}}, "zzz", ROOT)
        finally:
            cleanup(d)

    def test_lightgbm_params_profiles(self):
        res = load_resources()
        c = factory.lgbm_params(load_config(), res); b = factory.lgbm_params(load_config(profile="baseline"), res)
        self.assertTrue(c["deterministic"] and c["force_row_wise"]); self.assertNotIn("deterministic", b)
        for p in (c, b):
            self.assertEqual((p["n_estimators"], p["num_leaves"], p["random_state"], p["n_jobs"]), (1200, 63, 42, -1))
        self.assertNotIn("class_weight", c); self.assertNotIn("max_depth", c)

    def test_seeding(self):
        info = seed_everything(load_config()); self.assertEqual(info["python"], 0)
        a = np.random.rand(); seed_everything(load_config()); self.assertEqual(a, np.random.rand())


class TabICLCheckpointTests(unittest.TestCase):
    """The TabICL pretrained checkpoint is pinned (version + sha256), loaded from a verified local file, never auto-downloaded."""
    PIN = ("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt", "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0")

    def setUp(self):
        self.t = tmpdir(); factory._verified_tabicl_checkpoint.cache_clear()
        self.ck = self.t / "tabicl.ckpt"; self.ck.write_bytes(b"pretend checkpoint")
        import hashlib
        self.cfg = load_config(overrides={"tabicl": {"checkpoint": {"sha256": hashlib.sha256(b"pretend checkpoint").hexdigest()}}})
        self.calls = []

    def tearDown(self):
        factory._verified_tabicl_checkpoint.cache_clear(); cleanup(self.t)

    def _modules(self, hf_result):
        import types
        hf = types.ModuleType("huggingface_hub")
        def hf_hub_download(**kw):
            self.calls.append(kw)
            if isinstance(hf_result, Exception):
                raise hf_result
            return str(hf_result)
        hf.hf_hub_download = hf_hub_download
        tab = types.ModuleType("tabicl")
        test = self
        class FakeTabICL:
            def __init__(self, **kw): test.kwargs = kw
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.tile([0.25, 0.75], (len(X), 1))
        tab.TabICLClassifier = FakeTabICL
        return {"huggingface_hub": hf, "tabicl": tab}

    def test_config_pins_original_checkpoint(self):
        ck = load_config()["tabicl"]["checkpoint"]
        self.assertEqual((ck["hf_repo"], ck["version"], ck["sha256"]), self.PIN)
        self.assertEqual(load_config(profile="baseline")["tabicl"]["checkpoint"], ck)

    def test_fit_predict_uses_verified_local_file_and_disables_download(self):
        with mock.patch.dict("sys.modules", self._modules(self.ck)):
            [p] = factory.tabicl_fit_predict(self.cfg, "cpu", np.zeros((4, 2)), np.array([0, 1, 0, 1]), [np.zeros((3, 2))])
        self.assertEqual(p.tolist(), [0.75] * 3)
        self.assertEqual(self.calls[0], {"repo_id": self.PIN[0], "filename": self.PIN[1], "local_files_only": True})
        self.assertEqual(self.kwargs, {"n_estimators": 4, "random_state": 42, "device": "cpu", "checkpoint_version": self.PIN[1],
                                       "model_path": self.ck, "allow_auto_download": False})

    def test_hash_mismatch_or_missing_cache_blocks(self):
        bad = load_config()                                       # real pinned sha256 != the pretend file
        with mock.patch.dict("sys.modules", self._modules(self.ck)), self.assertRaises(RM.ResourceError) as cm:
            factory.tabicl_fit_predict(bad, "cpu", np.zeros((2, 2)), np.array([0, 1]), [np.zeros((1, 2))])
        self.assertIn("sha256 mismatch", str(cm.exception))
        factory._verified_tabicl_checkpoint.cache_clear()
        with mock.patch.dict("sys.modules", self._modules(FileNotFoundError("not cached"))), self.assertRaises(RM.ResourceError) as cm:
            factory.tabicl_fit_predict(self.cfg, "cpu", np.zeros((2, 2)), np.array([0, 1]), [np.zeros((1, 2))])
        self.assertIn("never downloads", str(cm.exception))

    @unittest.skipUnless(have("huggingface_hub"), "BLOCKED: huggingface_hub not installed")
    def test_real_cached_checkpoint_matches_pin(self):
        try:
            p = factory._verified_tabicl_checkpoint(*self.PIN)
        except RM.ResourceError as e:
            if "not in the local Hugging Face cache" in str(e):
                self.skipTest("BLOCKED: pinned TabICL checkpoint not in the local Hugging Face cache")
            raise
        self.assertEqual(p.name, self.PIN[1])


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.t = tmpdir(); self.cfg = load_config(); self.lg = logging.getLogger("emb"); self.lg.addHandler(logging.NullHandler())
        self.ck = self.t / "stand_in_checkpoint.pt"; self.ck.write_bytes(b"synthetic stand-in; contents are never loaded (embedding function is mocked)")
        self.sha = RM.sha256_file(self.ck); self.calls = 0

    def tearDown(self): cleanup(self.t)

    def fake(self, smiles, ckpt, sha, dev, bs):
        self.calls += 1; return np.random.default_rng(len(smiles)).normal(size=(len(smiles), 5))

    def run_embed(self, smi):
        with mock.patch.object(EMB, "chemprop_embed", self.fake):
            return EMB.embed_with_cache("chemprop", smi, self.ck, self.sha, "cpu", self.cfg, self.t, self.lg)

    def test_cache_reuse_and_invalidation(self):
        smi = ["C", "CC", "CCC"]
        a, m1 = self.run_embed(smi); b, m2 = self.run_embed(smi)
        self.assertEqual((self.calls, m1["from_cache"], m2["from_cache"]), (1, False, True)); self.assertTrue(np.array_equal(a, b))
        self.run_embed(["C", "CC", "CCCC"]); self.assertEqual(self.calls, 2)                  # different molecules -> new key
        f = self.t / "embeddings" / f"chemprop-{m1['key']}.npy"; arr = np.load(f); arr[0, 0] += 1; np.save(f, arr)   # tamper
        _, m3 = self.run_embed(smi); self.assertEqual((self.calls, m3["from_cache"]), (3, False))                    # recomputed

    def test_bad_checkpoint_blocks_embedding(self):
        with self.assertRaises(RM.ResourceError):
            EMB.embed_with_cache("chemprop", ["C"], self.t / "missing.pt", self.sha, "cpu", self.cfg, self.t, self.lg)
        with self.assertRaises(RM.ResourceError):
            EMB.embed_with_cache("chemprop", ["C"], self.ck, "0" * 64, "cpu", self.cfg, self.t, self.lg)


@unittest.skipUnless(have("rdkit"), "BLOCKED: rdkit not installed in the packaging environment")
class RDKitTests(unittest.TestCase):
    def test_feature_dimensions_and_descriptor_order(self):
        import json
        from cyp_tdi import featurization as F
        cfg = load_config()["features"]
        with open(ROOT / "assets/reference/jb_descriptor_names.json") as fh: ref = json.load(fh)
        self.assertEqual(F.descriptor_names(), ref)
        X, ok = F.featurize(["CCO", "c1ccccc1O", "not_a_smiles"], cfg)
        self.assertEqual(X.shape, (3, 3456)); self.assertEqual(ok.tolist(), [True, True, False]); self.assertTrue(np.isnan(X[2]).all())
        with self.assertRaises(Exception): F.require_all_valid(ok, ["a", "b", "c"], "t")


@unittest.skipUnless(have("lightgbm"), "BLOCKED: lightgbm not installed in the packaging environment")
class LightGBMTests(unittest.TestCase):
    def test_make_lgbm_accepts_params(self):
        m = factory.make_lgbm(load_config(), load_resources()); self.assertEqual(m.get_params()["num_leaves"], 63)

    def test_booster_save_load_in_non_ascii_path(self):
        """Regression: model_training failed with 'Model file ...base_full.txt is not available for writes' because the
        project path contains emoji; LightGBM's native file I/O cannot open non-ASCII paths on Windows."""
        import sys, lightgbm as lgb
        from cyp_tdi.models.factory import save_booster, load_booster
        d = tmpdir()
        try:
            X = np.random.default_rng(0).normal(size=(120, 6)); y = (X[:, 0] > 0).astype(int)
            b = lgb.LGBMClassifier(n_estimators=8, verbosity=-1).fit(X, y).booster_
            uni = d / "proj_🔵_é" / "model_training"; uni.mkdir(parents=True)
            save_booster(b, uni / "base_full.txt")
            p = uni / "base_full.txt"; self.assertGreater(p.stat().st_size, 0)
            r = load_booster(p)
            self.assertEqual(r.num_feature(), 6); self.assertTrue(np.allclose(r.predict(X), b.predict(X)))
            self.assertEqual(p.read_bytes(), b.model_to_string().encode("utf-8"))           # written verbatim
            asc = d / "ascii"; asc.mkdir(); b.save_model(str(asc / "native.txt"))            # native writer, ASCII path
            self.assertEqual(p.read_bytes(), (asc / "native.txt").read_bytes().replace(b"\r\n", b"\n"))  # same model text
            (asc / "ours.txt").write_bytes(p.read_bytes())
            self.assertTrue(np.allclose(lgb.Booster(model_file=str(asc / "ours.txt")).predict(X), b.predict(X)))
            if sys.platform == "win32":                                                        # documents the root cause
                with self.assertRaises(lgb.basic.LightGBMError):
                    b.save_model(str(uni / "native_fails.txt"))
        finally:
            cleanup(d)


if __name__ == "__main__":
    unittest.main()
