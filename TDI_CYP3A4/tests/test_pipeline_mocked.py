"""MOCKED integration tests: real step code, synthetic arrays, stub models (no LightGBM/TabICL/RDKit/torch).
These validate orchestration and data flow only - NOT scientific performance."""
import json, logging, pickle, unittest
from unittest import mock
import numpy as np, pandas as pd
from tests.helpers import ROOT, tmpdir, cleanup
from cyp_tdi.config import load_config, load_isoform_config, load_resources
from cyp_tdi.executor import Context
from cyp_tdi.state_manager import StateManager
from cyp_tdi.models import factory
from cyp_tdi.core.nn_features import tanimoto_binary
from cyp_tdi.validation import ValidationError
from cyp_tdi.steps import model_steps as MS, inference_steps as IS


def _j(p):
    with open(p) as fh:
        return json.load(fh)


class _Boost:
    def __init__(self, m): self.m = m
    def save_model(self, path):
        with open(path, "wb") as fh: pickle.dump(self.m, fh)
    def num_feature(self): return self.m.coef_.shape[1]
    def predict(self, X): return self.m.predict_proba(X)[:, 1]


class _Stub:
    def __init__(self): 
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=300)
    def fit(self, X, y): self.m.fit(X, y); self.booster_ = _Boost(self.m); return self
    def predict_proba(self, X): return self.m.predict_proba(X)


def _stub_tabicl(cfg, device, Xtr, ytr, Xte_list):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=300).fit(Xtr, ytr); return [m.predict_proba(X)[:, 1] for X in Xte_list]


def _load_booster(path):
    with open(path, "rb") as fh: return _Boost(pickle.load(fh))


class MockedPipeline(unittest.TestCase):
    N, NB = 160, 40

    def setUp(self):
        self.t = tmpdir(); r = np.random.default_rng(0); N, NB = self.N, self.NB
        self.cfg = load_config(overrides={"pca": {"n_components": 8}, "ensemble": {"greedy_rounds": 4, "fracs": {"start": 0.1, "stop": 0.5, "step": 0.05}},
                                          "threshold": {"n_trials": 40, "n_bins": 8}})
        iso = load_isoform_config("CYP3A4"); run = self.t / "run"; (run / "state").mkdir(parents=True)
        lg = logging.getLogger("mock"); lg.addHandler(logging.NullHandler())
        data = self.t / "data"; data.mkdir()
        c = ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"]
        pd.DataFrame([["a", "C", "TRUE"]], columns=c).to_csv(data / "CYP3A4_Positive.csv", index=False)
        pd.DataFrame([["b", "CC", "FALSE"]], columns=c).to_csv(data / "CYP3A4_Negative.csv", index=False)
        self.blind = pd.DataFrame({"Molecule_Name": [f"B{i}" for i in range(NB)], "SMILES": [f"C{'C' * i}" for i in range(NB)]})
        self.blind.to_csv(data / "TEST-BLINDED.csv", index=False)
        self.ctx = Context("CYP3A4", self.cfg, iso, load_resources(), run, data, ROOT / "assets", self.t / "cache", ROOT, lg, StateManager(run))
        y = (r.random(N) < 0.3).astype(int); self.y = y
        X = r.normal(size=(N, 20)).astype(np.float32); X[:, 0] += y
        A = (r.random((N + NB, 64)) < 0.25).astype(np.float32); S_all = tanimoto_binary(A, A).astype(np.float32)
        folds = np.arange(N) % 5; groups = [f"g{i}" for i in range(N)]
        E1 = r.normal(size=(N, 30)).astype(np.float32); E1[:, 0] += y; E2 = r.normal(size=(N, 25)).astype(np.float32); E2[:, 1] += y
        def put(sid, name, arr):
            d = run / sid; d.mkdir(exist_ok=True); p = d / name
            if name.endswith(".npy"): np.save(p, arr)
            else: arr.to_csv(p, index=False)
        put("data_clean", "clean_data.csv", pd.DataFrame({"Molecule_Name": [f"T{i}" for i in range(N)], "SMILES": ["C"] * N, "TDI_Label": y}))
        put("features", "X_train.npy", X); (run / "features" / "feature_names.json").write_text(json.dumps([f"f{i}" for i in range(20)]))
        put("similarity", "sim_train_train.npy", S_all[:N, :N]); put("scaffold_cv", "folds.npy", folds)
        put("scaffold_cv", "scaffold_groups.csv", pd.DataFrame({"group": groups}))
        put("embeddings", "chemeleon_train.npy", E1); put("embeddings", "chemprop_train.npy", E2)
        put("blind_features", "blind_ids.csv", self.blind); Xb = r.normal(size=(NB, 20)).astype(np.float32)
        put("blind_features", "X_blind.npy", Xb); put("blind_features", "S_blind_train.npy", S_all[N:, :N])
        put("blind_embeddings", "chemeleon_blind.npy", r.normal(size=(NB, 30)).astype(np.float32))
        put("blind_embeddings", "chemprop_blind.npy", r.normal(size=(NB, 25)).astype(np.float32))
        self.patches = [mock.patch.object(factory, "make_lgbm", lambda cfg, res: _Stub()),
                        mock.patch.object(factory, "tabicl_fit_predict", _stub_tabicl),
                        mock.patch.object(factory, "save_booster", lambda b, path: b.save_model(path)),
                        mock.patch.object(factory, "load_booster", _load_booster)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches: p.stop()
        cleanup(self.t)

    def A(self, sid, name): return self.ctx.run_dir / sid / name

    def run_members(self):
        for fn in (MS.oof_base, MS.oof_emb, MS.oof_tab, MS.oof_tabcp): fn(self.ctx)

    def test_oof_members_complete_and_pca_fold_scope(self):
        self.run_members()
        for tag in ("base", "emb", "tab", "tabcp"):
            o = np.load(self.A(f"oof_{tag}", f"oof_{tag}.npy")); self.assertEqual(o.shape, (self.N,)); self.assertTrue(np.isfinite(o).all())
        rep = _j(self.A("oof_tab", "pca_fold_report_tab.json"))
        folds = np.arange(self.N) % 5
        for r in rep:
            self.assertEqual(r["scope"], "fold_train_only"); self.assertEqual(r["n_samples_fit"], int((folds != r["fold"]).sum()))
        # baseline profile reproduces the original global PCA fit (documented leak)
        self.ctx.cfg["pca"]["fold_scope"] = "global_legacy"; MS.oof_tab(self.ctx)
        rep = _j(self.A("oof_tab", "pca_fold_report_tab.json"))
        self.assertTrue(all(r["n_samples_fit"] == self.N for r in rep))

    def test_ensemble_threshold_training_prediction_submission(self):
        self.run_members(); MS.ensemble_selection(self.ctx); MS.threshold_selection(self.ctx)
        w = _j(self.A("ensemble_selection", "ensemble_weights.json"))
        self.assertAlmostEqual(sum(w["average_weights"].values()), 1.0)
        meta = _j(self.A("ensemble_selection", "selection_metadata.json")); self.assertEqual(set(meta["selected_weights"]), {"base", "emb", "tab", "tabcp"})
        th = _j(self.A("threshold_selection", "threshold_report.json")); f = th["selected_fraction"]
        self.assertTrue(0 < f < 1); self.assertEqual(th["oof_positive_count_at_fraction"], int(np.floor(self.N * f + 1e-9)))
        IS.model_training(self.ctx); IS.prediction(self.ctx)
        sc = pd.read_csv(self.A("prediction", "blind_scores.csv")); self.assertEqual(len(sc), self.NB); self.assertTrue(np.isfinite(sc.ensemble_score).all())
        IS.submission(self.ctx)
        sub = pd.read_csv(self.A("submission", "submission_CYP3A4_is_TDI.csv"))
        self.assertEqual(list(sub.columns), ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"])
        k = int(np.floor(self.NB * f + 1e-9)); self.assertEqual(int(sub.CYP3A4_is_TDI.sum()), k)
        sm = _j(self.A("submission", "submission_meta.json"))
        self.assertEqual((sm["predicted_positive_count"], sm["intended_positive_count"]), (k, k))
        self.assertTrue((self.ctx.run_dir / "state" / "run_summary.json").exists())
        self.assertTrue((sub.Molecule_Name.values == self.blind.Molecule_Name.values).all())

    def test_fraction_override_and_baseline_off_by_one(self):
        self.run_members(); self.ctx.cfg["threshold"]["fraction_override"] = 0.455
        MS.ensemble_selection(self.ctx); MS.threshold_selection(self.ctx); IS.model_training(self.ctx); IS.prediction(self.ctx); IS.submission(self.ctx)
        k = lambda: int(pd.read_csv(self.A("submission", "submission_CYP3A4_is_TDI.csv")).CYP3A4_is_TDI.sum())
        self.assertEqual(k(), int(np.floor(self.NB * 0.455 + 1e-9)))
        self.ctx.cfg["selection"]["count_convention"] = "legacy_k_plus_1"; IS.submission(self.ctx)
        self.assertEqual(k(), int(self.NB * 0.455) + 1)

    def test_prediction_fails_fast(self):
        self.run_members(); MS.ensemble_selection(self.ctx); MS.threshold_selection(self.ctx); IS.model_training(self.ctx)
        np.save(self.A("blind_features", "X_blind.npy"), np.zeros((self.NB - 1, 20), dtype=np.float32))
        with self.assertRaises(ValidationError): IS.prediction(self.ctx)              # reordered/missing blind rows
        np.save(self.A("blind_features", "X_blind.npy"), np.zeros((self.NB, 19), dtype=np.float32))
        with self.assertRaises(ValidationError): IS.prediction(self.ctx)              # inconsistent matrix width
        w = _j(self.A("ensemble_selection", "ensemble_weights.json")); w["average_weights"]["emb"] += 0.5
        (self.A("ensemble_selection", "ensemble_weights.json")).write_text(json.dumps(w))
        np.save(self.A("blind_features", "X_blind.npy"), np.zeros((self.NB, 20), dtype=np.float32))
        with self.assertRaises(ValidationError): IS.prediction(self.ctx)              # bad weights

    def test_submission_detects_id_mismatch_and_missing_scores(self):
        self.run_members(); MS.ensemble_selection(self.ctx); MS.threshold_selection(self.ctx); IS.model_training(self.ctx); IS.prediction(self.ctx)
        p = self.A("prediction", "blind_scores.csv"); sc = pd.read_csv(p); good = sc.copy()
        sc.loc[3, "ensemble_score"] = np.nan; sc.to_csv(p, index=False)
        with self.assertRaises(ValidationError): IS.submission(self.ctx)
        sc = good.iloc[::-1].reset_index(drop=True); sc.to_csv(p, index=False)
        with self.assertRaises(ValidationError): IS.submission(self.ctx)

    def test_ensemble_requires_all_members(self):
        self.run_members(); (self.A("oof_tabcp", "oof_tabcp.npy")).unlink()
        with self.assertRaises(FileNotFoundError): MS.ensemble_selection(self.ctx)   # never silently blends a partial set

    def test_compare_runs_script(self):
        import shutil, sys, io, contextlib
        sys.path.insert(0, str(ROOT / "scripts")); import compare_runs
        self.run_members(); MS.ensemble_selection(self.ctx); MS.threshold_selection(self.ctx)
        IS.model_training(self.ctx); IS.prediction(self.ctx); IS.submission(self.ctx)
        other = self.t / "run2"; shutil.copytree(self.ctx.run_dir, other)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): out = compare_runs.main(self.ctx.run_dir, other)
        self.assertTrue(all(v["identical"] for v in out.values())); self.assertIn("label agreement 1.0000", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
