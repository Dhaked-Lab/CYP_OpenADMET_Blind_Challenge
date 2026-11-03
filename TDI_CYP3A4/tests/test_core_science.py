import json, unittest
import numpy as np
from tests.helpers import ROOT
from cyp_tdi.core.folds import make_folds, assert_scaffold_purity, assign_none_buckets
from cyp_tdi.core.nn_features import nn_pos_features, nn_block, tanimoto_binary, n_nn_columns
from cyp_tdi.core.pca_utils import fold_pca_split, fit_pca
from cyp_tdi.core import metrics, ensemble as ens, threshold as thr
from cyp_tdi.featurization import feature_layout
from cyp_tdi.config import load_config


def _orig_nn(S_rows, pos_cols, self_pos=None, NN_TOPK=10, NN_SIM_THR=0.7):   # verbatim original common.nn_pos_features
    Sp = S_rows[:, pos_cols].copy()
    if self_pos is not None:
        pos_of = {c: j for j, c in enumerate(pos_cols)}
        for i, q in enumerate(self_pos):
            j = pos_of.get(q)
            if j is not None:
                Sp[i, j] = -1.0
    maxp = Sp.max(1)
    topk = np.partition(-Sp, min(NN_TOPK - 1, Sp.shape[1] - 1), axis=1)[:, :NN_TOPK]
    return np.column_stack([maxp, -topk.mean(1), (Sp > NN_SIM_THR).sum(1)]).astype(np.float32)


def synth_S(n, seed=0):
    r = np.random.default_rng(seed); A = (r.random((n, 64)) < 0.2).astype(np.float32)
    return tanimoto_binary(A, A).astype(np.float32)


class FoldTests(unittest.TestCase):
    def test_deterministic_pure_balanced(self):
        r = np.random.default_rng(1)
        groups = [f"s{i}" for i in r.integers(0, 120, 600)]
        f1, f2 = make_folds(groups, 5, 7), make_folds(groups, 5, 7)
        self.assertTrue((f1 == f2).all()); assert_scaffold_purity(groups, f1)
        self.assertLess(np.bincount(f1).max() - np.bincount(f1).min(), 60)
        fs = make_folds(groups, 5, 7, "stable"); assert_scaffold_purity(groups, fs)
        self.assertFalse((make_folds(groups, 5, 8) == f1).all())

    def test_purity_check_detects_leak(self):
        with self.assertRaises(AssertionError):
            assert_scaffold_purity(["a", "a", "b"], np.array([0, 1, 1]))

    def test_none_buckets(self):
        k = assign_none_buckets([""] * 85 + ["x"], 40)
        self.assertEqual((k[0], k[40], k[80], k[85]), ("__NONE__0", "__NONE__1", "__NONE__2", "x"))


class NNTests(unittest.TestCase):
    def test_matches_original_verbatim(self):
        S = synth_S(120); pos = np.arange(0, 120, 3); trc = np.arange(0, 100)
        a = nn_pos_features(S[trc], pos, self_pos=trc); b = _orig_nn(S[trc], pos, self_pos=trc)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(nn_block(S[trc], pos, {"mode": "baseline", "topk": 10, "sim_threshold": 0.7}, self_pos=trc), b))

    def test_self_excluded_and_validation_uses_only_train_positives(self):
        S = synth_S(80); pos = np.where(np.arange(80) % 4 == 0)[0]
        f_self = nn_pos_features(S[pos], pos, self_pos=pos); f_no = nn_pos_features(S[pos], pos)
        self.assertTrue((f_no[:, 0] > 0.999).all());         self.assertTrue(((f_no[:, 2] - f_self[:, 2]) == 1).all())

    def test_features_do_not_depend_on_validation_labels(self):
        n = 100; S = synth_S(n); folds = np.arange(n) % 5; y = (np.arange(n) % 3 == 0).astype(int)
        trc, vac = np.where(folds != 0)[0], np.where(folds == 0)[0]
        f1 = nn_pos_features(S[vac], trc[y[trc] == 1])
        y2 = y.copy(); y2[vac] = 1 - y2[vac]
        f2 = nn_pos_features(S[vac], trc[y2[trc] == 1])
        self.assertTrue(np.array_equal(f1, f2))

    def test_empty_reference_fails_fast(self):
        with self.assertRaises(ValueError): nn_pos_features(synth_S(10), np.array([], dtype=int))

    def test_fewer_than_topk_positives_safe(self):
        f = nn_pos_features(synth_S(30)[:5], np.array([20, 21, 22]))
        self.assertTrue(np.isfinite(f).all())

    def test_optional_modes(self):
        S = synth_S(60); pos = np.arange(0, 60, 2); groups = np.array([i // 6 for i in range(60)]); trc = np.arange(60)
        base = nn_block(S, pos, {"mode": "baseline", "topk": 10, "sim_threshold": 0.7}, self_pos=trc)
        ex = nn_block(S, pos, {"mode": "exclude_same_scaffold", "topk": 10, "sim_threshold": 0.7}, self_pos=trc, groups_rows=groups, groups_all=groups)
        self.assertTrue((ex[:, 0] <= base[:, 0] + 1e-6).all()); self.assertFalse(np.array_equal(ex, base))
        dc = nn_block(S, pos, {"mode": "drop_count", "topk": 10, "sim_threshold": 0.7}, self_pos=trc)
        self.assertEqual(dc.shape[1], 2); self.assertEqual(n_nn_columns({"mode": "drop_count"}), 2)
        self.assertTrue(np.array_equal(dc, base[:, :2]))


class PCATests(unittest.TestCase):
    cfg = {"n_components": 6, "whiten": True}

    def test_fold_fit_uses_only_training_rows(self):
        r = np.random.default_rng(0); E = r.normal(size=(120, 20)); trn = np.arange(0, 96); val = np.arange(96, 120)
        _, _, pf = fold_pca_split(E, trn, val, self.cfg, 0, "fold_train_only")
        ref = fit_pca(E[trn], 6, True, 0)
        self.assertTrue(np.allclose(pf.components_, ref.components_)); self.assertEqual(pf.n_samples_, 96)
        E2 = E.copy(); E2[val] = r.normal(size=(24, 20)) * 50 + 9          # wreck validation rows
        _, _, pf2 = fold_pca_split(E2, trn, val, self.cfg, 0, "fold_train_only")
        self.assertTrue(np.allclose(pf.components_, pf2.components_))      # fit unaffected by validation data
        _, _, pg = fold_pca_split(E, trn, val, self.cfg, 0, "global_legacy")
        self.assertEqual(pg.n_samples_, 120)                              # legacy: validation rows leak into fit
        _, _, pg2 = fold_pca_split(E2, trn, val, self.cfg, 0, "global_legacy")
        self.assertFalse(np.allclose(pg.components_, pg2.components_))

    def test_dimensions_dtype_and_unknown_scope(self):
        E = np.random.default_rng(1).normal(size=(50, 12))
        a, b, _ = fold_pca_split(E, np.arange(40), np.arange(40, 50), self.cfg, 0, "fold_train_only")
        self.assertEqual((a.shape, b.shape, a.dtype), ((40, 6), (10, 6), np.float32))
        with self.assertRaises(ValueError): fold_pca_split(E, np.arange(5), np.arange(5), self.cfg, 0, "weird")

    def test_global_cache_reused(self):
        E = np.random.default_rng(2).normal(size=(50, 12)); cache = {}
        _, _, p1 = fold_pca_split(E, np.arange(40), np.arange(40, 50), self.cfg, 0, "global_legacy", cache)
        _, _, p2 = fold_pca_split(E, np.arange(10, 50), np.arange(10), self.cfg, 0, "global_legacy", cache)
        self.assertIs(p1, p2)


class RankFractionTests(unittest.TestCase):
    def test_exact_count_vs_legacy_off_by_one(self):    # JB-04
        s = np.random.default_rng(0).normal(size=750)
        self.assertEqual(metrics.k_for_fraction(750, 0.455, "exact_k"), 341)
        self.assertEqual(int(metrics.positive_mask(s, 0.455, "exact_k").sum()), 341)
        self.assertEqual(int(metrics.positive_mask(s, 0.455, "legacy_k_plus_1").sum()), 342)
        self.assertEqual(int(metrics.positive_mask(s, 0.435, "legacy_k_plus_1").sum()), 327)   # audit's CYP2D6-folder count
        self.assertEqual(int(metrics.positive_mask(s, 0.435, "exact_k").sum()), 326)

    def test_float_edge_floor(self):
        self.assertEqual(metrics.k_for_fraction(100, 0.29, "exact_k"), 29)

    def test_ties_deterministic_and_lowest_index_first(self):
        s = np.array([1.0, 2.0, 2.0, 2.0, 0.0, 2.0]); m = metrics.select_top_k(s, 2)
        self.assertEqual(np.where(m)[0].tolist(), [1, 2])
        self.assertTrue((metrics.select_top_k(s, 3) == metrics.select_top_k(s, 3)).all())
        self.assertEqual(int(metrics.select_top_k(np.zeros(10), 4).sum()), 4)

    def test_invalid_inputs(self):
        for bad in (np.array([1, np.nan, 3.0]), np.array([1, np.inf, 3.0])):
            with self.assertRaises(ValueError): metrics.positive_mask(bad, 0.5)
        with self.assertRaises(ValueError): metrics.select_top_k(np.arange(5.0), 0)
        with self.assertRaises(ValueError): metrics.select_top_k(np.arange(5.0), 5)

    def test_permutation_invariance_except_ties(self):
        s = np.random.default_rng(3).normal(size=200); p = np.random.default_rng(4).permutation(200)
        a = metrics.select_top_k(s, 50); b = metrics.select_top_k(s[p], 50)
        self.assertTrue((a[p] == b).all())

    def test_best_by_frac_runs(self):
        r = np.random.default_rng(5); y = (r.random(300) < 0.25).astype(int); s = y + r.normal(size=300)
        m, f = metrics.best_by_frac(s, y, np.round(np.arange(0.1, 0.5, 0.05), 2), "exact_k")
        self.assertGreater(m, 0.2)


class EnsembleTests(unittest.TestCase):
    def data(self, seed=0, n=250):
        r = np.random.default_rng(seed); y = (r.random(n) < 0.25).astype(int); folds = np.arange(n) % 5
        mem = {k: y * a + r.normal(size=n) for k, a in (("base", 1.0), ("emb", 0.7), ("tab", 0.4), ("tabcp", 0.9))}
        return mem, y, folds

    def test_weights_valid_deterministic(self):
        mem, y, folds = self.data(); F = np.round(np.arange(0.1, 0.5, 0.05), 2)
        a = ens.nested_ensemble(mem, y, folds, 5, F, "exact_k", 12); b = ens.nested_ensemble(mem, y, folds, 5, F, "exact_k", 12)
        self.assertEqual(a[2], b[2]); self.assertAlmostEqual(sum(a[2].values()), 1.0)
        for w in a[1]: self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_tie_break_prefers_earliest_member(self):
        r = np.random.default_rng(1); y = (r.random(100) < 0.3).astype(int); z = y + r.normal(size=100)
        zs = {"base": z.copy(), "emb": z.copy()}                  # identical members => exact tie each round
        w = ens.greedy_w(zs, y, np.round(np.arange(0.1, 0.5, 0.05), 2), "exact_k", 4)
        self.assertEqual(w["base"], 1.0)

    def test_outer_fold_rows_do_not_choose_their_own_weights(self):
        mem, y, folds = self.data(); F = np.round(np.arange(0.1, 0.5, 0.05), 2)
        pooled, hist, _ = ens.nested_ensemble(mem, y, folds, 5, F, "exact_k", 6)
        zs = {k: metrics.zscore(v) for k, v in mem.items()}
        for f in range(5):                                   # independent recomputation from rows outside fold f only
            trn = folds != f
            self.assertEqual(hist[f], ens.greedy_w({k: v[trn] for k, v in zs.items()}, y[trn], F, "exact_k", 6))
            va = folds == f
            self.assertTrue(np.allclose(pooled[va], sum(hist[f][k] * zs[k][va] for k in zs), atol=1e-5))
        y2 = y.copy(); y2[folds == 2] = 1 - y2[folds == 2]  # flipping outer-fold labels must not change that fold's weights
        _, h2, _ = ens.nested_ensemble(mem, y2, folds, 5, F, "exact_k", 6)
        self.assertEqual(hist[2], h2[2]); self.assertNotEqual(hist[0], h2[0]) if hist[0] != h2[0] else None


class ThresholdTests(unittest.TestCase):
    def test_rescale_and_determinism(self):
        r = np.linspace(0.6, 0.02, 24); out = thr.rescale_to_pi(r, 0.3); self.assertAlmostEqual(float(out.mean()), 0.3, 3)
        rng = np.random.default_rng(0); y = (rng.random(400) < 0.2).astype(int); p = y + rng.normal(size=400)
        F = np.round(np.arange(0.1, 0.5, 0.05), 2)
        a = thr.select_fraction(p, y, F, 7, [0.85, 1.0, 1.2], 12, 60, 0.005, 3)
        b = thr.select_fraction(p, y, F, 7, [0.85, 1.0, 1.2], 12, 60, 0.005, 3)
        self.assertEqual(a, b); self.assertIn(a["selected_fraction"], list(map(float, F)) + [float(np.median(a["plateau"]))])


class FeatureLayoutTests(unittest.TestCase):
    def test_dimensions_and_order_against_jb_cache(self):
        with open(ROOT / "assets" / "reference" / "jb_descriptor_names.json") as fh: ref = json.load(fh)
        lay = feature_layout(load_config()["features"], len(ref))
        self.assertEqual(len(ref), 217); self.assertEqual(lay["total"], 3456)
        self.assertEqual((lay["desc"], lay["m2"], lay["m3"], lay["mk"]), ((0, 217), (217, 2265), (2265, 3289), (3289, 3456)))
        from cyp_tdi.featurization import feature_names
        names = feature_names(load_config()["features"], ref)
        self.assertEqual((len(names), names[217], names[2265], names[3289], names[-1]), (3456, "m2_0", "m3_0", "mk_0", "mk_166"))


if __name__ == "__main__":
    unittest.main()
