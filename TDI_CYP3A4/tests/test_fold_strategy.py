"""cv.fold_strategy: original_all_isoforms ([JB] folds over the whole all-isoform training table, then CYP3A4 rows) vs cyp_only.
No model is trained here; the RDKit test only computes scaffold keys and fold ids."""
import copy, hashlib, logging, unittest
import numpy as np, pandas as pd
from tests.helpers import ROOT, tmpdir, cleanup, have
from cyp_tdi.config import load_config, load_isoform_config, validate_config, ConfigError, config_subset, fingerprint
from cyp_tdi.core.folds import make_folds, universe_folds, assert_scaffold_purity
from cyp_tdi.registry import Registry, step, REGISTRY
from cyp_tdi.dependency_resolver import downstream
from cyp_tdi.executor import Executor, Context
from cyp_tdi.state_manager import StateManager
from cyp_tdi.steps import data_steps as DS

UNIVERSE_SHA = "3f52513c7a233fd9d186081c3642166e343ca390f60c06a21f52a3cfa95b9eff"


def _universe():
    """Synthetic all-isoform table: 60 rows in 25 scaffold groups; the 'CYP3A4' subset is every row not divisible by 3."""
    ids = [f"m{i}" for i in range(60)]
    groups = [f"g{(i * 7) % 25}" for i in range(60)]
    target = [s for i, s in enumerate(ids) if i % 3]
    return ids, groups, target


class UniverseFoldTests(unittest.TestCase):
    def test_folds_built_over_universe_then_subset(self):
        ids, groups, target = _universe()
        f, g, fu = universe_folds(ids, groups, target, k=5, seed=7)
        full = make_folds(groups, 5, 7)
        pos = {s: i for i, s in enumerate(ids)}
        self.assertTrue((fu == full).all())                                             # built over ALL universe rows
        self.assertEqual(f.tolist(), [int(full[pos[s]]) for s in target])                # ... then the target rows selected
        self.assertEqual(g, [groups[pos[s]] for s in target])
        assert_scaffold_purity(g, f)
        subset_only = make_folds([groups[pos[s]] for s in target], 5, 7)                # the cyp_only construction
        self.assertFalse((subset_only == f).all())                                       # the two strategies really differ

    def test_row_alignment_independent_of_target_order(self):
        ids, groups, target = _universe()
        f, _, _ = universe_folds(ids, groups, target)
        perm = np.random.default_rng(0).permutation(len(target))
        fp, _, _ = universe_folds(ids, groups, [target[i] for i in perm])
        self.assertEqual(fp.tolist(), f[perm].tolist())                                  # each molecule keeps its fold

    def test_missing_or_duplicate_ids_fail(self):
        ids, groups, target = _universe()
        with self.assertRaises(ValueError): universe_folds(ids, groups, target + ["not_in_universe"])
        with self.assertRaises(ValueError): universe_folds(ids + ["m0"], groups + ["gX"], target)
        with self.assertRaises(ValueError): universe_folds(ids, groups, target + [target[0]])
        with self.assertRaises(ValueError): universe_folds(ids, groups[:-1], target)


class FoldStrategyConfigTests(unittest.TestCase):
    def test_reproduction_uses_original_strategy_and_cyp_only_is_available(self):
        for prof in ("corrected", "baseline"):
            cv = load_config(profile=prof)["cv"]
            self.assertEqual(cv["fold_strategy"], "original_all_isoforms")
            self.assertEqual((cv["fold_universe"]["path"], cv["fold_universe"]["sha256"]),
                             ("assets/reference/jb_fold_universe_smiles.csv", UNIVERSE_SHA))
        self.assertEqual(load_config(overrides={"cv": {"fold_strategy": "cyp_only"}})["cv"]["fold_strategy"], "cyp_only")

    def test_invalid_strategy_or_missing_universe_rejected(self):
        c = load_config()
        bad = copy.deepcopy(c); bad["cv"]["fold_strategy"] = "bogus"
        with self.assertRaises(ConfigError): validate_config(bad)
        bad = copy.deepcopy(c); del bad["cv"]["fold_universe"]
        with self.assertRaises(ConfigError): validate_config(bad)
        ok = copy.deepcopy(bad); ok["cv"]["fold_strategy"] = "cyp_only"; validate_config(ok)    # cyp_only needs no universe

    def test_universe_asset_integrity(self):
        p = ROOT / "assets" / "reference" / "jb_fold_universe_smiles.csv"
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), UNIVERSE_SHA)
        u = pd.read_csv(p, keep_default_na=False)["SMILES"].tolist()
        self.assertEqual((len(u), len(set(u))), (6145, 6145))
        from cyp_tdi.discovery import discover_inputs, load_training_frames
        if not (ROOT / "data" / "CYP3A4_Positive.csv").exists():
            self.skipTest("BLOCKED: training data not present in data/ (see data/README.md)")
        iso = load_isoform_config("CYP3A4"); us = set(u)
        smi = [s for _, _, df, *_ in load_training_frames(discover_inputs(ROOT / "data", iso), iso) for s in df[iso["smiles_column"]]]
        self.assertEqual(len(smi), 3584)
        self.assertTrue(all(s in us for s in smi))                                       # every CYP3A4 row is in the universe

    def test_universe_sha_mismatch_blocks(self):
        d = tmpdir()
        try:
            (d / "u.csv").write_text("SMILES\nC\n")
            cv = {**load_config()["cv"], "fold_universe": {"path": str(d / "u.csv"), "sha256": UNIVERSE_SHA}}
            with self.assertRaises(DS.ValidationError): DS.load_fold_universe(cv, ROOT)
            cv["fold_universe"] = {"path": str(d / "missing.csv"), "sha256": UNIVERSE_SHA}
            with self.assertRaises(DS.ValidationError): DS.load_fold_universe(cv, ROOT)
        finally:
            cleanup(d)


class FoldInvalidationTests(unittest.TestCase):
    def test_strategy_changes_fingerprint_of_folds_and_every_oof_member(self):
        from cyp_tdi import steps  # noqa: F401  (registers steps)
        a, b = load_config(), load_config(overrides={"cv": {"fold_strategy": "cyp_only"}})
        for sid in ("scaffold_cv", "oof_base", "oof_emb", "oof_tab", "oof_tabcp"):
            keys = REGISTRY.get(sid).config_keys
            self.assertNotEqual(fingerprint(config_subset(a, keys)), fingerprint(config_subset(b, keys)), sid)
        self.assertIs(REGISTRY.get("scaffold_cv").inputs_fn, DS.fold_universe_inputs)
        down = downstream(REGISTRY, "scaffold_cv")                                       # reached through dependency output hashes
        for sid in ("oof_base", "oof_emb", "oof_tab", "oof_tabcp", "ensemble_selection", "threshold_selection",
                    "model_training", "prediction", "submission"):
            self.assertIn(sid, down)

    def test_executor_reruns_folds_and_dependents_on_strategy_or_universe_change(self):
        """Real Executor/StateManager and the real fold_universe_inputs on a two-step stand-in (folds -> member)."""
        t = tmpdir()
        try:
            (t / "u.csv").write_text("SMILES\nC\nCC\n")
            reg, calls = Registry(), []
            def F(ctx):
                calls.append("F"); (ctx.step_dir("F") / "folds.txt").write_text(ctx.cfg["cv"]["fold_strategy"] + (t / "u.csv").read_text())
            def G(ctx):
                calls.append("G"); (ctx.step_dir("G") / "oof.txt").write_text(ctx.artifact("F", "folds.txt").read_text())
            step("F", "folds", outputs=("folds.txt",), config_keys=("cv", "seeds.folds"), inputs_fn=DS.fold_universe_inputs, registry=reg)(F)
            step("G", "oof", depends_on=("F",), outputs=("oof.txt",), config_keys=("seeds.lightgbm",), registry=reg)(G)
            cfg = load_config(overrides={"cv": {"fold_universe": {"path": str(t / "u.csv")}}})
            run = t / "run"; run.mkdir(); lg = logging.getLogger("folds"); lg.addHandler(logging.NullHandler())
            ctx = Context("CYP3A4", cfg, {}, {}, run, t, t, t, t, lg, StateManager(run))
            Executor(reg, ctx).run(["G"]); self.assertEqual(calls, ["F", "G"])
            calls.clear(); Executor(reg, ctx).run(["G"]); self.assertEqual(calls, [])              # resume: nothing stale
            calls.clear(); (t / "u.csv").write_text("SMILES\nC\nCCC\n")                           # universe file edited
            Executor(reg, ctx).run(["G"]); self.assertEqual(calls, ["F", "G"])
            calls.clear(); ctx.cfg["cv"]["fold_strategy"] = "cyp_only"                             # strategy switched
            self.assertEqual(Executor(reg, ctx).status_of("F")[0], "stale")
            Executor(reg, ctx).run(["G"]); self.assertEqual(calls, ["F", "G"])                     # old member output never reused
            calls.clear(); (t / "u.csv").write_text("SMILES\nO\n")                                 # cyp_only ignores the universe
            Executor(reg, ctx).run(["G"]); self.assertEqual(calls, [])
        finally:
            cleanup(t)


@unittest.skipUnless(have("rdkit"), "BLOCKED: rdkit not installed")
@unittest.skipUnless((ROOT / "data" / "CYP3A4_Positive.csv").exists(), "BLOCKED: training data not present in data/")
class RealDataFoldTests(unittest.TestCase):
    """Fold ids only (scaffold keys + make_folds) on the CYP3A4 training SMILES; no model, no pipeline step."""
    def test_original_and_cyp_only_fold_sizes(self):
        from cyp_tdi.discovery import discover_inputs, load_training_frames
        iso = load_isoform_config("CYP3A4")
        smi = [s for _, _, df, *_ in load_training_frames(discover_inputs(ROOT / "data", iso), iso) for s in df[iso["smiles_column"]]]
        cfg = load_config(); seed = cfg["seeds"]["folds"]
        f, g, info = DS.compute_folds(smi, cfg["cv"], seed, ROOT)
        self.assertEqual(np.bincount(f).tolist(), [687, 727, 724, 721, 725])            # = rows per fold in the original OOF files
        self.assertEqual((info["n_universe"], info["universe_fold_sizes"]), (6145, [1229] * 5))
        assert_scaffold_purity(g, f)
        fc, gc, info_c = DS.compute_folds(smi, {**cfg["cv"], "fold_strategy": "cyp_only"}, seed, ROOT)
        self.assertEqual(np.bincount(fc).tolist(), [717, 717, 717, 717, 716]); self.assertEqual(info_c, {"fold_strategy": "cyp_only"})
        self.assertEqual(g, gc)                                                           # same scaffold groups, different partition
        self.assertFalse((pd.crosstab(f, fc).gt(0).sum(axis=1) == 1).all())


if __name__ == "__main__":
    unittest.main()
