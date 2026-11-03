import subprocess, sys, unittest, logging
from tests.helpers import ROOT, tmpdir, cleanup
from cyp_tdi.registry import Registry, RegistryError, step, REGISTRY
from cyp_tdi.dependency_resolver import topo_order, validate_graph, CycleError, downstream
from cyp_tdi.executor import Executor, Context, StepFailed
from cyp_tdi.state_manager import StateManager
from cyp_tdi.config import load_config


class RegistryTests(unittest.TestCase):
    def test_real_registry(self):
        from cyp_tdi import steps  # noqa
        validate_graph(REGISTRY)
        ids = REGISTRY.ids(); self.assertEqual(len(ids), len(set(ids)))
        order = topo_order(REGISTRY, ids)
        pos = {s: i for i, s in enumerate(order)}
        for s in REGISTRY.specs():
            self.assertTrue(s.outputs and s.description and s.version)
            for d in s.depends_on:
                self.assertLess(pos[d], pos[s.id])
        for need in ("data_audit", "features", "scaffold_cv", "embeddings", "ensemble_selection", "threshold_selection",
                     "prediction", "submission", "model_training"):
            self.assertIn(need, ids)
        self.assertEqual(topo_order(REGISTRY, ["oof_tab"]), ["data_audit", "data_clean", "scaffold_cv", "embeddings", "oof_tab"])
        self.assertIn("submission", downstream(REGISTRY, "prediction"))

    def test_cycle_and_unknown(self):
        r = Registry(); f = lambda c: None
        step("a", "A", depends_on=("c",), outputs=("o",), registry=r)(f)
        step("b", "B", depends_on=("a",), outputs=("o",), registry=r)(f)
        step("c", "C", depends_on=("b",), outputs=("o",), registry=r)(f)
        with self.assertRaises(CycleError): validate_graph(r)
        r2 = Registry(); step("a", "A", depends_on=("zzz",), outputs=("o",), registry=r2)(f)
        with self.assertRaises(RegistryError): validate_graph(r2)
        with self.assertRaises(RegistryError): step("a", "dup", outputs=("o",), registry=r)(f)
        with self.assertRaises(RegistryError): r.get("nope")
        with self.assertRaises(RegistryError): step("noout", "x", registry=Registry())(f)


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), *args], capture_output=True, text=True, cwd=ROOT)

    def test_list_steps(self):
        r = self.run_cli("--list-steps"); self.assertEqual(r.returncode, 0); self.assertIn("ensemble_selection", r.stdout)

    def test_dry_run_plans(self):
        r = self.run_cli("--isoform", "CYP3A4", "--steps", "data_audit,features", "--dry-run")
        self.assertEqual(r.returncode, 0); self.assertIn("data_audit -> data_clean -> features", r.stdout)
        r = self.run_cli("--isoform", "CYP3A4", "--run-all", "--force", "--dry-run"); self.assertEqual(r.returncode, 0)
        self.assertIn("submission", r.stdout)

    def test_rejections(self):
        self.assertEqual(self.run_cli("--isoform", "CYP2D6", "--run-all").returncode, 2)
        self.assertEqual(self.run_cli("--isoform", "CYP3A4", "--step", "nonexistent").returncode, 2)
        self.assertEqual(self.run_cli("--run-all").returncode, 2)                 # missing isoform
        self.assertEqual(self.run_cli("--isoform", "CYP3A4").returncode, 2)       # nothing to do
        self.assertEqual(self.run_cli("--run-all", "--step", "features", "--isoform", "CYP3A4").returncode, 2)  # exclusive
        self.assertEqual(self.run_cli("--status", "--isoform", "CYP2D6").returncode, 2)

    def test_status_in_temp_output(self):
        d = tmpdir()
        try:
            r = self.run_cli("--status", "--isoform", "CYP3A4", "--output-dir", str(d))
            self.assertEqual(r.returncode, 0); self.assertIn("missing", r.stdout)
            self.assertTrue((d / "CYP3A4" / "state").exists())
            r = self.run_cli("--status", "--output-dir", str(d), "--profile", "baseline")
            self.assertIn("baseline", r.stdout)
        finally:
            cleanup(d)


def mini(tmp, fail_b=False):
    """Mini pipeline A -> B -> C with real Executor/StateManager."""
    reg = Registry(); calls = []
    inp = tmp / "input.txt"
    if not inp.exists(): inp.write_text("v1")

    def A(ctx): calls.append("A"); (ctx.step_dir("A") / "a.txt").write_text(inp.read_text())
    def B(ctx):
        calls.append("B")
        if fail_b: raise RuntimeError("boom")
        (ctx.step_dir("B") / "b.txt").write_text(ctx.artifact("A", "a.txt").read_text() + str(ctx.cfg["seeds"]["lightgbm"]))
    def Cc(ctx): calls.append("C"); (ctx.step_dir("C") / "c.txt").write_text(ctx.artifact("B", "b.txt").read_text())
    step("A", "a", outputs=("a.txt",), inputs_fn=lambda c: [inp], registry=reg)(A)
    step("B", "b", depends_on=("A",), outputs=("b.txt",), config_keys=("seeds.lightgbm",), registry=reg)(B)
    step("C", "c", depends_on=("B",), outputs=("c.txt",), registry=reg)(Cc)
    cfg = load_config(); run = tmp / "run"; run.mkdir(exist_ok=True)
    lg = logging.getLogger("mini"); lg.addHandler(logging.NullHandler())
    ctx = Context("CYP3A4", cfg, {}, {}, run, tmp, tmp, tmp, tmp, lg, StateManager(run))
    return reg, ctx, calls, inp


class StateTests(unittest.TestCase):
    def setUp(self): self.t = tmpdir()
    def tearDown(self): cleanup(self.t)

    def test_resume_and_invalidation(self):
        reg, ctx, calls, inp = mini(self.t)
        Executor(reg, ctx).run(["C"]); self.assertEqual(calls, ["A", "B", "C"])
        calls.clear(); Executor(reg, ctx).run(["C"]); self.assertEqual(calls, [])            # safe resume
        inp.write_text("v2"); Executor(reg, ctx).run(["C"]); self.assertEqual(calls, ["A", "B", "C"])  # input change
        calls.clear(); ctx.cfg["seeds"]["lightgbm"] = 99
        Executor(reg, ctx).run(["C"]); self.assertEqual(calls, ["B", "C"])                   # config change: A preserved
        calls.clear(); Executor(reg, ctx).run(["B"], force=True); self.assertEqual(calls, ["B"])
        calls.clear(); Executor(reg, ctx).run(["C"]); self.assertEqual(calls, [])             # B output identical -> C valid
        calls.clear(); Executor(reg, ctx).run(["C"], force=True, force_all_in_plan=True); self.assertEqual(calls, ["A", "B", "C"])

    def test_tamper_and_corrupt_state(self):
        reg, ctx, calls, inp = mini(self.t); Executor(reg, ctx).run(["C"]); calls.clear()
        (ctx.run_dir / "A" / "a.txt").write_text("tampered")
        Executor(reg, ctx).run(["C"]); self.assertEqual(calls[0], "A")
        calls.clear(); (ctx.run_dir / "state" / "B.json").write_text("{not json")
        Executor(reg, ctx).run(["C"]); self.assertEqual(calls, ["B"])        # only the corrupt-state step reruns
        self.assertTrue(list((ctx.run_dir / "state").glob("B.corrupt")))

    def test_failure_keeps_upstream(self):
        reg, ctx, calls, inp = mini(self.t, fail_b=True)
        with self.assertRaises(StepFailed): Executor(reg, ctx).run(["C"])
        self.assertEqual(calls, ["A", "B"])
        self.assertEqual(Executor(reg, ctx).status_of("A")[0], "complete")
        self.assertEqual(Executor(reg, ctx).status_of("B")[0], "failed")
        reg2, ctx2, calls2, _ = mini(self.t, fail_b=False)
        Executor(reg2, ctx2).run(["C"]); self.assertEqual(calls2, ["B", "C"])   # A not rerun

    def test_missing_input_blocks(self):
        reg, ctx, calls, inp = mini(self.t); inp.unlink()
        with self.assertRaises((StepFailed, FileNotFoundError)): Executor(reg, ctx).run(["A"])

    def test_output_dirs_from_step_ids(self):
        reg, ctx, calls, inp = mini(self.t); Executor(reg, ctx).run(["C"])
        for s in ("A", "B", "C", "state"): self.assertTrue((ctx.run_dir / s).is_dir())


if __name__ == "__main__":
    unittest.main()
