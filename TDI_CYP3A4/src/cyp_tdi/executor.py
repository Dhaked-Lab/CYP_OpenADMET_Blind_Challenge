"""Plan + run steps with dependency resolution, fingerprints, safe resume and forced reruns."""
from __future__ import annotations
import time, traceback
from dataclasses import dataclass
from pathlib import Path
from .config import fingerprint, config_subset
from .dependency_resolver import topo_order, validate_graph
from .registry import Registry
from .state_manager import StateManager, COMPLETE, FAILED, RUNNING


class StepFailed(RuntimeError):
    pass


@dataclass
class Context:
    isoform: str
    cfg: dict
    iso_cfg: dict
    res: dict
    run_dir: Path
    data_dir: Path
    assets_dir: Path
    cache_dir: Path
    root: Path
    logger: object
    state: StateManager
    device: str = "cpu"

    def step_dir(self, sid) -> Path:
        d = self.run_dir / sid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact(self, sid, name) -> Path:
        p = self.run_dir / sid / name
        if not p.exists():
            raise FileNotFoundError(f"upstream artifact missing: {p} (run step {sid!r})")
        return p


class Executor:
    def __init__(self, reg: Registry, ctx: Context):
        validate_graph(reg)
        self.reg, self.ctx = reg, ctx
        self.log = ctx.logger

    # -- fingerprints -----------------------------------------------------------
    def expected_fingerprint(self, sid) -> str:
        spec = self.reg.get(sid)
        c = self.ctx
        deps = {}
        for d in spec.depends_on:
            ds = c.state.load(d)
            if not ds or ds.get("status") != COMPLETE:
                raise StepFailed(f"dependency {d!r} of {sid!r} is not complete")
            deps[d] = {n: m["sha256"] for n, m in sorted(ds["outputs"].items())}
        ins = {}
        if spec.inputs_fn:
            for p in spec.inputs_fn(c):
                p = Path(p)
                if not p.exists():
                    raise FileNotFoundError(f"input file for step {sid!r} not found: {p}")
                ins[str(p.name)] = c.state.file_hash(p)
        return fingerprint({"step": sid, "version": spec.version, "isoform": c.isoform,
                            "profile": c.cfg["profile"], "config": config_subset(c.cfg, spec.config_keys),
                            "inputs": ins, "deps": deps})

    def status_of(self, sid) -> tuple[str, str]:
        """Return (status, reason). status in complete|stale|failed|missing|blocked."""
        c = self.ctx
        st = c.state.load(sid)
        if st is None:
            return "missing", "no state"
        if st["status"] == FAILED:
            return "failed", st.get("error", "")[:200]
        if st["status"] == RUNNING:
            return "stale", "previous run did not finish"
        spec = self.reg.get(sid)
        if not c.state.outputs_intact(c.run_dir / sid, st.get("outputs")):
            return "stale", "outputs missing or modified"
        try:
            exp = self.expected_fingerprint(sid)
        except StepFailed as e:
            return "stale", str(e)
        except FileNotFoundError as e:
            return "blocked", str(e)
        if exp != st.get("fingerprint"):
            return "stale", "inputs/config/dependencies/implementation changed"
        return COMPLETE, ""

    # -- planning ---------------------------------------------------------------
    def plan(self, targets, force=False, force_all=False):
        order = topo_order(self.reg, targets)
        return order

    def run(self, targets, force: bool = False, force_all_in_plan: bool = False):
        order = self.plan(targets)
        forced = set(order) if force_all_in_plan else (set(targets) if force else set())
        self.log.info("plan: " + " -> ".join(order))
        results = {}
        for sid in order:
            spec = self.reg.get(sid)
            if sid not in forced:
                status, why = self.status_of(sid)
                if status == COMPLETE:
                    self.log.info(f"[{sid}] up to date - skipped")
                    results[sid] = "cached"
                    continue
                self.log.info(f"[{sid}] {status}" + (f" ({why})" if why else "") + " - running")
            else:
                self.log.info(f"[{sid}] forced - running")
            self._execute(spec)
            results[sid] = "ran"
        return results

    def _execute(self, spec):
        c = self.ctx
        sdir = c.step_dir(spec.id)
        for name in spec.outputs:                       # never reuse stale outputs
            p = sdir / name
            if p.exists():
                p.unlink()
        try:
            fp = self.expected_fingerprint(spec.id)
        except FileNotFoundError as e:
            raise StepFailed(str(e))
        started = c.state.now(); t0 = time.time(); phase = "setup"
        c.state.save(spec.id, status=RUNNING, started=started, fingerprint=fp, version=spec.version,
                     isoform=c.isoform, profile=c.cfg["profile"])
        try:
            from .seeds import seed_everything
            from .resource_manager import resolve_device
            seed_everything(c.cfg, frameworks=spec.requires)       # torch only for steps that need it
            if "torch" in spec.requires:
                c.device = resolve_device(c.res)
            phase = "run"
            spec.run(c)
            phase = "validate"
            if spec.validate_fn:
                spec.validate_fn(c)
            outs = c.state.hash_outputs(sdir, spec.outputs)
        except BaseException as e:
            c.state.save(spec.id, status=FAILED, started=started, failed=c.state.now(), fingerprint=fp,
                         version=spec.version, isoform=c.isoform, profile=c.cfg["profile"],
                         phase=phase, step_function_invoked=(phase != "setup"), error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
            self.log.error(f"[{spec.id}] FAILED during {phase}: {type(e).__name__}: {e}")
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            raise StepFailed(f"step {spec.id!r} failed: {e}") from e
        c.state.save(spec.id, status=COMPLETE, started=started, completed=c.state.now(),
                     seconds=round(time.time() - t0, 2), fingerprint=fp, version=spec.version,
                     isoform=c.isoform, profile=c.cfg["profile"], outputs=outs,
                     output_dir=str(sdir))
        self.log.info(f"[{spec.id}] complete in {time.time() - t0:.1f}s")
