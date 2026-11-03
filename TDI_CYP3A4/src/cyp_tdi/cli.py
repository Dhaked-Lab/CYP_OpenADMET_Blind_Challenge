"""Command line interface."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from . import SUPPORTED_ISOFORMS
from .config import (load_config, load_isoform_config, load_resources, ConfigError, PROJECT_ROOT)
from .dependency_resolver import topo_order, validate_graph
from .executor import Context, Executor, StepFailed
from .logging_utils import get_logger
from .registry import REGISTRY, RegistryError
from .resource_manager import ResourceError, apply_cli_overrides
from .deps import DependencyError
from .state_manager import StateManager
from .discovery import InputError
from .validation import ValidationError

EXIT_USAGE, EXIT_FAIL = 2, 1


def build_parser():
    p = argparse.ArgumentParser(prog="run_pipeline.py", description="CYP3A4 TDI pipeline (corrected JacksonBurns reproduction)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list-steps", action="store_true", help="list registered steps and dependencies")
    g.add_argument("--status", action="store_true", help="show per-step status (all isoforms unless --isoform)")
    g.add_argument("--run-all", action="store_true", help="run the full pipeline (resuming valid steps)")
    g.add_argument("--step", metavar="ID", help="run one step (+ missing/stale dependencies)")
    g.add_argument("--steps", metavar="ID,ID", help="run selected steps (+ dependencies)")
    p.add_argument("--isoform", help=f"isoform to process (supported: {', '.join(SUPPORTED_ISOFORMS)})")
    p.add_argument("--force", action="store_true", help="rerun requested step(s); with --run-all rerun everything")
    p.add_argument("--dry-run", action="store_true", help="print the execution plan and exit")
    p.add_argument("--profile", choices=("corrected", "baseline"), help="corrected (default) or original-behaviour baseline")
    p.add_argument("--config", help="path to pipeline.yaml")
    p.add_argument("--resources", help="path to resources.yaml")
    p.add_argument("--data-dir"); p.add_argument("--output-dir")
    p.add_argument("--cpu-threads", type=int)
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    g2.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--gpu-device", type=int)
    return p


def _load_registry():
    from . import steps  # noqa: F401  (registers steps)
    validate_graph(REGISTRY)
    return REGISTRY


def cmd_list_steps(reg):
    print(f"{'ID':<22}{'DEPENDS ON':<70}DESCRIPTION")
    for sid in topo_order(reg, reg.ids()):
        s = reg.get(sid)
        print(f"{sid:<22}{', '.join(s.depends_on) or '-':<72}{s.description}")
    return 0


def make_context(args, isoform, root=PROJECT_ROOT):
    cfg = load_config(args.config, args.profile, root=root)
    iso_cfg = load_isoform_config(isoform, root=root)
    res = apply_cli_overrides(load_resources(args.resources, root=root), args.cpu_threads, args.gpu, args.gpu_device)
    paths = cfg["paths"]
    rel = lambda p: Path(p) if Path(p).is_absolute() else Path(root) / p
    data_dir = rel(args.data_dir or paths["data_dir"]); out_root = rel(args.output_dir or paths["output_dir"])
    run_dir = out_root / (isoform if cfg["profile"] == "corrected" else f"{isoform}.baseline")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    logger = get_logger("cyp_tdi", run_dir / "logs" / "pipeline.log")
    return Context(isoform=isoform, cfg=cfg, iso_cfg=iso_cfg, res=res, run_dir=run_dir, data_dir=data_dir,
                   assets_dir=rel(paths["assets_dir"]), cache_dir=rel(paths["cache_dir"]), root=Path(root),
                   logger=logger, state=StateManager(run_dir))


def cmd_status(reg, args, isoforms):
    for iso in isoforms:
        ctx = make_context(args, iso)
        ex = Executor(reg, ctx)
        print(f"\nisoform {iso} | profile {ctx.cfg['profile']} | run dir {ctx.run_dir}")
        print(f"{'STEP':<22}{'STATUS':<10}DETAIL")
        for sid in topo_order(reg, reg.ids()):
            st, why = ex.status_of(sid)
            print(f"{sid:<22}{st:<10}{why}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        reg = _load_registry()
        if args.list_steps:
            return cmd_list_steps(reg)
        if args.isoform and args.isoform not in SUPPORTED_ISOFORMS:
            print(f"error: unsupported isoform {args.isoform!r}; supported: {list(SUPPORTED_ISOFORMS)}", file=sys.stderr)
            return EXIT_USAGE
        if args.status:
            return cmd_status(reg, args, [args.isoform] if args.isoform else list(SUPPORTED_ISOFORMS))
        if not (args.run_all or args.step or args.steps):
            print("error: choose one of --list-steps, --status, --run-all, --step, --steps", file=sys.stderr)
            return EXIT_USAGE
        if not args.isoform:
            print("error: --isoform is required to run steps", file=sys.stderr)
            return EXIT_USAGE
        if args.run_all:
            targets = reg.ids()
        else:
            targets = [s.strip() for s in (args.steps or args.step).split(",") if s.strip()]
            if not targets:
                print("error: no step ids given", file=sys.stderr); return EXIT_USAGE
            unknown = [t for t in targets if t not in reg]
            if unknown:
                print(f"error: unknown step(s) {unknown}; use --list-steps", file=sys.stderr); return EXIT_USAGE
        if args.dry_run:
            print("plan:", " -> ".join(topo_order(reg, targets))); return 0
        ctx = make_context(args, args.isoform)
        ctx.logger.info(f"isoform={ctx.isoform} profile={ctx.cfg['profile']} data={ctx.data_dir} (device resolved per step, only for steps needing PyTorch)")
        Executor(reg, ctx).run(targets, force=args.force, force_all_in_plan=args.force and args.run_all)
        return 0
    except (ConfigError, ResourceError, RegistryError, InputError, ValidationError, DependencyError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr); return EXIT_FAIL
    except StepFailed as e:
        print(f"error: {e}", file=sys.stderr); return EXIT_FAIL
