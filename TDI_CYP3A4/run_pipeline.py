#!/usr/bin/env python3
"""Entry point. (1) torch is imported FIRST when steps will run, before numpy/scikit-learn/rdkit do native work (Windows DLL
shadowing, see src/cyp_tdi/deps.py); (2) thread limits from config/resources.yaml (or --cpu-threads) are exported before numpy loads."""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cyp_tdi.deps import wants_torch_preload, preload_torch   # imports only importlib: safe before any numeric library


def _early_threads(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resources"); pre.add_argument("--cpu-threads", type=int)
    a, _ = pre.parse_known_args(argv)
    try:
        import yaml
        rp = Path(a.resources) if a.resources else ROOT / "config" / "resources.yaml"
        res = yaml.safe_load(open(rp)) if rp.exists() else {}
    except Exception:
        res = {}
    res = res or {}
    if a.cpu_threads is not None:
        res.setdefault("cpu", {})["threads"] = a.cpu_threads
    from cyp_tdi.resource_manager import apply_thread_env
    apply_thread_env(res)


if __name__ == "__main__":
    if wants_torch_preload(sys.argv[1:], os.environ):
        err = preload_torch()
        if err:
            print(f"warning: torch could not be imported at start-up ({err}). Steps that do not need PyTorch still run; "
                  "steps that need it will stop with a DependencyError.", file=sys.stderr)
    _early_threads(sys.argv[1:])
    from cyp_tdi.cli import main
    sys.exit(main())
