#!/usr/bin/env python3
"""Environment diagnostic. Each heavy import runs in a SUBPROCESS so a DLL crash cannot kill this report.
Reports facts only; it does not modify anything. Exit code 1 if a required framework cannot be initialised."""
import argparse, importlib.util, os, platform, subprocess, sys
from importlib import metadata

DIST = {"sklearn": "scikit-learn", "yaml": "PyYAML"}
PKGS = ["numpy", "pandas", "scipy", "sklearn", "yaml", "rdkit", "lightgbm", "torch", "chemprop", "lightning", "tabicl"]
PROBES = {
    "torch": "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| cuda available', torch.cuda.is_available(), "
             "'| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')",
    "chemprop": "import chemprop; from chemprop import data, featurizers, nn, models; print('chemprop import ok')",
    "lightgbm": "import lightgbm; print('lightgbm import ok')",
    "rdkit": "from rdkit import Chem; print('rdkit import ok', Chem.MolFromSmiles('CCO') is not None)",
    "tabicl": "import tabicl; print('tabicl import ok')",
}


def ver(m):
    try: return metadata.version(DIST.get(m, m))
    except metadata.PackageNotFoundError: return None


ORDER_SNIPPETS = {
    "numpy BLAS": "import numpy as np; a = np.random.rand(300, 300); (a @ a).sum(); np.linalg.svd(a)",
    "scipy": "import numpy as np, scipy.linalg as l; l.svd(np.random.rand(50, 50))",
    "scikit-learn": "import numpy as np; from sklearn.decomposition import PCA; PCA(3).fit(np.random.rand(100, 10))",
    "pandas": "import pandas as pd; pd.DataFrame({'a': [1, 2]}).sum()",
    "rdkit": "from rdkit import Chem; Chem.MolFromSmiles('CCO')",
    "lightgbm": "import lightgbm",
}
EXTRA_DLLS = {"libiomp5md.dll", "vcomp140.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"}
CHILD = """
import os, glob, importlib.util
{pre}
try:
    import torch
    print('RESULT OK')
except BaseException as e:
    print('RESULT FAIL', repr(e)[:300])
    try:
        import psutil
        spec = importlib.util.find_spec('torch'); libdir = os.path.join(os.path.dirname(spec.origin), 'lib')
        names = {{os.path.basename(p).lower() for p in glob.glob(os.path.join(libdir, '*.dll'))}} | set({extra!r})
        for m in psutil.Process().memory_maps():
            if os.path.basename(m.path).lower() in names and not os.path.normcase(m.path).startswith(os.path.normcase(libdir)):
                print('CONFLICT-CANDIDATE', m.path)
    except Exception as e2:
        print('CONFLICT-SCAN-UNAVAILABLE', type(e2).__name__, e2)
"""


def order_probes():
    """Import torch in a fresh process AFTER each other native library has done real work. Facts only."""
    print("\nImport-order probes (torch imported AFTER the named library has done work; separate process each):")
    base = subprocess.run([sys.executable, "-c", CHILD.format(pre="", extra=sorted(EXTRA_DLLS))], capture_output=True, text=True)
    print(f"  {'torch first (nothing else)':28s} {'OK' if 'RESULT OK' in base.stdout else 'FAIL'}")
    bad = 0 if 'RESULT OK' in base.stdout else 1
    for name, snippet in ORDER_SNIPPETS.items():
        r = subprocess.run([sys.executable, "-c", CHILD.format(pre=snippet, extra=sorted(EXTRA_DLLS))], capture_output=True, text=True)
        ok = "RESULT OK" in r.stdout
        print(f"  {'after ' + name:28s} {'OK' if ok else 'FAIL'}")
        if not ok:
            bad += 1
            for line in r.stdout.splitlines() + r.stderr.strip().splitlines()[-2:]:
                if line.startswith(("RESULT FAIL", "CONFLICT")) or "Error" in line:
                    print("      " + line[:400])
    print("  Interpretation: all OK -> no import-order problem; OK first but FAIL after some library -> that library's DLL shadows one torch needs\n"
          "  (CONFLICT-CANDIDATE lines name the already-loaded DLL files); FAIL even first -> the torch install itself is broken.")
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--order-probes", action="store_true", help="also test torch import order (slower)")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]} | {platform.platform()} | executable {sys.executable}")
    print("\nInstalled distributions (metadata only):")
    for m in PKGS:
        spec = importlib.util.find_spec(m) if m != "yaml" or True else None
        print(f"  {m:10s} {ver(m) or 'NOT INSTALLED':20s} {'(importable path found)' if spec else ''}")
    bad = 0
    print("\nImport probes (separate process each):")
    for name, code in PROBES.items():
        if ver(name) is None:
            print(f"  {name:9s} skipped - not installed"); continue
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        ok = r.returncode == 0
        tail = (r.stdout.strip() or r.stderr.strip()).splitlines()
        print(f"  {name:9s} {'OK  ' if ok else 'FAIL'} {tail[-1] if tail else '(no output; process exit code ' + str(r.returncode) + ')'}")
        if not ok and name in ("torch", "chemprop"): bad += 1
    if platform.system() == "Windows":
        print("\nWindows DLL hints (heuristic, not proof):")
        dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and os.path.isdir(d)]
        for dll in ("libiomp5md.dll", "vcomp140.dll", "cudart64_12.dll", "cudart64_13.dll", "msvcp140.dll"):
            hits = [d for d in dirs if os.path.exists(os.path.join(d, dll))]
            print(f"  {dll:18s} found in {len(hits)} PATH dir(s){': ' + '; '.join(hits[:3]) if hits else ''}")
        print("  Several different copies of libiomp5md.dll / CUDA runtimes on PATH, a pip+conda mixture of torch/CUDA/MKL, or a\n"
              "  missing/old Microsoft Visual C++ 2015-2022 x64 runtime are common causes of 'WinError 127' while loading torch\\lib\\shm.dll.")
    if args.order_probes and ver("torch"):
        order_probes()
    print("\n" + ("RESULT: required frameworks initialise." if not bad else "RESULT: PyTorch/Chemprop cannot initialise - see README 'Install -> Windows + NVIDIA GPU'."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
