"""Optional/heavy dependency boundary. Heavy frameworks are imported ONLY by code that needs them, through
`import_framework`, which turns both 'not installed' and 'installed but cannot initialise' (e.g. Windows
WinError 127 while loading torch\\lib\\shm.dll) into one actionable DependencyError. Nothing is swallowed."""
from __future__ import annotations
import importlib, importlib.util
from importlib import metadata

# import-module name -> PyPI distribution name (only where they differ)
DISTRIBUTION = {"sklearn": "scikit-learn", "yaml": "PyYAML", "rdkit": "rdkit"}

INSTALL_HINT = {
    "torch": "install PyTorch with the official selector command for your OS/Python/CUDA (https://pytorch.org/get-started/locally/); "
             "see README 'Installation'",
    "chemprop": "pip install \"chemprop==2.3.*\" (after PyTorch)",
    "tabicl": "pip install \"tabicl==2.2.0\" (after PyTorch)",
    "lightgbm": "pip install \"lightgbm==4.7.*\"",
    "rdkit": "conda install -c conda-forge rdkit   (or pip install rdkit)",
}


class DependencyError(RuntimeError):
    """kind == 'missing' (not installed) or 'broken' (installed but import/initialisation failed)."""
    def __init__(self, message, name=None, kind="missing"):
        super().__init__(message)
        self.name, self.kind = name, kind


def is_installed(module: str) -> bool:
    """True if the module can be found. Does NOT import it (so a broken install is still 'installed')."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def import_framework(module: str, purpose: str = ""):
    why = f" ({purpose})" if purpose else ""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] != module.split(".")[0]:
            raise DependencyError(f"{module}{why} is installed but its dependency {e.name!r} is missing: {e}", module, "broken") from e
        raise DependencyError(f"{module} is required{why} but is not installed: {INSTALL_HINT.get(module, 'pip install ' + module)}",
                              module, "missing") from e
    except Exception as e:      # OSError (DLL load), ImportError (symbol mismatch), RuntimeError ... : keep the original text
        hint = ""
        if "WinError 127" in str(e) or "WinError 126" in str(e) or "shm.dll" in str(e):
            hint = (" This is a Windows DLL-loading failure inside the installed package, not a pipeline bug. If `python -c \"import torch\"` "
                    "works in a FRESH process, the cause is an import-order conflict: another native library (e.g. MKL/OpenMP from numpy/scipy/"
                    "scikit-learn, rdkit, lightgbm) was loaded into THIS process first and shadows a DLL torch needs. Import torch first "
                    "(run_pipeline.py does this automatically for runs) or run `python scripts/check_environment.py --order-probes`. "
                    "If a fresh-process import also fails, the PyTorch/CUDA/VC++ install is the problem: see README 'Troubleshooting'.")
        raise DependencyError(f"{module} is installed but failed to initialise{why}: {type(e).__name__}: {e}.{hint}", module, "broken") from e


def distribution_version(module: str):
    """Installed version from distribution metadata; never imports the package. None if not installed."""
    try:
        return metadata.version(DISTRIBUTION.get(module, module))
    except metadata.PackageNotFoundError:
        return None


PRELOAD_ERROR = None   # text of the exception from preload_torch(), if it failed


def wants_torch_preload(argv, environ) -> bool:
    """Preload only for commands that execute steps. --list-steps/--status/--help/--dry-run never touch torch."""
    if environ.get("CYP_TDI_NO_TORCH_PRELOAD"):
        return False
    if any(a in argv for a in ("-h", "--help", "--list-steps", "--status", "--dry-run")):
        return False
    return any(a in ("--run-all", "--step", "--steps") or a.startswith(("--step=", "--steps=")) for a in argv)


def preload_torch():
    """Import torch BEFORE numpy/scipy/scikit-learn/rdkit/lightgbm do native numerical work in this process.
    Rationale (hypothesis consistent with a Windows report: `import torch` succeeded in a fresh process and alone, but failed with
    WinError 127 on shm.dll once other native libraries had run in the same process): Windows resolves DLLs by file name, so an
    already-loaded same-named DLL from another package (typically Intel OpenMP / MKL / VC runtime) can shadow the one torch ships.
    This function never hides a failure: it records it (PRELOAD_ERROR) and torch-requiring steps re-import and raise DependencyError."""
    global PRELOAD_ERROR
    if not is_installed("torch"):
        return None
    try:
        importlib.import_module("torch")
        PRELOAD_ERROR = None
    except Exception as e:                 # recorded, re-raised later by import_framework for steps that need torch
        PRELOAD_ERROR = f"{type(e).__name__}: {e}"
    return PRELOAD_ERROR
