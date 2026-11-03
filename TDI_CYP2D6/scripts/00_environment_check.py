"""STEP 00 -- Environment check.

Run this FIRST, before any other script. Verifies that every third-party
package imported by the pipeline can be imported and prints its version.
Exits with status 0 if all required packages are present, 1 otherwise.

Run:  python scripts/00_environment_check.py
"""

from __future__ import annotations

import sys

# import name -> pip package name
REQUIRED = {
    "rdkit": "rdkit",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "yaml": "PyYAML",
    "scipy": "scipy",
}


def _try_import(mod_name: str):
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> None:
    print(f"Python: {sys.version}")
    print()
    print("Required dependencies:")
    all_ok = True
    for mod, pip_name in REQUIRED.items():
        ok, info = _try_import(mod)
        print(f"  [{'OK' if ok else 'MISSING'}] {mod:12s} {info if ok else ''}")
        if not ok:
            all_ok = False
            print(f"           -> pip install {pip_name}")

    print()
    if all_ok:
        print("ENVIRONMENT OK -- all required packages are installed.")
    else:
        print("ENVIRONMENT INCOMPLETE -- install the missing packages above before running step 01.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
