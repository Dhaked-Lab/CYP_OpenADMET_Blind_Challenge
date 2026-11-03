"""Shared helpers for the test-suite (no heavy dependencies required)."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"


def have(module: str) -> bool:
    """True if ``module`` can be found without importing it."""
    return importlib.util.find_spec(module) is not None


def load_script(filename: str):
    """Import a numbered script (e.g. ``04_scaffold_folds.py``) as a module."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    path = SCRIPTS / filename
    name = "script_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_level_assignments(path: Path) -> dict[str, ast.expr]:
    """Map module-level ``NAME = <expr>`` assignments to their AST value nodes."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
    return out


def literal_kwargs(node: ast.expr):
    """Evaluate literal expressions that may contain ``dict(a=1, b="x")`` calls."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        return {kw.arg: literal_kwargs(kw.value) for kw in node.keywords}
    if isinstance(node, ast.Dict):
        return {ast.literal_eval(k): literal_kwargs(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)
