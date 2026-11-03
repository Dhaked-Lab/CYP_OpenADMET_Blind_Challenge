"""Minimal step logger: prints to stdout AND appends to logs/<step>.log."""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

from .paths import LOGS_DIR


class StepLogger:
    def __init__(self, step_name: str):
        self.step_name = step_name
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOGS_DIR / f"{step_name}.log"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.log(f"=== {step_name} started {_dt.datetime.now().isoformat()} ===")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def require_file(self, path: Path, hint: str = "") -> None:
        if not Path(path).exists():
            self.log(f"FATAL: required input file missing: {path}")
            if hint:
                self.log(f"  -> {hint}")
            self._fh.close()
            sys.exit(1)

    def fatal(self, msg: str) -> None:
        self.log(f"FATAL: {msg}")
        self._fh.close()
        sys.exit(1)

    def done(self) -> None:
        self.log(f"=== {self.step_name} finished {_dt.datetime.now().isoformat()} ===")
        self._fh.close()
