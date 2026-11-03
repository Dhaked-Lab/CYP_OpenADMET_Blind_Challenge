"""Central path definitions.

All paths are resolved relative to the repository root (the directory that
contains ``src/``, ``scripts/``, ``data/``, ``config/`` and ``outputs/``), so
the pipeline can be launched from any working directory.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"

ISOFORMS = ("CYP2D6",)


def isoform_dir(isoform: str) -> Path:
    """Output directory for one isoform (``outputs/<isoform>/``)."""
    if isoform not in ISOFORMS:
        raise ValueError(f"Unknown isoform {isoform!r}, expected one of {ISOFORMS}")
    return OUTPUTS_DIR / isoform


def ensure_dirs() -> None:
    """Create the output and log directories if they do not exist."""
    for iso in ISOFORMS:
        isoform_dir(iso).mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Canonical raw file names per isoform (positive/negative pairs)
RAW_FILES = {
    "CYP2D6": {
        "positive": DATA_RAW / "CYP2D6_Positive.csv",
        "negative": DATA_RAW / "CYP2D6_Negative.csv",
    },
}

TEST_BLINDED_PATH = DATA_RAW / "TEST-BLINDED.csv"
