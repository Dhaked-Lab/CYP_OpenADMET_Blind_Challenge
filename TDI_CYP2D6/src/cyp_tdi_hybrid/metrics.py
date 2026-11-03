"""TDI classification metrics, using the exact same metric functions as the
official leaderboard (vendored, unmodified, under reference/).

DOCUMENTED FACT (reference/evaluation/config.py): CLASSIFICATION_METRICS =
[MCC, Accuracy, Precision(zero_division=0), Recall(zero_division=0),
F1(zero_division=0)], leaderboard sorted by MCC.

Adapted from the Jermy reproduction's metrics.py, but computed strictly
per-isoform (no combined/masked frame), so there is no "MA" macro column.
"""

from __future__ import annotations

import sys

import numpy as np

from .paths import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reference.evaluation.config import CLASSIFICATION_METRICS  # noqa: E402


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """y_true/y_pred: boolean arrays, no NaNs (mask before calling)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {name: float(func(y_true, y_pred)) for name, func in CLASSIFICATION_METRICS}
