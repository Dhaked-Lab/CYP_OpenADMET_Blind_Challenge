"""Nested greedy z-space ensemble ([JB] tdi_blend_nested.greedy_w), deterministic tie-breaking."""
import numpy as np
from .metrics import best_by_frac, zscore


def greedy_w(zs, y, fracs, convention, rounds=12):
    """Round-robin add-one-member-at-a-time, each round picks the member that most improves the running
    z-average's best-by-fraction MCC. Strict '>' => ties go to the EARLIEST member in `zs` order
    (config ensemble.member_order), which makes tie-breaking deterministic."""
    keys = list(zs); w = {k: 0.0 for k in keys}
    cur_z, wsum = None, 0
    for _ in range(rounds):
        bestk, bestr, cand_z = None, -2, None
        for k in keys:
            z = (wsum * cur_z + zs[k]) / (wsum + 1) if wsum else zs[k]
            r, _ = best_by_frac(z, y, fracs, convention)
            if r > bestr:
                bestr, bestk, cand_z = r, k, z
        w[bestk] += 1; wsum += 1; cur_z = cand_z
    return {k: v / wsum for k, v in w.items()}


def nested_ensemble(members: dict, y, folds, k, fracs, convention, rounds):
    """Weights for outer fold f are chosen on the OTHER folds' OOF predictions (original method).
    See docs/corrections_log.md (JB-08) for the residual second-order dependence."""
    zs = {m: zscore(v) for m, v in members.items()}
    pooled = np.zeros(len(y), dtype=np.float32)
    hist = []
    for f in range(k):
        trn, va = folds != f, folds == f
        w = greedy_w({m: v[trn] for m, v in zs.items()}, y[trn], fracs, convention, rounds)
        hist.append(w)
        pooled[va] = sum(w[m] * zs[m][va] for m in zs)
    avg = {m: float(np.mean([w.get(m, 0.0) for w in hist])) for m in zs}
    return pooled, hist, avg
