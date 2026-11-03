"""Rank-fraction selection ([JB] tdi_fraction_opt, with the flat [RECON] prior of the original port)."""
import numpy as np
from .metrics import mcc_counts


def rank_conditional_curve(scores, y, n_bins=24):
    order = np.argsort(-scores)
    ys = np.asarray(y)[order]
    bins = np.array_split(ys, n_bins)
    return np.array([b.mean() for b in bins]), np.array([len(b) for b in bins])


def rescale_to_pi(rates, pi, lo=0.002, hi=0.98):
    r = np.clip(rates, lo, hi)
    a, b = -30.0, 30.0
    for _ in range(60):
        m = (a + b) / 2
        v = 1 / (1 + np.exp(-(np.log(r / (1 - r)) + m)))
        a, b = (a, m) if v.mean() > pi else (m, b)
    return 1 / (1 + np.exp(-(np.log(r / (1 - r)) + (a + b) / 2)))


def emcc(probs_by_rank, f, rng, n_trials=800):
    n = len(probs_by_rank); k = max(1, min(n - 1, int(round(f * n))))
    tot = 0.0
    for _ in range(n_trials):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tot += mcc_counts(tp, fp, tn, fn)
    return tot / n_trials


def select_fraction(p, y, fracs, seed, multipliers, n_bins, n_trials, plateau_tol, min_plateau):
    rates, _ = rank_conditional_curve(p, y, n_bins)
    pi_obs = float(np.mean(y))
    pi_grid = np.clip(np.array([pi_obs * m for m in multipliers]), 0.02, 0.9)
    rng = np.random.default_rng(seed)
    curves = {float(pi): [emcc(rescale_to_pi(rates, pi), f, rng, n_trials) for f in fracs] for pi in pi_grid}
    weighted = np.mean(np.array(list(curves.values())), axis=0)
    worst = np.min(np.array(list(curves.values())), axis=0)
    ib, im = int(np.argmax(weighted)), int(np.argmax(worst))
    cand = float(fracs[ib])
    plateau = fracs[weighted >= weighted[ib] - plateau_tol]
    if len(plateau) > min_plateau:
        cand = float(np.median(plateau))
    return dict(pi_observed_train=pi_obs, posterior_optimal_frac=float(fracs[ib]),
                posterior_optimal_expected_mcc=float(weighted[ib]), minimax_optimal_frac=float(fracs[im]),
                minimax_optimal_expected_mcc=float(worst[im]), selected_fraction=cand,
                plateau=[float(x) for x in plateau])
