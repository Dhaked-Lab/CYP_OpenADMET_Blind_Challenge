"""MCC / rank-fraction selection with an explicit, testable count convention."""
import math
import numpy as np


def mcc_counts(tp, fp, tn, fn):
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def confusion_from_pred(pr, y):
    pr = np.asarray(pr).astype(int); y = np.asarray(y).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def mcc_at(p, y, t):
    return float(mcc_counts(*confusion_from_pred(np.asarray(p) >= t, y)))


def check_scores(scores):
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("non-finite scores cannot be ranked")
    return s


def k_for_fraction(n, f, convention="exact_k"):
    """exact_k: floor(n*f) (epsilon guards float error); legacy_k_plus_1: original int(n*f)+1."""
    base = int(math.floor(n * f + 1e-9)) if convention == "exact_k" else int(n * f)
    return base if convention == "exact_k" else base + 1


def select_top_k(scores, k):
    """Exactly k positives: score descending, ties -> lowest row index (deterministic)."""
    s = check_scores(scores)
    n = len(s)
    if not (1 <= k <= n - 1):
        raise ValueError(f"k={k} out of range for n={n} (need 1 <= k <= n-1)")
    order = np.lexsort((np.arange(n), -s))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def positive_mask(scores, f, convention="exact_k"):
    s = check_scores(scores)
    n = len(s)
    if convention == "legacy_k_plus_1":
        thr = np.sort(s)[::-1][int(n * f)]      # original frac_threshold; '>=' keeps ties
        return s >= thr
    return select_top_k(s, k_for_fraction(n, f, "exact_k"))


def cut_score(scores, mask):
    s = np.asarray(scores)
    return float(s[mask].min())


def best_by_frac(z, y, fracs, convention="exact_k"):
    rows = []
    for f in fracs:
        pr = positive_mask(z, f, convention)
        rows.append((float(mcc_counts(*confusion_from_pred(pr, y))), float(f)))
    return max(rows)


def full_metrics(pr, y):
    tp, fp, tn, fn = confusion_from_pred(pr, y)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(MCC=float(mcc_counts(tp, fp, tn, fn)), Accuracy=(tp + tn) / len(y), Precision=prec, Recall=rec,
                F1=(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, TP=tp, FP=fp, TN=tn, FN=fn)


def zscore(p, mean=None, std=None):
    mean = p.mean() if mean is None else mean
    std = p.std() if std is None else std
    return (p - mean) / std


def report_oof(tag, oof, y, folds, k, fracs, convention):
    from sklearn.metrics import roc_auc_score
    mb, fb = best_by_frac(oof, y, fracs, convention)
    mask = positive_mask(oof, fb, convention)
    t = cut_score(oof, mask)
    per_fold = [dict(fold=int(f), n=int((folds == f).sum()), pos=int(y[folds == f].sum()),
                     AUC=float(roc_auc_score(y[folds == f], oof[folds == f])),
                     MCC_at_pooled_frac=mcc_at(oof[folds == f], y[folds == f], t)) for f in range(k)]
    grid = np.linspace(0.05, 0.85, 81)
    g = [mcc_at(oof, y, tt) for tt in grid]
    return dict(tag=tag, count_convention=convention, pooled_best_frac_MCC=mb, best_frac=fb,
                n_selected_at_best_frac=int(mask.sum()), metrics_at_best_frac=full_metrics(mask, y),
                prob_grid_best_MCC=float(np.max(g)), prob_grid_best_thr=float(grid[int(np.argmax(g))]),
                MCC_at_0p5=mcc_at(oof, y, 0.5), AUC=float(roc_auc_score(y, oof)), per_fold=per_fold)
