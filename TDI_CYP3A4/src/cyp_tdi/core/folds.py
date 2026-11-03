"""Scaffold-grouped greedy-balanced K-fold ([JB] run_regression.make_folds), over the target rows (cyp_only) or over the
original all-isoform training table followed by selection of the target rows (original_all_isoforms)."""
import numpy as np


def make_folds(group_keys, k=5, seed=7, argsort_kind="legacy"):
    """argsort_kind='legacy' reproduces the original np.argsort default (quicksort, not stable);
    'stable' is platform independent but may assign tied-size scaffold groups differently."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(group_keys, return_inverse=True)
    sizes = np.bincount(inv)
    order = rng.permutation(len(uniq))
    key = -sizes[order]
    order = order[np.argsort(key) if argsort_kind == "legacy" else np.argsort(key, kind="stable")]
    fold = np.empty(len(group_keys), dtype=int)
    load = np.zeros(k)
    for g in order:
        f = int(np.argmin(load))
        fold[inv == g] = f
        load[f] += sizes[g]
    return fold


def universe_folds(universe_ids, universe_groups, target_ids, k=5, seed=7, argsort_kind="legacy"):
    """[JB] run_tdi.py / run_tdi_emb.py / tdi_tabicl*.py: folds are built over the WHOLE training table (all isoforms),
    then each isoform uses the rows it has labels for. Returns (target_folds, target_groups, universe_fold_array), aligned
    with target_ids. Every target id must occur exactly once in the universe."""
    universe_ids = list(universe_ids)
    if len(universe_ids) != len(universe_groups):
        raise ValueError("universe ids and groups differ in length")
    pos = {}
    for i, s in enumerate(universe_ids):
        if s in pos:
            raise ValueError(f"duplicate id in fold universe: {s!r}")
        pos[s] = i
    missing = [s for s in target_ids if s not in pos]
    if missing:
        raise ValueError(f"{len(missing)} target row(s) are not in the fold universe, e.g. {missing[:3]}")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("duplicate target ids")
    fu = make_folds(universe_groups, k=k, seed=seed, argsort_kind=argsort_kind)
    idx = np.array([pos[s] for s in target_ids], dtype=int)
    return fu[idx], [universe_groups[i] for i in idx], fu


def assign_none_buckets(keys, none_bucket=40):
    fixed, bucket = [], 0
    for k in keys:
        if k == "":
            fixed.append(f"__NONE__{bucket // none_bucket}"); bucket += 1
        else:
            fixed.append(k)
    return fixed


def assert_scaffold_purity(groups, folds):
    import pandas as pd
    g = pd.Series(list(groups))
    spans = g.groupby(g).apply(lambda s: len(set(folds[s.index.values])) > 1)
    if spans.any():
        raise AssertionError(f"{int(spans.sum())} scaffold group(s) span more than one fold")
