"""Nearest-positive similarity block ([JB] run_tdi.py) with fold-safe reference sets."""
import numpy as np


def tanimoto_binary(A, B):
    d = A @ B.T
    ca = (A > 0).sum(1)[:, None]
    cb = (B > 0).sum(1)[None, :]
    return d / np.maximum(ca + cb - d, 1e-9)


def nn_pos_features(S_rows, pos_cols, self_pos=None, topk=10, thr=0.7, exclude_mask=None):
    """S_rows: (n_query, n_ref_all) similarity to the whole reference pool; pos_cols: column indices of
    reference POSITIVES (must come from permitted training data only).
    self_pos: per-row column index of the query itself (self exclusion -> -1).
    exclude_mask: optional boolean (n_query, len(pos_cols)); True entries are set to -1
                  (used by the optional same-scaffold experiment).
    Returns [max sim, mean of top-k sims, count(sim>thr)] exactly as [JB]."""
    pos_cols = np.asarray(pos_cols)
    if pos_cols.size == 0:
        raise ValueError("empty positive reference set for nearest-positive features")
    Sp = S_rows[:, pos_cols].copy()
    if self_pos is not None:
        pos_of = {int(c): j for j, c in enumerate(pos_cols)}
        for i, q in enumerate(self_pos):
            j = pos_of.get(int(q))
            if j is not None:
                Sp[i, j] = -1.0
    if exclude_mask is not None:
        if exclude_mask.shape != Sp.shape:
            raise ValueError("exclude_mask shape mismatch")
        Sp[exclude_mask] = -1.0
    maxp = Sp.max(1)
    kth = min(topk - 1, Sp.shape[1] - 1)
    top = np.partition(-Sp, kth, axis=1)[:, :topk]
    return np.column_stack([maxp, -top.mean(1), (Sp > thr).sum(1)]).astype(np.float32)


def nn_block(S_rows, pos_cols, nn_cfg, self_pos=None, groups_rows=None, groups_all=None):
    """Mode-aware wrapper. mode=baseline reproduces the original 3 columns."""
    mode = nn_cfg["mode"]
    exclude = None
    if mode == "exclude_same_scaffold" and groups_rows is not None:
        gp = np.asarray(groups_all)[np.asarray(pos_cols)]
        exclude = (np.asarray(groups_rows)[:, None] == gp[None, :])
    f = nn_pos_features(S_rows, pos_cols, self_pos=self_pos, topk=nn_cfg["topk"],
                        thr=nn_cfg["sim_threshold"], exclude_mask=exclude)
    if mode == "drop_count":
        f = f[:, :2]
    return f


def n_nn_columns(nn_cfg) -> int:
    return 2 if nn_cfg["mode"] == "drop_count" else 3
