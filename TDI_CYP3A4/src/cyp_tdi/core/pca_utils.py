"""PCA fitting with explicit fold boundaries (JB-03)."""
import numpy as np


def fit_pca(E_fit, n_components=256, whiten=True, seed=0):
    """[JB] PCA(256, whiten=True, random_state=0); svd_solver left at sklearn default 'auto'
    (resolved solver is recorded by callers via pca_info)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, whiten=whiten, random_state=seed).fit(E_fit)


def pca_info(pca) -> dict:
    import sklearn
    return {"sklearn": sklearn.__version__, "n_components": int(pca.n_components_),
            "svd_solver_resolved": str(getattr(pca, "_fit_svd_solver", "unknown")),
            "whiten": bool(pca.whiten), "n_samples_fit": int(pca.n_samples_)}


def fold_pca_split(E, train_idx, apply_idx, pca_cfg, seed, scope, global_cache=None):
    """Return (Etrain_pc, Eapply_pc) float32.
    scope='fold_train_only': the PCA is fitted ONLY on E[train_idx] and applied (no refit) to E[apply_idx].
    scope='global_legacy'  : original behaviour, PCA fitted on ALL rows (fold boundary violated).
    Works for outer folds and, with inner indices, for inner CV."""
    E = np.asarray(E)
    if scope == "fold_train_only":
        pca = fit_pca(E[train_idx], pca_cfg["n_components"], pca_cfg["whiten"], seed)
    elif scope == "global_legacy":
        if global_cache is not None and "pca" in global_cache:
            pca = global_cache["pca"]
        else:
            pca = fit_pca(E, pca_cfg["n_components"], pca_cfg["whiten"], seed)
            if global_cache is not None:
                global_cache["pca"] = pca
    else:
        raise ValueError(f"unknown pca scope {scope!r}")
    return (pca.transform(E[train_idx]).astype(np.float32), pca.transform(E[apply_idx]).astype(np.float32), pca)
