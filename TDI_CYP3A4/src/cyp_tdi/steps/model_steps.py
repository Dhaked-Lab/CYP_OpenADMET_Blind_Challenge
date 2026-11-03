"""embeddings, OOF member steps, ensemble_selection, threshold_selection (original 06-10)."""
from __future__ import annotations
import numpy as np
from ..registry import step
from ..io_utils import write_json, save_npy, library_versions, sha256_array
from ..validation import ValidationError, validate_weights
from ..core import metrics, ensemble as ens, threshold as thr
from ..core.nn_features import nn_block
from ..core.pca_utils import fold_pca_split, pca_info
from ..models import factory
from . import _common as C

OOF_KEYS = ("seeds", "cv", "nn_features", "selection", "ensemble.fracs")


def _nn_needs_groups(ctx):
    return ctx.cfg["nn_features"]["mode"] == "exclude_same_scaffold"


def _finish_oof(ctx, sid, tag, oof, y, folds):
    if not np.isfinite(oof).all():
        raise ValidationError(f"{tag}: OOF contains missing predictions")
    sd = ctx.step_dir(sid)
    save_npy(sd / f"oof_{tag}.npy", oof.astype(np.float32))
    rep = metrics.report_oof(tag, oof, y, folds, ctx.cfg["cv"]["k"], C.fracs(ctx), C.convention(ctx))
    write_json(rep, sd / f"cv_report_{tag}.json")
    ctx.logger.info(f"[{tag}] OOF AUC {rep['AUC']:.4f} | MCC(best frac {rep['best_frac']:.2f}) {rep['pooled_best_frac_MCC']:.4f}")


# ------------------------------------------------------------------------------------ embeddings
@step("embeddings", "Frozen CheMeleon (2048-d) and ChemProp-medium embeddings for the training set, provenance-checked cache",
      depends_on=("data_clean",), version="1.0", config_keys=("embeddings", "seeds.torch"),
      outputs=("chemeleon_train.npy", "chemprop_train.npy", "embeddings_meta.json"),
      inputs_fn=lambda ctx: C.checkpoint_files(ctx, "chemeleon", "chemprop"), resources={"gpu": "optional"}, requires=("torch",))
def embeddings(ctx):
    from ..embeddings import embed_with_cache
    d, _ = C.load_clean(ctx); smi = d["SMILES"].tolist(); sd = ctx.step_dir("embeddings"); meta = {}
    for name, fn in (("chemeleon", "chemeleon_train.npy"), ("chemprop", "chemprop_train.npy")):
        path, sha = C.ckpt(ctx, name)
        arr, m = embed_with_cache(name, smi, path, sha, ctx.device, ctx.cfg, ctx.cache_dir, ctx.logger)
        save_npy(sd / fn, arr); meta[name] = m
    write_json(meta, sd / "embeddings_meta.json")


# ------------------------------------------------------------------------------------ OOF members
def _lgbm_oof(ctx, sid, tag, blocks):
    d, y = C.load_clean(ctx); S = C.load_S(ctx); folds = C.load_folds(ctx)
    groups = C.load_groups(ctx) if _nn_needs_groups(ctx) else None
    nn = ctx.cfg["nn_features"]; k = ctx.cfg["cv"]["k"]
    Xblock = np.hstack(blocks)
    oof = np.full(len(d), np.nan, dtype=np.float32)
    for f in range(k):
        trc, vac = np.where(folds != f)[0], np.where(folds == f)[0]
        pos = trc[y[trc] == 1]                           # reference positives: training fold ONLY
        ntr = nn_block(S[trc], pos, nn, self_pos=trc, groups_rows=None if groups is None else groups[trc], groups_all=groups)
        nva = nn_block(S[vac], pos, nn, self_pos=vac)
        m = factory.make_lgbm(ctx.cfg, ctx.res)
        m.fit(np.hstack([Xblock[trc], ntr]), y[trc])
        oof[vac] = m.predict_proba(np.hstack([Xblock[vac], nva]))[:, 1]
        ctx.logger.info(f"[{tag}] fold {f}: n_train={len(trc)} n_val={len(vac)}")
    _finish_oof(ctx, sid, tag, oof, y, folds)


@step("oof_base", "Member 'base': LightGBM on descriptors+fingerprints+nearest-positive features, scaffold-CV OOF (original 06)",
      depends_on=("features", "similarity", "scaffold_cv"), version="1.0", config_keys=OOF_KEYS + ("lightgbm",),
      outputs=("oof_base.npy", "cv_report_base.json"))
def oof_base(ctx):
    X, _ = C.load_X(ctx)
    _lgbm_oof(ctx, "oof_base", "base", [X])


@step("oof_emb", "Member 'emb': CheMeleon embedding concatenated to base features + LightGBM (original 07a)",
      depends_on=("features", "similarity", "scaffold_cv", "embeddings"), version="1.0", config_keys=OOF_KEYS + ("lightgbm",),
      outputs=("oof_emb.npy", "cv_report_emb.json"))
def oof_emb(ctx):
    X, _ = C.load_X(ctx); E = np.load(ctx.artifact("embeddings", "chemeleon_train.npy"))
    _lgbm_oof(ctx, "oof_emb", "emb", [X, E])


def _tabicl_oof(ctx, sid, tag, emb_file):
    d, y = C.load_clean(ctx); folds = C.load_folds(ctx)
    E = np.load(ctx.artifact("embeddings", emb_file))
    scope = ctx.cfg["pca"]["fold_scope"]; seed = ctx.cfg["seeds"]["pca"]
    oof = np.full(len(d), np.nan, dtype=np.float32); cache = {}; infos = []
    for f in range(ctx.cfg["cv"]["k"]):
        trc, vac = np.where(folds != f)[0], np.where(folds == f)[0]
        Ptr, Pva, pca = fold_pca_split(E, trc, vac, ctx.cfg["pca"], seed, scope, cache)   # JB-03 fix
        infos.append({"fold": f, **pca_info(pca), "scope": scope})
        [p] = factory.tabicl_fit_predict(ctx.cfg, ctx.device, Ptr, y[trc], [Pva])
        oof[vac] = p
        ctx.logger.info(f"[{tag}] fold {f} done (PCA scope={scope}, fit on {infos[-1]['n_samples_fit']} rows)")
    write_json(infos, ctx.step_dir(sid) / f"pca_fold_report_{tag}.json")
    _finish_oof(ctx, sid, tag, oof, y, folds)


@step("oof_tab", "Member 'tab': TabICL on PCA-256 of CheMeleon embedding; PCA fitted inside each training fold (original 07b, JB-03 fix)",
      depends_on=("scaffold_cv", "embeddings"), version="1.0", config_keys=OOF_KEYS + ("pca", "tabicl"),
      outputs=("oof_tab.npy", "cv_report_tab.json", "pca_fold_report_tab.json"), resources={"gpu": "optional"}, requires=("torch",))
def oof_tab(ctx):
    _tabicl_oof(ctx, "oof_tab", "tab", "chemeleon_train.npy")


@step("oof_tabcp", "Member 'tabcp': TabICL on PCA-256 of ChemProp-medium embedding; fold-contained PCA (original 08, JB-03 fix)",
      depends_on=("scaffold_cv", "embeddings"), version="1.0", config_keys=OOF_KEYS + ("pca", "tabicl"),
      outputs=("oof_tabcp.npy", "cv_report_tabcp.json", "pca_fold_report_tabcp.json"), resources={"gpu": "optional"}, requires=("torch",))
def oof_tabcp(ctx):
    _tabicl_oof(ctx, "oof_tabcp", "tabcp", "chemprop_train.npy")


# ------------------------------------------------------------------------------------ ensemble
@step("ensemble_selection", "Nested per-fold greedy z-space ensemble weights, deterministic tie-break (original 09)",
      depends_on=("data_clean", "scaffold_cv", "oof_base", "oof_emb", "oof_tab", "oof_tabcp"), version="1.0",
      config_keys=("ensemble", "selection", "cv.k"),
      outputs=("nested_oof.npy", "ensemble_weights.json", "cv_report_ensemble.json", "selection_metadata.json"))
def ensemble_selection(ctx):
    d, y = C.load_clean(ctx); folds = C.load_folds(ctx); cfg = ctx.cfg["ensemble"]
    members = {}
    for name in cfg["member_order"]:                       # tie-break order = original insertion order
        if name in cfg["members"]:
            members[name] = np.load(ctx.artifact(f"oof_{name}", f"oof_{name}.npy"))
    missing = set(cfg["members"]) - set(members)
    if missing:
        raise ValidationError(f"missing OOF members {sorted(missing)}")
    pooled, hist, avg = ens.nested_ensemble(members, y, folds, ctx.cfg["cv"]["k"], C.fracs(ctx), C.convention(ctx), cfg["greedy_rounds"])
    validate_weights(avg, list(members))
    sd = ctx.step_dir("ensemble_selection")
    save_npy(sd / "nested_oof.npy", pooled)
    stats = {k: [float(v.mean()), float(v.std())] for k, v in members.items()}
    write_json({"per_fold_weights": hist, "average_weights": avg, "member_mean_std_for_zscore": stats}, sd / "ensemble_weights.json")
    rep = metrics.report_oof("ensemble", pooled, y, folds, ctx.cfg["cv"]["k"], C.fracs(ctx), C.convention(ctx))
    rep["standalone"] = {k: dict(zip(("best_frac_MCC", "best_frac"), metrics.best_by_frac(metrics.zscore(v), y, C.fracs(ctx), C.convention(ctx))))
                         for k, v in members.items()}
    write_json(rep, sd / "cv_report_ensemble.json")
    write_json({"selected_weights": avg, "member_order": list(members), "greedy_rounds": cfg["greedy_rounds"],
                "count_convention": C.convention(ctx), "member_oof_sha256": {k: sha256_array(v) for k, v in members.items()},
                "library_versions": library_versions(),
                "note": "Weights/fraction may legitimately differ between runs when member OOFs differ (embedding-member noise); "
                        "see docs/corrections_log.md JB-02."}, sd / "selection_metadata.json")
    ctx.logger.info(f"ensemble weights: { {k: round(v, 3) for k, v in avg.items()} }")


@step("threshold_selection", "Rank-fraction selection by expected MCC over prevalence hypotheses (original 10; flat [RECON] prior)",
      depends_on=("data_clean", "ensemble_selection"), version="1.0", config_keys=("threshold", "ensemble.fracs", "seeds.threshold_mc", "selection"),
      outputs=("threshold_report.json",))
def threshold_selection(ctx):
    d, y = C.load_clean(ctx); p = np.load(ctx.artifact("ensemble_selection", "nested_oof.npy")); t = ctx.cfg["threshold"]
    r = thr.select_fraction(p, y, C.fracs(ctx), ctx.cfg["seeds"]["threshold_mc"], t["pi_multipliers"], t["n_bins"],
                            t["n_trials"], t["plateau_tolerance"], t["min_plateau"])
    computed = r["selected_fraction"]
    if t["fraction_override"] is not None:
        r["selected_fraction"] = float(t["fraction_override"]); r["fraction_source"] = "config.threshold.fraction_override"
    else:
        r["fraction_source"] = "threshold_selection (flat prior, original method)"
    r["computed_fraction"] = computed
    mask = metrics.positive_mask(p, r["selected_fraction"], C.convention(ctx))
    r.update(count_convention=C.convention(ctx), n_oof=int(len(p)), oof_positive_count_at_fraction=int(mask.sum()),
             oof_cut_score=metrics.cut_score(p, mask), achieved_oof_mcc_at_fraction=float(metrics.mcc_counts(*metrics.confusion_from_pred(mask, y))),
             reconstruction_note="[RECON] flat prior over pi in multipliers x observed train rate; not leaderboard-tuned")
    write_json(r, ctx.step_dir("threshold_selection") / "threshold_report.json")
    ctx.logger.info(f"selected fraction {r['selected_fraction']} ({r['fraction_source']})")
