"""model_training, blind_features, blind_embeddings, prediction, submission (original 11-12)."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from ..registry import step
from ..io_utils import write_json, save_npy, read_json, library_versions, sha256_file
from ..validation import (ValidationError, validate_weights, validate_scores, validate_feature_matrix,
                          validate_blind_frame, validate_submission)
from ..discovery import discover_inputs, load_blind
from ..core import metrics
from ..core.nn_features import nn_block, tanimoto_binary
from ..core.pca_utils import fit_pca, pca_info
from ..models import factory
from . import _common as C


def _nn_groups(ctx):
    return C.load_groups(ctx) if ctx.cfg["nn_features"]["mode"] == "exclude_same_scaffold" else None


# ------------------------------------------------------------------------------------ model_training
@step("model_training", "Freeze: final LightGBM members + full-training PCA fits (original 11); TabICL is refit at prediction time",
      depends_on=("features", "similarity", "embeddings", "scaffold_cv"), version="1.0",
      config_keys=("lightgbm", "seeds", "nn_features", "pca"),
      outputs=("base_full.txt", "emb_full.txt", "pca_chemeleon_full.pkl", "pca_chemprop_full.pkl", "training_manifest.json"))
def model_training(ctx):
    d, y = C.load_clean(ctx); X, names = C.load_X(ctx); S = C.load_S(ctx)
    Ech = np.load(ctx.artifact("embeddings", "chemeleon_train.npy")); Ecp = np.load(ctx.artifact("embeddings", "chemprop_train.npy"))
    nn = ctx.cfg["nn_features"]; groups = _nn_groups(ctx); sd = ctx.step_dir("model_training")
    pos_all = np.where(y == 1)[0]
    nnf = nn_block(S, pos_all, nn, self_pos=np.arange(len(d)), groups_rows=groups, groups_all=groups)
    man = {"n_train": len(d), "n_positive": int(len(pos_all)), "nn_columns": int(nnf.shape[1]), "members": {}}
    for tag, blocks in (("base", [X]), ("emb", [X, Ech])):
        Xf = np.hstack(blocks + [nnf])
        m = factory.make_lgbm(ctx.cfg, ctx.res); m.fit(Xf, y)
        factory.save_booster(m.booster_, sd / f"{tag}_full.txt")
        man["members"][tag] = {"n_features": int(Xf.shape[1]), "file": f"{tag}_full.txt"}
        ctx.logger.info(f"{tag}: trained on full training set ({Xf.shape[1]} features)")
    pc = ctx.cfg["pca"]
    for name, E in (("chemeleon", Ech), ("chemprop", Ecp)):
        pca = fit_pca(E, pc["n_components"], pc["whiten"], ctx.cfg["seeds"]["pca"])   # full permitted training set only
        with open(sd / f"pca_{name}_full.pkl", "wb") as fh:
            pickle.dump(pca, fh)
        man["members"][f"pca_{name}"] = pca_info(pca)
    man["library_versions"] = library_versions()
    write_json(man, sd / "training_manifest.json")


# ------------------------------------------------------------------------------------ blind_features
@step("blind_features", "Validate + featurize the blind compounds and compute similarity to the training set (first half of original 12)",
      depends_on=("data_clean", "features"), version="1.0", config_keys=("features", "nn_features"),
      outputs=("blind_ids.csv", "X_blind.npy", "S_blind_train.npy", "blind_features_meta.json"), inputs_fn=C.blind_input_file)
def blind_features(ctx):
    from .. import featurization as F
    iso = ctx.iso_cfg
    blind = load_blind(discover_inputs(ctx.data_dir, iso), iso)
    validate_blind_frame(blind, iso)
    X, ok = F.featurize(blind[iso["smiles_column"]].tolist(), ctx.cfg["features"])
    F.require_all_valid(ok, blind[iso["id_column"]].tolist(), "blind set")
    Xtr, names = C.load_X(ctx)
    validate_feature_matrix(X, Xtr.shape[1], "X_blind")
    B_tr, B_te = F.m2_binary(Xtr, names), F.m2_binary(X, names)
    S = tanimoto_binary(B_te, B_tr).astype(np.float32)
    if S.shape != (len(blind), len(Xtr)) or not np.isfinite(S).all():
        raise ValidationError("blind similarity matrix has wrong shape or non-finite values")
    sd = ctx.step_dir("blind_features")
    blind.to_csv(sd / "blind_ids.csv", index=False); save_npy(sd / "X_blind.npy", X); save_npy(sd / "S_blind_train.npy", S)
    write_json({"n_blind": len(blind), "n_features": int(X.shape[1])}, sd / "blind_features_meta.json")


# ------------------------------------------------------------------------------------ blind_embeddings
@step("blind_embeddings", "Frozen CheMeleon / ChemProp-medium embeddings for the blind compounds",
      depends_on=("blind_features",), version="1.0", config_keys=("embeddings", "seeds.torch"),
      outputs=("chemeleon_blind.npy", "chemprop_blind.npy", "blind_embeddings_meta.json"),
      inputs_fn=lambda ctx: C.checkpoint_files(ctx, "chemeleon", "chemprop"), resources={"gpu": "optional"}, requires=("torch",))
def blind_embeddings(ctx):
    from ..embeddings import embed_with_cache
    ids = pd.read_csv(ctx.artifact("blind_features", "blind_ids.csv"), dtype=str, keep_default_na=False)
    smi = ids[ctx.iso_cfg["smiles_column"]].tolist(); sd = ctx.step_dir("blind_embeddings"); meta = {}
    for name, fn in (("chemeleon", "chemeleon_blind.npy"), ("chemprop", "chemprop_blind.npy")):
        path, sha = C.ckpt(ctx, name)
        arr, m = embed_with_cache(name, smi, path, sha, ctx.device, ctx.cfg, ctx.cache_dir, ctx.logger)
        if arr.shape[0] != len(smi):
            raise ValidationError(f"{name}: embedding rows {arr.shape[0]} != blind compounds {len(smi)}")
        save_npy(sd / fn, arr); meta[name] = m
    write_json(meta, sd / "blind_embeddings_meta.json")


# ------------------------------------------------------------------------------------ prediction
@step("prediction", "Score blind compounds with every member and blend with frozen weights/OOF z-score statistics (second half of original 12)",
      depends_on=("data_clean", "embeddings", "model_training", "blind_features", "blind_embeddings", "ensemble_selection"),
      version="1.0", config_keys=("ensemble", "tabicl", "seeds", "pca", "nn_features"),
      outputs=("blind_scores.csv", "prediction_meta.json"), resources={"gpu": "optional"}, requires=("torch",))
def prediction(ctx):
    d, y = C.load_clean(ctx); groups = None
    ids = pd.read_csv(ctx.artifact("blind_features", "blind_ids.csv"), dtype=str, keep_default_na=False)
    n = len(ids); Xte = np.load(ctx.artifact("blind_features", "X_blind.npy")); Ste = np.load(ctx.artifact("blind_features", "S_blind_train.npy"))
    Ech_te = np.load(ctx.artifact("blind_embeddings", "chemeleon_blind.npy")); Ecp_te = np.load(ctx.artifact("blind_embeddings", "chemprop_blind.npy"))
    Ech = np.load(ctx.artifact("embeddings", "chemeleon_train.npy")); Ecp = np.load(ctx.artifact("embeddings", "chemprop_train.npy"))
    for nm, a in (("X_blind", Xte), ("S_blind", Ste), ("chemeleon_blind", Ech_te), ("chemprop_blind", Ecp_te)):
        if a.shape[0] != n:
            raise ValidationError(f"{nm} has {a.shape[0]} rows, expected {n} (blind compound order mismatch)")
    pos_all = np.where(y == 1)[0]
    nnf = nn_block(Ste, pos_all, ctx.cfg["nn_features"])           # reference = full training positives
    members_cfg = ctx.cfg["ensemble"]["members"]; man = read_json(ctx.artifact("model_training", "training_manifest.json"))
    preds = {}
    for tag, blocks in (("base", [Xte]), ("emb", [Xte, Ech_te])):
        if tag in members_cfg:
            Xf = np.hstack(blocks + [nnf]); b = factory.load_booster(ctx.artifact("model_training", f"{tag}_full.txt"))
            if b.num_feature() != Xf.shape[1] or man["members"][tag]["n_features"] != Xf.shape[1]:
                raise ValidationError(f"{tag}: inconsistent matrix dimensions (model {b.num_feature()}, blind {Xf.shape[1]})")
            preds[tag] = np.asarray(b.predict(Xf))
    import sklearn
    for tag, name, Etr, Ete in (("tab", "chemeleon", Ech, Ech_te), ("tabcp", "chemprop", Ecp, Ecp_te)):
        if tag in members_cfg:
            with open(ctx.artifact("model_training", f"pca_{name}_full.pkl"), "rb") as fh:
                pca = pickle.load(fh)
            if man["members"][f"pca_{name}"]["sklearn"] != sklearn.__version__:
                raise ValidationError("scikit-learn version differs from the one that fitted the PCA; rerun model_training with --force")
            [p] = factory.tabicl_fit_predict(ctx.cfg, ctx.device, pca.transform(Etr).astype(np.float32), y,
                                             [pca.transform(Ete).astype(np.float32)])
            preds[tag] = np.asarray(p)
    for k, v in preds.items():
        validate_scores(v, n, f"member {k}")
    wf = read_json(ctx.artifact("ensemble_selection", "ensemble_weights.json"))
    w = wf["average_weights"]; validate_weights(w, members_cfg)
    if set(preds) != set(w):
        raise ValidationError(f"scored members {sorted(preds)} != weighted members {sorted(w)}")  # JB-06
    z = np.zeros(n, dtype=np.float64); cols = {}
    for k in members_cfg:
        mu, sd_ = wf["member_mean_std_for_zscore"][k]            # OOF statistics, never the blind set's own
        zk = (preds[k] - mu) / sd_; z += w[k] * zk; cols[f"p_{k}"] = preds[k]; cols[f"z_{k}"] = zk
    validate_scores(z, n, "ensemble score")
    out = pd.DataFrame({"Molecule_Name": ids[ctx.iso_cfg["id_column"]], "SMILES": ids[ctx.iso_cfg["smiles_column"]], **cols,
                        "ensemble_score": z})
    sd = ctx.step_dir("prediction"); out.to_csv(sd / "blind_scores.csv", index=False)
    write_json({"n": n, "weights": w, "members": list(members_cfg), "device": ctx.device, "library_versions": library_versions(),
                "note": "TabICL on GPU is not guaranteed bitwise reproducible"}, sd / "prediction_meta.json")


# ------------------------------------------------------------------------------------ submission
def _fraction(ctx):
    return float(read_json(ctx.artifact("threshold_selection", "threshold_report.json"))["selected_fraction"])


@step("submission", "Rank-fraction positive selection (exact count, deterministic ties), validation and submission files",
      depends_on=("prediction", "threshold_selection"), version="1.0", config_keys=("selection", "ensemble.members"),
      outputs=("submission_CYP3A4_is_TDI.csv", "blind_predictions_CYP3A4.csv", "submission_meta.json"), inputs_fn=C.blind_input_file)
def submission(ctx):
    from ..config import fingerprint
    iso = ctx.iso_cfg
    blind = load_blind(discover_inputs(ctx.data_dir, iso), iso); validate_blind_frame(blind, iso)
    sc = pd.read_csv(ctx.artifact("prediction", "blind_scores.csv"), dtype={"Molecule_Name": str, "SMILES": str})
    n = len(blind); validate_scores(sc["ensemble_score"].values, n, "ensemble score")
    if not (sc["Molecule_Name"].values == blind[iso["id_column"]].values).all():
        raise ValidationError("prediction compound order does not match the blind file")
    f = _fraction(ctx); conv = C.convention(ctx)
    mask = metrics.positive_mask(sc["ensemble_score"].values, f, conv)
    k = metrics.k_for_fraction(n, f, conv)
    n_pos = int(mask.sum())
    expected = k if conv == "exact_k" else n_pos            # legacy convention may include tied scores
    if conv == "exact_k" and n_pos != k:
        raise ValidationError(f"selected {n_pos} positives, intended {k}")
    sub = pd.DataFrame({iso["id_column"]: blind[iso["id_column"]], iso["smiles_column"]: blind[iso["smiles_column"]],
                        iso["label_column"]: mask})
    sub = sub[list(iso["submission"]["columns"])]
    validate_submission(sub, blind, iso, expected)
    sd = ctx.step_dir("submission")
    sub.to_csv(sd / "submission_CYP3A4_is_TDI.csv", index=False)
    pred = sc.copy(); pred["Predicted_TDI_Label"] = mask
    pred.to_csv(sd / "blind_predictions_CYP3A4.csv", index=False)
    meta = {"isoform": iso["isoform"], "profile": ctx.cfg["profile"], "n_blind": n, "selected_fraction": f,
            "count_convention": conv, "intended_positive_count": k, "predicted_positive_count": n_pos,
            "cut_score": metrics.cut_score(sc["ensemble_score"].values, mask),
            "n_scores_tied_at_cut": int((sc["ensemble_score"].values == metrics.cut_score(sc["ensemble_score"].values, mask)).sum()),
            "tie_break": ctx.cfg["selection"]["tie_break"],
            "weights": read_json(ctx.artifact("ensemble_selection", "ensemble_weights.json"))["average_weights"],
            "config_fingerprint": fingerprint({k_: v for k_, v in ctx.cfg.items() if k_ != "_root"}),
            "library_versions": library_versions(), "submission_sha256": sha256_file(sd / "submission_CYP3A4_is_TDI.csv")}
    write_json(meta, sd / "submission_meta.json")
    write_json(meta, ctx.run_dir / "state" / "run_summary.json")
    ctx.logger.info(f"submission: {n_pos}/{n} positives (fraction {f}, convention {conv})")
