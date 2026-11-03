"""CYP2D6 STEP 11 -- ADDITIVE experiment: leakage-safe hyperparameter tuning of
RF / ExtraTrees / LightGBM / XGBoost vs. the fixed-parameter baseline.

NOTHING in steps 01-10 is modified and nothing in outputs/CYP2D6/ (baseline) is
overwritten. All new files go to outputs/CYP2D6/hp_tuning/.

Design
------
Outer evaluation : the EXISTING Murcko scaffold folds (04_scaffold_folds.csv), untouched.
Nested tuning    : for each outer fold f, the 4 training folds are tuned with an INNER
                   4-fold CV whose groups are the existing fold ids of those 4 folds
                   (leave-one-existing-fold-out => scaffold-safe, no new split invented).
                   Fold f is never seen during tuning, imputation, or thresholding.
Inner score      : inner OOF probabilities are pooled; the score is the best MCC over the
                   baseline threshold grid (0.10-0.90, step 0.02) on those pooled INNER
                   predictions. The threshold is chosen from training-fold data only.
                   Also saved: per-inner-fold MCC at that threshold (mean/std) and PR-AUC.
Search           : ParameterSampler, fixed seed, SAME candidate list for every outer fold
                   (so stability across folds is comparable). The baseline fixed config is
                   added as trial 0 (control; disable with --no-fixed-candidate).
Final params     : one extra tuning run on all 1497 molecules (inner CV = the 5 existing
                   folds) picks the single config per model used for the blind-test models.
                   This does NOT touch the nested OOF predictions.
Ensemble/thresh  : equal mean of the 4 tuned OOF probabilities; threshold grid and rule
                   identical to step 06 (argmax MCC on the OOF ensemble).

Usage:  python scripts/11_hyperparameter_tuning.py --n-iter 30          # full run
        python scripts/11_hyperparameter_tuning.py --n-iter 2           # smoke test
        python scripts/11_hyperparameter_tuning.py --stage report       # rebuild outputs/report only
Stages: nested -> final -> evaluate -> blind -> report (default: all). Nested/final tuning
results are cached per (model, fold) so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from cyp_tdi_hybrid.metrics import compute_classification_metrics
from cyp_tdi_hybrid.paths import isoform_dir
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import ParameterSampler

ISO = "CYP2D6"
SEED = 42                       # tuning/search seed (model seeds stay 42, as in baseline)
GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.02), 4)   # identical to steps 06/07
MODELS = ["RF", "ExtraTrees", "LightGBM", "XGBoost"]
SHORT = {"RF": "RF", "ExtraTrees": "ET", "LightGBM": "LGBM", "XGBoost": "XGB"}
OUT = isoform_dir(ISO)
HP = OUT / "hp_tuning"
NESTED = HP / "nested"

SPACES = {
    "RF": dict(n_estimators=[400, 600, 800, 1000], max_depth=[6, 8, 10, 12, None],
               min_samples_leaf=[1, 2, 4, 8], max_features=["sqrt", "log2", 0.5, 0.75]),
    "ExtraTrees": dict(n_estimators=[400, 600, 800, 1000], max_depth=[6, 8, 10, 12, None],
                       min_samples_leaf=[1, 2, 4, 8], max_features=["sqrt", "log2", 0.5, 0.75]),
    "LightGBM": dict(n_estimators=[300, 500, 700, 1000], learning_rate=[0.01, 0.03, 0.05, 0.08],
                     num_leaves=[15, 31, 63], max_depth=[-1, 4, 6, 8],
                     min_child_samples=[10, 20, 40, 60], subsample=[0.7, 0.8, 0.9, 1.0],
                     colsample_bytree=[0.7, 0.8, 0.9, 1.0]),
    "XGBoost": dict(n_estimators=[300, 500, 700, 1000], max_depth=[3, 4, 5, 6, 7],
                    learning_rate=[0.01, 0.03, 0.05, 0.08], min_child_weight=[1, 3, 5, 10],
                    subsample=[0.7, 0.8, 0.9, 1.0], colsample_bytree=[0.7, 0.8, 0.9, 1.0],
                    gamma=[0, 0.1, 0.3, 0.5], reg_alpha=[0, 0.1, 0.5, 1.0],
                    reg_lambda=[1, 2, 5, 10]),
}


# ---------------------------------------------------------------- helpers
def load_fixed_params() -> dict:
    """Source of truth: import the constants straight from scripts/05 (not re-typed)."""
    spec = importlib.util.spec_from_file_location("step05", ROOT / "scripts" / "05_train_models_oof.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {"RF": dict(m.RF_PARAMS), "ExtraTrees": dict(m.ET_PARAMS),
            "LightGBM": dict(m.LGBM_PARAMS), "XGBoost": dict(m.XGB_PARAMS)}


def make_model(name: str, params: dict):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    if name == "RF":
        return RandomForestClassifier(**params)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    if name == "LightGBM":
        return LGBMClassifier(**params)
    return XGBClassifier(**params)


def merged(name: str, fixed: dict, tuned: dict) -> dict:
    """Fixed (non-searched) params from the baseline + tuned params on top."""
    return {**fixed[name], **tuned}


def fit_predict(name, params, X_tr, y_tr, X_va):
    y = y_tr.astype(int) if name == "XGBoost" else y_tr.astype(bool)   # as baseline
    return make_model(name, params).fit(X_tr, y).predict_proba(X_va)[:, 1]


def best_threshold(y, p):
    scores = [(t, matthews_corrcoef(y, p >= t)) for t in GRID]
    return max(scores, key=lambda s: s[1])


def metrics_row(y, p, thr):
    m = compute_classification_metrics(y, p >= thr)
    return {"ROC_AUC": float(roc_auc_score(y, p)), "PR_AUC": float(average_precision_score(y, p)),
            **m, "Threshold": float(thr), "Pred_Positive": int((p >= thr).sum())}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    feat = pd.read_csv(OUT / "02_features_ecfp4_rdkit2d.csv")
    alert = pd.read_csv(OUT / "03_structural_alerts.csv")
    fold = pd.read_csv(OUT / "04_scaffold_folds.csv")
    assert (feat["Molecule_Name"] == alert["Molecule_Name"]).all()
    assert (feat["Molecule_Name"] == fold["Molecule_Name"]).all()
    # identical to step 05 _feature_matrix
    feat_cols = [c for c in feat.columns if c.startswith("ecfp4_") or c not in ("Molecule_Name", "SMILES", "is_TDI")]
    alert_cols = [c for c in alert.columns if c.startswith("alert_")]
    Xdf = pd.concat([feat[feat_cols].reset_index(drop=True), alert[alert_cols].reset_index(drop=True)], axis=1)
    return feat, Xdf, feat["is_TDI"].astype(bool).to_numpy(), fold["fold"].to_numpy()


# ---------------------------------------------------------------- tuning core
def build_cache(X, y, groups):
    """Inner splits: imputer fit on inner-train only. Reused across all trials."""
    cache = []
    for g in np.unique(groups):
        va = groups == g
        imp = SimpleImputer(strategy="median").fit(X[~va])
        cache.append((np.where(va)[0], imp.transform(X[~va]), y[~va], imp.transform(X[va])))
    return cache


def score_candidate(name, params, cache, n):
    oof = np.full(n, np.nan)
    for va_idx, Xtr, ytr, Xva in cache:
        oof[va_idx] = fit_predict(name, params, Xtr, ytr, Xva)
    return oof


def tune(name, fixed, X, y, groups, n_iter, include_fixed, tag):
    cache = build_cache(X, y, groups)
    cands = list(ParameterSampler(SPACES[name], n_iter=n_iter, random_state=SEED))
    if include_fixed:
        # baseline config expressed in the search-space keys; keys the baseline leaves unset
        # take the library default so the control equals the real fixed model.
        defaults = {"min_samples_leaf": 1, "max_features": "sqrt", "max_depth": -1, "min_child_samples": 20}
        fx = {k: fixed[name].get(k, defaults.get(k)) for k in SPACES[name]}
        cands = [{"__is_fixed__": True, **fx}] + cands
    rows = []
    for i, c in enumerate(cands):
        is_fixed = c.pop("__is_fixed__", False)
        params = merged(name, fixed, c)
        t0 = time.time()
        oof = score_candidate(name, params, cache, len(y))
        m = np.isfinite(oof)
        yy, pp = y[m], oof[m]
        thr, mcc = best_threshold(yy, pp)
        fold_mcc = [matthews_corrcoef(y[v], oof[v] >= thr) for v, *_ in cache]
        rows.append({"trial": i, "is_fixed_config": is_fixed, "params": json.dumps(c, sort_keys=True),
                     "inner_pooled_mcc": mcc, "inner_threshold": thr,
                     "mean_fold_mcc": float(np.mean(fold_mcc)), "std_fold_mcc": float(np.std(fold_mcc)),
                     "inner_pr_auc": float(average_precision_score(yy, pp)),
                     "inner_roc_auc": float(roc_auc_score(yy, pp)), "seconds": round(time.time() - t0, 1)})
        print(f"  [{tag} {name}] trial {i + 1}/{len(cands)} pooled MCC={mcc:.4f} ({rows[-1]['seconds']}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["inner_pooled_mcc", "inner_pr_auc"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def best_from(df):
    top = df.iloc[0]
    return json.loads(top["params"]), float(top["inner_pooled_mcc"])


# ---------------------------------------------------------------- stages
def stage_nested(args, fixed, X, y, folds):
    NESTED.mkdir(parents=True, exist_ok=True)
    for f in sorted(np.unique(folds)):
        tr = folds != f
        for name in MODELS:
            path = NESTED / f"{SHORT[name]}_outer{f}_results.csv"
            if path.exists() and not args.force:
                print(f"  cached: {path.name}")
                continue
            df = tune(name, fixed, X[tr], y[tr], folds[tr], args.n_iter, not args.no_fixed_candidate, f"outer{f}")
            df.to_csv(path, index=False)


def stage_final(args, fixed, X, y, folds):
    HP.mkdir(parents=True, exist_ok=True)
    best = {}
    for name in MODELS:
        path = HP / f"{SHORT[name]}_tuning_results.csv"
        if path.exists() and not args.force:
            df = pd.read_csv(path)
        else:
            df = tune(name, fixed, X, y, folds, args.n_iter, not args.no_fixed_candidate, "full")
            df.to_csv(path, index=False)
        p, s = best_from(df)
        best[name] = {"model": name, "best_params": p, "inner_cv_score": s,
                      "inner_score_definition": "pooled inner-OOF MCC at best threshold on grid 0.10-0.90/0.02",
                      "fixed_params_kept": {k: v for k, v in fixed[name].items() if k not in p}}
        (HP / f"{SHORT[name]}_best_params.json").write_text(json.dumps(best[name], indent=2))
    (HP / "tuned_parameters.json").write_text(json.dumps(best, indent=2))
    return best


def stage_evaluate(args, fixed, X, y, folds, feat):
    """Nested tuned OOF -> tuned ensemble -> own threshold; compare with baseline files."""
    HP.mkdir(parents=True, exist_ok=True)
    oof = {n: np.full(len(y), np.nan) for n in MODELS}
    nested_best = {}
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xva = imp.transform(X[tr]), imp.transform(X[va])
        for name in MODELS:
            df = pd.read_csv(NESTED / f"{SHORT[name]}_outer{f}_results.csv")
            p, s = best_from(df)
            nested_best[f"{SHORT[name]}_outer{f}"] = {"best_params": p, "inner_cv_score": s}
            oof[name][va] = fit_predict(name, merged(name, fixed, p), Xtr, y[tr], Xva)
            print(f"  outer {f} {name}: fit with {p}", flush=True)
    (HP / "nested_best_params_per_outer_fold.json").write_text(json.dumps(nested_best, indent=2))

    out = feat[["Molecule_Name", "SMILES", "is_TDI"]].copy()
    for n in MODELS:
        out[f"{n}_oof_prob"] = oof[n]
    out["ensemble_oof_prob"] = out[[f"{n}_oof_prob" for n in MODELS]].mean(axis=1)
    out.to_csv(HP / "tuned_oof_predictions.csv", index=False)

    # threshold: identical rule to step 06
    p_t = out["ensemble_oof_prob"].to_numpy()
    scan = pd.DataFrame([{"threshold": float(t), **compute_classification_metrics(y, p_t >= t)} for t in GRID])
    scan.to_csv(HP / "tuned_threshold_scan.csv", index=False)
    br = scan.loc[scan["MCC"].idxmax()]
    thr_t = float(br["threshold"])
    (HP / "tuned_threshold.json").write_text(json.dumps(
        {"best_threshold": thr_t, "MCC": float(br["MCC"]), "n_oof": int(len(y)), "grid_min": 0.10,
         "grid_max": 0.90, "grid_step": 0.02, "ensemble_members": MODELS}, indent=2))

    # baseline (read-only)
    fx = pd.read_csv(OUT / "05_oof_predictions.csv")
    thr_f = json.loads((OUT / "07_best_threshold.json").read_text())["best_threshold"]
    assert (fx["Molecule_Name"] == out["Molecule_Name"]).all()

    rows = []
    for name in MODELS + ["Ensemble4"]:
        col = "ensemble_oof_prob" if name == "Ensemble4" else f"{name}_oof_prob"
        for arm, d, thr in (("fixed", fx, thr_f), ("tuned", out, thr_t)):
            rows.append({"model": name, "arm": arm, **metrics_row(y, d[col].to_numpy(), thr)})
            rows.append({"model": name, "arm": arm + "@0.50", **metrics_row(y, d[col].to_numpy(), 0.5)})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(HP / "fixed_vs_tuned_oof_metrics.csv", index=False)

    # paired bootstrap on delta MCC (ensemble, each arm at its own frozen threshold)
    rng = np.random.default_rng(SEED)
    pf, pt = fx["ensemble_oof_prob"].to_numpy() >= thr_f, p_t >= thr_t
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        d.append(matthews_corrcoef(y[i], pt[i]) - matthews_corrcoef(y[i], pf[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    e = cmp[cmp.model == "Ensemble4"].set_index("arm")
    delta = float(e.loc["tuned", "MCC"] - e.loc["fixed", "MCC"])
    verdict = ("IMPROVED" if delta > 0 and lo > 0 else "NO IMPROVEMENT" if delta <= 0 else "INCONCLUSIVE")
    summ = {"fixed_ensemble_mcc": float(e.loc["fixed", "MCC"]), "tuned_ensemble_mcc": float(e.loc["tuned", "MCC"]),
            "delta_mcc": delta, "bootstrap_delta_mcc_95ci": [float(lo), float(hi)], "n_bootstrap": 2000,
            "verdict_rule": "IMPROVED if delta>0 and 95% CI lower bound>0; NO IMPROVEMENT if delta<=0; else INCONCLUSIVE",
            "verdict": verdict, "fixed_threshold": thr_f, "tuned_threshold": thr_t}
    (HP / "comparison_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return summ


def stage_blind(args, fixed, X, y, folds, feat, best):
    from cyp_tdi_hybrid.data import load_test_blinded
    from cyp_tdi_hybrid.features import featurize_classical
    from cyp_tdi_hybrid.structural_alerts import structural_alert_features
    test_df = load_test_blinded()
    assert test_df is not None and len(test_df) == 750, "TEST-BLINDED.csv must have 750 rows"
    thr = json.loads((HP / "tuned_threshold.json").read_text())["best_threshold"]   # frozen OOF threshold
    imp = SimpleImputer(strategy="median").fit(X)
    Xall = imp.transform(X)
    models, params_used = {}, {}
    for name in MODELS:
        params_used[name] = merged(name, fixed, best[name]["best_params"])
        yy = y.astype(int) if name == "XGBoost" else y
        models[name] = make_model(name, params_used[name]).fit(Xall, yy)

    feats = featurize_classical(test_df["SMILES"].tolist())
    alerts = structural_alert_features(test_df["SMILES"].tolist())
    ecfp = [c for c in feats.columns if c.startswith("ecfp4_")]
    desc = [c for c in feats.columns if not c.startswith("ecfp4_")]
    acol = [c for c in alerts.columns if c.startswith("alert_")]
    Xt = pd.concat([feats[ecfp + desc].reset_index(drop=True), alerts[acol].reset_index(drop=True)], axis=1)
    feature_names = list(load_data()[1].columns)
    Xt = imp.transform(Xt.reindex(columns=feature_names))
    probs = {n: models[n].predict_proba(Xt)[:, 1] for n in MODELS}
    ens = sum(probs[n] for n in MODELS) / 4
    out = test_df[["Molecule_Name", "SMILES"]].copy() if "Molecule_Name" in test_df.columns else pd.DataFrame({"SMILES": test_df["SMILES"]})
    for n in MODELS:
        out[f"{n}_prob"] = probs[n]
    out["ensemble_prob"], out["threshold_used"] = ens, thr
    out[f"{ISO}_is_TDI_predicted"] = ens >= thr
    out.to_csv(HP / "tuned_test_predictions.csv", index=False)
    with open(HP / "tuned_final_models.pkl", "wb") as f_:
        pickle.dump({"models": models, "imputer": imp, "feature_names": feature_names,
                     "ensemble_members": MODELS, "params": params_used}, f_)
    (HP / "tuned_model_parameters.json").write_text(json.dumps(params_used, indent=2, default=str))
    shutil.copy(OUT / "08_final_test_predictions.csv", HP / "fixed_test_predictions.csv")   # copy, baseline untouched
    n_t = int((ens >= thr).sum())
    n_f = int(pd.read_csv(HP / "fixed_test_predictions.csv")[f"{ISO}_is_TDI_predicted"].sum())
    (HP / "blind_counts.json").write_text(json.dumps({"n_test": len(out), "tuned_pred_positive": n_t,
                                                      "fixed_pred_positive": n_f, "tuned_threshold": thr}, indent=2))
    print(f"  blind: tuned predicted positive {n_t}/{len(out)}; fixed {n_f}")


def sanity_and_repro(args, fixed, X, y, folds, feat):
    from cyp_tdi_hybrid.splits import scaffold_groups
    checks = {}
    checks["n_train"] = int(len(y)); checks["n_pos"] = int(y.sum()); checks["n_neg"] = int((~y).sum())
    checks["n_features"] = int(X.shape[1])
    checks["n_ecfp4"] = int(sum(c.startswith("ecfp4_") for c in load_data()[1].columns))
    checks["n_alerts"] = int(sum(c.startswith("alert_") for c in load_data()[1].columns))
    checks["n_rdkit2d"] = checks["n_features"] - checks["n_ecfp4"] - checks["n_alerts"]
    g = np.array(scaffold_groups(feat["SMILES"].tolist()), dtype=object)
    checks["scaffold_groups_spanning_multiple_folds"] = int(
        pd.DataFrame({"g": g, "f": folds}).groupby("g")["f"].nunique().gt(1).sum())
    o = pd.read_csv(HP / "tuned_oof_predictions.csv")
    checks["oof_rows"] = int(len(o)); checks["oof_nan"] = int(o.filter(like="_oof_prob").isna().sum().sum())
    checks["ensemble_is_exact_mean_of_4"] = bool(np.allclose(
        o[[f"{n}_oof_prob" for n in MODELS]].sum(axis=1) / 4, o["ensemble_oof_prob"]))
    checks["expected"] = {"n_train": 1497, "n_pos": 324, "n_neg": 1173, "n_features": 2278,
                          "n_ecfp4": 2048, "n_rdkit2d": 217, "n_alerts": 13}
    checks["all_pass"] = bool(
        checks["n_train"] == 1497 and checks["n_pos"] == 324 and checks["n_neg"] == 1173
        and checks["n_features"] == 2278 and checks["n_ecfp4"] == 2048 and checks["n_rdkit2d"] == 217
        and checks["n_alerts"] == 13 and checks["scaffold_groups_spanning_multiple_folds"] == 0
        and checks["oof_rows"] == 1497 and checks["oof_nan"] == 0 and checks["ensemble_is_exact_mean_of_4"])
    (HP / "sanity_checks.json").write_text(json.dumps(checks, indent=2))

    import lightgbm, rdkit, sklearn, xgboost
    n_trials = {}
    for n in MODELS:
        p = HP / f"{SHORT[n]}_tuning_results.csv"
        n_trials[n] = int(len(pd.read_csv(p))) if p.exists() else None
    repro = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
             "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "rdkit": rdkit.__version__,
             "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
             "search_seed": SEED, "model_seed": 42, "scaffold_fold_seed_baseline": 7,
             "search_method": "sklearn ParameterSampler (random search), same candidates for every outer fold, fixed baseline config added as trial 0",
             "n_trials_requested": args.n_iter, "n_trials_per_model_full_data": n_trials,
             "outer_cv": "existing 5-fold Murcko scaffold (04_scaffold_folds.csv)",
             "inner_cv": "leave-one-existing-fold-out within the training folds (4 inner folds; 5 for the final full-data tuning)",
             "inner_score": "pooled inner-OOF MCC at best threshold on grid 0.10-0.90 step 0.02",
             "search_spaces": {k: {a: [str(x) for x in b] for a, b in v.items()} for k, v in SPACES.items()},
             "feature_count": int(X.shape[1]), "train_rows": int(len(y)),
             "sha256": {"clean_train_data": sha256(OUT / "01_clean_cyp2d6.csv"),
                        "scaffold_folds": sha256(OUT / "04_scaffold_folds.csv"),
                        "features": sha256(OUT / "02_features_ecfp4_rdkit2d.csv"),
                        "alerts": sha256(OUT / "03_structural_alerts.csv"),
                        "feature_matrix_values": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(X)).tobytes()).hexdigest(),
                        "step05_script": sha256(ROOT / "scripts" / "05_train_models_oof.py"),
                        "step06_script": sha256(ROOT / "scripts" / "06_threshold_optimization.py"),
                        "this_script": sha256(Path(__file__)),
                        "baseline_oof": sha256(OUT / "05_oof_predictions.csv")}}
    (HP / "reproducibility.json").write_text(json.dumps(repro, indent=2))
    return checks, repro


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def stage_report(args, fixed, X, y, folds, feat, best):
    summ = json.loads((HP / "comparison_summary.json").read_text())
    cmp = pd.read_csv(HP / "fixed_vs_tuned_oof_metrics.csv")
    checks, repro = sanity_and_repro(args, fixed, X, y, folds, feat)
    blind = json.loads((HP / "blind_counts.json").read_text()) if (HP / "blind_counts.json").exists() else None

    ind = []
    for n in MODELS + ["Ensemble4"]:
        a = cmp[(cmp.model == n) & (cmp.arm == "fixed")].iloc[0]
        b = cmp[(cmp.model == n) & (cmp.arm == "tuned")].iloc[0]
        ind.append({"MODEL": n, "FIXED MCC": a.MCC, "TUNED MCC": b.MCC, "Δ MCC": b.MCC - a.MCC,
                    "Δ ROC-AUC": b.ROC_AUC - a.ROC_AUC, "Δ PR-AUC": b.PR_AUC - a.PR_AUC})
    full = cmp[cmp.arm.isin(["fixed", "tuned"])][["model", "arm", "ROC_AUC", "PR_AUC", "MCC", "Accuracy", "Precision",
                                                   "Recall", "F1", "Threshold", "Pred_Positive"]]
    ens = full[full.model == "Ensemble4"]
    d = summ["delta_mcc"]
    sec = []
    sec.append(f"FIXED ENSEMBLE MCC: {summ['fixed_ensemble_mcc']:.4f}\nTUNED ENSEMBLE MCC: {summ['tuned_ensemble_mcc']:.4f}\n"
               f"DELTA: {d:+.4f}\n\nTUNING VERDICT: [{summ['verdict']}]  (rule: {summ['verdict_rule']}; "
               f"paired-bootstrap 95% CI of Δ MCC = [{summ['bootstrap_delta_mcc_95ci'][0]:+.4f}, {summ['bootstrap_delta_mcc_95ci'][1]:+.4f}])\n")
    sec.append("## 1. Objective\nTest whether tuned RF/ET/LGBM/XGB (nested, leakage-safe) beat the fixed hyperparameters under the unchanged CYP2D6 TDI workflow.")
    sec.append("## 2. Existing fixed workflow\nECFP4(2048)+RDKit2D(217)+13 alerts = 2278 features -> 5-fold Murcko scaffold CV (seed 7) -> RF/ET/LGBM/XGB OOF -> equal mean -> threshold grid 0.10-0.90/0.02 (max MCC) -> final fit -> blind test. Unchanged.")
    sec.append("## 3. Exact fixed parameters (read from scripts/05)\n```json\n" + json.dumps(fixed, indent=2) + "\n```")
    sec.append("## 4. Search methodology\n" + repro["search_method"] + f". Seed {SEED}; trials per model = {args.n_iter} (+1 fixed control). Inner score: {repro['inner_score']}.")
    sec.append("## 5. Search spaces\n```json\n" + json.dumps(repro["search_spaces"], indent=2) + "\n```")
    sec.append("## 6. CV methodology\nOuter: existing Murcko folds. Inner: " + repro["inner_cv"] + ". Per-fold best params: nested_best_params_per_outer_fold.json. Final blind-test params come from a separate tuning run on all 1497 molecules (inner CV = 5 existing folds); they are not used for the OOF numbers.")
    for i, n in enumerate(MODELS, start=7):
        sec.append(f"## {i}. Best parameters for {n}\n```json\n" + json.dumps(best[n], indent=2) + "\n```\nFull trial table: `" + f"{SHORT[n]}_tuning_results.csv`.")
    sec.append("## 11. Fixed vs tuned individual-model comparison (each arm at its own ensemble threshold)\n" + md_table(pd.DataFrame(ind)))
    sec.append("## 12. Fixed vs tuned ensemble comparison\n" + md_table(ens))
    sec.append(f"## 13. Threshold comparison\nFixed ensemble threshold = {summ['fixed_threshold']:.2f}; tuned ensemble threshold = {summ['tuned_threshold']:.2f} (own OOF-optimised, same method).")
    sec.append("## 14. Blind prediction counts\n" + (json.dumps(blind, indent=2) if blind else "Blind stage not run."))
    sec.append("## 15. Reproducibility information\n```json\n" + json.dumps(repro, indent=2) + "\n```")
    sec.append("## 16. Sanity checks\n```json\n" + json.dumps(checks, indent=2) + "\n```")
    sec.append("## 17. Final conclusion\nBased only on the numbers above: ΔMCC = " + f"{d:+.4f}" + f", verdict {summ['verdict']}. "
               "With a small number of positives, treat deltas inside the bootstrap CI as noise; inspect the *_tuning_results.csv files for optimum stability before trusting any single winning config.")
    (HP / "HYPERPARAMETER_TUNING_REPORT.md").write_text(
        "\n\n".join(sec),
        encoding="utf-8"
    )
    print(f"Wrote {HP / 'HYPERPARAMETER_TUNING_REPORT.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--stage", default="all", choices=["all", "nested", "final", "evaluate", "blind", "report"])
    ap.add_argument("--no-fixed-candidate", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached tuning results")
    args = ap.parse_args()

    HP.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_params()
    (HP / "fixed_parameters.json").write_text(json.dumps(fixed, indent=2))
    feat, Xdf, y, folds = load_data()
    X = Xdf.to_numpy(dtype=float)
    s = args.stage
    if s in ("all", "nested"):
        stage_nested(args, fixed, X, y, folds)
    best = None
    if s in ("all", "final", "blind", "report"):
        best = stage_final(args, fixed, X, y, folds)
    if s in ("all", "evaluate"):
        stage_evaluate(args, fixed, X, y, folds, feat)
    if s in ("all", "blind"):
        stage_blind(args, fixed, X, y, folds, feat, best)
    if s in ("all", "report"):
        stage_report(args, fixed, X, y, folds, feat, best)


if __name__ == "__main__":
    main()
