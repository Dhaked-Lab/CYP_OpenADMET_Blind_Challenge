#!/usr/bin/env python3
"""
10_xgb_complementarity_analysis.py
==================================
ANALYSIS-ONLY. Does XGBoost add complementary information to the
RF + ExtraTrees + LightGBM ensemble, or does the gain come mainly from a
change in the number of predicted positives?

What this script does NOT do
  * no training / retraining, no CV-fold changes, no feature changes
  * no threshold changes (the frozen threshold is only READ)
  * no submission is generated or modified
  * TEST-BLINDED / blind files are never opened (any path containing
    "blind", "test" or "submission" is skipped by the auto-detector)

Inputs (auto-detected, all overridable from the command line)
  * Step 05 OOF probabilities for RF / ET / LGBM / XGB (+ labels)
  * Step 06 frozen ensemble threshold (json / txt / csv)
  * training CSV (only for SMILES / Molecule_Name / label if the OOF file
    does not already carry them)

Run:
    python scripts/10_xgb_complementarity_analysis.py
Optional overrides:
    --oof PATH --threshold 0.xx --old-threshold 0.xx --threshold-file PATH
    --train-csv PATH --outdir DIR --isoform CYP2D6 --n-boot 500
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["RF", "ET", "LGBM", "XGB"]
ALIAS = {
    "RF": r"(?<![a-z])(randomforest|random_forest|rf)(?![a-z])",
    "ET": r"(?<![a-z])(extratrees|extra_trees|extratree|et)(?![a-z])",
    "LGBM": r"(?<![a-z])(lightgbm|light_gbm|lgbm|lgb)(?![a-z])",
    "XGB": r"(?<![a-z])(xgboost|xgb)(?![a-z])",
}
BAD_COL = ("label", "auc", "mcc", "thresh", "binary", "class", "fold", "rank",
           "pred_pos", "is_pos")
LABEL_CANDS = ["is_tdi", "label", "y", "y_true", "target", "is_active",
               "active", "activity", "class"]
NAME_CANDS = ["molecule_name", "moleculename", "mol_name", "name", "mol_id",
              "molecule_id", "compound_id", "id"]
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".idea"}
SKIP_NAME_TOKENS = ("blind", "test", "submission")
THR_PRIORITY = ["ensemble_threshold", "frozen_threshold", "best_threshold",
                "optimal_threshold", "threshold", "best_thr", "thr"]
OLD_RE = re.compile(r"(?<![a-z])old(?![a-z])|3model|3_model|three|rf_et_lgbm")


def is_old(key):
    return bool(OLD_RE.search(str(key).lower()))

HI_AGREE, HI_DISAGREE = 0.10, 0.30
SCAN = np.round(np.arange(0.10, 0.9001, 0.02), 2)
FINE = np.round(np.arange(0.01, 0.9901, 0.01), 2)
OUT_NAMES = ["01_model_comparison.csv", "02_ensemble_comparison.csv",
             "03_prediction_changes.csv", "04_xgb_threshold_scan.csv",
             "05_model_disagreement.csv", "06_xgb_complementarity_report.md"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fail(msg):
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)
    sys.exit(2)


def info(msg):
    print(msg, flush=True)


def walk_files(root, exts, outdir=None):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        if outdir is not None and Path(dp).resolve() == Path(outdir).resolve():
            dn[:] = []
            continue
        for f in fn:
            p = Path(dp) / f
            low = f.lower()
            if p.suffix.lower() not in exts:
                continue
            if any(t in low for t in SKIP_NAME_TOKENS):
                continue
            if any(t in str(p.parent).lower() for t in ("blind",)):
                continue
            if f in OUT_NAMES or f == "xgb_ensemble_threshold_scan.csv":
                continue
            yield p


def path_score(p, isoform):
    s = str(p).lower()
    sc = 0
    if isoform.lower() in s:
        sc += 3
    if "butina" in s:
        sc -= 3
    if "nn_pos" in s or "nn_neg" in s or "nnpos" in s or "nnneg" in s:
        sc -= 3
    if "murcko" in s or "scaffold" in s:
        sc += 1
    nm = p.name.lower()
    if "oof" in nm:
        sc += 2
    if nm.startswith("05"):
        sc += 1
    return sc


def map_columns(cols):
    """Map model -> list of best-ranked candidate columns."""
    out = {}
    for m in MODELS:
        cands = []
        for c in cols:
            lc = str(c).lower()
            if not re.search(ALIAS[m], lc):
                continue
            sc = 2 * ("oof" in lc) + 2 * ("prob" in lc)
            if any(b in lc for b in BAD_COL):
                sc -= 3
            cands.append((sc, c))
        if cands:
            top = max(s for s, _ in cands)
            out[m] = [c for s, c in cands if s == top]
        else:
            out[m] = []
    return out


def mapping_ok(mp):
    return all(len(mp[m]) >= 1 for m in MODELS)


def pick_label(cols):
    low = {str(c).lower(): c for c in cols}
    for k in LABEL_CANDS:
        if k in low:
            return low[k]
    iscols = [c for c in cols if str(c).lower().startswith("is_")]
    return iscols[0] if len(iscols) == 1 else None


def pick_named(cols, cands):
    low = {str(c).lower(): c for c in cols}
    for k in cands:
        if k in low:
            return low[k]
    return None


def pick_smiles(cols):
    for c in cols:
        if "smiles" in str(c).lower():
            return c
    return None


# --------------------------------------------------------------------------
# OOF detection / loading
# --------------------------------------------------------------------------
def read_any(p):
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".npz":
        z = np.load(p, allow_pickle=True)
        return pd.DataFrame({k: np.asarray(z[k]).ravel() for k in z.files
                             if np.asarray(z[k]).ndim == 1})
    if ext in (".pkl", ".pickle", ".joblib"):
        try:
            import joblib
            obj = joblib.load(p)
        except Exception:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame({k: np.asarray(v).ravel() for k, v in obj.items()
                                 if hasattr(v, "__len__") and not isinstance(v, str)
                                 and np.asarray(v).ndim == 1})
    raise ValueError(f"cannot read {p}")


def header_of(p):
    ext = p.suffix.lower()
    try:
        if ext in (".csv", ".tsv"):
            return list(pd.read_csv(p, nrows=0, sep="," if ext == ".csv" else "\t").columns)
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(p).schema.names)
        if ext == ".npz":
            return list(np.load(p, allow_pickle=True).files)
        if ext in (".pkl", ".pickle", ".joblib"):
            if re.search("oof|prob", p.name.lower()):
                return list(read_any(p).columns)
    except Exception:
        return []
    return []


def find_oof_table(root, isoform, outdir):
    found = []
    for p in walk_files(root, {".csv", ".tsv", ".parquet", ".npz", ".pkl", ".pickle", ".joblib"}, outdir):
        if p.stat().st_size > 400e6:
            continue
        mp = map_columns(header_of(p))
        if mapping_ok(mp):
            found.append((path_score(p, isoform), p, mp))
    if not found:
        return None
    found.sort(key=lambda t: -t[0])
    best = found[0][0]
    tied = [f for f in found if f[0] == best]
    if len(tied) > 1:
        lines = "\n".join(f"   score={s}: {p}" for s, p, _ in found[:10])
        fail("Several candidate OOF files match equally well:\n" + lines +
             "\nRe-run with --oof <path> to choose one.")
    return found[0][1], found[0][2], found


def find_oof_split(root, isoform, outdir):
    """Fallback: one OOF file per model (.npy/.csv) sitting in one directory."""
    by_dir = {}
    for p in walk_files(root, {".npy", ".csv"}, outdir):
        stem = p.stem.lower()
        if "oof" not in stem:
            continue
        for m in MODELS:
            if re.search(ALIAS[m], stem):
                by_dir.setdefault(p.parent, {}).setdefault(m, []).append(p)
    good = [(path_score(d / "oof", isoform), d, f) for d, f in by_dir.items()
            if all(m in f and len(f[m]) == 1 for m in MODELS)]
    if not good:
        return None
    good.sort(key=lambda t: -t[0])
    if len(good) > 1 and good[0][0] == good[1][0]:
        fail("Several directories contain per-model OOF files:\n" +
             "\n".join(f"   {d}" for _, d, _ in good) + "\nUse --oof <dir-or-file>.")
    _, d, f = good[0]
    data = {}
    for m in MODELS:
        p = f[m][0]
        if p.suffix == ".npy":
            arr = np.load(p, allow_pickle=True)
        else:
            t = pd.read_csv(p)
            num = t.select_dtypes("number")
            if num.shape[1] == 1:
                arr = num.iloc[:, 0].values
            else:
                mp = map_columns(num.columns)[m] or [c for c in num.columns if "prob" in c.lower() or "oof" in c.lower()]
                if len(mp) != 1:
                    fail(f"Cannot decide which column of {p} holds the probability: {list(t.columns)}")
                arr = num[mp[0]].values
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 1]
        data[m] = arr.ravel()
    return pd.DataFrame({f"{m}_oof_prob": data[m] for m in MODELS}), d, f


def resolve_models(df, mp):
    chosen = {}
    for m in MODELS:
        c = mp[m]
        if len(c) > 1:
            fail(f"Ambiguous column for {m} in OOF file: {c}. Rename or pass --oof for a cleaner file.")
        chosen[m] = c[0]
    return chosen


def find_train_csv(root, n, isoform, outdir):
    best = []
    for p in walk_files(root, {".csv"}, outdir):
        cols = header_of(p)
        if not cols or pick_smiles(cols) is None or pick_label(cols) is None:
            continue
        if any(t in p.name.lower() for t in ("oof", "pred", "fold", "feature", "scan")):
            continue
        try:
            nrow = len(pd.read_csv(p, usecols=[0]))
        except Exception:
            continue
        if nrow != n:
            continue
        sc = path_score(p, isoform) + 3 * ("train" in p.name.lower())
        best.append((sc, p))
    if not best:
        return None
    best.sort(key=lambda t: -t[0])
    return best[0][1]


def load_oof(args, root, outdir):
    """Return DataFrame with Molecule_Name, SMILES, label, RF/ET/LGBM/XGB and provenance dict."""
    prov = {}
    if args.oof:
        p = Path(args.oof)
        if p.is_dir():
            sp = find_oof_split(p, args.isoform, outdir)
            if sp is None:
                fail(f"No usable OOF files in {p}")
            raw, d, files = sp
            prov["oof"] = f"per-model files in {d}"
            cols = {m: f"{m}_oof_prob" for m in MODELS}
        else:
            if not p.exists():
                fail(f"--oof file not found: {p}")
            raw = read_any(p)
            mp = map_columns(raw.columns)
            if not mapping_ok(mp):
                fail(f"{p} lacks recognisable RF/ET/LGBM/XGB probability columns. Columns: {list(raw.columns)}")
            cols = resolve_models(raw, mp)
            prov["oof"] = str(p)
    else:
        res = find_oof_table(root, args.isoform, outdir)
        if res is not None:
            p, mp, allc = res
            raw = read_any(p)
            cols = resolve_models(raw, mp)
            prov["oof"] = str(p)
            if len(allc) > 1:
                prov["oof_other_candidates"] = [str(x[1]) for x in allc[1:6]]
        else:
            sp = find_oof_split(root, args.isoform, outdir)
            if sp is None:
                fail("Could not find Step 05 OOF probabilities for RF/ET/LGBM/XGB.\n"
                     f"Searched under: {root}\n"
                     "Expected either one table with a probability column per model, or one "
                     "*oof*.npy/.csv per model in a single directory.\n"
                     "Run Step 05 first, or pass --oof <file-or-dir>.")
            raw, d, files = sp
            cols = {m: f"{m}_oof_prob" for m in MODELS}
            prov["oof"] = f"per-model files in {d}"

    out = pd.DataFrame({m: pd.to_numeric(raw[cols[m]], errors="coerce").values for m in MODELS})
    prov["oof_columns"] = cols
    n = len(out)
    if out.isna().any().any():
        fail("OOF probabilities contain NaN / non-numeric values (a model may be missing rows).")
    if ((out < -1e-9) | (out > 1 + 1e-9)).any().any():
        fail("OOF values fall outside [0, 1]; these do not look like probabilities.")

    lab = pick_label(raw.columns)
    nm = pick_named(raw.columns, NAME_CANDS)
    sm = pick_smiles(raw.columns)
    if lab is not None:
        out["label"] = pd.to_numeric(raw[lab], errors="coerce").values
    if nm is not None:
        out["Molecule_Name"] = raw[nm].astype(str).values
    if sm is not None:
        out["SMILES"] = raw[sm].astype(str).values

    if "label" not in out or "SMILES" not in out or "Molecule_Name" not in out:
        tp = Path(args.train_csv) if args.train_csv else find_train_csv(root, n, args.isoform, outdir)
        if tp is None or not tp.exists():
            if "label" not in out:
                fail("OOF file has no label column and no training CSV with matching row count "
                     f"({n}) was found. Pass --train-csv <path>.")
            info("[warn] SMILES/Molecule_Name unavailable; using row index.")
        else:
            if any(t in tp.name.lower() for t in SKIP_NAME_TOKENS):
                fail(f"Refusing to read {tp} (looks like a blind/test file).")
            tr = pd.read_csv(tp)
            prov["train_csv"] = str(tp)
            tl, tn, ts = pick_label(tr.columns), pick_named(tr.columns, NAME_CANDS), pick_smiles(tr.columns)
            if len(tr) != n:
                fail(f"Training CSV rows ({len(tr)}) != OOF rows ({n}); cannot align safely.")
            if "Molecule_Name" in out and tn is not None:
                key = tr[tn].astype(str)
                if set(out["Molecule_Name"]) != set(key) or out["Molecule_Name"].duplicated().any():
                    fail("Molecule_Name sets differ between OOF file and training CSV (or contain duplicates).")
                tr = tr.assign(_k=key).set_index("_k").loc[out["Molecule_Name"]].reset_index(drop=True)
                prov["alignment"] = "by Molecule_Name"
            else:
                prov["alignment"] = "POSITIONAL (OOF row order assumed == training CSV order)"
                info("[warn] aligning OOF and training CSV positionally. Verify row order is unchanged.")
            if "label" not in out:
                out["label"] = pd.to_numeric(tr[tl], errors="coerce").values
            elif tl is not None:
                if not np.array_equal(out["label"].values, pd.to_numeric(tr[tl], errors="coerce").values):
                    fail("Labels in OOF file disagree with training CSV after alignment.")
            if "SMILES" not in out and ts is not None:
                out["SMILES"] = tr[ts].astype(str).values
            if "Molecule_Name" not in out:
                out["Molecule_Name"] = tr[tn].astype(str).values if tn is not None else [f"mol_{i}" for i in range(n)]
    if "Molecule_Name" not in out:
        out["Molecule_Name"] = [f"mol_{i}" for i in range(n)]
    if "SMILES" not in out:
        out["SMILES"] = ""
    if out["label"].isna().any() or not set(out["label"].unique()) <= {0, 1, 0.0, 1.0}:
        fail("Labels are missing or not binary 0/1.")
    out["label"] = out["label"].astype(int)

    # informational cross-check against a stored ensemble column, if any
    for c in raw.columns:
        lc = str(c).lower()
        if "ensemble" in lc and "oof" in lc and not is_old(lc) and not any(t in lc for t in ("pred", "label")):
            try:
                v = pd.to_numeric(raw[c], errors="coerce").values
                new4 = out[MODELS].mean(axis=1).values
                prov["stored_ensemble_check"] = (f"column '{c}': max|stored - mean(4)| = "
                                                 f"{np.nanmax(np.abs(v - new4)):.2e}")
            except Exception:
                pass
            break
    return out, prov


# --------------------------------------------------------------------------
# threshold detection
# --------------------------------------------------------------------------
def collect_numbers(obj, prefix=""):
    res = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            res += collect_numbers(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if 0.0 < float(obj) < 1.0 and ("thresh" in prefix.lower() or "thr" in prefix.lower()):
            res.append((prefix, float(obj)))
    return res


def parse_threshold_file(p):
    ext = p.suffix.lower()
    if ext == ".json":
        return collect_numbers(json.loads(p.read_text()))
    if ext == ".csv":
        t = pd.read_csv(p)
        res = []
        for c in t.columns:
            if "thresh" in c.lower() or c.lower() == "thr":
                v = pd.to_numeric(t[c], errors="coerce").dropna()
                if len(v) == 1:
                    res.append((c, float(v.iloc[0])))
        return res
    txt = p.read_text()
    m = re.findall(r"([A-Za-z_ ]*thr[A-Za-z_ ]*)\s*[:=]\s*([0-9.]+)", txt)
    if m:
        return [(a.strip(), float(b)) for a, b in m if 0 < float(b) < 1]
    try:
        v = float(txt.strip())
        return [("threshold", v)] if 0 < v < 1 else []
    except ValueError:
        return []


def find_thresholds(args, root, oof_path, outdir):
    if args.threshold is not None:
        return args.threshold, args.old_threshold, "command line (--threshold)"
    cands = []
    for p in walk_files(root, {".json", ".txt", ".csv"}, outdir):
        if "threshold" not in p.name.lower():
            continue
        sc = path_score(p, args.isoform)
        if oof_path is not None and p.parent == oof_path.parent:
            sc += 4
        if p.name.startswith("06"):
            sc += 3
        cands.append((sc, p))
    if args.threshold_file:
        tf = Path(args.threshold_file)
        if not tf.exists():
            fail(f"--threshold-file not found: {tf}")
        cands = [(99, tf)]
    if not cands:
        fail("Could not find the Step 06 threshold file (searched for *threshold*.json/.txt/.csv).\n"
             "Run Step 06, or pass --threshold <value> / --threshold-file <path>.")
    cands.sort(key=lambda t: -t[0])
    if len(cands) > 1 and cands[0][0] == cands[1][0]:
        fail("Several threshold files match equally well:\n" +
             "\n".join(f"   score={s}: {p}" for s, p in cands[:8]) +
             "\nRe-run with --threshold-file <path> (or --threshold <value>).")
    p = cands[0][1]
    nums = parse_threshold_file(p)
    if not nums:
        fail(f"No numeric threshold found in {p}. Pass --threshold <value>.")
    old = [(k, v) for k, v in nums if is_old(k)]
    new = [(k, v) for k, v in nums if (k, v) not in old]
    non_model = [(k, v) for k, v in new if not any(re.search(ALIAS[m], k.lower()) for m in MODELS)]
    new = non_model or new
    if not new:
        fail(f"Only 'old'-style thresholds found in {p}: {nums}. Pass --threshold.")
    if len({round(v, 9) for _, v in new}) > 1:
        def prio(k):
            last = k.split("/")[-1].lower()
            return THR_PRIORITY.index(last) if last in THR_PRIORITY else 99
        new.sort(key=lambda kv: prio(kv[0]))
        if prio(new[0][0]) == 99 or (len(new) > 1 and prio(new[0][0]) == prio(new[1][0])
                                     and new[0][1] != new[1][1]):
            fail(f"Ambiguous thresholds in {p}: {new}. Pass --threshold <value>.")
    thr = new[0][1]
    old_thr = args.old_threshold if args.old_threshold is not None else (old[0][1] if old else None)
    return thr, old_thr, f"{p}  (key '{new[0][0]}')"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def conf(y, pred):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    return (float(np.sum(pred & y)), float(np.sum(~pred & ~y)),
            float(np.sum(pred & ~y)), float(np.sum(~pred & y)))


def mcc_c(tp, tn, fp, fn):
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return 0.0 if d == 0 else (tp * tn - fp * fn) / np.sqrt(d)


def mcc_pred(y, pred):
    return mcc_c(*conf(y, pred))


def cls_metrics(y, pred):
    tp, tn, fp, fn = conf(y, pred)
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_pred_pos=int(tp + fp), mcc=mcc_c(tp, tn, fp, fn), accuracy=(tp + tn) / n,
                precision=prec, recall=rec, f1=f1, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def rank_metrics(y, p):
    if len(set(y)) < 2:
        return dict(roc_auc=np.nan, pr_auc=np.nan)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))


def full(y, p, thr):
    d = dict(threshold=float(thr))
    d.update(cls_metrics(y, p >= thr))
    d.update(rank_metrics(y, p))
    return d


def optimal_threshold(y, p):
    best = (-2, None)
    for t in FINE:
        m = mcc_pred(y, p >= t)
        if m > best[0] + 1e-12:
            best = (m, float(t))
    return best[1], best[0]


def topk_pred(p, k):
    pred = np.zeros(len(p), dtype=int)
    pred[np.argsort(-p, kind="stable")[:k]] = 1
    return pred


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analysis-only XGB complementarity check on existing OOF results.")
    ap.add_argument("--oof")
    ap.add_argument("--train-csv")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--old-threshold", type=float)
    ap.add_argument("--threshold-file")
    ap.add_argument("--outdir")
    ap.add_argument("--isoform", default="CYP2D6")
    ap.add_argument("--project-root")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    here = Path(__file__).resolve()
    root = Path(args.project_root).resolve() if args.project_root else here.parent.parent
    if not (root / "outputs").exists() and Path.cwd().joinpath("outputs").exists():
        root = Path.cwd()
    info(f"Project root : {root}")

    # temporary outdir guess so detectors skip our own outputs
    guess_out = Path(args.outdir) if args.outdir else None
    df, prov = load_oof(args, root, guess_out)
    oof_path = Path(prov["oof"]) if Path(prov["oof"]).exists() else None
    outdir = Path(args.outdir) if args.outdir else (
        (oof_path.parent if oof_path else root / "outputs") / "10_xgb_complementarity_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    thr_new, thr_old, thr_src = find_thresholds(args, root, oof_path, outdir)
    y = df["label"].values
    n = len(df)
    P = {m: df[m].values for m in MODELS}
    old_p = (P["RF"] + P["ET"] + P["LGBM"]) / 3.0
    new_p = (P["RF"] + P["ET"] + P["LGBM"] + P["XGB"]) / 4.0
    opt_old_thr, opt_old_mcc = optimal_threshold(y, old_p)
    opt_new_thr, opt_new_mcc = optimal_threshold(y, new_p)
    old_thr_derived = thr_old is None
    if thr_old is None:
        thr_old = opt_old_thr
    prov_thr_old = ("OOF-optimal for the old 3-model ensemble (DERIVED: no old threshold found; "
                    "pass --old-threshold to use the stored one)" if old_thr_derived else "stored / supplied")

    info(f"OOF source   : {prov['oof']}")
    info(f"Threshold    : {thr_new:.4f}  <- {thr_src}")
    info(f"Old threshold: {thr_old:.4f}  <- {prov_thr_old}")
    info(f"Molecules    : {n}  (positives {int(y.sum())}, {100*y.mean():.2f}%)")

    # ---- 1. individual models ------------------------------------------------
    rows = []
    for m in MODELS:
        for tt, t in (("A_natural_0.50", 0.5), ("B_frozen_ensemble_thr", thr_new)):
            rows.append(dict(model=m, threshold_type=tt, **full(y, P[m], t)))
        ot, _ = optimal_threshold(y, P[m])
        rows.append(dict(model=m, threshold_type="C_model_own_OOF_optimum", **full(y, P[m], ot)))
    m01 = pd.DataFrame(rows)
    m01.to_csv(outdir / OUT_NAMES[0], index=False)

    # ---- 2/3. ensembles -------------------------------------------------------
    scen = []

    def add(scn, ens, p, t):
        scen.append(dict(scenario=scn, ensemble=ens, **full(y, p, t)))

    add("frozen_own_threshold", "old_3model", old_p, thr_old)
    add("frozen_own_threshold", "new_4model", new_p, thr_new)
    add("same_threshold(frozen_new)", "old_3model", old_p, thr_new)
    add("same_threshold(frozen_new)", "new_4model", new_p, thr_new)
    add("OOF_optimum_own", "old_3model", old_p, opt_old_thr)
    add("OOF_optimum_own", "new_4model", new_p, opt_new_thr)
    k_old = int((old_p >= thr_old).sum())
    k_new = int((new_p >= thr_new).sum())
    for k in sorted({k_old, k_new}):
        for ens, p in (("old_3model", old_p), ("new_4model", new_p)):
            pr = topk_pred(p, k)
            d = dict(scenario=f"matched_positive_count(k={k})", ensemble=ens, threshold=np.nan)
            d.update(cls_metrics(y, pr))
            d.update(rank_metrics(y, p))
            scen.append(d)
    # leave-one-out ablation at each ensemble's own OOF optimum
    for drop in MODELS:
        keep = [m for m in MODELS if m != drop]
        p = np.mean([P[m] for m in keep], axis=0)
        ot, _ = optimal_threshold(y, p)
        add("leave_one_out_at_own_OOF_optimum", f"drop_{drop}", p, ot)
    s02 = pd.DataFrame(scen)
    diffs = []
    num_cols = ["threshold", "n_pred_pos", "mcc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for sc in s02["scenario"].unique():
        if sc.startswith("leave_one_out"):
            continue
        a = s02[(s02.scenario == sc) & (s02.ensemble == "old_3model")]
        b = s02[(s02.scenario == sc) & (s02.ensemble == "new_4model")]
        if len(a) and len(b):
            d = {c: float(b.iloc[0][c] - a.iloc[0][c]) for c in num_cols}
            diffs.append(dict(scenario=sc, ensemble="DIFF(new-old)", **d))
    s02 = pd.concat([s02, pd.DataFrame(diffs)], ignore_index=True)
    s02.to_csv(outdir / OUT_NAMES[1], index=False)

    def pick(scn, ens, col):
        r = s02[(s02.scenario == scn) & (s02.ensemble == ens)]
        return float(r.iloc[0][col])

    # ---- 4. contribution stats ------------------------------------------------
    pe = pd.DataFrame({**P, "old3": old_p, "new4": new_p})
    pear = pe.corr()
    spear = pe.rank().corr()
    mad_xgb_old = float(np.mean(np.abs(P["XGB"] - old_p)))
    mad_new_old = float(np.mean(np.abs(new_p - old_p)))
    resid = pd.DataFrame({m: P[m] - y for m in MODELS})
    err_corr = resid.corr()
    stats = [("pearson_XGB_vs_" + k, pear.loc["XGB", k]) for k in ["RF", "ET", "LGBM", "old3"]]
    stats += [("spearman_XGB_vs_" + k, spear.loc["XGB", k]) for k in ["RF", "ET", "LGBM", "old3"]]
    stats += [("residual_corr_XGB_vs_" + k, err_corr.loc["XGB", k]) for k in ["RF", "ET", "LGBM"]]
    stats += [("mean_abs_diff_XGB_vs_old3", mad_xgb_old), ("mean_abs_diff_new4_vs_old3", mad_new_old),
              ("max_abs_diff_new4_vs_old3", float(np.max(np.abs(new_p - old_p)))),
              ("mean_signed_shift_new4_minus_old3", float(np.mean(new_p - old_p)))]
    pd.DataFrame(stats, columns=["metric", "value"]).to_csv(outdir / "02b_xgb_contribution_stats.csv", index=False)

    # ---- 5. prediction changes -------------------------------------------------
    old_pred = (old_p >= thr_old).astype(int)
    new_pred = (new_p >= thr_new).astype(int)
    old_pred_same = (old_p >= thr_new).astype(int)
    new_pred_same = new_pred

    def quad(a, b):
        return dict(PP=int(((a == 1) & (b == 1)).sum()), PN=int(((a == 1) & (b == 0)).sum()),
                    NP=int(((a == 0) & (b == 1)).sum()), NN=int(((a == 0) & (b == 0)).sum()))

    q1, q2 = quad(old_pred, new_pred), quad(old_pred_same, new_pred_same)
    ctype = np.where((old_pred == 1) & (new_pred == 1), "A_Pos->Pos",
             np.where((old_pred == 1) & (new_pred == 0), "B_Pos->Neg",
              np.where((old_pred == 0) & (new_pred == 1), "C_Neg->Pos", "D_Neg->Neg")))
    out3 = pd.DataFrame({
        "Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"], "is_TDI": y,
        "RF_probability": P["RF"], "ET_probability": P["ET"], "LGBM_probability": P["LGBM"],
        "XGB_probability": P["XGB"], "old_3model_probability": old_p, "new_4model_probability": new_p,
        "old_prediction": old_pred, "new_prediction": new_pred, "probability_change": new_p - old_p,
        "change_type": ctype, "old_prediction_at_frozen_new_thr": old_pred_same,
        "changed_primary": (old_pred != new_pred).astype(int),
        "changed_same_threshold": (old_pred_same != new_pred_same).astype(int)})
    ch = out3[(out3.changed_primary == 1) | (out3.changed_same_threshold == 1)]
    ch.sort_values("probability_change", ascending=False).to_csv(outdir / OUT_NAMES[2], index=False)

    # ---- 6. flipped groups ------------------------------------------------------
    def group(mask, to_pos):
        yy = y[mask]
        rev = new_pred.copy()
        rev[mask] = 0 if to_pos else 1
        return dict(n=int(mask.sum()), label_pos=int(yy.sum()), label_neg=int((1 - yy).sum()),
                    precision_of_group=float(yy.mean()) if mask.sum() else np.nan,
                    mcc_with_group=mcc_pred(y, new_pred), mcc_without_group=mcc_pred(y, rev),
                    mcc_contribution=mcc_pred(y, new_pred) - mcc_pred(y, rev))
    gC = group((old_pred == 0) & (new_pred == 1), True)
    gB = group((old_pred == 1) & (new_pred == 0), False)
    gC_same = group((old_pred_same == 0) & (new_pred_same == 1), True)
    gB_same = group((old_pred_same == 1) & (new_pred_same == 0), False)

    # ---- 7/8. threshold scan -----------------------------------------------------
    scan_rows = []
    for t in SCAN:
        r = cls_metrics(y, new_p >= t)
        ro = cls_metrics(y, old_p >= t)
        scan_rows.append(dict(threshold=float(t), predicted_positive_count=r["n_pred_pos"], MCC=r["mcc"],
                              accuracy=r["accuracy"], precision=r["precision"], recall=r["recall"], F1=r["f1"],
                              old3_positive_count=ro["n_pred_pos"], old3_MCC=ro["mcc"]))
    scan = pd.DataFrame(scan_rows)
    scan.to_csv(outdir / OUT_NAMES[3], index=False)
    scan.iloc[:, :7].to_csv(outdir / "xgb_ensemble_threshold_scan.csv", index=False)
    imax = int(scan["MCC"].idxmax())
    mcc_new_at_oldthr = mcc_pred(y, new_p >= thr_old)
    cnt_new_at_oldthr = int((new_p >= thr_old).sum())

    # decomposition of the MCC change
    total = pick("frozen_own_threshold", "new_4model", "mcc") - pick("frozen_own_threshold", "old_3model", "mcc")
    prob_eff = pick("same_threshold(frozen_new)", "new_4model", "mcc") - pick("same_threshold(frozen_new)", "old_3model", "mcc")
    thr_eff = pick("same_threshold(frozen_new)", "old_3model", "mcc") - pick("frozen_own_threshold", "old_3model", "mcc")
    mk = lambda k, ens: pick(f"matched_positive_count(k={k})", ens, "mcc")
    matched = {k: (mk(k, "old_3model"), mk(k, "new_4model")) for k in sorted({k_old, k_new})}

    # ---- 9. disagreement ------------------------------------------------------------
    M = np.column_stack([P[m] for m in MODELS])
    spread = M.max(1) - M.min(1)
    amax = np.array(MODELS)[M.argmax(1)]
    amin = np.array(MODELS)[M.argmin(1)]
    cls = np.where(spread <= HI_AGREE, "high_agreement", np.where(spread >= HI_DISAGREE, "high_disagreement", "moderate"))
    out5 = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "SMILES": df["SMILES"], "is_TDI": y,
                         "RF_probability": P["RF"], "ET_probability": P["ET"], "LGBM_probability": P["LGBM"],
                         "XGB_probability": P["XGB"], "new_4model_probability": new_p,
                         "prob_range_max_minus_min": spread, "prob_std": M.std(1), "agreement_class": cls,
                         "highest_model": amax, "lowest_model": amin, "new_prediction": new_pred,
                         "final_predicted_positive": new_pred})
    out5.sort_values("prob_range_max_minus_min", ascending=False).to_csv(outdir / OUT_NAMES[4], index=False)
    cls_rows = []
    for c in ["high_agreement", "moderate", "high_disagreement"]:
        mk_ = cls == c
        pp = mk_ & (new_pred == 1)
        cls_rows.append(dict(agreement_class=c, n=int(mk_.sum()), pct=100 * mk_.mean(),
                             label_rate=float(y[mk_].mean()) if mk_.any() else np.nan,
                             predicted_pos=int(pp.sum()),
                             precision_of_pred_pos=float(y[pp].mean()) if pp.any() else np.nan))
    cls_df = pd.DataFrame(cls_rows)
    pos_mask = new_pred == 1
    xgb_extreme_pos = float(np.mean((amax[pos_mask] == "XGB") | (amin[pos_mask] == "XGB"))) if pos_mask.any() else np.nan

    # ---- bootstrap (paired) ------------------------------------------------------------
    rng = np.random.default_rng(42)
    boot = {k: [] for k in ["d_roc", "d_pr", "d_mcc_same_thr", "d_mcc_frozen"]}
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        o, nw = old_p[idx], new_p[idx]
        boot["d_roc"].append(roc_auc_score(yb, nw) - roc_auc_score(yb, o))
        boot["d_pr"].append(average_precision_score(yb, nw) - average_precision_score(yb, o))
        boot["d_mcc_same_thr"].append(mcc_pred(yb, nw >= thr_new) - mcc_pred(yb, o >= thr_new))
        boot["d_mcc_frozen"].append(mcc_pred(yb, nw >= thr_new) - mcc_pred(yb, o >= thr_old))
    bs = {}
    for k, v in boot.items():
        v = np.array(v)
        bs[k] = dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                     p_pos=float((v > 0).mean()))

    # ---- 10. assessment --------------------------------------------------------------
    r_old = float(pear.loc["XGB", "old3"])
    d_roc = pick("OOF_optimum_own", "DIFF(new-old)", "roc_auc")
    d_pr = pick("OOF_optimum_own", "DIFF(new-old)", "pr_auc")
    d_mcc_opt = pick("OOF_optimum_own", "DIFF(new-old)", "mcc")
    d_matched = [b - a for a, b in matched.values()]
    for_, against = [], []
    (against if r_old >= 0.97 else for_ if r_old <= 0.90 else []).append(
        f"XGB vs old-ensemble Pearson r = {r_old:.3f} (>=0.97 = largely redundant, <=0.90 = distinct)")
    ci = bs["d_pr"]
    (for_ if ci["lo"] > 0 else against).append(
        f"Delta PR-AUC = {d_pr:+.4f}, bootstrap 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
        + ("" if ci["lo"] > 0 else " (CI includes 0 / negative)"))
    ci = bs["d_roc"]
    (for_ if ci["lo"] > 0 else against).append(
        f"Delta ROC-AUC = {d_roc:+.4f}, bootstrap 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
        + ("" if ci["lo"] > 0 else " (CI includes 0 / negative)"))
    dm = min(d_matched)
    (for_ if dm >= 0.01 else against).append(
        f"Delta MCC at matched positive counts = {', '.join(f'{d:+.4f}' for d in d_matched)} "
        f"(materiality bar +0.010 in all matched settings)")
    ci = bs["d_mcc_same_thr"]
    (for_ if ci["lo"] > 0 else against).append(
        f"Delta MCC at one shared threshold = {prob_eff:+.4f}, bootstrap 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    count_driven = abs(thr_eff) > abs(prob_eff)
    (against if count_driven else for_).append(
        f"MCC change decomposition: probability effect {prob_eff:+.4f} vs threshold/count effect {thr_eff:+.4f}")
    verdict = ("LEANS COMPLEMENTARY" if len(for_) >= 4 else
               "MARGINAL / NOT DEMONSTRATED" if len(for_) >= 2 else "LARGELY REDUNDANT (no demonstrable probability-level gain)")

    # ---- report ------------------------------------------------------------------------
    def r4(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    L = []
    L.append(f"# XGBoost complementarity analysis ({args.isoform}, Murcko Scaffold-CV OOF)\n")
    L.append("**Analysis-only.** No retraining, no change to folds/features/thresholds, no blind labels read, "
             "no submission written.\n")
    L.append("## Provenance\n")
    L.append(f"- OOF source: `{prov['oof']}`  \n- OOF columns: `{prov['oof_columns']}`")
    if "train_csv" in prov:
        L.append(f"- Labels/SMILES from: `{prov['train_csv']}` ({prov.get('alignment')})")
    if "stored_ensemble_check" in prov:
        L.append(f"- Stored-ensemble cross-check: {prov['stored_ensemble_check']}")
    L.append(f"- Frozen (new 4-model) threshold: **{thr_new:.4f}** from `{thr_src}`")
    L.append(f"- Old 3-model threshold: **{thr_old:.4f}** ({prov_thr_old})")
    L.append(f"- Molecules: {n}; positives {int(y.sum())} ({100*y.mean():.2f}%). Positive call = prob >= threshold.")
    L.append("- Note: these are pooled OOF metrics for one isoform; the leaderboard MA-* numbers are macro-averaged "
             "over isoforms on the blind set, so they are not expected to match.\n")
    L.append("## 1. Individual models\n")
    L.append(md_table(m01[["model", "threshold_type", "threshold", "n_pred_pos", "mcc", "accuracy", "precision",
                           "recall", "f1", "roc_auc", "pr_auc"]]) + "\n")
    L.append("## 2-3. Ensembles (old 3-model vs new 4-model)\n")
    L.append(md_table(s02[["scenario", "ensemble", "threshold", "n_pred_pos", "mcc", "accuracy", "precision",
                           "recall", "f1", "roc_auc", "pr_auc"]]) + "\n")
    L.append(f"OOF-optimal thresholds (0.01 grid): old {opt_old_thr:.2f} (MCC {opt_old_mcc:.4f}), "
             f"new {opt_new_thr:.2f} (MCC {opt_new_mcc:.4f}).\n")
    L.append("## 4. XGBoost contribution\n")
    L.append(md_table(pd.DataFrame(stats, columns=["metric", "value"]), "{:.4f}") + "\n")
    L.append("Pairwise Pearson correlation of OOF probabilities:\n")
    L.append(md_table(pear.round(4).reset_index().rename(columns={"index": ""})) + "\n")
    L.append("Because new = old + (XGB - old)/4, `mean|new-old|` is mechanically exactly "
             f"`mean|XGB-old|/4` ({mad_xgb_old:.4f}/4 = {mad_xgb_old/4:.4f}); the ensemble can move at most "
             "a quarter of the way toward XGBoost.\n")
    L.append("## 5. Prediction changes\n")
    qt = pd.DataFrame([
        dict(scenario="old@old_thr vs new@frozen_thr (primary)", **{k: v for k, v in q1.items()}),
        dict(scenario="both@frozen_thr (probability effect only)", **{k: v for k, v in q2.items()})])
    qt = qt.rename(columns={"PP": "A Pos->Pos", "PN": "B Pos->Neg", "NP": "C Neg->Pos", "NN": "D Neg->Neg"})
    L.append(md_table(qt) + "\n")
    for nm_, q in (("primary", q1), ("same-threshold", q2)):
        L.append(f"- {nm_} %: " + ", ".join(f"{k}={100*v/n:.2f}%" for k, v in q.items()))
    L.append("\nChanged molecules: `03_prediction_changes.csv` (union of both scenarios, flagged).\n")
    L.append("## 6. Flipped-prediction groups\n")
    gt = pd.DataFrame([dict(group="C: old neg -> new pos (primary)", **gC), dict(group="B: old pos -> new neg (primary)", **gB),
                       dict(group="C at shared thr", **gC_same), dict(group="B at shared thr", **gB_same)])
    L.append(md_table(gt) + "\n")
    L.append("`mcc_contribution` = MCC(new) - MCC(new with that group's prediction reverted to the old call). "
             "`precision_of_group` is the share of true positives among the flipped molecules "
             f"(overall positive rate {y.mean():.4f}).\n")
    L.append("## 7-8. Threshold scan and positive-count vs MCC\n")
    L.append(md_table(scan[["threshold", "predicted_positive_count", "MCC", "precision", "recall", "F1"]]) + "\n")
    L.append(f"- Maximum MCC on scan grid: **{scan.loc[imax,'MCC']:.4f}** at threshold {scan.loc[imax,'threshold']:.2f} "
             f"with **{int(scan.loc[imax,'predicted_positive_count'])}** predicted positives.")
    L.append(f"- New ensemble at the old threshold {thr_old:.4f}: MCC {mcc_new_at_oldthr:.4f}, {cnt_new_at_oldthr} positives.")
    L.append(f"- Old ensemble at its threshold: MCC {pick('frozen_own_threshold','old_3model','mcc'):.4f}, {k_old} positives. "
             f"New ensemble at frozen threshold: MCC {pick('frozen_own_threshold','new_4model','mcc'):.4f}, {k_new} positives. "
             f"Positive-count difference (new - old): {k_new - k_old:+d}.")
    L.append(f"- MCC change decomposition: total {total:+.4f} = probability effect {prob_eff:+.4f} "
             f"(shared threshold {thr_new:.4f}) + threshold/count effect {thr_eff:+.4f}.")
    for k, (a, b) in matched.items():
        L.append(f"- Same number of positives (top-{k}): old {a:.4f} vs new {b:.4f} (diff {b-a:+.4f}).")
    L.append("\nFewer positives is not treated as inherently better: the matched-count rows above compare both "
             "ensembles at an identical positive budget, which removes the count effect.\n")
    L.append("## 9. Model disagreement\n")
    L.append(f"Range = max - min over RF/ET/LGBM/XGB; high agreement <= {HI_AGREE}, high disagreement >= {HI_DISAGREE}. "
             f"Mean range {spread.mean():.4f}, median {np.median(spread):.4f}, 95th pct {np.percentile(spread,95):.4f}.\n")
    L.append(md_table(cls_df) + "\n")
    L.append(f"Among final predicted positives, XGB is the highest or lowest model in "
             f"{100*xgb_extreme_pos:.1f}% of cases (25% expected if all four models were exchangeable).\n")
    top = out5[out5.final_predicted_positive == 1].sort_values("new_4model_probability", ascending=False).head(30)
    L.append("Final predicted positives, side by side (top 30 by ensemble probability; all rows in `05_model_disagreement.csv`):\n")
    L.append(md_table(top[["Molecule_Name", "is_TDI", "RF_probability", "ET_probability", "LGBM_probability",
                           "XGB_probability", "new_4model_probability", "prob_range_max_minus_min"]]) + "\n")
    L.append("## 10. Does XGBoost provide complementary information?\n")
    L.append(f"**Assessment: {verdict}** ({len(for_)} supporting / {len(against)} non-supporting criteria; "
             "rule-based summary, not a significance proof)\n")
    L.append("Supporting:\n" + ("\n".join(f"- {s}" for s in for_) or "- none") + "\n")
    L.append("Not supporting:\n" + ("\n".join(f"- {s}" for s in against) or "- none") + "\n")
    L.append(f"Other evidence: {q1['PN']+q1['NP']} molecules ({100*(q1['PN']+q1['NP'])/n:.2f}%) change class in the primary "
             f"comparison ({q2['PN']+q2['NP']} at a shared threshold); newly added positives have precision "
             f"{gC['precision_of_group']:.3f} vs ensemble precision {pick('frozen_own_threshold','new_4model','precision'):.3f}; "
             f"individual XGB ROC-AUC {m01[(m01.model=='XGB')&(m01.threshold_type.str.startswith('A'))].roc_auc.iloc[0]:.4f} / "
             f"PR-AUC {m01[(m01.model=='XGB')&(m01.threshold_type.str.startswith('A'))].pr_auc.iloc[0]:.4f}. "
             "Leave-one-out rows in `02_ensemble_comparison.csv` show how much each member contributes.\n")
    L.append("Caveats: scaffold-CV OOF molecules are not independent, so bootstrap intervals are optimistic; "
             "differences of a few molecules in a rare-positive set can swing MCC; OOF behaviour may not transfer "
             "to the blind set.\n")
    (outdir / OUT_NAMES[5]).write_text("\n".join(L))

    # ---- terminal summary ----------------------------------------------------------------
    info("\n" + "=" * 72)
    info("XGB COMPLEMENTARITY SUMMARY (OOF only)")
    info("=" * 72)
    for m in MODELS:
        r = m01[(m01.model == m) & (m01.threshold_type == "A_natural_0.50")].iloc[0]
        info(f"{m:5s} @0.50: MCC {r.mcc:.4f}  P {r.precision:.3f}  R {r.recall:.3f}  ROC {r.roc_auc:.4f}  "
             f"PR {r.pr_auc:.4f}  npos {int(r.n_pred_pos)}")
    o, nw = (s02[(s02.scenario == "frozen_own_threshold") & (s02.ensemble == e)].iloc[0] for e in ("old_3model", "new_4model"))
    info(f"old3 @{thr_old:.3f}: MCC {o.mcc:.4f} P {o.precision:.3f} R {o.recall:.3f} F1 {o.f1:.3f} npos {int(o.n_pred_pos)}")
    info(f"new4 @{thr_new:.3f}: MCC {nw.mcc:.4f} P {nw.precision:.3f} R {nw.recall:.3f} F1 {nw.f1:.3f} npos {int(nw.n_pred_pos)}")
    info(f"Delta ROC {d_roc:+.4f} | Delta PR {d_pr:+.4f} | Delta MCC(own optimum) {d_mcc_opt:+.4f}")
    info(f"XGB vs old3 r={r_old:.3f}; mean|XGB-old3|={mad_xgb_old:.4f}; mean|new-old|={mad_new_old:.4f}")
    info(f"Changes (primary): A={q1['PP']} B={q1['PN']} C={q1['NP']} D={q1['NN']} | shared-thr flips: {q2['PN']+q2['NP']}")
    info(f"C (neg->pos): n={gC['n']} TP={gC['label_pos']} FP={gC['label_neg']} contrib={gC['mcc_contribution']:+.4f}")
    info(f"B (pos->neg): n={gB['n']} TP lost={gB['label_pos']} FP removed={gB['label_neg']} contrib={gB['mcc_contribution']:+.4f}")
    info(f"MCC change {total:+.4f} = probability effect {prob_eff:+.4f} + threshold/count effect {thr_eff:+.4f}")
    for k, (a, b) in matched.items():
        info(f"Matched positives k={k}: old {a:.4f} vs new {b:.4f} ({b-a:+.4f})")
    info(f"Scan max MCC {scan.loc[imax,'MCC']:.4f} @ {scan.loc[imax,'threshold']:.2f} "
         f"({int(scan.loc[imax,'predicted_positive_count'])} positives)")
    info(f"VERDICT: {verdict}  ({len(for_)} for / {len(against)} against)")
    info(f"Outputs in: {outdir}")


if __name__ == "__main__":
    main()
