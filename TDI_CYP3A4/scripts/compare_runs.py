#!/usr/bin/env python3
"""Compare two completed run directories (e.g. outputs/CYP3A4.baseline vs outputs/CYP3A4, or two repeat runs).
Usage: python scripts/compare_runs.py outputs/CYP3A4.baseline outputs/CYP3A4
Needs only numpy/pandas. Reports OOF correlations, AUC/MCC, ensemble weights, fraction, positive counts, agreement."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

MEMBERS = ("base", "emb", "tab", "tabcp")


def _j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main(a, b):
    a, b = Path(a), Path(b); out = {}
    print(f"A = {a}\nB = {b}\n")
    for m in MEMBERS:
        pa, pb = a / f"oof_{m}" / f"oof_{m}.npy", b / f"oof_{m}" / f"oof_{m}.npy"
        if pa.exists() and pb.exists():
            x, y = np.load(pa), np.load(pb); ra, rb = _j(pa.parent / f"cv_report_{m}.json"), _j(pb.parent / f"cv_report_{m}.json")
            out[m] = dict(corr=float(np.corrcoef(x, y)[0, 1]), identical=bool(np.array_equal(x, y)), auc_a=ra["AUC"], auc_b=rb["AUC"])
            print(f"OOF {m:6s} corr={out[m]['corr']:.5f} identical={out[m]['identical']} AUC {ra['AUC']:.4f} -> {rb['AUC']:.4f}")
    wa, wb = _j(a / "ensemble_selection" / "ensemble_weights.json"), _j(b / "ensemble_selection" / "ensemble_weights.json")
    if wa and wb:
        print("weights A:", {k: round(v, 3) for k, v in wa["average_weights"].items()}); print("weights B:", {k: round(v, 3) for k, v in wb["average_weights"].items()})
    ea, eb = _j(a / "ensemble_selection" / "cv_report_ensemble.json"), _j(b / "ensemble_selection" / "cv_report_ensemble.json")
    if ea and eb:
        print(f"nested ensemble AUC {ea['AUC']:.4f} -> {eb['AUC']:.4f}; best-frac MCC {ea['pooled_best_frac_MCC']:.4f} -> {eb['pooled_best_frac_MCC']:.4f}")
    ta, tb = _j(a / "threshold_selection" / "threshold_report.json"), _j(b / "threshold_selection" / "threshold_report.json")
    if ta and tb:
        print(f"selected fraction {ta['selected_fraction']} -> {tb['selected_fraction']}")
    sa, sb = a / "submission" / "submission_CYP3A4_is_TDI.csv", b / "submission" / "submission_CYP3A4_is_TDI.csv"
    if sa.exists() and sb.exists():
        da, db = pd.read_csv(sa), pd.read_csv(sb); lab = "CYP3A4_is_TDI"
        same_ids = (da["Molecule_Name"] == db["Molecule_Name"]).all()
        print(f"positives {int(da[lab].sum())} -> {int(db[lab].sum())}; same compound order: {bool(same_ids)}; "
              f"label agreement {float((da[lab] == db[lab]).mean()):.4f} ({int((da[lab] != db[lab]).sum())} differ)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
