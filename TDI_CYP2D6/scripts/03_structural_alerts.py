"""CYP2D6 STEP 03 -- RECONSTRUCTED_TEAMPOZESCAF_ALERTS: 13 structural-alert
binary features.

IMPORTANT: the TeamPozeSCAF report's original 13 alert definitions were not
found anywhere in the provided material (report, Jermy ZIP, CSVs). The 13
alerts computed here are a documented RECONSTRUCTION -- see
config/structural_alerts.yaml for every SMARTS pattern and its chemical
rationale, and docs/STRUCTURAL_ALERTS_RECONSTRUCTION.md for the search trail.

Requires: outputs/CYP2D6/01_clean_cyp2d6.csv (step 01)
Writes:   outputs/CYP2D6/03_structural_alerts.csv
          outputs/CYP2D6/03_structural_alerts_summary.txt

Run:  python scripts/03_structural_alerts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from cyp_tdi_hybrid.logging_utils import StepLogger
from cyp_tdi_hybrid.paths import ensure_dirs, isoform_dir
from cyp_tdi_hybrid.structural_alerts import load_alert_defs, self_test, structural_alert_features

ISO = "CYP2D6"


def main() -> None:
    ensure_dirs()
    log = StepLogger(f"{ISO}_03_structural_alerts")
    out_dir = isoform_dir(ISO)
    data_path = out_dir / "01_clean_cyp2d6.csv"
    log.require_file(data_path, hint="Run scripts/01_data_audit.py first.")

    defs = load_alert_defs()
    log.log(f"[{ISO}] STEP 03: RECONSTRUCTED_TEAMPOZESCAF_ALERTS ({len(defs)} alerts)")
    log.log("  STATUS: RECONSTRUCTION -- original TeamPozeSCAF SMARTS were not found "
            "in any provided material. See config/structural_alerts.yaml.")

    st = self_test()
    log.log("  Self-test hits on known example molecules:")
    for label, hits in st.items():
        log.log(f"    {label}: {hits}")

    df = pd.read_csv(data_path)
    alert_df = structural_alert_features(df["SMILES"].tolist())
    out = pd.concat([df[["Molecule_Name", "SMILES", "is_TDI"]], alert_df], axis=1)
    out_path = out_dir / "03_structural_alerts.csv"
    out.to_csv(out_path, index=False)

    hit_rates = alert_df.mean().round(4).to_dict()
    summary_lines = [f"CYP2D6 RECONSTRUCTED_TEAMPOZESCAF_ALERTS ({len(defs)} alerts)",
                      "STATUS: RECONSTRUCTION, not the original TeamPozeSCAF definitions.", ""]
    for a in defs:
        col = f"alert_{a['name']}"
        summary_lines.append(f"  {a['name']:32s} smarts={a['smarts']!r:35s} "
                              f"hit_rate={hit_rates.get(col, float('nan')):.4f}")
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "03_structural_alerts_summary.txt").write_text(summary)
    log.log(summary)
    log.log(f"  Wrote {out_path}")
    log.log(f"  Wrote {out_dir / '03_structural_alerts_summary.txt'}")
    log.done()


if __name__ == "__main__":
    main()
