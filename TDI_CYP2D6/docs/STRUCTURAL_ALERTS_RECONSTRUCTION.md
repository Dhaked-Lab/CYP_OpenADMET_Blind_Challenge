# Structural alerts: reconstruction, not recovery

## What the TeamPozeSCAF report says

Section 1.3 of the methodology report states:

> "13 mechanism-based structural-alert features were added to the ECFP4
> representation. Each alert was encoded as a binary feature (0/1) indicating
> the presence or absence of a predefined chemical motif associated with CYP
> TDI."

This tells us: 13 alerts, binary encoding, associated with CYP2D6 TDI
specifically (Section 1.3 sits under the CYP2D6-pipeline description, and
Section 3.2 -- the CYP3A4 architecture -- never mentions alerts again). It
does **not** give SMARTS patterns, alert names, a reference, or any
generation code.

## Where we searched

Full-text search (`grep -ri "smarts\|alert"`) across every file in:
- The TeamPozeSCAF methodology report PDF (extracted text)
- The entire Jermy TDI reproduction ZIP: all of `scripts/`, `src/`,
  `configs/`, `docs/`, `reference/`, `README.md`, `run_instructions.md`
- The four raw training CSVs (column headers and a sample of values)

**Result: zero matches.** No SMARTS strings, no alert names, no
alert-generation function, no reference to a specific alert set (e.g. a named
published alert collection) anywhere in the provided material.

## What we built instead

`config/structural_alerts.yaml` defines 13 independently-chosen binary
alerts, each representing a chemical motif documented in the pharmacology /
medicinal-chemistry literature as a mechanism-based ("suicide")
CYP-inactivator pharmacophore -- e.g. furan and thiophene rings (reactive
epoxide/sulfoxide formation), terminal alkynes and terminal alkenes (reactive
oxirene/epoxide formation), the methylenedioxyphenyl group (carbene-heme
complex formation, as in paroxetine), cyclopropylamines and
1,4-dihydropyridines (radical-mediated heme alkylation), and several
nitrogen/sulfur motifs (hydrazines, hydrazones, primary arylamines,
thioureas, nitroaromatics, thiazolidinediones) associated with reactive
intermediate formation on CYP oxidation.

Every SMARTS pattern, its exact string, and its chemical rationale are in
`config/structural_alerts.yaml`. The features are computed by
`src/cyp_tdi_hybrid/structural_alerts.py` and are labeled throughout the
codebase and all output files as `RECONSTRUCTED_TEAMPOZESCAF_ALERTS`.

## What this means for the results

Any CYP2D6 model result that includes these features should be understood as
using **a** reasonable 13-alert structural-alert feature set for CYP TDI, not
**the** TeamPozeSCAF one. If TeamPozeSCAF's original 13 alerts become
available later, swap the `alerts:` list in `config/structural_alerts.yaml`
(names/SMARTS/rationale) -- no other code changes are needed, since
`structural_alert_features()` reads the count and definitions from that file
dynamically (it will raise an error if the file does not contain exactly 13
entries, as a guard against silently changing the feature dimensionality).
