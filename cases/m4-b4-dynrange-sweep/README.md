# B4 Dynamic Range Sweep (MVP)

Goal: compare floor/substeps on/off under the same stress configuration to
produce auditable evidence of dynamic-range strategy impact.

Variants (case IDs):
- m4-b4-dynrange-control: nfloor=0, substeps=1 (control)
- m4-b4-dynrange-floor: nfloor>0, substeps=1
- m4-b4-dynrange-substeps: nfloor=0, substeps>1
- m4-b4-dynrange-both: nfloor>0, substeps>1 (treatment)

Run:
`python tools/run_case.py --case m4-b4-dynrange-sweep --stage all --update-evidence`

PASS/FAIL:
- sweep_pass == true (treatment is stable and not worse than control, or
  treatment passes while control fails).
