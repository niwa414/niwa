# B3 Electron Energy MVP Sweep

On/off gate for the electron-energy tracer path. The treatment case enables
electron-energy evolution (particle radius_rms proxy), while the control keeps
Te fixed.

Run:
`python tools/run_case.py --case m4-b3-electron-energy-sweep --stage all --update-evidence`

PASS/FAIL:
- sweep_pass == true
- treatment shows nonzero Te variation while control remains constant.
