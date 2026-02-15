# m26-d3-helion-readiness-audit

Purpose: compute Helion-style P0/P1 readiness from existing evidence and emit an actionable gap backlog.

Outputs:
- `analysis/metrics.json`
- `analysis/helion_readiness.json`
- `analysis/helion_readiness.md`
- `analysis/helion_gap_backlog.md`

Run:
- `python tools/run_case.py --case m26-d3-helion-readiness-audit --stage all`
