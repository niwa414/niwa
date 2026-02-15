# M1-A2 Compression Gate

Lightweight compression gate using external EMF drive (frc_merge pgen).

Run:
```
python tools/run_case.py --case m1-a2-compress-gate --stage all --update-evidence
```

Compression indicator:
- Compute rho threshold from initial snapshot:
  rho_thr = rho_min + frac * (rho_max - rho_min), default frac=0.5.
- For each snapshot, compute the effective axial length along x1 where rho > rho_thr.
- Compression ratio = length_initial / length_final.
