# Helion Live Tilt Trade Study

This is the live (non-replay) end-to-end path:

1. Run WarpX 3D hybrid case.
2. Compute merge/tilt/gamma + formation + energy KPIs.
3. Gate with PASS/FAIL.
4. Generate engineering recommendation for offline adjustment.

Single knob:
- `h5_seed_config.json::opmd_double_seed_shift[0]`

Cases:
- `baseline`: shift = `0.20`
- `knob_minus`: shift = `0.18`
- `knob_plus`: shift = `0.22`

Run all:

```bash
python tools/run_helion_live_trade_study.py
```

Regenerate report only:

```bash
python tools/run_helion_live_trade_study.py --skip-run
```

Report output:
- `outputs/helion-live-tilt-tradestudy/report.md`
