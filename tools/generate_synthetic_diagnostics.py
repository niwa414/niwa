#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def finite(x: float) -> bool:
    return math.isfinite(x)


def read_coil_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "step": float(r["step"]),
                        "time": float(r["time"]),
                        "phi": float(r["phi"]),
                        "area": float(r.get("area", "nan")),
                        "bn_avg": float(r.get("bn_avg", "nan")),
                    }
                )
            except Exception:
                continue
    return rows


def read_pnum_txt(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append(
                    {
                        "step": float(parts[0]),
                        "time": float(parts[1]),
                        "total_weight": float(parts[4]),
                    }
                )
            except Exception:
                continue
    return rows


def read_rhomax_txt(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append(
                    {
                        "step": float(parts[0]),
                        "time": float(parts[1]),
                        "rho_max": float(parts[2]),
                    }
                )
            except Exception:
                continue
    return rows


def numeric_derivative(times: list[float], values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    out = []
    for i in range(n):
        if i == 0:
            dt = times[1] - times[0]
            dv = values[1] - values[0]
        elif i == n - 1:
            dt = times[-1] - times[-2]
            dv = values[-1] - values[-2]
        else:
            dt = times[i + 1] - times[i - 1]
            dv = values[i + 1] - values[i - 1]
        if dt == 0.0:
            out.append(float("nan"))
        else:
            out.append(dv / dt)
    return out


def write_signal_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "time", "signal"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic diagnostics (B-dot + interferometer proxy).")
    ap.add_argument("--run-dir", required=True, help="Simulation run dir containing diag/diags reducedfiles.")
    ap.add_argument("--out-dir", required=True, help="Output synthetic diagnostics directory.")
    ap.add_argument("--metrics", required=True, help="Output metrics json path.")
    ap.add_argument("--phase-scale", type=float, default=1.0, help="Scale for interferometer phase proxy.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    metrics_path = Path(args.metrics).resolve()

    coil_path = run_dir / "diag" / "reducedfiles" / "COIL.txt"
    pnum_path = run_dir / "diags" / "reducedfiles" / "PNUM.txt"
    rhomax_path = run_dir / "diags" / "reducedfiles" / "RHOMAX.txt"

    coil_rows = read_coil_csv(coil_path)
    times = [r["time"] for r in coil_rows]
    steps = [r["step"] for r in coil_rows]
    phi = [r["phi"] for r in coil_rows]
    dphi_dt = numeric_derivative(times, phi)
    bdot = [-x for x in dphi_dt]

    bdot_rows = []
    flux_rows = []
    flux0 = phi[0] if phi else 0.0
    flux_span = (max(phi) - min(phi)) if phi else 0.0
    for i in range(len(coil_rows)):
        bdot_rows.append({"step": steps[i], "time": times[i], "signal": bdot[i]})
        flux_norm = (phi[i] - flux0) / flux_span if flux_span > 0.0 else 0.0
        flux_rows.append({"step": steps[i], "time": times[i], "signal": flux_norm})

    pnum_rows = read_pnum_txt(pnum_path)
    rhomax_rows = read_rhomax_txt(rhomax_path)
    interferometer_rows = []
    interferometer_source = "none"
    if pnum_rows:
        interferometer_source = "pnum_total_weight"
        base = pnum_rows[0]["total_weight"]
        for r in pnum_rows:
            val = 0.0 if base == 0.0 else args.phase_scale * (r["total_weight"] / base - 1.0)
            interferometer_rows.append({"step": r["step"], "time": r["time"], "signal": val})
    elif rhomax_rows:
        interferometer_source = "rhomax_proxy"
        base = rhomax_rows[0]["rho_max"]
        for r in rhomax_rows:
            val = 0.0 if base == 0.0 else args.phase_scale * (r["rho_max"] / base - 1.0)
            interferometer_rows.append({"step": r["step"], "time": r["time"], "signal": val})

    bdot_out = out_dir / "signal_bdot.csv"
    flux_out = out_dir / "signal_flux.csv"
    interferometer_out = out_dir / "signal_interferometer.csv"
    if bdot_rows:
        write_signal_csv(bdot_out, bdot_rows)
    if flux_rows:
        write_signal_csv(flux_out, flux_rows)
    if interferometer_rows:
        write_signal_csv(interferometer_out, interferometer_rows)

    numeric_values = []
    numeric_values.extend([r["signal"] for r in bdot_rows])
    numeric_values.extend([r["signal"] for r in flux_rows])
    numeric_values.extend([r["signal"] for r in interferometer_rows])
    no_nan = all(finite(v) for v in numeric_values) if numeric_values else False

    metrics = {
        "synthetic_dir_exists": out_dir.exists(),
        "coil_source_exists": coil_path.exists(),
        "pnum_source_exists": pnum_path.exists(),
        "rhomax_source_exists": rhomax_path.exists(),
        "bdot_signal_present": len(bdot_rows) >= 2,
        "flux_signal_present": len(flux_rows) >= 2,
        "interferometer_signal_present": len(interferometer_rows) >= 2,
        "bdot_points": len(bdot_rows),
        "flux_points": len(flux_rows),
        "interferometer_points": len(interferometer_rows),
        "interferometer_source": interferometer_source,
        "bdot_file": str(bdot_out) if bdot_out.exists() else None,
        "flux_file": str(flux_out) if flux_out.exists() else None,
        "interferometer_file": str(interferometer_out) if interferometer_out.exists() else None,
        "no_nan_in_metrics": no_nan,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
