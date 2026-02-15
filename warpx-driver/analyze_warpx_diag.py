#!/usr/bin/env python3
"""
WarpX RZ diagnostics helper (thetaMode plotfiles)
-------------------------------------------------
- Single-diag summary: extrema, total field energy, mode RMS.
- Series mode: walk diag*/, emit CSV of mode RMS (rho/Bz) vs time, optional growth fit.

Examples
  PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
    python warpx-driver/analyze_warpx_diag.py --diag-path outputs/warpx/<run>
  PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
    python warpx-driver/analyze_warpx_diag.py --diag-path outputs/warpx/<run> --series \
      --max-mode 2 --output-csv outputs/analysis/mode_spectrum.csv --fit-start 5e-8 --fit-end 2e-7
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import yt

UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "magnetic_unit": (1.0, "T"),
}


def list_diags(diag_root: Path):
    return sorted([p for p in diag_root.iterdir() if p.name.startswith("diag") and "old" not in p.name])


def find_latest_diag(diag_root: Path) -> Path:
    diags = list_diags(diag_root)
    if not diags:
        raise SystemExit(f"No diag* found under {diag_root}")
    return diags[-1]


def compute_mode_metrics(ad, field: str, max_mode: int):
    arr = ad["boxlib", field].to_ndarray()
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim < 3:
        return {}
    nmodes = min(arr.shape[0], max_mode + 1)
    metrics = {}
    for m in range(nmodes):
        mode_arr = np.asarray(arr[m])
        amp = np.abs(mode_arr)
        metrics[f"{field}_m{m}_rms"] = float(np.sqrt(np.mean(amp * amp)))
        metrics[f"{field}_m{m}_max"] = float(np.max(amp))
    return metrics


def total_field_energy(ds, Br, Bt, Bz, Er, Et, Ez):
    Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bz ** 2)
    Emag = np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2)

    dx = ds.domain_width.v[0]
    dz = ds.domain_width.v[-1]
    nx = ds.domain_dimensions[0]
    # For RZ thetaMode, use 2*pi*r volume element; theta extent = domain_width[1]
    x = ds.index.grid_left_edge[0].v[0] + (np.arange(nx) + 0.5) * dx
    volume = 2 * np.pi * x[:, None, None] * dx * dz
    if len(ds.domain_width) > 1:
        volume = volume * ds.domain_width.v[1]
    energy_density = 0.5 * (Emag ** 2 + Bmag ** 2)
    return float(np.sum(energy_density * volume)), Bmag, Emag


def summarize_single(diag_root: Path, max_mode: int):
    diag = find_latest_diag(diag_root)
    print(f"Using diag: {diag}")

    ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
    ad = ds.all_data()

    rho = ad["boxlib", "rho"].to_ndarray()
    Br = ad["boxlib", "Br"].to_ndarray()
    Bt = ad["boxlib", "Bt"].to_ndarray()
    Bz = ad["boxlib", "Bz"].to_ndarray()
    Er = ad["boxlib", "Er"].to_ndarray()
    Et = ad["boxlib", "Et"].to_ndarray()
    Ez = ad["boxlib", "Ez"].to_ndarray()

    total_energy, Bmag, Emag = total_field_energy(ds, Br, Bt, Bz, Er, Et, Ez)
    print(f"rho min/max: {rho.min():.3e}, {rho.max():.3e}")
    print(f"|B| max: {Bmag.max():.3e}, |E| max: {Emag.max():.3e}")
    print(f"Approx total field energy: {total_energy:.3e} (assuming SI units)")

    metrics = compute_mode_metrics(ad, "rho", max_mode)
    if metrics:
        for key, val in metrics.items():
            print(f"{key}: {val:.3e}")
    else:
        print("Theta-mode decomposition not detected (rho ndim < 3).")


def fit_growth(times, amps, t0=None, t1=None):
    if len(times) < 2:
        return None
    sel = []
    for t, a in zip(times, amps):
        if a is None or a <= 0:
            continue
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            continue
        sel.append((t, a))
    if len(sel) < 2:
        return None
    ts = np.array([p[0] for p in sel])
    ys = np.log(np.array([p[1] for p in sel]))
    slope, intercept = np.polyfit(ts, ys, 1)
    return float(slope), float(intercept)


def summarize_series(diag_root: Path, fields, max_mode: int, output_csv: Path, fit_start, fit_end, growth_field: str, plot_path: Optional[Path]):
    diags = list_diags(diag_root)
    if not diags:
        raise SystemExit(f"No diag* found under {diag_root}")

    rows = []
    times = []
    for diag in diags:
        ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        t = float(ds.current_time.to_value())
        row = {"diag": diag.name, "time_s": t}
        for field in fields:
            row.update(compute_mode_metrics(ad, field, max_mode))
        rows.append(row)
        times.append(t)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for row in rows for k in row.keys()})
        with output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote mode spectrum CSV to {output_csv}")

    if growth_field:
        amps = [row.get(growth_field) for row in rows]
        fit = fit_growth(times, amps, fit_start, fit_end)
        if fit is not None:
            slope, intercept = fit
            print(
                f"Growth fit for {growth_field}: gamma={slope:.3e}  "
                f"(t in s, fit window {fit_start or 'full'} to {fit_end or 'full'})"
            )
        else:
            print(f"No growth fit available for {growth_field} (insufficient points or zeros).")

    if plot_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for key in rows[0].keys():
                if key == "time_s" or key == "diag":
                    continue
                if not key.endswith("_rms"):
                    continue
                series = [row.get(key) for row in rows]
                if all(val is None for val in series):
                    continue
                ax.plot(times, series, label=key)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mode RMS")
            ax.set_yscale("log")
            ax.legend()
            fig.tight_layout()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)
            print(f"Saved mode spectrum plot to {plot_path}")
        except Exception as exc:
            print(f"Plot skipped: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Analyze WarpX RZ diags (thetaMode).")
    ap.add_argument("--diag-path", default="outputs/warpx", help="Path to WarpX diag root.")
    ap.add_argument("--series", action="store_true", help="Process all diag* and emit CSV time series.")
    ap.add_argument("--max-mode", type=int, default=2, help="Highest theta mode to include (0-based).")
    ap.add_argument(
        "--fields",
        default="rho,Bz",
        help="Comma-separated fields to track modes for in series mode (default: rho,Bz).",
    )
    ap.add_argument("--output-csv", default=None, help="CSV output path for series mode (default: diag_path/mode_spectrum.csv).")
    ap.add_argument("--fit-start", type=float, default=None, help="Start time (s) for growth fit window.")
    ap.add_argument("--fit-end", type=float, default=None, help="End time (s) for growth fit window.")
    ap.add_argument(
        "--growth-field",
        default="rho_m1_rms",
        help="Which column to fit growth rate on (series mode).",
    )
    ap.add_argument(
        "--plot",
        default=None,
        help="Optional path to save mode RMS vs time (log-y).",
    )
    args = ap.parse_args()

    diag_root = Path(args.diag_path)

    if args.series:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
        output_csv = Path(args.output_csv) if args.output_csv else diag_root / "mode_spectrum.csv"
        plot_path = Path(args.plot) if args.plot else None
        summarize_series(diag_root, fields, args.max_mode, output_csv, args.fit_start, args.fit_end, args.growth_field, plot_path)
    else:
        summarize_single(diag_root, args.max_mode)


if __name__ == "__main__":
    main()
