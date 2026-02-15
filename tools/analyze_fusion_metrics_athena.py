#!/usr/bin/env python3
"""
Standard FRC/Fusion metrics for Athena++ 2D VTK series.

Assumptions:
  - 2D cartesian runs from `frc_merge.cpp` (x1 ~ axial z, x2 ~ radial r).
  - VTK contains rho, vel (or v), and Bcc/bcc/b fields.

Metrics (per snapshot):
  - Centroid (r,z) from rho-weighted volume.
  - Flux function proxy A_z(x,y) from integrating Bx along y.
  - Separatrix radius r_s from first A_z sign change at midplane.
  - Doublet separation from A_z O-points along x.
  - Reconnection Ez proxy from ideal E = v x B at X-point.
  - Hall quadrupole proxy from out-of-plane Bz in dense core.

Outputs CSV/JSON time series plus optional PNG plot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def add_athena_vis_path() -> None:
    """Resolve athena_read location from env or local checkout."""
    env_path = os.environ.get("ATHENA_VIS_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "athena-24.0" / "vis" / "python",
            repo_root / "athena-public-version-21.0" / "vis" / "python",
        ]
    )
    for path in candidates:
        if path.exists():
            sys.path.append(str(path))
            return
    raise SystemExit(
        "athena_read not found. Set ATHENA_VIS_PATH or keep athena-24.0/vis/python available."
    )


add_athena_vis_path()
import athena_read  # type: ignore  # noqa: E402


def get_vtk_time(filename: Path) -> Optional[float]:
    """Extract time from the ASCII VTK header."""
    try:
        with filename.open("r", errors="replace") as fh:
            for _ in range(6):
                line = fh.readline()
                if "time=" in line:
                    return float(line.split("time=")[1].split()[0])
    except Exception:
        return None
    return None


def _local_max_mask(field: np.ndarray, size: int = 10) -> np.ndarray:
    """Return a boolean mask of local maxima; SciPy if available, else simple neighbor check."""
    try:
        from scipy.ndimage import maximum_filter  # type: ignore

        return maximum_filter(field, size=size) == field
    except Exception:
        mask = np.ones_like(field, dtype=bool)
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mask &= field >= np.roll(field, shift, axis=(0, 1))
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        return mask


def analyze_snapshot(vtk_file: Path, rho_threshold: float) -> dict:
    x_faces, y_faces, z_faces, data = athena_read.vtk(str(vtk_file))

    rho_in = data.get("rho")
    vel_in = data.get("vel") if "vel" in data else data.get("v")
    b_in = data.get("Bcc")
    if b_in is None:
        b_in = data.get("bcc")
    if b_in is None:
        b_in = data.get("b")
    if rho_in is None or vel_in is None or b_in is None:
        missing = [
            name
            for name, arr in (("rho", rho_in), ("vel/v", vel_in), ("Bcc/bcc/b", b_in))
            if arr is None
        ]
        raise ValueError(f"VTK missing fields: {', '.join(missing)}")

    rho = np.asarray(rho_in[0], dtype=np.float64)
    vel = np.asarray(vel_in[0], dtype=np.float64)
    bcc = np.asarray(b_in[0], dtype=np.float64)

    bx = bcc[:, :, 0]
    by = bcc[:, :, 1]
    bz = bcc[:, :, 2]
    vx = vel[:, :, 0]
    vy = vel[:, :, 1]

    xc = 0.5 * (x_faces[:-1] + x_faces[1:])
    yc = 0.5 * (y_faces[:-1] + y_faces[1:])
    x_grid, y_grid = np.meshgrid(xc, yc)

    # Centroid (cartesian volume element)
    mass = rho
    total_mass = float(np.sum(mass))
    if total_mass > 0:
        x_centroid = float(np.sum(mass * x_grid) / total_mass)
        y_centroid = float(np.sum(mass * y_grid) / total_mass)
    else:
        x_centroid = float("nan")
        y_centroid = float("nan")

    # Flux function proxy Az from integrating Bx along y
    dy = np.diff(y_faces)
    az = np.cumsum(bx * dy[:, None], axis=0)
    az_max = float(np.max(az))

    local_max = _local_max_mask(az, size=10)
    peaks = np.argwhere(local_max & (az > 0.1 * az_max)) if az_max > 0 else []
    o_points = [(xc[i], yc[j], az[j, i]) for j, i in peaks]  # peaks in (y,x)
    o_points.sort(key=lambda p: p[0])  # sort by x

    separation = 0.0
    mid_x_idx = len(xc) // 2
    if len(o_points) >= 2:
        separation = float(abs(o_points[-1][0] - o_points[0][0]))
        x_idx1 = int(np.argmin(np.abs(xc - o_points[0][0])))
        x_idx2 = int(np.argmin(np.abs(xc - o_points[-1][0])))
        if x_idx1 > x_idx2:
            x_idx1, x_idx2 = x_idx2, x_idx1
        mid_x_idx = (x_idx1 + x_idx2) // 2
    elif np.isfinite(x_centroid):
        mid_x_idx = int(np.argmin(np.abs(xc - x_centroid)))

    # Separatrix radius r_s from first Az sign change along +y at midplane x
    az_prof = az[:, mid_x_idx]
    pos_mask = yc >= 0.0
    az_pos = az_prof[pos_mask]
    y_pos = yc[pos_mask]
    signs = np.sign(az_pos)
    cross = np.where(signs[:-1] * signs[1:] < 0)[0]
    r_s = float("nan")
    if cross.size > 0:
        j0 = int(cross[0])
        a0, a1 = az_pos[j0], az_pos[j0 + 1]
        if a1 != a0:
            r_s = float(
                y_pos[j0] - a0 * (y_pos[j0 + 1] - y_pos[j0]) / (a1 - a0)
            )

    # Reconnection Ez proxy (ideal E = v x B) at X-point
    ez = vx * by - vy * bx
    x_point_ez = float(ez[int(np.argmax(az[:, mid_x_idx])), mid_x_idx])

    # Hall quadrupole proxy from out-of-plane Bz in dense core
    rho_max = float(np.max(rho))
    core_mask = rho > (rho_threshold * rho_max) if rho_max > 0 else None
    bz_core = bz[core_mask] if core_mask is not None and np.any(core_mask) else bz
    bz_max = float(np.max(bz_core))
    bz_min = float(np.min(bz_core))
    hall_quad_amp = 0.5 * (bz_max - bz_min)

    t_val = get_vtk_time(vtk_file)
    time_s = float(t_val) if t_val is not None else float("nan")

    return {
        "time_s": time_s,
        "centroid_r_m": abs(y_centroid),
        "centroid_z_m": x_centroid,
        "r_s_m": r_s,
        "psi_max": az_max,
        "num_o_points": len(o_points),
        "separation_m": separation,
        "Et_xpoint": x_point_ez,
        "hall_quadrupole_amp": hall_quad_amp,
        "B_out_max_core": bz_max,
        "B_out_min_core": bz_min,
    }


def process_series(run_dir: Path, vtk_pattern: str, rho_threshold: float) -> list[dict]:
    pattern_path = Path(vtk_pattern)
    pattern = (
        str(run_dir / vtk_pattern)
        if not pattern_path.is_absolute()
        else vtk_pattern
    )
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No VTK files matched: {pattern}")

    results: list[dict] = []
    print(f"Processing {len(files)} VTK files...")
    for f in files:
        try:
            metrics = analyze_snapshot(f, rho_threshold=rho_threshold)
            results.append(metrics)
            print(
                f"  t={metrics['time_s']:.3e}: z_c={metrics['centroid_z_m']:.3f}, "
                f"sep={metrics['separation_m']:.3f}, psi_max={metrics['psi_max']:.3e}"
            )
        except Exception as exc:
            print(f"  Skipping {f}: {exc}")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Compute standardized FRC metrics from Athena++ VTK series."
    )
    ap.add_argument("run_dir", type=Path, help="Athena++ run directory.")
    ap.add_argument(
        "--vtk-pattern",
        default="*.vtk",
        help="Glob for VTK files. If relative, interpreted under run_dir.",
    )
    ap.add_argument("--output-prefix", type=Path, default=None)
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument(
        "--rho-threshold",
        type=float,
        default=0.1,
        help="Core mask threshold as fraction of rho_max for Hall quadrupole proxy.",
    )
    args = ap.parse_args()

    if args.output_prefix:
        if args.output_csv is None:
            args.output_csv = args.output_prefix.with_suffix(".csv")
        if args.output_json is None:
            args.output_json = args.output_prefix.with_suffix(".json")
        if args.plot is None:
            args.plot = args.output_prefix.with_suffix(".png")

    results = process_series(args.run_dir, args.vtk_pattern, args.rho_threshold)
    if not results:
        raise SystemExit("No metrics produced.")

    times = [r["time_s"] for r in results]
    seps = [r["separation_m"] for r in results]
    psi_maxes = [r["psi_max"] for r in results]

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(results[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Wrote metrics CSV to {args.output_csv}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump({"metrics": results}, fh, indent=2, sort_keys=True)
        print(f"Wrote metrics JSON to {args.output_json}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots()
            ax1.plot(times, seps, color="tab:blue")
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("separation (m)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(times, psi_maxes, color="tab:red")
            ax2.set_ylabel("psi_max", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot)
            print(f"Saved metrics plot to {args.plot}")
        except Exception as exc:
            print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
