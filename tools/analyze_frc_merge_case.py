#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def add_athena_vis_path() -> None:
    env_path = os.environ.get("ATHENA_VIS_PATH")
    candidates = []
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
    raise SystemExit("athena_read not found. Set ATHENA_VIS_PATH.")


add_athena_vis_path()
import athena_read  # type: ignore  # noqa: E402


def vtk_time(path: Path) -> float:
    try:
        with path.open("r", errors="replace") as handle:
            for _ in range(5):
                line = handle.readline()
                if "time=" in line:
                    return float(line.split("time=")[1].split()[0])
    except Exception:
        pass
    return 0.0


def compute_divb(x, y, B, x_faces, y_faces):
    Bx = B[:, :, 0]
    By = B[:, :, 1]
    dBx_dx = np.gradient(Bx, x, axis=1, edge_order=1)
    dBy_dy = np.gradient(By, y, axis=0, edge_order=1)
    divb = dBx_dx + dBy_dy
    bmag = np.sqrt(B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2)
    bmag_max = float(np.max(bmag)) if bmag.size else 0.0
    divb_linf = float(np.max(np.abs(divb))) if divb.size else 0.0
    min_dx = min(float(np.min(np.diff(x_faces))), float(np.min(np.diff(y_faces))))
    divb_mean = float(np.mean(np.abs(divb))) if divb.size else 0.0
    divb_rel = (divb_mean * min_dx / bmag_max) if bmag_max > 0 else None
    return divb, divb_linf, divb_rel, divb_mean, min_dx, bmag_max


def find_peaks(profile, x_vals, threshold):
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > threshold:
            peaks.append((profile[i], x_vals[i]))
    peaks.sort(key=lambda item: item[0], reverse=True)
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FRC merge run and emit metrics/plots.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing hst/vtk files.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument(
        "--merge-sep-threshold",
        type=float,
        default=1.0,
        help="Peak separation threshold for declaring merge (in x units).",
    )
    parser.add_argument(
        "--peak-threshold-frac",
        type=float,
        default=0.5,
        help="Peak detection threshold as fraction of (max-min) above min.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hst_file = run_dir / "frc_merge.hst"
    if not hst_file.exists():
        candidates = list(run_dir.glob("*.hst"))
        if candidates:
            hst_file = candidates[0]

    hst_data = None
    if hst_file.exists():
        hst_data = athena_read.hst(str(hst_file))

    vtk_files = sorted(run_dir.glob("*.vtk"))
    num_outputs = len(vtk_files)

    ran_to_completion = bool(hst_data) and num_outputs > 0
    no_nan_in_metrics = True

    mass_rel_drift = None
    total_energy_rel_drift = None
    if hst_data and "mass" in hst_data:
        mass = hst_data["mass"]
        if np.isnan(mass).any():
            no_nan_in_metrics = False
        if mass[0] != 0.0:
            mass_rel_drift = float(np.max(np.abs((mass - mass[0]) / mass[0])))

    if hst_data and "tot-E" in hst_data:
        tot_e = hst_data["tot-E"]
        if np.isnan(tot_e).any():
            no_nan_in_metrics = False
        if tot_e[0] != 0.0:
            total_energy_rel_drift = float(np.max(np.abs((tot_e - tot_e[0]) / tot_e[0])))

    divb_linf = None
    divb_rel = None
    divb_mean = None
    divb_dx_min = None
    divb_bmax = None
    divb_series = []

    separation_series = []
    merged_time = None
    merge_threshold = args.merge_sep_threshold
    saw_double_peak = False
    peak_sep_initial = None
    peak_sep_final = None

    profile_y_index = None
    profile_y_value = None

    for vtk_file in vtk_files:
        try:
            x_faces, y_faces, _, data = athena_read.vtk(str(vtk_file))
            x = 0.5 * (x_faces[:-1] + x_faces[1:])
            y = 0.5 * (y_faces[:-1] + y_faces[1:])
            rho = data["rho"][0]
            if "Bcc" in data:
                B = data["Bcc"][0]
            elif "b" in data:
                B = data["b"][0]
            else:
                raise RuntimeError(f"Missing magnetic field in {vtk_file}")

            divb, divb_linf_snap, divb_rel_snap, divb_mean_snap, divb_dx_min_snap, divb_bmax_snap = compute_divb(
                x, y, B, x_faces, y_faces
            )
            if np.isnan(divb).any() or np.isnan(rho).any():
                no_nan_in_metrics = False

            t = vtk_time(vtk_file)
            divb_series.append((t, divb_rel_snap if divb_rel_snap is not None else 0.0))

            y_idx = int(np.argmin(np.abs(y)))
            if profile_y_index is None:
                profile_y_index = y_idx
                profile_y_value = float(y[y_idx])
            profile = rho[y_idx, :]
            p_min = float(np.min(profile))
            p_max = float(np.max(profile))
            threshold = p_min + args.peak_threshold_frac * (p_max - p_min)
            peaks = find_peaks(profile, x, threshold)

            if len(peaks) >= 2:
                saw_double_peak = True
                sep = abs(peaks[0][1] - peaks[1][1])
            else:
                sep = 0.0

            separation_series.append((t, sep))
            if peak_sep_initial is None and len(peaks) >= 2:
                peak_sep_initial = sep
            peak_sep_final = sep

            if merged_time is None and saw_double_peak and sep <= merge_threshold:
                merged_time = t

            if vtk_file == vtk_files[-1]:
                divb_linf = divb_linf_snap
                divb_rel = divb_rel_snap
                divb_mean = divb_mean_snap
                divb_dx_min = divb_dx_min_snap
                divb_bmax = divb_bmax_snap
        except Exception:
            no_nan_in_metrics = False

    merged_time_exists = merged_time is not None
    merged_time_frac = None
    if merged_time is not None and separation_series:
        end_time = max(t for t, _ in separation_series)
        if end_time > 0.0:
            merged_time_frac = merged_time / end_time

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": num_outputs,
        "no_nan_in_metrics": no_nan_in_metrics,
        "mass_rel_drift": mass_rel_drift,
        "total_energy_rel_drift": total_energy_rel_drift,
        "divB_Linf": divb_linf,
        "divB_rel": divb_rel,
        "divB_mean_abs": divb_mean,
        "divB_dx_min": divb_dx_min,
        "divB_bmax": divb_bmax,
        "divB_region": "full_domain",
        "divB_rel_def": "mean|divB| * dx_min / max|B| over full domain",
        "merge_axis": "x1",
        "merge_profile_y_index": profile_y_index,
        "merge_profile_y_value": profile_y_value,
        "merge_peak_threshold_frac": args.peak_threshold_frac,
        "merge_threshold": merge_threshold,
        "merged_time": merged_time,
        "merged_time_exists": merged_time_exists,
        "merged_time_frac": merged_time_frac,
        "peak_separation_initial": peak_sep_initial,
        "peak_separation_final": peak_sep_final,
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    if hst_data and "time" in hst_data:
        time = hst_data["time"]
        plt.figure(figsize=(6, 4))
        if "tot-E" in hst_data:
            plt.plot(time, hst_data["tot-E"], label="tot-E")
        if "1-ME" in hst_data and "2-ME" in hst_data and "3-ME" in hst_data:
            me = hst_data["1-ME"] + hst_data["2-ME"] + hst_data["3-ME"]
            plt.plot(time, me, label="magnetic")
        if "1-KE" in hst_data and "2-KE" in hst_data and "3-KE" in hst_data:
            ke = hst_data["1-KE"] + hst_data["2-KE"] + hst_data["3-KE"]
            plt.plot(time, ke, label="kinetic")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energy Budget")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "energy_budget.png")
        plt.close()

    if divb_series:
        divb_series.sort(key=lambda item: item[0])
        times = [t for t, _ in divb_series]
        vals = [v for _, v in divb_series]
        plt.figure(figsize=(6, 4))
        plt.plot(times, vals, marker="o")
        plt.xlabel("Time")
        plt.ylabel("divB_rel")
        plt.title("divB Relative (mean|divB|) vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "divb_vs_time.png")
        plt.close()

    if separation_series:
        separation_series.sort(key=lambda item: item[0])
        times = [t for t, _ in separation_series]
        seps = [s for _, s in separation_series]
        plt.figure(figsize=(6, 4))
        plt.plot(times, seps, marker="o")
        plt.axhline(merge_threshold, color="red", linestyle="--", label="merge_threshold")
        plt.xlabel("Time")
        plt.ylabel("Peak separation")
        plt.title("Merge Indicator (Peak Separation)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "merge_indicator_vs_time.png")
        plt.close()

    if vtk_files:
        last_file = vtk_files[-1]
        try:
            x_faces, y_faces, _, data = athena_read.vtk(str(last_file))
            x = 0.5 * (x_faces[:-1] + x_faces[1:])
            y = 0.5 * (y_faces[:-1] + y_faces[1:])
            rho = data["rho"][0]
            X, Y = np.meshgrid(x, y)
            plt.figure(figsize=(6, 4))
            plt.pcolormesh(X, Y, rho, shading="auto")
            plt.colorbar(label="rho")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Density Snapshot")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(plots_dir / "density_snapshot.png")
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
