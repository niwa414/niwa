#!/usr/bin/env python3
import argparse
import json
import os
import re
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


def parse_input_params(path: Path) -> dict[str, str]:
    params = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        params[key] = value
    return params


def load_waveform(path: Path, col: int) -> list[float]:
    if not path.exists():
        return []
    data = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "t":
            continue
        if len(parts) <= col:
            continue
        try:
            data.append(float(parts[col]))
        except ValueError:
            continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compression gate run and emit metrics/plots.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing hst/vtk files.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--input", required=True, help="Input file path for driver metadata.")
    parser.add_argument(
        "--threshold-frac",
        type=float,
        default=0.5,
        help="Threshold fraction for compression indicator (rho_min + frac*(rho_max-rho_min)).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hst_file = run_dir / "bext_emf_smoke.hst"
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

    compression_radius_series = []
    compression_length_series = []
    compression_length_initial = None
    compression_length_final = None
    compression_area_initial = None
    compression_area_final = None
    compression_radius_initial = None
    compression_radius_final = None
    compression_ratio = None
    compression_radius_min = None
    compression_time_min = None
    threshold_value = None
    cell_area = None

    for idx, vtk_file in enumerate(vtk_files):
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

            if idx == 0:
                rho_min = float(np.min(rho))
                rho_max = float(np.max(rho))
                threshold_value = rho_min + args.threshold_frac * (rho_max - rho_min)
                dx = np.diff(x_faces)
                dy = np.diff(y_faces)
                cell_area = np.outer(dy, dx)

            mask = rho > threshold_value if threshold_value is not None else rho > 0.0
            if np.any(mask):
                x_mask = np.broadcast_to(x[np.newaxis, :], rho.shape)[mask]
                x_min = float(np.min(x_mask))
                x_max = float(np.max(x_mask))
                length = x_max - x_min
                if cell_area is not None:
                    area = float(np.sum(cell_area[mask]))
                else:
                    area = 0.0
                radius = np.sqrt(area / np.pi) if area > 0.0 else 0.0
            else:
                length = 0.0
                area = 0.0
                radius = 0.0

            compression_length_series.append((t, length))
            compression_radius_series.append((t, radius))
            if idx == 0:
                compression_length_initial = length
                compression_area_initial = area
                compression_radius_initial = radius
            if idx == len(vtk_files) - 1:
                compression_length_final = length
                compression_area_final = area
                compression_radius_final = radius
                divb_linf = divb_linf_snap
                divb_rel = divb_rel_snap
                divb_mean = divb_mean_snap
                divb_dx_min = divb_dx_min_snap
                divb_bmax = divb_bmax_snap
        except Exception:
            no_nan_in_metrics = False

    if compression_radius_series:
        min_entry = min(compression_radius_series, key=lambda item: item[1])
        compression_time_min = min_entry[0]
        compression_radius_min = min_entry[1]

    if compression_radius_initial and compression_radius_min and compression_radius_min > 0.0:
        compression_ratio = compression_radius_initial / compression_radius_min

    driver_meta = {}
    input_path = Path(args.input)
    if input_path.exists():
        params = parse_input_params(input_path)
        driver_meta = {
            "driver_profile": params.get("b_ext_profile"),
            "driver_mirror_ratio": params.get("mirror_ratio"),
            "driver_waveform_file": params.get("b_mirror_delta_waveform"),
            "driver_waveform_column": params.get("b_mirror_delta_column"),
            "driver_waveform_scale": params.get("b_mirror_delta_scale"),
            "driver_waveform_bias": params.get("b_mirror_delta_bias"),
            "apply_bext_emf": params.get("apply_bext_emf"),
            "piston_bc": params.get("piston_bc"),
            "piston_use_depth": params.get("piston_use_depth"),
            "piston_depth_max": params.get("piston_depth_max"),
            "piston_waveform": params.get("piston_waveform"),
            "piston_waveform_column": params.get("piston_waveform_column"),
            "piston_waveform_scale": params.get("piston_waveform_scale"),
            "piston_waveform_bias": params.get("piston_waveform_bias"),
            "piston_waveform_tshift": params.get("piston_waveform_tshift"),
        }
        waveform_file = params.get("b_mirror_delta_waveform")
        col_str = params.get("b_mirror_delta_column", "1")
        scale_str = params.get("b_mirror_delta_scale", "0.0")
        bias_str = params.get("b_mirror_delta_bias", "0.0")
        try:
            col_index = max(0, int(col_str) - 1)
            scale = float(scale_str)
            bias = float(bias_str)
        except ValueError:
            col_index = 1
            scale = 0.0
            bias = 0.0
        if waveform_file:
            wf_path = Path(waveform_file)
            if not wf_path.is_absolute():
                wf_path = (input_path.parent.parent / waveform_file).resolve()
            data = load_waveform(wf_path, col_index)
            if data:
                driver_meta["driver_waveform_frac_max"] = max(data)
                driver_meta["driver_bext_peak_est"] = bias + scale * max(data)

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
        "compression_indicator": "area_equiv_radius",
        "compression_threshold_frac": args.threshold_frac,
        "compression_threshold_value": threshold_value,
        "compression_length_initial": compression_length_initial,
        "compression_length_final": compression_length_final,
        "compression_area_initial": compression_area_initial,
        "compression_area_final": compression_area_final,
        "compression_radius_initial": compression_radius_initial,
        "compression_radius_final": compression_radius_final,
        "compression_radius_min": compression_radius_min,
        "compression_time_min": compression_time_min,
        "compression_ratio": compression_ratio,
    }
    metrics.update(driver_meta)

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

    if compression_radius_series:
        compression_radius_series.sort(key=lambda item: item[0])
        times = [t for t, _ in compression_radius_series]
        vals = [v for _, v in compression_radius_series]
        plt.figure(figsize=(6, 4))
        plt.plot(times, vals, marker="o")
        plt.xlabel("Time")
        plt.ylabel("Effective radius")
        plt.title("Compression Indicator vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "compression_indicator_vs_time.png")
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
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Density Snapshot")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(plots_dir / "density_snapshot.png")
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
