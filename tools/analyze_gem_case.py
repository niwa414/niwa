#!/usr/bin/env python3
import argparse
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


def load_vtk_fields(path: Path):
    x_faces, y_faces, z_faces, data = athena_read.vtk(str(path))
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    if "Bcc" in data:
        B = data["Bcc"][0]
        V = data["vel"][0]
    elif "b" in data:
        B = data["b"][0]
        V = data["v"][0]
    else:
        raise RuntimeError(f"Missing magnetic field in {path}")
    return x, y, B, V


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


def reconnection_rate(x, y, B, V):
    Bx = B[:, :, 0]
    By = B[:, :, 1]
    Vx = V[:, :, 0]
    Vy = V[:, :, 1]
    Ez = Vx * By - Vy * Bx
    y_2d = y[:, np.newaxis]
    mask_center = np.abs(y_2d) < 0.05
    if not np.any(mask_center):
        idx_center = int(np.argmin(np.abs(y)))
        mask_center = np.zeros_like(y_2d, dtype=bool)
        mask_center[idx_center] = True
    mask_up = y_2d > (0.3 * float(y.max()))
    if not np.any(mask_up):
        mask_up = np.zeros_like(y_2d, dtype=bool)
        mask_up[-1] = True
    mask_center = np.repeat(mask_center, len(x), axis=1)
    mask_up = np.repeat(mask_up, len(x), axis=1)
    Ez_center = float(np.mean(Ez[mask_center]))
    Bx_up = float(np.mean(Bx[mask_up]))
    if Bx_up == 0.0:
        return 0.0
    return Ez_center / Bx_up


add_athena_vis_path()
import athena_read  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hall-GEM run and emit metrics/plots.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing hst/vtk files.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--athinput", default=None, help="Optional Athena++ input file for audit fields.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hst_file = run_dir / "hall_gem.hst"
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
    reconnection_series = []

    for vtk_file in vtk_files:
        try:
            x_faces, y_faces, _, data = athena_read.vtk(str(vtk_file))
            x = 0.5 * (x_faces[:-1] + x_faces[1:])
            y = 0.5 * (y_faces[:-1] + y_faces[1:])
            if "Bcc" in data:
                B = data["Bcc"][0]
                V = data["vel"][0]
            elif "b" in data:
                B = data["b"][0]
                V = data["v"][0]
            else:
                raise RuntimeError(f"Missing magnetic field in {vtk_file}")
            (
                divb,
                divb_linf_snap,
                divb_rel_snap,
                divb_mean_snap,
                divb_dx_min_snap,
                divb_bmax_snap,
            ) = compute_divb(x, y, B, x_faces, y_faces)
            if np.isnan(divb).any():
                no_nan_in_metrics = False
            rate = reconnection_rate(x, y, B, V)
            if np.isnan(rate):
                no_nan_in_metrics = False
            t = vtk_time(vtk_file)
            divb_series.append((t, divb_rel_snap if divb_rel_snap is not None else 0.0))
            reconnection_series.append((t, rate))
            if vtk_file == vtk_files[-1]:
                divb_linf = divb_linf_snap
                divb_rel = divb_rel_snap
                divb_mean = divb_mean_snap
                divb_dx_min = divb_dx_min_snap
                divb_bmax = divb_bmax_snap
        except Exception:
            no_nan_in_metrics = False

    reconnection_rate_peak = None
    reconnection_rate_peak_time = None
    if reconnection_series:
        reconnection_series.sort(key=lambda item: item[0])
        peak_time, peak_rate = max(reconnection_series, key=lambda item: abs(item[1]))
        reconnection_rate_peak = float(peak_rate)
        reconnection_rate_peak_time = float(peak_time)

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
        "reconnection_rate_peak": reconnection_rate_peak,
        "reconnection_rate_peak_raw": reconnection_rate_peak,
        "reconnection_rate_peak_time": reconnection_rate_peak_time,
        "reconnection_rate_norm_def": "Ez=Vx*By - Vy*Bx; rate=Ez_center/Bx_up; center=|y|<0.05 (fallback row), upstream=y>0.3*y_max (fallback top row)",
    }

    eta_hall_value = None
    if args.athinput:
        input_path = Path(args.athinput)
        if input_path.exists():
            for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "eta_hall" not in line:
                    continue
                stripped = line.split("#", 1)[0]
                if "eta_hall" not in stripped:
                    continue
                parts = stripped.split("=", 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                if key != "eta_hall":
                    continue
                try:
                    eta_hall_value = float(parts[1].split()[0])
                except Exception:
                    eta_hall_value = None
                break
    metrics["eta_hall_value"] = eta_hall_value
    metrics["hall_enabled"] = (
        eta_hall_value is not None and abs(float(eta_hall_value)) > 0.0
    )
    if args.athinput:
        metrics["athinput_path"] = str(args.athinput)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json

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
        plt.title("divB Relative (Linf) vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "divb_vs_time.png")
        plt.close()

    if vtk_files:
        last_file = vtk_files[-1]
        try:
            x, y, B, _ = load_vtk_fields(last_file)
            X, Y = np.meshgrid(x, y)
            Bz = B[:, :, 2]
            limit = max(abs(float(Bz.min())), abs(float(Bz.max())))
            if limit == 0.0:
                limit = 0.1
            plt.figure(figsize=(6, 4))
            plt.pcolormesh(X, Y, Bz, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
            plt.colorbar(label="Bz")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Current Sheet Snapshot (Bz)")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(plots_dir / "current_sheet_snapshot.png")
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
