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
    b1 = B[:, :, 0]
    b2 = B[:, :, 1]
    db1_dx = np.gradient(b1, x, axis=1, edge_order=1)
    db2_dy = np.gradient(b2, y, axis=0, edge_order=1)
    divb = db1_dx + db2_dy
    bmag = np.sqrt(B[:, :, 0] ** 2 + B[:, :, 1] ** 2 + B[:, :, 2] ** 2)
    bmag_max = float(np.max(bmag)) if bmag.size else 0.0
    divb_mean = float(np.mean(np.abs(divb))) if divb.size else 0.0
    min_dx = min(float(np.min(np.diff(x_faces))), float(np.min(np.diff(y_faces))))
    divb_rel = (divb_mean * min_dx / bmag_max) if bmag_max > 0 else None
    return divb_rel


def compute_psi_from_b1(x2, b1):
    pos_idx = np.where(x2 >= 0.0)[0]
    if pos_idx.size == 0:
        return None, None, None
    r = x2[pos_idx]
    neg_idx = np.array([int(np.argmin(np.abs(x2 + x2[i]))) for i in pos_idx])
    b1_sym = 0.5 * (b1[pos_idx, :] + b1[neg_idx, :])
    rbz = r[:, None] * b1_sym
    psi = np.zeros_like(rbz)
    for i in range(1, r.size):
        dr = r[i] - r[i - 1]
        psi[i] = psi[i - 1] + 0.5 * (rbz[i] + rbz[i - 1]) * dr
    return r, psi, rbz


def psi_indicator(rbz, psi, eps_frac):
    psi_peak_to_edge_max = 0.0
    psi_closed_exists = False
    if psi.size == 0:
        return None, False
    for idx in range(psi.shape[1]):
        psi_col = psi[:, idx]
        psi_edge = psi_col[-1]
        delta = float(np.max(np.abs(psi_col - psi_edge))) if psi_col.size else 0.0
        scale = float(np.max(np.abs(psi_col))) if psi_col.size else 0.0
        denom = max(scale, 1.0e-12)
        psi_peak_to_edge_max = max(psi_peak_to_edge_max, delta / denom)

        rbz_col = rbz[:, idx]
        rbz_scale = float(np.max(np.abs(rbz_col))) if rbz_col.size else 0.0
        rbz_eps = rbz_scale * eps_frac
        mask = np.abs(rbz_col) > rbz_eps
        if np.count_nonzero(mask) >= 2:
            signs = np.sign(rbz_col[mask])
            if np.any(signs[:-1] * signs[1:] < 0):
                psi_closed_exists = True
    return psi_peak_to_edge_max, psi_closed_exists


def longest_true_run(flags):
    max_run = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def longest_monotonic_run(diffs: np.ndarray, direction: float, tol: float = 0.0) -> int:
    if diffs.size == 0:
        return 0
    if direction >= 0:
        ok = diffs >= -tol
    else:
        ok = diffs <= tol
    max_run = 0
    current = 0
    for flag in ok:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A2 formation + translation gate.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing hst/vtk files.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument(
        "--formation-contrast-threshold",
        type=float,
        default=1.3,
        help="Density contrast threshold (peak / background) required for formation.",
    )
    parser.add_argument(
        "--formation-persist",
        type=int,
        default=2,
        help="Number of consecutive outputs required to declare formation.",
    )
    parser.add_argument(
        "--psi-peak-to-edge-threshold",
        type=float,
        default=0.05,
        help="Normalized psi peak-to-edge threshold required for formation.",
    )
    parser.add_argument(
        "--psi-eps-frac",
        type=float,
        default=1e-3,
        help="Fraction of max |r*Bz| used to filter psi sign changes.",
    )
    parser.add_argument(
        "--axis-window-frac",
        type=float,
        default=0.1,
        help="Fraction of |x2|max used for axis averaging window.",
    )
    parser.add_argument(
        "--edge-window-frac",
        type=float,
        default=0.8,
        help="Fraction of |x2|max used for edge averaging window.",
    )
    parser.add_argument(
        "--b1-eps-frac",
        type=float,
        default=1e-3,
        help="Fraction of max |B1| used as sign threshold in reversal detection.",
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

    hst_data = athena_read.hst(str(hst_file)) if hst_file.exists() else None
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

    reversal_series = []
    contrast_series = []
    psi_peak_series = []
    psi_closed_series = []
    centroid_series = []
    peak_rho_series = []
    divb_series = []
    formation_time = None

    first_snapshot = None
    last_snapshot = None
    x1_edges = None
    x2_edges = None
    x1c = None
    x2c = None
    psi_snapshot = None
    psi_snapshot_r = None
    psi_snapshot_x1 = None
    psi_snapshot_time = None
    psi_snapshot_score = None

    for vtk_file in vtk_files:
        try:
            x1_faces, x2_faces, _, data = athena_read.vtk(str(vtk_file))
            x1 = 0.5 * (x1_faces[:-1] + x1_faces[1:])
            x2 = 0.5 * (x2_faces[:-1] + x2_faces[1:])
            rho = data["rho"][0]
            if "Bcc" in data:
                B = data["Bcc"][0]
            elif "b" in data:
                B = data["b"][0]
            else:
                raise RuntimeError(f"Missing magnetic field in {vtk_file}")

            if np.isnan(rho).any() or np.isnan(B).any():
                no_nan_in_metrics = False

            b1 = B[:, :, 0]
            y_abs = np.abs(x2)
            y_max = float(np.max(y_abs)) if y_abs.size else 0.0
            axis_mask = y_abs <= args.axis_window_frac * y_max if y_max > 0 else None
            edge_mask = y_abs >= args.edge_window_frac * y_max if y_max > 0 else None

            if axis_mask is None or not np.any(axis_mask):
                axis_idx = int(np.argmin(np.abs(x2)))
                axis_vals = b1[axis_idx, :]
            else:
                axis_vals = np.mean(b1[axis_mask, :], axis=0)

            if edge_mask is None or not np.any(edge_mask):
                edge_idx = int(np.argmax(np.abs(x2)))
                edge_vals = b1[edge_idx, :]
            else:
                edge_vals = np.mean(b1[edge_mask, :], axis=0)

            b1_scale = float(np.max(np.abs(b1))) if b1.size else 0.0
            b1_eps = args.b1_eps_frac * b1_scale
            reversal = (axis_vals * edge_vals < 0.0) & (np.abs(axis_vals) > b1_eps) & (
                np.abs(edge_vals) > b1_eps
            )
            reversal_fraction = float(np.count_nonzero(reversal) / reversal.size) if reversal.size else 0.0

            t = vtk_time(vtk_file)
            reversal_series.append((t, reversal_fraction))

            dx1 = np.diff(x1_faces)
            dx2 = np.diff(x2_faces)
            cell_area = dx2[:, None] * dx1[None, :]
            weight = rho * cell_area
            total = float(np.sum(weight))
            centroid_x1 = float(np.sum(weight * x1[None, :]) / total) if total > 0 else None
            centroid_series.append((t, centroid_x1))

            peak_rho = float(np.max(rho)) if rho.size else 0.0
            peak_rho_series.append((t, peak_rho))
            rho_bg = float(np.percentile(rho, 10)) if rho.size else 0.0
            rho_bg = max(rho_bg, 1.0e-12)
            core_contrast = peak_rho / rho_bg
            contrast_series.append((t, core_contrast))

            r_grid, psi, rbz = compute_psi_from_b1(x2, b1)
            if psi is not None and rbz is not None:
                psi_peak_to_edge, psi_closed_exists = psi_indicator(rbz, psi, args.psi_eps_frac)
                if psi_peak_to_edge is not None:
                    psi_peak_series.append((t, psi_peak_to_edge))
                    psi_closed_series.append((t, psi_closed_exists))
                    if psi_snapshot_score is None or psi_peak_to_edge > psi_snapshot_score:
                        psi_snapshot = psi.copy()
                        psi_snapshot_r = r_grid.copy()
                        psi_snapshot_x1 = x1.copy()
                        psi_snapshot_time = t
                        psi_snapshot_score = psi_peak_to_edge

            divb_rel = compute_divb(x1, x2, B, x1_faces, x2_faces)
            divb_series.append((t, divb_rel if divb_rel is not None else 0.0))

            if first_snapshot is None:
                first_snapshot = rho.copy()
                x1_edges = x1_faces
                x2_edges = x2_faces
                x1c = x1
                x2c = x2
            last_snapshot = rho.copy()
            x1_edges = x1_faces
            x2_edges = x2_faces
            x1c = x1
            x2c = x2
        except Exception:
            no_nan_in_metrics = False

    formation_reversal_fraction_max = None
    formation_contrast_max = None
    psi_peak_to_edge_max = None
    psi_closed_exists = False
    psi_closed_persist_outputs = None
    psi_closed_persist_frac = None
    formation_time_frac = None
    frc_formed = False
    if reversal_series:
        formation_reversal_fraction_max = max(val for _, val in reversal_series)
    if contrast_series:
        formation_contrast_max = max(val for _, val in contrast_series)
    if psi_peak_series:
        psi_peak_to_edge_max = max(val for _, val in psi_peak_series)
        psi_closed_exists = any(val for _, val in psi_closed_series)
        end_time = max(t for t, _ in psi_peak_series)
        values = [val for _, val in psi_peak_series]
        closed_vals = [val for _, val in psi_closed_series]
        psi_closed_persist_outputs = longest_true_run(closed_vals)
        if closed_vals:
            psi_closed_persist_frac = psi_closed_persist_outputs / len(closed_vals)
        times = [t for t, _ in psi_peak_series]
        persist = max(1, args.formation_persist)
        for i in range(0, len(values) - persist + 1):
            window = values[i : i + persist]
            closed_window = closed_vals[i : i + persist]
            if all(closed_window) and all(v >= args.psi_peak_to_edge_threshold for v in window):
                formation_time = times[i]
                frc_formed = True
                break
        if formation_time is not None and end_time > 0.0:
            formation_time_frac = formation_time / end_time

    centroid_shift = None
    centroid_shift_abs = None
    centroid_shift_frac = None
    translation_monotonic_run = 0
    translation_monotonic_frac = None
    if centroid_series and x1c is not None:
        centroids = np.array([c for _, c in centroid_series if c is not None], dtype=float)
        times = np.array([t for t, c in centroid_series if c is not None], dtype=float)
        if centroids.size:
            centroid_shift = float(centroids[-1] - centroids[0])
            centroid_shift_abs = float(abs(centroid_shift))
            domain_len = float(x1c[-1] - x1c[0]) if x1c.size else None
            if domain_len and domain_len > 0.0:
                centroid_shift_frac = centroid_shift_abs / domain_len
            diffs = np.diff(centroids)
            direction = centroid_shift if centroid_shift is not None else 0.0
            translation_monotonic_run = longest_monotonic_run(diffs, direction, tol=0.0)
            if diffs.size:
                if direction >= 0:
                    translation_monotonic_frac = float(np.mean(diffs >= 0.0))
                else:
                    translation_monotonic_frac = float(np.mean(diffs <= 0.0))

    peak_rho_ratio_end = None
    if peak_rho_series:
        peak_vals = np.array([val for _, val in peak_rho_series], dtype=float)
        if peak_vals.size and peak_vals[0] != 0.0:
            peak_rho_ratio_end = float(peak_vals[-1] / peak_vals[0])

    divb_rel = None
    if divb_series:
        divb_rel = divb_series[-1][1]

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": num_outputs,
        "no_nan_in_metrics": no_nan_in_metrics,
        "mass_rel_drift": mass_rel_drift,
        "total_energy_rel_drift": total_energy_rel_drift,
        "formation_contrast_threshold": args.formation_contrast_threshold,
        "formation_persist": args.formation_persist,
        "psi_peak_to_edge_threshold": args.psi_peak_to_edge_threshold,
        "psi_eps_frac": args.psi_eps_frac,
        "formation_reversal_fraction_max": formation_reversal_fraction_max,
        "formation_contrast_max": formation_contrast_max,
        "psi_peak_to_edge_max": psi_peak_to_edge_max,
        "psi_closed_exists": psi_closed_exists,
        "psi_closed_persist_outputs": psi_closed_persist_outputs,
        "psi_closed_persist_frac": psi_closed_persist_frac,
        "formation_time": formation_time,
        "formation_time_frac": formation_time_frac,
        "frc_formed": frc_formed,
        "centroid_shift": centroid_shift,
        "centroid_shift_abs": centroid_shift_abs,
        "centroid_shift_frac": centroid_shift_frac,
        "translation_monotonic_run": translation_monotonic_run,
        "translation_monotonic_frac": translation_monotonic_frac,
        "peak_rho_ratio_end": peak_rho_ratio_end,
        "divB_rel": divb_rel,
        "formation_indicator": "psi peak-to-edge from B1-integrated flux",
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if psi_peak_series:
        t_vals = [t for t, _ in psi_peak_series]
        psi_vals = [v for _, v in psi_peak_series]
        plt.figure(figsize=(6, 4))
        plt.plot(t_vals, psi_vals, marker="o")
        plt.axhline(args.psi_peak_to_edge_threshold, color="r", linestyle="--", label="threshold")
        if formation_time is not None:
            plt.axvline(formation_time, color="k", linestyle=":", label="formation_time")
        plt.xlabel("Time")
        plt.ylabel("psi peak-to-edge (normalized)")
        plt.title("Psi Formation Indicator vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "psi_indicator_vs_time.png")
        plt.close()

    if contrast_series:
        t_vals = [t for t, _ in contrast_series]
        frac_vals = [v for _, v in contrast_series]
        plt.figure(figsize=(6, 4))
        plt.plot(t_vals, frac_vals, marker="o")
        plt.axhline(args.formation_contrast_threshold, color="r", linestyle="--", label="threshold")
        plt.xlabel("Time")
        plt.ylabel("Core density contrast")
        plt.title("Density Contrast vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "density_contrast_vs_time.png")
        plt.close()

    if centroid_series:
        t_vals = [t for t, c in centroid_series if c is not None]
        c_vals = [c for _, c in centroid_series if c is not None]
        if t_vals and c_vals:
            plt.figure(figsize=(6, 4))
            plt.plot(t_vals, c_vals, marker="o")
            plt.xlabel("Time")
            plt.ylabel("Centroid x1 (axial)")
            plt.title("Centroid Shift vs Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "centroid_z_vs_time.png")
            plt.close()

    if first_snapshot is not None and last_snapshot is not None and x1_edges is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        axes[0].pcolormesh(x1_edges, x2_edges, first_snapshot, shading="auto")
        axes[0].set_title("rho (early)")
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")
        axes[1].pcolormesh(x1_edges, x2_edges, last_snapshot, shading="auto")
        axes[1].set_title("rho (late)")
        axes[1].set_xlabel("x1")
        fig.tight_layout()
        fig.savefig(plots_dir / "snapshot_pre_post.png")
        plt.close(fig)

    if psi_snapshot is not None and psi_snapshot_r is not None and psi_snapshot_x1 is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        X, R = np.meshgrid(psi_snapshot_x1, psi_snapshot_r)
        levels = 16
        contour = ax.contour(X, R, psi_snapshot, levels=levels)
        ax.clabel(contour, inline=True, fontsize=7)
        ax.set_xlabel("x1 (axial)")
        ax.set_ylabel("r (|x2|)")
        title = "Psi contours"
        if psi_snapshot_time is not None:
            title = f"Psi contours (t={psi_snapshot_time:.3f})"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plots_dir / "psi_contours_snapshot.png")
        plt.close(fig)

    if divb_series:
        t_vals = [t for t, _ in divb_series]
        v_vals = [v for _, v in divb_series]
        plt.figure(figsize=(6, 4))
        plt.plot(t_vals, v_vals, marker="o")
        plt.xlabel("Time")
        plt.ylabel("divB_rel")
        plt.title("divB_rel vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "divb_vs_time.png")
        plt.close()


if __name__ == "__main__":
    main()
