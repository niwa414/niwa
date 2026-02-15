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


def parse_athinput(path: Path) -> dict:
    config = {}
    block = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("<comment>"):
            continue
        if line.startswith("</comment>"):
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line.strip("<>/").strip()
            config.setdefault(block, {})
            continue
        if "=" not in line or block is None:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.split("#")[0].split("!")[0].strip()
        config[block][key] = val
    return config


def get_float(cfg: dict, block: str, key: str, default: float | None) -> float | None:
    try:
        return float(cfg.get(block, {}).get(key, default))
    except Exception:
        return default


def get_str(cfg: dict, block: str, key: str, default: str | None) -> str | None:
    val = cfg.get(block, {}).get(key, default)
    return str(val) if val is not None else None


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


def to_array(series):
    if series is None:
        return None
    return np.array(series, dtype=float)


def trapz_or_none(t, y):
    if t is None or y is None:
        return None
    if len(t) < 2 or len(t) != len(y):
        return None
    return float(np.trapz(y, t))


def build_uniform_axis(xmin: float | None, xmax: float | None, n: int | None):
    if xmin is None or xmax is None or n is None:
        return None, None
    if n <= 0:
        return None, None
    edges = np.linspace(xmin, xmax, n + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def r_wall_profile(cfg: dict):
    profile = (get_str(cfg, "problem", "cyl_wall_profile", "none") or "none").lower()
    rmax = get_float(cfg, "problem", "cyl_wall_rmax", None)
    x1_min = get_float(cfg, "problem", "cyl_wall_x1_min", None)
    x1_max = get_float(cfg, "problem", "cyl_wall_x1_max", None)
    x1_step = get_float(cfg, "problem", "cyl_wall_x1_step", None)
    r0 = get_float(cfg, "problem", "cyl_wall_r0", None)
    r1 = get_float(cfg, "problem", "cyl_wall_r1", None)

    def r_wall(x1):
        if profile == "step" and x1_step is not None:
            if r0 is None or r1 is None:
                return rmax if rmax is not None else 0.0
            return r0 if x1 < x1_step else r1
        if profile == "linear" and x1_min is not None and x1_max is not None:
            if r0 is None or r1 is None:
                return rmax if rmax is not None else 0.0
            if x1_max <= x1_min:
                return r0
            if x1 <= x1_min:
                return r0
            if x1 >= x1_max:
                return r1
            t = (x1 - x1_min) / (x1_max - x1_min)
            return r0 + t * (r1 - r0)
        return rmax if rmax is not None else 0.0

    meta = {
        "profile": profile,
        "rmax": rmax,
        "x1_min": x1_min,
        "x1_max": x1_max,
        "x1_step": x1_step,
        "r0": r0,
        "r1": r1,
    }
    return r_wall, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A4 EB wall gate.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing hst/vtk files.")
    parser.add_argument("--athinput", required=True, help="Athena++ input file path.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_athinput(Path(args.athinput))
    r_wall_fn, wall_meta = r_wall_profile(cfg)
    wall_enabled = (get_str(cfg, "problem", "cyl_wall", "false") or "").lower() == "true"
    piston_enabled = (
        (get_str(cfg, "problem", "piston_bc", "false") or "").lower() == "true"
    )
    piston_use_depth = (
        (get_str(cfg, "problem", "piston_use_depth", "false") or "").lower() == "true"
    )
    piston_depth_max = get_float(cfg, "problem", "piston_depth_max", None)
    piston_waveform = get_str(cfg, "problem", "piston_waveform", None)
    piston_waveform_scale = get_float(cfg, "problem", "piston_waveform_scale", 1.0)
    piston_waveform_bias = get_float(cfg, "problem", "piston_waveform_bias", 0.0)
    piston_waveform_tshift = get_float(cfg, "problem", "piston_waveform_tshift", 0.0)
    piston_waveform_column = get_float(cfg, "problem", "piston_waveform_column", 1.0)
    piston_waveform_frac_max = None
    piston_depth_est = None
    piston_metric = None
    piston_metric_source = None

    if piston_waveform:
        col_index = 0
        try:
            col_index = max(0, int(float(piston_waveform_column)) - 1)
        except Exception:
            col_index = 0
        wf_path = Path(piston_waveform)
        if not wf_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            wf_path = (repo_root / piston_waveform).resolve()
            if not wf_path.exists():
                wf_path = (Path(args.athinput).parent / piston_waveform).resolve()
        data = load_waveform(wf_path, col_index)
        if data:
            piston_waveform_frac_max = float(piston_waveform_bias) + float(
                piston_waveform_scale
            ) * max(data)
            if piston_use_depth and piston_depth_max is not None:
                piston_depth_est = piston_depth_max * piston_waveform_frac_max
            piston_metric = (
                piston_depth_est
                if piston_depth_est is not None
                else piston_waveform_frac_max
            )
            piston_metric_source = "input_waveform_est"

    hst_file = run_dir / "frc_merge.hst"
    if not hst_file.exists():
        candidates = list(run_dir.glob("*.hst"))
        if candidates:
            hst_file = candidates[0]

    hst_data = athena_read.hst(str(hst_file)) if hst_file.exists() else None
    vtk_files = sorted(run_dir.glob("*.vtk"))
    num_outputs = len(vtk_files)

    ran_to_completion = False
    if hst_data:
        t_series = hst_data.get("time")
        tlim = get_float(cfg, "time", "tlim", None)
        if t_series is not None and len(t_series) > 0:
            t_last = float(t_series[-1])
            if tlim is None:
                ran_to_completion = True
            else:
                ran_to_completion = t_last >= 0.99 * tlim
        else:
            ran_to_completion = True
    no_nan_in_metrics = True

    mass_in = None
    mass_out = None
    if hst_data:
        mass_in = hst_data.get("mass_cyl_in")
        mass_out = hst_data.get("mass_cyl_out")

    leak_mass_frac_max = None
    mass_rel_drift_in = None
    mass_out_initial = None
    if mass_in is not None and mass_out is not None and len(mass_in) > 0:
        mass_in0 = mass_in[0]
        mass_out_initial = float(mass_out[0])
        if np.isnan(mass_in).any() or np.isnan(mass_out).any():
            no_nan_in_metrics = False
        if mass_in0 != 0.0:
            mass_out_delta = mass_out - mass_out[0]
            leak_mass_frac_max = float(np.max(mass_out_delta) / mass_in0)
            mass_rel_drift_in = float(np.max(np.abs(mass_in - mass_in0) / mass_in0))

    mass_budget_M_start = None
    mass_budget_M_end = None
    mass_budget_deltaM = None
    mass_budget_int_flux_total = None
    mass_budget_int_flux_x1 = None
    mass_budget_int_flux_ox1 = None
    mass_budget_int_flux_x2 = None
    mass_budget_int_flux_ox2 = None
    mass_budget_int_flux_x3 = None
    mass_budget_int_flux_ox3 = None
    mass_budget_residual = None
    mass_budget_residual_rel = None
    mass_budget_primary_face = None
    mass_budget_int_blocked_total = None
    mass_budget_int_blocked_x1 = None
    mass_budget_int_blocked_ox1 = None
    mass_budget_int_blocked_x2 = None
    mass_budget_int_blocked_ox2 = None
    mass_budget_int_blocked_x3 = None
    mass_budget_int_blocked_ox3 = None
    mass_budget_blocked_primary_face = None
    mass_budget_blocked_vs_boundary_ratio = None
    mass_budget_int_blocked_geom_total = None
    mass_budget_int_blocked_geom_x1 = None
    mass_budget_int_blocked_geom_ox1 = None
    mass_budget_int_blocked_geom_x2 = None
    mass_budget_int_blocked_geom_ox2 = None
    mass_budget_int_blocked_geom_x3 = None
    mass_budget_int_blocked_geom_ox3 = None
    mass_budget_residual_geom = None
    mass_budget_residual_geom_rel = None
    blocked_area_frac_mean = None
    blocked_area_frac_min = None
    blocked_geom_explains_residual = None
    leak_mass_frac_max_geom = None

    if hst_data:
        t_series = to_array(hst_data.get("time"))
        mass_domain = to_array(hst_data.get("mass_domain"))
        flux_x1 = to_array(hst_data.get("fmass_x1"))
        flux_ox1 = to_array(hst_data.get("fmass_ox1"))
        flux_x2 = to_array(hst_data.get("fmass_x2"))
        flux_ox2 = to_array(hst_data.get("fmass_ox2"))
        flux_x3 = to_array(hst_data.get("fmass_x3"))
        flux_ox3 = to_array(hst_data.get("fmass_ox3"))
        flux_blocked_total = to_array(hst_data.get("fmass_blocked_total"))
        flux_blocked_x1 = to_array(hst_data.get("fmass_blocked_x1"))
        flux_blocked_ox1 = to_array(hst_data.get("fmass_blocked_ox1"))
        flux_blocked_x2 = to_array(hst_data.get("fmass_blocked_x2"))
        flux_blocked_ox2 = to_array(hst_data.get("fmass_blocked_ox2"))
        flux_blocked_x3 = to_array(hst_data.get("fmass_blocked_x3"))
        flux_blocked_ox3 = to_array(hst_data.get("fmass_blocked_ox3"))
        flux_blocked_geom_total = to_array(hst_data.get("fmass_blocked_geom_total"))
        flux_blocked_geom_x1 = to_array(hst_data.get("fmass_blocked_geom_x1"))
        flux_blocked_geom_ox1 = to_array(hst_data.get("fmass_blocked_geom_ox1"))
        flux_blocked_geom_x2 = to_array(hst_data.get("fmass_blocked_geom_x2"))
        flux_blocked_geom_ox2 = to_array(hst_data.get("fmass_blocked_geom_ox2"))
        flux_blocked_geom_x3 = to_array(hst_data.get("fmass_blocked_geom_x3"))
        flux_blocked_geom_ox3 = to_array(hst_data.get("fmass_blocked_geom_ox3"))
        blocked_area_frac_sum = to_array(hst_data.get("blocked_area_frac_sum"))
        blocked_area_frac_min_series = to_array(hst_data.get("blocked_area_frac_min"))
        blocked_area_frac_count = to_array(hst_data.get("blocked_area_frac_count"))
        for series in (
            t_series,
            mass_domain,
            flux_x1,
            flux_ox1,
            flux_x2,
            flux_ox2,
            flux_x3,
            flux_ox3,
            flux_blocked_total,
            flux_blocked_x1,
            flux_blocked_ox1,
            flux_blocked_x2,
            flux_blocked_ox2,
            flux_blocked_x3,
            flux_blocked_ox3,
            flux_blocked_geom_total,
            flux_blocked_geom_x1,
            flux_blocked_geom_ox1,
            flux_blocked_geom_x2,
            flux_blocked_geom_ox2,
            flux_blocked_geom_x3,
            flux_blocked_geom_ox3,
            blocked_area_frac_sum,
            blocked_area_frac_min_series,
            blocked_area_frac_count,
        ):
            if series is None:
                continue
            if np.isnan(series).any():
                no_nan_in_metrics = False
        if t_series is not None and mass_domain is not None:
            if len(t_series) == len(mass_domain) and len(t_series) > 1:
                mass_budget_M_start = float(mass_domain[0])
                mass_budget_M_end = float(mass_domain[-1])
                mass_budget_deltaM = mass_budget_M_end - mass_budget_M_start
        flux_ints = {
            "x1": trapz_or_none(t_series, flux_x1),
            "ox1": trapz_or_none(t_series, flux_ox1),
            "x2": trapz_or_none(t_series, flux_x2),
            "ox2": trapz_or_none(t_series, flux_ox2),
            "x3": trapz_or_none(t_series, flux_x3),
            "ox3": trapz_or_none(t_series, flux_ox3),
        }
        if all(val is not None for val in flux_ints.values()):
            mass_budget_int_flux_x1 = flux_ints["x1"]
            mass_budget_int_flux_ox1 = flux_ints["ox1"]
            mass_budget_int_flux_x2 = flux_ints["x2"]
            mass_budget_int_flux_ox2 = flux_ints["ox2"]
            mass_budget_int_flux_x3 = flux_ints["x3"]
            mass_budget_int_flux_ox3 = flux_ints["ox3"]
            mass_budget_int_flux_total = sum(flux_ints.values())
            mass_budget_primary_face = max(flux_ints, key=lambda k: abs(flux_ints[k]))

        blocked_ints = {
            "x1": trapz_or_none(t_series, flux_blocked_x1),
            "ox1": trapz_or_none(t_series, flux_blocked_ox1),
            "x2": trapz_or_none(t_series, flux_blocked_x2),
            "ox2": trapz_or_none(t_series, flux_blocked_ox2),
            "x3": trapz_or_none(t_series, flux_blocked_x3),
            "ox3": trapz_or_none(t_series, flux_blocked_ox3),
        }
        blocked_total = trapz_or_none(t_series, flux_blocked_total)
        if blocked_total is not None and all(val is not None for val in blocked_ints.values()):
            mass_budget_int_blocked_total = blocked_total
            mass_budget_int_blocked_x1 = blocked_ints["x1"]
            mass_budget_int_blocked_ox1 = blocked_ints["ox1"]
            mass_budget_int_blocked_x2 = blocked_ints["x2"]
            mass_budget_int_blocked_ox2 = blocked_ints["ox2"]
            mass_budget_int_blocked_x3 = blocked_ints["x3"]
            mass_budget_int_blocked_ox3 = blocked_ints["ox3"]
            mass_budget_blocked_primary_face = max(
                blocked_ints, key=lambda k: abs(blocked_ints[k])
            )
            if mass_budget_int_flux_total is not None:
                denom = max(abs(mass_budget_int_flux_total), 1.0e-30)
                mass_budget_blocked_vs_boundary_ratio = (
                    abs(mass_budget_int_blocked_total) / denom
                )

        blocked_geom_ints = {
            "x1": trapz_or_none(t_series, flux_blocked_geom_x1),
            "ox1": trapz_or_none(t_series, flux_blocked_geom_ox1),
            "x2": trapz_or_none(t_series, flux_blocked_geom_x2),
            "ox2": trapz_or_none(t_series, flux_blocked_geom_ox2),
            "x3": trapz_or_none(t_series, flux_blocked_geom_x3),
            "ox3": trapz_or_none(t_series, flux_blocked_geom_ox3),
        }
        blocked_geom_total = trapz_or_none(t_series, flux_blocked_geom_total)
        if blocked_geom_total is not None and all(
            val is not None for val in blocked_geom_ints.values()
        ):
            mass_budget_int_blocked_geom_total = blocked_geom_total
            mass_budget_int_blocked_geom_x1 = blocked_geom_ints["x1"]
            mass_budget_int_blocked_geom_ox1 = blocked_geom_ints["ox1"]
            mass_budget_int_blocked_geom_x2 = blocked_geom_ints["x2"]
            mass_budget_int_blocked_geom_ox2 = blocked_geom_ints["ox2"]
            mass_budget_int_blocked_geom_x3 = blocked_geom_ints["x3"]
            mass_budget_int_blocked_geom_ox3 = blocked_geom_ints["ox3"]

        if (
            mass_budget_deltaM is not None
            and mass_budget_int_flux_total is not None
            and mass_budget_int_blocked_total is not None
        ):
            mass_budget_residual = (
                mass_budget_deltaM
                + mass_budget_int_flux_total
                + mass_budget_int_blocked_total
            )
            denom = max(abs(mass_budget_M_start or 0.0), 1.0e-30)
            mass_budget_residual_rel = abs(mass_budget_residual) / denom
            if mass_budget_int_blocked_geom_total is not None:
                mass_budget_residual_geom = (
                    mass_budget_deltaM
                    + mass_budget_int_flux_total
                    + mass_budget_int_blocked_geom_total
                )
                mass_budget_residual_geom_rel = abs(mass_budget_residual_geom) / denom
                blocked_geom_explains_residual = (
                    mass_budget_residual_geom_rel < mass_budget_residual_rel
                )
                leak_mass_frac_max_geom = mass_budget_residual_geom_rel
        elif mass_budget_deltaM is not None and mass_budget_int_flux_total is not None:
            mass_budget_residual = mass_budget_deltaM + mass_budget_int_flux_total
            denom = max(abs(mass_budget_M_start or 0.0), 1.0e-30)
            mass_budget_residual_rel = abs(mass_budget_residual) / denom

        if (
            blocked_area_frac_sum is not None
            and blocked_area_frac_count is not None
            and len(blocked_area_frac_sum) > 0
            and len(blocked_area_frac_count) > 0
        ):
            count = float(blocked_area_frac_count[-1])
            if count > 0.0:
                blocked_area_frac_mean = float(blocked_area_frac_sum[-1] / count)
        if blocked_area_frac_min_series is not None and len(blocked_area_frac_min_series) > 0:
            blocked_area_frac_min = float(blocked_area_frac_min_series[-1])

    wall_cells_count = None
    wall_cells_fraction = None
    blocked_flux_faces_count = None
    blocked_flux_faces_fraction = None
    eb_mask_applied = False

    x1f = x2f = x3f = None
    x1c = x2c = x3c = None
    if vtk_files:
        x1f, x2f, x3f, _data = athena_read.vtk(str(vtk_files[-1]))
        x1c = 0.5 * (x1f[:-1] + x1f[1:])
        x2c = 0.5 * (x2f[:-1] + x2f[1:])
        x3c = 0.5 * (x3f[:-1] + x3f[1:])
    else:
        x1f, x1c = build_uniform_axis(
            get_float(cfg, "mesh", "x1min", None),
            get_float(cfg, "mesh", "x1max", None),
            int(get_float(cfg, "mesh", "nx1", None) or 0),
        )
        x2f, x2c = build_uniform_axis(
            get_float(cfg, "mesh", "x2min", None),
            get_float(cfg, "mesh", "x2max", None),
            int(get_float(cfg, "mesh", "nx2", None) or 0),
        )
        x3f, x3c = build_uniform_axis(
            get_float(cfg, "mesh", "x3min", None),
            get_float(cfg, "mesh", "x3max", None),
            int(get_float(cfg, "mesh", "nx3", None) or 0),
        )

    if x1f is not None and x2f is not None and x3f is not None:
        r2_grid = x2c[None, :, None] ** 2 + x3c[:, None, None] ** 2
        rmax_vals = np.array([r_wall_fn(float(x1)) for x1 in x1c], dtype=float)
        inside_mask = r2_grid <= (rmax_vals[None, None, :] ** 2)
        total_cells = int(inside_mask.size)
        inside = int(np.count_nonzero(inside_mask))
        wall_cells_count = total_cells - inside
        wall_cells_fraction = wall_cells_count / total_cells if total_cells > 0 else None

        if inside_mask.shape[1] > 1:
            diff_x2 = inside_mask[:, 1:, :] ^ inside_mask[:, :-1, :]
            blocked_x2 = int(np.count_nonzero(diff_x2))
        else:
            blocked_x2 = 0
        if inside_mask.shape[0] > 1:
            diff_x3 = inside_mask[1:, :, :] ^ inside_mask[:-1, :, :]
            blocked_x3 = int(np.count_nonzero(diff_x3))
        else:
            blocked_x3 = 0
        blocked_flux_faces_count = blocked_x2 + blocked_x3
        total_faces = 0
        if inside_mask.shape[1] > 1:
            total_faces += inside_mask.shape[0] * (inside_mask.shape[1] - 1) * inside_mask.shape[2]
        if inside_mask.shape[0] > 1:
            total_faces += (inside_mask.shape[0] - 1) * inside_mask.shape[1] * inside_mask.shape[2]
        blocked_flux_faces_fraction = (
            blocked_flux_faces_count / total_faces if total_faces > 0 else None
        )
        eb_mask_applied = bool(
            wall_enabled
            and wall_cells_count is not None
            and wall_cells_count > 0
            and blocked_flux_faces_count > 0
        )

        # Mask snapshot at mid-plane (x3 ~ 0)
        mid_k = len(x3c) // 2
        x3_mid = x3c[mid_k]
        r2_slice = x2c ** 2 + x3_mid ** 2
        mask2d = np.zeros((len(x2c), len(x1c)), dtype=float)
        for i, x1 in enumerate(x1c):
            rmax = r_wall_fn(float(x1))
            mask2d[:, i] = (r2_slice <= rmax * rmax).astype(float)

        plt.figure(figsize=(7, 4))
        x1_edges = x1f
        x2_edges = x2f
        plt.pcolormesh(x1_edges, x2_edges, mask2d, shading="auto")
        rline = np.array([r_wall_fn(float(x1)) for x1 in x1c])
        plt.plot(x1c, rline, color="w", linewidth=1.0)
        plt.plot(x1c, -rline, color="w", linewidth=1.0)
        plt.xlabel("x1")
        plt.ylabel("x2 (slice)")
        plt.title("EB Mask Snapshot (x3 mid-plane)")
        plt.colorbar(label="mask (1=inside)")
        plt.tight_layout()
        plt.savefig(plots_dir / "eb_mask_snapshot.png")
        plt.close()
        if wall_meta.get("profile") == "step":
            plt.figure(figsize=(7, 4))
            plt.pcolormesh(x1_edges, x2_edges, mask2d, shading="auto")
            plt.plot(x1c, rline, color="w", linewidth=1.0)
            plt.plot(x1c, -rline, color="w", linewidth=1.0)
            plt.xlabel("x1")
            plt.ylabel("x2 (slice)")
            plt.title("EB Mask Snapshot (step wall)")
            plt.colorbar(label="mask (1=inside)")
            plt.tight_layout()
            plt.savefig(plots_dir / "eb_mask_snapshot_step.png")
            plt.close()

    # Leak plot
    if hst_data and mass_in is not None and mass_out is not None:
        t = hst_data.get("time")
        plt.figure(figsize=(6, 4))
        plt.plot(t, mass_out, marker="o", label="mass_out")
        plt.plot(t, mass_in, marker="o", label="mass_in")
        plt.xlabel("Time")
        plt.ylabel("Mass")
        plt.title("Mass Inside/Outside Wall vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "leak_mass_out_vs_time.png")
        plt.close()

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": num_outputs,
        "no_nan_in_metrics": no_nan_in_metrics,
        "wall_enabled": wall_enabled,
        "wall_profile": wall_meta.get("profile"),
        "wall_rmax": wall_meta.get("rmax"),
        "wall_x1_min": wall_meta.get("x1_min"),
        "wall_x1_max": wall_meta.get("x1_max"),
        "wall_x1_step": wall_meta.get("x1_step"),
        "wall_r0": wall_meta.get("r0"),
        "wall_r1": wall_meta.get("r1"),
        "wall_cells_count": wall_cells_count,
        "wall_cells_fraction": wall_cells_fraction,
        "blocked_flux_faces_count": blocked_flux_faces_count,
        "blocked_flux_faces_fraction": blocked_flux_faces_fraction,
        "eb_mask_applied": eb_mask_applied,
        "leak_mass_frac_max": leak_mass_frac_max,
        "mass_rel_drift_in": mass_rel_drift_in,
        "mass_out_initial": mass_out_initial,
        "mass_budget_M_start": mass_budget_M_start,
        "mass_budget_M_end": mass_budget_M_end,
        "mass_budget_deltaM": mass_budget_deltaM,
        "mass_budget_int_flux_total": mass_budget_int_flux_total,
        "mass_budget_int_flux_x1": mass_budget_int_flux_x1,
        "mass_budget_int_flux_ox1": mass_budget_int_flux_ox1,
        "mass_budget_int_flux_x2": mass_budget_int_flux_x2,
        "mass_budget_int_flux_ox2": mass_budget_int_flux_ox2,
        "mass_budget_int_flux_x3": mass_budget_int_flux_x3,
        "mass_budget_int_flux_ox3": mass_budget_int_flux_ox3,
        "mass_budget_residual": mass_budget_residual,
        "mass_budget_residual_rel": mass_budget_residual_rel,
        "mass_budget_primary_face": mass_budget_primary_face,
        "mass_budget_int_blocked_total": mass_budget_int_blocked_total,
        "mass_budget_int_blocked_x1": mass_budget_int_blocked_x1,
        "mass_budget_int_blocked_ox1": mass_budget_int_blocked_ox1,
        "mass_budget_int_blocked_x2": mass_budget_int_blocked_x2,
        "mass_budget_int_blocked_ox2": mass_budget_int_blocked_ox2,
        "mass_budget_int_blocked_x3": mass_budget_int_blocked_x3,
        "mass_budget_int_blocked_ox3": mass_budget_int_blocked_ox3,
        "mass_budget_blocked_primary_face": mass_budget_blocked_primary_face,
        "mass_budget_blocked_vs_boundary_ratio": mass_budget_blocked_vs_boundary_ratio,
        "mass_budget_int_blocked_geom_total": mass_budget_int_blocked_geom_total,
        "mass_budget_int_blocked_geom_x1": mass_budget_int_blocked_geom_x1,
        "mass_budget_int_blocked_geom_ox1": mass_budget_int_blocked_geom_ox1,
        "mass_budget_int_blocked_geom_x2": mass_budget_int_blocked_geom_x2,
        "mass_budget_int_blocked_geom_ox2": mass_budget_int_blocked_geom_ox2,
        "mass_budget_int_blocked_geom_x3": mass_budget_int_blocked_geom_x3,
        "mass_budget_int_blocked_geom_ox3": mass_budget_int_blocked_geom_ox3,
        "mass_budget_residual_geom": mass_budget_residual_geom,
        "mass_budget_residual_geom_rel": mass_budget_residual_geom_rel,
        "blocked_area_frac_mean": blocked_area_frac_mean,
        "blocked_area_frac_min": blocked_area_frac_min,
        "blocked_geom_explains_residual": blocked_geom_explains_residual,
        "leak_mass_frac_max_geom": leak_mass_frac_max_geom,
        "piston_bc": piston_enabled,
        "piston_use_depth": piston_use_depth,
        "piston_depth_max": piston_depth_max,
        "piston_waveform": piston_waveform,
        "piston_waveform_column": piston_waveform_column,
        "piston_waveform_scale": piston_waveform_scale,
        "piston_waveform_bias": piston_waveform_bias,
        "piston_waveform_tshift": piston_waveform_tshift,
        "piston_waveform_frac_max": piston_waveform_frac_max,
        "piston_depth_est": piston_depth_est,
        "piston_metric": piston_metric,
        "piston_metric_source": piston_metric_source,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
