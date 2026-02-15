#!/usr/bin/env python3
"""
Advanced analysis for Fusion/FRC simulations in WarpX (RZ).

Standard metrics (per diag snapshot):
  - Centroid (r,z) from rho-weighted volume.
  - Excluded-flux proxy Psi_max and separatrix radius r_s.
  - Doublet separation from Psi O-points.
  - Reconnection Et at the X-point (midplane between O-points).
  - Hall quadrupole proxy from Bt in the core.

Outputs a CSV/JSON time series plus an optional PNG summary plot.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yt

# Explicit SI unit overrides to avoid yt default "code_length" reporting.
UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "magnetic_unit": (1.0, "T"),
}

# Physical constants (SI)
M_U = 1.66053906660e-27  # atomic mass unit (kg)
MU0 = 4.0e-7 * np.pi  # vacuum permeability (H/m)


def list_diags(diag_root: Path):
    return sorted(
        [p for p in diag_root.iterdir() if p.name.startswith("diag") and "old" not in p.name]
    )


def infer_mass_density(diag_root: Path, metadata_path: Optional[Path] = None) -> Optional[float]:
    """Infer upstream mass density from WarpX run metadata (n0, ion_amu)."""
    meta = metadata_path
    if meta is None:
        candidates = sorted(diag_root.glob("warpx_run_*.json"))
        if candidates:
            meta = candidates[-1]
    if meta is None or not meta.exists():
        return None
    try:
        with meta.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        args = data.get("args", {}) if isinstance(data, dict) else {}
        n0 = args.get("n0") or args.get("hybrid_n0") or args.get("hybrid_n0_fallback")
        ion_amu = args.get("ion_amu", 1.0)
        if n0 is None:
            return None
        n0_val = float(n0)
        if n0_val <= 0.0:
            return None
        return n0_val * float(ion_amu) * M_U
    except Exception:
        return None

def get_field_data(ds, field_name):
    """Extract a uniform grid array for a WarpX field."""
    level = 0
    dims = ds.domain_dimensions
    left = ds.domain_left_edge
    right = ds.domain_right_edge

    cg = ds.covering_grid(level=level, left_edge=left, dims=dims)
    arr = cg[("boxlib", field_name)].to_ndarray()
    return arr, left, right


def _local_max_mask(field: np.ndarray, size: int = 10):
    """Return a boolean mask of local maxima; SciPy if available, else simple neighbor check."""
    try:
        from scipy.ndimage import maximum_filter  # type: ignore

        return maximum_filter(field, size=size) == field
    except Exception:
        mask = np.ones_like(field, dtype=bool)
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mask &= field >= np.roll(field, shift, axis=(0, 1))
        # suppress edges
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        return mask

def analyze_frc_metrics(
    ds,
    rho_threshold: float = 0.1,
    xpoint_window_cells: int = 8,
    upstream_dr_cells: int = 4,
    upstream_b_min_T: float = 0.0,
    mass_density_ref: Optional[float] = None,
):
    """Compute standardized FRC metrics for a single WarpX diag."""
    rho, left, right = get_field_data(ds, "rho")
    br, _, _ = get_field_data(ds, "Br")
    bz, _, _ = get_field_data(ds, "Bz")
    bt, _, _ = get_field_data(ds, "Bt")
    et, _, _ = get_field_data(ds, "Et")

    # yt may report edges in code units; convert to plain floats (SI via units_override)
    if hasattr(left, "to_value"):
        left = np.asarray(left.to_value(), dtype=np.float64)
    else:
        left = np.asarray(left, dtype=np.float64)
    if hasattr(right, "to_value"):
        right = np.asarray(right.to_value(), dtype=np.float64)
    else:
        right = np.asarray(right, dtype=np.float64)

    rho = rho.squeeze()
    br = br.squeeze()
    bz = bz.squeeze()
    bt = bt.squeeze()
    et = et.squeeze()

    nr, nz = rho.shape
    dr = float((right[0] - left[0]) / nr)
    dz = float((right[1] - left[1]) / nz)

    r = np.linspace(left[0] + dr / 2, right[0] - dr / 2, nr)
    z = np.linspace(left[1] + dz / 2, right[1] - dz / 2, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")

    # 1) Centroid (rho-weighted with cylindrical volume element 2*pi*r)
    weight = rho * rr
    total_weight = float(np.sum(weight))
    if total_weight > 0.0:
        z_centroid = float(np.sum(weight * zz) / total_weight)
        r_centroid = float(np.sum(weight * rr) / total_weight)
    else:
        z_centroid = float("nan")
        r_centroid = float("nan")

    # 2) Flux function Psi(r,z) = int_0^r Bz r dr (2*pi factor omitted)
    psi = np.cumsum(bz * rr * dr, axis=0)
    psi_max = float(np.max(psi))

    local_max = _local_max_mask(psi, size=10)
    global_max = float(np.max(psi))
    peaks = (
        np.argwhere(local_max & (psi > 0.1 * global_max)) if global_max > 0 else []
    )
    o_points = [(r[i], z[j], psi[i, j]) for i, j in peaks]
    o_points.sort(key=lambda p: p[1])

    separation = 0.0
    mid_z_idx = nz // 2
    if len(o_points) >= 2:
        separation = float(abs(o_points[-1][1] - o_points[0][1]))
        z_idx1 = int(np.argmin(np.abs(z - o_points[0][1])))
        z_idx2 = int(np.argmin(np.abs(z - o_points[-1][1])))
        if z_idx1 > z_idx2:
            z_idx1, z_idx2 = z_idx2, z_idx1
        mid_z_idx = (z_idx1 + z_idx2) // 2
    elif np.isfinite(z_centroid):
        mid_z_idx = int(np.argmin(np.abs(z - z_centroid)))

    # Separatrix radius r_s from first Bz sign change at midplane
    bz_prof = bz[:, mid_z_idx]
    signs = np.sign(bz_prof)
    cross = np.where(signs[:-1] * signs[1:] < 0)[0]
    r_s = float("nan")
    if cross.size > 0:
        i0 = int(cross[0])
        b0, b1 = bz_prof[i0], bz_prof[i0 + 1]
        if b1 != b0:
            r_s = float(
                r[i0] - b0 * (r[i0 + 1] - r[i0]) / (b1 - b0)
            )

    # X-point indices: use separatrix radius if available, else fallback to psi ridge.
    if np.isfinite(r_s):
        x_r_idx = int(np.argmin(np.abs(r - r_s)))
        x_r_m = float(r_s)
    else:
        x_r_idx = int(np.argmax(psi[:, mid_z_idx]))
        x_r_m = float(r[x_r_idx])
    x_z_idx = mid_z_idx
    x_z_m = float(z[mid_z_idx])
    psi_xpoint = float(psi[x_r_idx, x_z_idx])

    # 3) Reconnection Et at X-point: average in a small window around (r_s, z_mid)
    win = max(1, int(xpoint_window_cells))
    r_lo = max(0, x_r_idx - win)
    r_hi = min(nr, x_r_idx + win + 1)
    z_lo = max(0, x_z_idx - win)
    z_hi = min(nz, x_z_idx + win + 1)
    et_win = et[r_lo:r_hi, z_lo:z_hi]
    x_point_et = float(np.mean(et_win)) if et_win.size else 0.0
    x_point_et_peak = float(np.max(np.abs(et_win))) if et_win.size else 0.0

    # Upstream ring outside separatrix: r in [r_s + Δr, r_s + 2Δr], z within X-point window.
    delta_cells = max(1, int(upstream_dr_cells))
    if np.isfinite(r_s):
        upstream_r_start = float(r_s + delta_cells * dr)
    else:
        upstream_r_start = float(r[x_r_idx] + delta_cells * dr)
    upstream_r_end = float(upstream_r_start + delta_cells * dr)

    idx_start = int(np.searchsorted(r, upstream_r_start))
    idx_end = int(np.searchsorted(r, upstream_r_end))
    idx_start = min(max(idx_start, 0), nr - 1)
    idx_end = min(max(idx_end, idx_start + 1), nr)

    bz_ring = bz[idx_start:idx_end, z_lo:z_hi]
    br_ring = br[idx_start:idx_end, z_lo:z_hi]
    if bz_ring.size:
        b_up_bz = float(np.mean(np.abs(bz_ring)))
        b_up_poloidal = float(
            np.mean(np.sqrt(bz_ring * bz_ring + br_ring * br_ring))
        )
    else:
        b_up_bz = float("nan")
        b_up_poloidal = float("nan")

    upstream_valid = bool(
        np.isfinite(b_up_poloidal) and b_up_poloidal >= float(upstream_b_min_T)
    )

    psi_ring = psi[idx_start:idx_end, z_lo:z_hi]
    psi_upstream_mean = float(np.mean(psi_ring)) if psi_ring.size else float("nan")
    upstream_r_mid = float(0.5 * (upstream_r_start + upstream_r_end))

    et_ring = et[idx_start:idx_end, z_lo:z_hi]
    et_upstream_mean = float(np.mean(et_ring)) if et_ring.size else float("nan")
    et_xpoint_detrended = (
        float(x_point_et - et_upstream_mean)
        if np.isfinite(et_upstream_mean)
        else float("nan")
    )
    et_xpoint_peak_detrended = (
        float(np.max(np.abs(et_win - et_upstream_mean)))
        if et_win.size and np.isfinite(et_upstream_mean)
        else float("nan")
    )

    vA_up = float("nan")
    if (
        mass_density_ref is not None
        and mass_density_ref > 0.0
        and upstream_valid
    ):
        vA_up = float(b_up_poloidal / np.sqrt(MU0 * mass_density_ref))

    reconnection_rate_norm = float("nan")
    reconnection_rate_norm_abs = float("nan")
    reconnection_rate_norm_peak = float("nan")
    reconnection_rate_norm_detrended = float("nan")
    reconnection_rate_norm_detrended_abs = float("nan")
    reconnection_rate_norm_detrended_peak = float("nan")
    if upstream_valid and np.isfinite(vA_up) and vA_up > 0.0:
        denom = b_up_poloidal * vA_up
        reconnection_rate_norm = float(x_point_et / denom)
        reconnection_rate_norm_abs = float(abs(x_point_et) / denom)
        reconnection_rate_norm_peak = float(x_point_et_peak / denom)
        if np.isfinite(et_xpoint_detrended):
            reconnection_rate_norm_detrended = float(et_xpoint_detrended / denom)
            reconnection_rate_norm_detrended_abs = float(abs(et_xpoint_detrended) / denom)
        if np.isfinite(et_xpoint_peak_detrended):
            reconnection_rate_norm_detrended_peak = float(et_xpoint_peak_detrended / denom)

    # 4) Hall quadrupole from Bt around X-point (quadrant decomposition)
    hall_quad_comp = float("nan")
    hall_quad_comp_detrended = float("nan")
    hall_quad_amp_detrended = float("nan")
    hall_quad_score = float("nan")
    bt_q1 = bt_q2 = bt_q3 = bt_q4 = float("nan")
    bt_q1_d = bt_q2_d = bt_q3_d = bt_q4_d = float("nan")
    bt_win_mean = float("nan")
    try:
        bt_win = bt[r_lo:r_hi, z_lo:z_hi]
        bt_win_mean = float(np.mean(bt_win)) if bt_win.size else float("nan")
        bt_fluct = bt_win - bt_win_mean
        rr_idx, zz_idx = np.meshgrid(
            (np.arange(r_lo, r_hi) - x_r_idx),
            (np.arange(z_lo, z_hi) - x_z_idx),
            indexing="ij",
        )
        q1_mask = (rr_idx >= 0) & (zz_idx >= 0)
        q2_mask = (rr_idx >= 0) & (zz_idx < 0)
        q3_mask = (rr_idx < 0) & (zz_idx >= 0)
        q4_mask = (rr_idx < 0) & (zz_idx < 0)
        if np.any(q1_mask):
            bt_q1 = float(np.mean(bt_win[q1_mask]))
            bt_q1_d = float(np.mean(bt_fluct[q1_mask]))
        if np.any(q2_mask):
            bt_q2 = float(np.mean(bt_win[q2_mask]))
            bt_q2_d = float(np.mean(bt_fluct[q2_mask]))
        if np.any(q3_mask):
            bt_q3 = float(np.mean(bt_win[q3_mask]))
            bt_q3_d = float(np.mean(bt_fluct[q3_mask]))
        if np.any(q4_mask):
            bt_q4 = float(np.mean(bt_win[q4_mask]))
            bt_q4_d = float(np.mean(bt_fluct[q4_mask]))
        if all(np.isfinite([bt_q1, bt_q2, bt_q3, bt_q4])):
            hall_quad_comp = 0.25 * (bt_q1 - bt_q2 - bt_q3 + bt_q4)
        if all(np.isfinite([bt_q1_d, bt_q2_d, bt_q3_d, bt_q4_d])):
            hall_quad_comp_detrended = 0.25 * (bt_q1_d - bt_q2_d - bt_q3_d + bt_q4_d)
            hall_quad_amp_detrended = float(abs(hall_quad_comp_detrended))
            denom = 0.25 * (
                abs(bt_q1_d) + abs(bt_q2_d) + abs(bt_q3_d) + abs(bt_q4_d)
            )
            if denom > 0:
                hall_quad_score = float(hall_quad_comp_detrended / denom)
    except Exception:
        hall_quad_comp = float("nan")
        hall_quad_comp_detrended = float("nan")

    # Fallback Hall quadrupole proxy from Bt in dense core
    rho_max = float(np.max(rho))
    core_mask = rho > (rho_threshold * rho_max) if rho_max > 0 else None
    bt_core = bt[core_mask] if core_mask is not None and np.any(core_mask) else bt
    bt_max = float(np.max(bt_core))
    bt_min = float(np.min(bt_core))
    hall_quad_amp_core = 0.5 * (bt_max - bt_min)
    hall_quad_amp = (
        float(abs(hall_quad_comp))
        if np.isfinite(hall_quad_comp)
        else hall_quad_amp_core
    )

    return {
        "time_s": float(ds.current_time),
        "centroid_r_m": r_centroid,
        "centroid_z_m": z_centroid,
        "r_s_m": r_s,
        "xpoint_r_m": x_r_m,
        "xpoint_z_m": x_z_m,
        "upstream_r_start_m": upstream_r_start,
        "upstream_r_end_m": upstream_r_end,
        "upstream_r_mid_m": upstream_r_mid,
        "upstream_dr_cells": delta_cells,
        "upstream_b_min_T": float(upstream_b_min_T),
        "upstream_valid": int(upstream_valid),
        "B_up_poloidal_T": b_up_poloidal,
        "B_up_Bz_T": b_up_bz,
        "vA_up_m_s": vA_up,
        "psi_upstream_mean": psi_upstream_mean,
        "psi_xpoint": psi_xpoint,
        "psi_max": psi_max,
        "num_o_points": len(o_points),
        "separation_m": separation,
        "Et_xpoint": x_point_et,
        "Et_xpoint_peak": x_point_et_peak,
        "Et_upstream_mean": et_upstream_mean,
        "Et_xpoint_detrended": et_xpoint_detrended,
        "Et_xpoint_peak_detrended": et_xpoint_peak_detrended,
        "reconnection_rate_norm": reconnection_rate_norm,
        "reconnection_rate_norm_abs": reconnection_rate_norm_abs,
        "reconnection_rate_norm_peak": reconnection_rate_norm_peak,
        "reconnection_rate_norm_detrended": reconnection_rate_norm_detrended,
        "reconnection_rate_norm_detrended_abs": reconnection_rate_norm_detrended_abs,
        "reconnection_rate_norm_detrended_peak": reconnection_rate_norm_detrended_peak,
        "Bt_max_core": bt_max,
        "Bt_min_core": bt_min,
        "hall_quadrupole_amp": hall_quad_amp,
        "hall_quadrupole_amp_core": hall_quad_amp_core,
        "hall_quadrupole_comp": hall_quad_comp,
        "hall_quadrupole_comp_detrended": hall_quad_comp_detrended,
        "hall_quadrupole_amp_detrended": hall_quad_amp_detrended,
        "hall_quadrupole_score": hall_quad_score,
        "Bt_win_mean": bt_win_mean,
        "Bt_q1_mean": bt_q1,
        "Bt_q2_mean": bt_q2,
        "Bt_q3_mean": bt_q3,
        "Bt_q4_mean": bt_q4,
        "Bt_q1_mean_detrended": bt_q1_d,
        "Bt_q2_mean_detrended": bt_q2_d,
        "Bt_q3_mean_detrended": bt_q3_d,
        "Bt_q4_mean_detrended": bt_q4_d,
        "xpoint_window_cells": win,
    }

def process_series(
    diag_path: Path,
    rho_threshold: float,
    xpoint_window_cells: int,
    upstream_dr_cells: int,
    upstream_b_min_T: float,
    mass_density_ref: Optional[float],
):
    diags = list_diags(diag_path)
    results = []
    print(f"Processing {len(diags)} diagnostics in {diag_path}...")
    for d in diags:
        try:
            ds = yt.load(str(d), units_override=UNITS_OVERRIDE)
            metrics = analyze_frc_metrics(
                ds,
                rho_threshold=rho_threshold,
                xpoint_window_cells=xpoint_window_cells,
                upstream_dr_cells=upstream_dr_cells,
                upstream_b_min_T=upstream_b_min_T,
                mass_density_ref=mass_density_ref,
            )
            results.append(metrics)
            print(
                f"  t={metrics['time_s']:.3e}: z_c={metrics['centroid_z_m']:.3f}, "
                f"sep={metrics['separation_m']:.3f}, psi_max={metrics['psi_max']:.3e}"
            )
        except Exception as exc:
            print(f"  Skipping {d}: {exc}")

    # Inductive Et correction over the time series:
    # Our psi is Ψ/(2π), so E_induced ≈ -(1/r) dpsi/dt.
    if len(results) >= 2:
        times = np.asarray([r.get("time_s", np.nan) for r in results], dtype=float)
        psi_x = np.asarray([r.get("psi_xpoint", np.nan) for r in results], dtype=float)
        r_x = np.asarray([r.get("xpoint_r_m", np.nan) for r in results], dtype=float)
        psi_up = np.asarray(
            [r.get("psi_upstream_mean", np.nan) for r in results], dtype=float
        )
        r_up = np.asarray([r.get("upstream_r_mid_m", np.nan) for r in results], dtype=float)

        dpsi_dt = np.full_like(psi_x, np.nan)
        dpsi_dt_up = np.full_like(psi_up, np.nan)

        for i in range(len(results)):
            if i == 0:
                dt = times[1] - times[0]
                if dt > 0 and np.isfinite(psi_x[0]) and np.isfinite(psi_x[1]):
                    dpsi_dt[i] = (psi_x[1] - psi_x[0]) / dt
                if dt > 0 and np.isfinite(psi_up[0]) and np.isfinite(psi_up[1]):
                    dpsi_dt_up[i] = (psi_up[1] - psi_up[0]) / dt
            elif i == len(results) - 1:
                dt = times[-1] - times[-2]
                if dt > 0 and np.isfinite(psi_x[-2]) and np.isfinite(psi_x[-1]):
                    dpsi_dt[i] = (psi_x[-1] - psi_x[-2]) / dt
                if dt > 0 and np.isfinite(psi_up[-2]) and np.isfinite(psi_up[-1]):
                    dpsi_dt_up[i] = (psi_up[-1] - psi_up[-2]) / dt
            else:
                dt = times[i + 1] - times[i - 1]
                if (
                    dt > 0
                    and np.isfinite(psi_x[i - 1])
                    and np.isfinite(psi_x[i + 1])
                ):
                    dpsi_dt[i] = (psi_x[i + 1] - psi_x[i - 1]) / dt
                if (
                    dt > 0
                    and np.isfinite(psi_up[i - 1])
                    and np.isfinite(psi_up[i + 1])
                ):
                    dpsi_dt_up[i] = (psi_up[i + 1] - psi_up[i - 1]) / dt

        for i, row in enumerate(results):
            row["dpsi_dt_xpoint"] = (
                float(dpsi_dt[i]) if np.isfinite(dpsi_dt[i]) else float("nan")
            )
            rx = r_x[i]
            if np.isfinite(rx) and rx > 1.0e-6 and np.isfinite(dpsi_dt[i]):
                et_induced = float(-dpsi_dt[i] / rx)
            else:
                et_induced = float("nan")
            row["Et_induced"] = et_induced

            et_x = row.get("Et_xpoint")
            if et_x is not None and np.isfinite(et_x) and np.isfinite(et_induced):
                et_corr = float(et_x - et_induced)
            else:
                et_corr = float("nan")
            row["Et_xpoint_corrected"] = et_corr

            et_up = row.get("Et_upstream_mean")
            if np.isfinite(et_corr) and et_up is not None and np.isfinite(et_up):
                et_corr_det = float(et_corr - et_up)
            else:
                et_corr_det = float("nan")
            row["Et_xpoint_corrected_detrended"] = et_corr_det

            # Upstream-ring inductive background using dpsi_upstream/dt.
            row["dpsi_dt_upstream"] = (
                float(dpsi_dt_up[i]) if np.isfinite(dpsi_dt_up[i]) else float("nan")
            )
            rup = r_up[i]
            if np.isfinite(rup) and rup > 1.0e-6 and np.isfinite(dpsi_dt_up[i]):
                et_induced_up = float(-dpsi_dt_up[i] / rup)
            else:
                et_induced_up = float("nan")
            row["Et_induced_upstream"] = et_induced_up

            if et_x is not None and np.isfinite(et_x) and np.isfinite(et_induced_up):
                et_corr_up = float(et_x - et_induced_up)
            else:
                et_corr_up = float("nan")
            row["Et_xpoint_corrected_upstream"] = et_corr_up

            if np.isfinite(et_corr_up) and et_up is not None and np.isfinite(et_up):
                et_corr_up_det = float(et_corr_up - et_up)
            else:
                et_corr_up_det = float("nan")
            row["Et_xpoint_corrected_upstream_detrended"] = et_corr_up_det

            b_up = row.get("B_up_poloidal_T")
            v_a_up = row.get("vA_up_m_s")
            if (
                b_up is not None
                and v_a_up is not None
                and np.isfinite(b_up)
                and np.isfinite(v_a_up)
                and b_up >= float(upstream_b_min_T)
                and v_a_up > 0.0
            ):
                denom = float(b_up * v_a_up)
            else:
                denom = float("nan")

            if np.isfinite(denom) and denom > 0.0:
                row["reconnection_rate_norm_corrected"] = (
                    float(et_corr / denom) if np.isfinite(et_corr) else float("nan")
                )
                row["reconnection_rate_norm_corrected_abs"] = (
                    float(abs(et_corr) / denom)
                    if np.isfinite(et_corr)
                    else float("nan")
                )
                row["reconnection_rate_norm_corrected_detrended"] = (
                    float(et_corr_det / denom)
                    if np.isfinite(et_corr_det)
                    else float("nan")
                )
                row["reconnection_rate_norm_corrected_detrended_abs"] = (
                    float(abs(et_corr_det) / denom)
                    if np.isfinite(et_corr_det)
                    else float("nan")
                )

                row["reconnection_rate_norm_corrected_upstream"] = (
                    float(et_corr_up / denom)
                    if np.isfinite(et_corr_up)
                    else float("nan")
                )
                row["reconnection_rate_norm_corrected_upstream_abs"] = (
                    float(abs(et_corr_up) / denom)
                    if np.isfinite(et_corr_up)
                    else float("nan")
                )
                row["reconnection_rate_norm_corrected_upstream_detrended"] = (
                    float(et_corr_up_det / denom)
                    if np.isfinite(et_corr_up_det)
                    else float("nan")
                )
                row["reconnection_rate_norm_corrected_upstream_detrended_abs"] = (
                    float(abs(et_corr_up_det) / denom)
                    if np.isfinite(et_corr_up_det)
                    else float("nan")
                )
            else:
                row["reconnection_rate_norm_corrected"] = float("nan")
                row["reconnection_rate_norm_corrected_abs"] = float("nan")
                row["reconnection_rate_norm_corrected_detrended"] = float("nan")
                row["reconnection_rate_norm_corrected_detrended_abs"] = float("nan")
                row["reconnection_rate_norm_corrected_upstream"] = float("nan")
                row["reconnection_rate_norm_corrected_upstream_abs"] = float("nan")
                row["reconnection_rate_norm_corrected_upstream_detrended"] = float("nan")
                row["reconnection_rate_norm_corrected_upstream_detrended_abs"] = float("nan")

    return results


def is_dr_specific_key(key: str) -> bool:
    """True if a metric depends on upstream Δr selection."""
    if key == "time_s":
        return False
    if key.startswith(("upstream_", "B_up_", "vA_up", "psi_upstream")):
        return True
    if key.startswith(
        (
            "Et_upstream",
            "Et_xpoint_detrended",
            "Et_xpoint_peak_detrended",
            "Et_xpoint_corrected_upstream",
        )
    ):
        return True
    if key == "Et_xpoint_corrected_detrended":
        return True
    if key.startswith(("dpsi_dt_upstream", "Et_induced_upstream")):
        return True
    if key.startswith("reconnection_rate_"):
        return True
    return False


def merge_dr_series(results_by_dr: dict[int, list[dict]]) -> list[dict]:
    """Merge multiple upstream-dr series into one list with suffixed columns."""
    dr_values = list(results_by_dr.keys())
    if not dr_values:
        return []
    base_dr = dr_values[0]
    base_results = results_by_dr[base_dr]
    if not base_results:
        return []

    sample_keys = list(base_results[0].keys())
    dr_specific_keys = {k for k in sample_keys if is_dr_specific_key(k)}
    base_keys = [k for k in sample_keys if k not in dr_specific_keys]

    merged: list[dict] = []
    for idx, base_row in enumerate(base_results):
        row = {k: base_row.get(k) for k in base_keys}
        for dr, series in results_by_dr.items():
            if idx >= len(series):
                continue
            src = series[idx]
            for k, v in src.items():
                if is_dr_specific_key(k):
                    row[f"{k}_dr{dr}"] = v
        merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Compute standardized FRC metrics from WarpX diags.")
    parser.add_argument("--diag-path", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="If set, writes <prefix>.csv/.json/.png unless overridden.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument(
        "--rho-threshold",
        type=float,
        default=0.1,
        help="Core mask threshold as fraction of rho_max for Hall quadrupole proxy.",
    )
    parser.add_argument(
        "--xpoint-window-cells",
        type=int,
        default=8,
        help="Half-width (cells) of the window around X-point for Et and Hall quadrupole metrics.",
    )
    parser.add_argument(
        "--upstream-dr-cells",
        type=int,
        default=4,
        help="Δr in cells for upstream sampling ring: r in [r_s+Δr, r_s+2Δr].",
    )
    parser.add_argument(
        "--upstream-dr-cells-list",
        type=str,
        default=None,
        help="Comma-separated list of Δr (cells) values to compare (e.g. 4,8,12).",
    )
    parser.add_argument(
        "--upstream-b-min-T",
        type=float,
        default=0.0,
        help="Minimum upstream B_poloidal (T) required to report normalized reconnection rates.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional WarpX run metadata JSON to infer n0 and ion_amu for v_A normalization.",
    )
    args = parser.parse_args()

    # Determine upstream Δr list for comparison.
    if args.upstream_dr_cells_list:
        try:
            raw_vals = [
                int(v.strip())
                for v in str(args.upstream_dr_cells_list).split(",")
                if v.strip()
            ]
        except ValueError:
            print("Error: --upstream-dr-cells-list must be comma-separated integers.")
            return
        if not raw_vals:
            print("Error: --upstream-dr-cells-list is empty.")
            return
        seen: set[int] = set()
        dr_values = [v for v in raw_vals if not (v in seen or seen.add(v))]
    else:
        dr_values = [int(args.upstream_dr_cells)]

    auto_csv = args.output_csv is None
    auto_json = args.output_json is None
    auto_plot = args.plot is None
    if args.output_prefix:
        prefix = args.output_prefix
        if len(dr_values) > 1:
            dr_tag = "dr" + "_".join(str(v) for v in dr_values)
            prefix = prefix.with_name(f"{prefix.name}_{dr_tag}")
        if auto_csv:
            args.output_csv = prefix.with_suffix(".csv")
        if auto_json:
            args.output_json = prefix.with_suffix(".json")
        if auto_plot:
            args.plot = prefix.with_suffix(".png")
    
    if not args.diag_path.exists():
        print(f"Error: {args.diag_path} does not exist.")
        return
        
    mass_density_ref = infer_mass_density(args.diag_path, args.metadata)
    if mass_density_ref is None:
        print("[normalize] no metadata density found; reconnection_rate_norm will be NaN.")
    else:
        print(f"[normalize] mass_density_ref={mass_density_ref:.3e} kg/m^3")

    if len(dr_values) == 1:
        results = process_series(
            args.diag_path,
            rho_threshold=args.rho_threshold,
            xpoint_window_cells=args.xpoint_window_cells,
            upstream_dr_cells=dr_values[0],
            upstream_b_min_T=args.upstream_b_min_T,
            mass_density_ref=mass_density_ref,
        )
    else:
        results_by_dr: dict[int, list[dict]] = {}
        for dr in dr_values:
            results_by_dr[dr] = process_series(
                args.diag_path,
                rho_threshold=args.rho_threshold,
                xpoint_window_cells=args.xpoint_window_cells,
                upstream_dr_cells=dr,
                upstream_b_min_T=args.upstream_b_min_T,
                mass_density_ref=mass_density_ref,
            )
        results = merge_dr_series(results_by_dr)
        print(f"[compare] merged upstream dr values: {dr_values}")
    
    # Summary
    if not results:
        print("No results.")
        return
        
    times = [r["time_s"] for r in results]
    seps = [r["separation_m"] for r in results]
    psi_maxes = [r["psi_max"] for r in results]
    
    print("\n--- Summary ---")
    print(f"Initial Separation: {seps[0]:.3f} m")
    print(f"Final Separation:   {seps[-1]:.3f} m")
    print(f"Peak Psi_max:       {max(psi_maxes):.3e} (SI if units_override applies)")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        single_run_keys = [
            "time_s",
            "centroid_r_m",
            "centroid_z_m",
            "r_s_m",
            "xpoint_r_m",
            "xpoint_z_m",
            "upstream_r_start_m",
            "upstream_r_end_m",
            "upstream_r_mid_m",
            "upstream_dr_cells",
            "upstream_b_min_T",
            "upstream_valid",
            "B_up_poloidal_T",
            "B_up_Bz_T",
            "vA_up_m_s",
            "psi_upstream_mean",
            "psi_xpoint",
            "dpsi_dt_xpoint",
            "dpsi_dt_upstream",
            "psi_max",
            "num_o_points",
            "separation_m",
            "Et_xpoint",
            "Et_xpoint_peak",
            "Et_induced",
            "Et_induced_upstream",
            "Et_xpoint_corrected",
            "Et_xpoint_corrected_detrended",
            "Et_xpoint_corrected_upstream",
            "Et_xpoint_corrected_upstream_detrended",
            "Et_upstream_mean",
            "Et_xpoint_detrended",
            "Et_xpoint_peak_detrended",
            "reconnection_rate_norm",
            "reconnection_rate_norm_abs",
            "reconnection_rate_norm_peak",
            "reconnection_rate_norm_detrended",
            "reconnection_rate_norm_detrended_abs",
            "reconnection_rate_norm_detrended_peak",
            "reconnection_rate_norm_corrected",
            "reconnection_rate_norm_corrected_abs",
            "reconnection_rate_norm_corrected_detrended",
            "reconnection_rate_norm_corrected_detrended_abs",
            "reconnection_rate_norm_corrected_upstream",
            "reconnection_rate_norm_corrected_upstream_abs",
            "reconnection_rate_norm_corrected_upstream_detrended",
            "reconnection_rate_norm_corrected_upstream_detrended_abs",
            "hall_quadrupole_amp",
            "hall_quadrupole_amp_core",
            "Bt_max_core",
            "Bt_min_core",
            "hall_quadrupole_comp",
            "hall_quadrupole_comp_detrended",
            "hall_quadrupole_amp_detrended",
            "hall_quadrupole_score",
            "Bt_win_mean",
            "Bt_q1_mean",
            "Bt_q2_mean",
            "Bt_q3_mean",
            "Bt_q4_mean",
            "Bt_q1_mean_detrended",
            "Bt_q2_mean_detrended",
            "Bt_q3_mean_detrended",
            "Bt_q4_mean_detrended",
            "xpoint_window_cells",
        ]
        if len(dr_values) == 1:
            keys = single_run_keys
        else:
            dr_specific = [k for k in single_run_keys if is_dr_specific_key(k)]
            base_keys = [k for k in single_run_keys if k not in dr_specific]
            keys = list(base_keys)
            for dr in dr_values:
                keys.extend([f"{k}_dr{dr}" for k in dr_specific])
        with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in keys})
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
            ax1.plot(times, seps, label="separation (m)", color="tab:blue")
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("separation (m)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(times, psi_maxes, label="psi_max", color="tab:red")
            ax2.set_ylabel("psi_max", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot)
            print(f"Saved metrics plot to {args.plot}")
        except Exception as exc:
            print(f"Plot skipped: {exc}")
    
    # Detect Merge vs Bounce
    # If separation decreases monotonically -> Merge
    # If separation decreases then increases -> Bounce
    min_sep = min(seps)
    min_idx = seps.index(min_sep)
    
    if min_idx < len(seps) - 1:
        rebound = seps[-1] - min_sep
        if rebound > 0.05: # Threshold 5cm
            print(f"Outcome: BOUNCE detected (Rebound {rebound*100:.1f} cm)")
        else:
            print("Outcome: MERGE (or stuck)")
    else:
        print("Outcome: MERGING (ongoing or complete)")

if __name__ == "__main__":
    main()
