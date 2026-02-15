#!/usr/bin/env python3
"""
Compute a Kirtley-style 0D compression scaling check from Athena++ VTK or WarpX diags.

Default scaling assumes isotropic compression with frozen-in flux:
  n ~ C^3, T ~ C^2, B ~ C^2 where C = (V0 / V)^(1/3).
Override exponents via CLI to match updated scalings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

K_B = 1.380649e-23
M_U = 1.66053906660e-27
MU0 = 4.0e-7 * np.pi


def _add_athena_vis_path() -> None:
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
            os.sys.path.append(str(path))
            return
    raise SystemExit("athena_read not found. Set ATHENA_VIS_PATH.")


def _get_vtk_time(filename: Path) -> Optional[float]:
    try:
        with filename.open("r", errors="replace") as fh:
            for _ in range(6):
                line = fh.readline()
                if "time=" in line:
                    return float(line.split("time=")[1].split()[0])
    except Exception:
        return None
    return None


def _fold_half_grid(values: np.ndarray, xc: np.ndarray, flip_sign: bool):
    pos_mask = xc >= 0.0
    neg_mask = xc < 0.0
    pos_vals = values[:, pos_mask]
    neg_vals = values[:, neg_mask][:, ::-1]
    if flip_sign:
        neg_vals = -neg_vals
    if pos_vals.shape[1] == neg_vals.shape[1]:
        folded = 0.5 * (pos_vals + neg_vals)
    else:
        folded = np.empty((values.shape[0], pos_vals.shape[1]), dtype=values.dtype)
        folded[:, 0] = pos_vals[:, 0]
        folded[:, 1:] = 0.5 * (pos_vals[:, 1:] + neg_vals)
    r_centers = np.abs(xc[pos_mask])
    return folded, r_centers


def _fold_fields(fields, r_coords, flip_signs):
    if r_coords.min() >= 0.0:
        return fields, r_coords
    folded_fields = []
    r_new = None
    for arr, flip in zip(fields, flip_signs):
        folded, r_vals = _fold_half_grid(arr.T, r_coords, flip_sign=flip)
        folded_fields.append(folded.T)
        r_new = r_vals
    return folded_fields, r_new if r_new is not None else r_coords


def _mask_from_threshold(
    rho: np.ndarray,
    b_mag: np.ndarray,
    press: Optional[np.ndarray],
    n: np.ndarray,
    threshold: float,
    mode: str,
    t_const_K: Optional[float],
):
    if mode == "rho":
        rho_max = float(np.max(rho))
        if rho_max <= 0.0:
            return np.zeros_like(rho, dtype=bool)
        return rho > (threshold * rho_max)
    if mode == "beta" or mode == "beta_rel":
        if press is None:
            if t_const_K is None:
                raise SystemExit("beta mask requires pressure or --T-const-eV.")
            press = n * K_B * t_const_K
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = 2.0 * MU0 * press / (b_mag * b_mag)
        if mode == "beta_rel":
            beta_max = float(np.max(beta))
            if not np.isfinite(beta_max) or beta_max <= 0.0:
                return np.zeros_like(rho, dtype=bool)
            return beta > (threshold * beta_max)
        return beta > threshold
    raise SystemExit(f"Unknown mask mode: {mode}")


def _boundary_bz_stats(bz: np.ndarray, ncell: int):
    if bz.ndim < 2:
        raise SystemExit("Field array must be at least 2D for excluded-flux mask.")
    ncell = max(1, int(ncell))
    ncell = min(ncell, bz.shape[-1])
    left = bz[..., :ncell]
    right = bz[..., -ncell:]
    vals = np.concatenate([left.ravel(), right.ravel()])
    if vals.size == 0:
        return 0.0, 0.0, 1.0
    median_signed = float(np.median(vals))
    median_mag = float(np.median(np.abs(vals)))
    sign = float(np.sign(median_signed)) if median_signed != 0.0 else 1.0
    return median_signed, median_mag, sign


def _athena_vtk_fields(vtk_path: Path, fold_r: bool):
    _add_athena_vis_path()
    import athena_read  # type: ignore

    x_faces, y_faces, z_faces, data = athena_read.vtk(str(vtk_path))
    rho_in = data.get("rho")
    press_in = data.get("press")
    if press_in is None:
        press_in = data.get("p")
    b_in = data.get("Bcc")
    if b_in is None:
        b_in = data.get("bcc")
    if b_in is None:
        b_in = data.get("b")
    if rho_in is None or b_in is None:
        raise SystemExit(f"VTK missing rho/B fields in {vtk_path}")

    rho = np.asarray(rho_in, dtype=np.float64)
    if rho.ndim == 2:
        rho = rho[None, :, :]
    press = None
    if press_in is not None:
        press = np.asarray(press_in, dtype=np.float64)
        if press.ndim == 2:
            press = press[None, :, :]
    bcc = np.asarray(b_in, dtype=np.float64)
    if bcc.ndim == 3:
        bcc = bcc[None, :, :, :]
    if bcc.ndim != 4:
        raise SystemExit(f"Unexpected Bcc shape {bcc.shape} in {vtk_path}")

    bx = bcc[..., 0]
    by = bcc[..., 1]
    bz = bcc[..., 2]

    xc = 0.5 * (x_faces[:-1] + x_faces[1:])
    yc = 0.5 * (y_faces[:-1] + y_faces[1:])
    is_3d = (rho.shape[0] > 1) or (len(z_faces) > 2)

    if not is_3d and fold_r and yc.min() < 0.0:
        rho_2d = rho[0]
        bx_2d = bx[0]
        by_2d = by[0]
        bz_2d = bz[0]
        fields = [rho_2d, bx_2d, by_2d, bz_2d]
        if press is not None:
            fields.append(press[0])
        fields, yc = _fold_fields(
            fields,
            yc,
            [False, False, True, False] + ([False] if press is not None else []),
        )
        rho = fields[0][None, :, :]
        bx = fields[1][None, :, :]
        by = fields[2][None, :, :]
        bz = fields[3][None, :, :]
        if press is not None:
            press = fields[4][None, :, :]

    return x_faces, y_faces, z_faces, xc, yc, rho, press, bx, by, bz, is_3d


def _read_waveform_csv(path: Optional[Path], column: int):
    if path is None:
        return None
    rows = []
    col_idx = int(column)
    if col_idx < 1:
        raise SystemExit("bext waveform column must be >= 1 (value column after time).")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p for p in raw.replace(",", " ").split() if p]
            if len(parts) <= col_idx:
                continue
            try:
                t_val = float(parts[0])
                v_val = float(parts[col_idx])
            except ValueError:
                continue
            rows.append((t_val, v_val))
    if not rows:
        raise SystemExit(f"No usable rows in waveform file: {path}")
    rows.sort(key=lambda pair: pair[0])
    times = np.array([r[0] for r in rows], dtype=float)
    vals = np.array([r[1] for r in rows], dtype=float)
    return {"times": times, "vals": vals}


def _interp_waveform(t_val: float, wf):
    times = wf["times"]
    vals = wf["vals"]
    if t_val <= times[0]:
        return float(vals[0])
    if t_val >= times[-1]:
        return float(vals[-1])
    idx = int(np.searchsorted(times, t_val)) - 1
    if idx < 0:
        idx = 0
    t0 = times[idx]
    t1 = times[idx + 1]
    v0 = vals[idx]
    v1 = vals[idx + 1]
    if t1 == t0:
        return float(v1)
    return float(v0 + (v1 - v0) * (t_val - t0) / (t1 - t0))


def _waveform_bext_values(t_val, wf, kind, scale, bias):
    if wf is None or not np.isfinite(t_val):
        return {"frac": float("nan"), "delta": float("nan"), "total": float("nan")}
    raw = _interp_waveform(float(t_val), wf)
    if kind == "delta":
        delta = raw
        total = bias + delta
        frac = delta / scale if scale != 0.0 else float("nan")
    elif kind == "total":
        total = raw
        delta = total - bias
        frac = delta / scale if scale != 0.0 else float("nan")
    else:
        frac = raw
        delta = scale * frac
        total = bias + delta
    return {"frac": float(frac), "delta": float(delta), "total": float(total)}


def _compute_beta(
    press: Optional[np.ndarray],
    n: np.ndarray,
    b_mag: np.ndarray,
    t_const_K: Optional[float],
):
    if press is None:
        if t_const_K is None:
            raise SystemExit("beta mask requires pressure or --T-const-eV.")
        press = n * K_B * t_const_K
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = 2.0 * MU0 * press / (b_mag * b_mag)
    return beta


def _mass_core_mask(rho: np.ndarray, vol: np.ndarray, mass_fraction: float,
                    domain_mask: Optional[np.ndarray] = None):
    if mass_fraction <= 0.0 or mass_fraction > 1.0:
        raise SystemExit("mass_core requires mass_fraction in (0, 1].")

    if domain_mask is not None:
        rho_flat = rho[domain_mask]
        vol_flat = vol[domain_mask]
    else:
        rho_flat = rho.ravel()
        vol_flat = vol.ravel()

    rho_flat = np.where(np.isfinite(rho_flat), rho_flat, -np.inf)
    vol_flat = np.where(np.isfinite(vol_flat) & (vol_flat > 0.0), vol_flat, 0.0)
    mass_flat = rho_flat * vol_flat
    total_mass = float(np.sum(mass_flat))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return np.zeros_like(rho, dtype=bool), 0.0, total_mass

    order = np.argsort(rho_flat)[::-1]
    mass_sorted = mass_flat[order]
    cum = np.cumsum(mass_sorted)
    target = mass_fraction * total_mass
    cutoff = int(np.searchsorted(cum, target, side="left"))
    cutoff = min(max(cutoff, 0), len(cum) - 1)
    select = order[: cutoff + 1]
    core_mask_flat = np.zeros_like(rho_flat, dtype=bool)
    core_mask_flat[select] = True

    mask = np.zeros_like(rho, dtype=bool)
    if domain_mask is not None:
        mask[domain_mask] = core_mask_flat
    else:
        mask = core_mask_flat.reshape(rho.shape)

    core_mass = float(np.sum(mass_flat[core_mask_flat]))
    return mask, core_mass, total_mass


def _stable_bext_sign(boundary_sign, sign_source, bias, wf_vals, frac_threshold):
    if sign_source == "boundary":
        return boundary_sign, True
    if sign_source == "bias":
        sign = float(np.sign(bias)) if bias != 0.0 else 1.0
        return sign, True
    frac = wf_vals.get("frac", float("nan"))
    if np.isfinite(frac) and abs(frac) < frac_threshold:
        sign = float(np.sign(bias)) if bias != 0.0 else 1.0
        return sign, False
    total = wf_vals.get("total", float("nan"))
    if np.isfinite(total) and total != 0.0:
        return float(np.sign(total)), True
    return 1.0, False


def _excluded_flux_mask(
    bz: np.ndarray, bext_sign: float, bext_mag: float, bz_frac: float
):
    if not np.isfinite(bext_mag) or bext_mag <= 0.0:
        return np.zeros_like(bz, dtype=bool)
    amp = np.abs(bz)
    return (bz * bext_sign < 0.0) & (amp > bz_frac * bext_mag)


def _beta_ref_threshold_athena(
    files: list[Path],
    fold_r: bool,
    bext_waveform: Optional[dict],
    bext_waveform_kind: str,
    bext_waveform_scale: float,
    bext_waveform_bias: float,
    bext_sign_source: str,
    bext_sign_frac_threshold: float,
    beta_ref_quantile: float,
    beta_ref_progress_max: float,
    wall_depth_max: Optional[float],
    amu: float,
    excl_component: str,
):
    for f in files:
        (
            x_faces,
            y_faces,
            _,
            xc,
            yc,
            rho,
            press,
            bx,
            by,
            bz,
            _,
        ) = _athena_vtk_fields(f, fold_r)

        b_mag = np.sqrt(bx * bx + by * by + bz * bz)
        n = rho / (amu * M_U)
        if excl_component == "b1":
            b_excl = bx
        elif excl_component == "b2":
            b_excl = by
        else:
            b_excl = bz
        bext_bz_median, bext_mag, bext_sign_boundary = _boundary_bz_stats(b_excl, 1)
        time_val = _get_vtk_time(f)
        wf_vals = _waveform_bext_values(
            time_val if time_val is not None else float("nan"),
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
        )
        _, bext_sign_valid = _stable_bext_sign(
            bext_sign_boundary,
            bext_sign_source,
            bext_waveform_bias,
            wf_vals,
            bext_sign_frac_threshold,
        )
        frac = wf_vals.get("frac", float("nan"))
        if not np.isfinite(frac):
            frac = 0.0
        if not bext_sign_valid or frac > beta_ref_progress_max:
            continue

        domain_mask = None
        if wall_depth_max is not None and np.isfinite(frac):
            x1min = float(x_faces[0])
            x1max = float(x_faces[-1])
            max_depth = 0.5 * (x1max - x1min) - 1.0e-12
            wall_depth = max(0.0, wall_depth_max * frac)
            wall_depth = min(wall_depth, max_depth)
            x1_mask = (xc >= x1min + wall_depth) & (xc <= x1max - wall_depth)
            domain_mask = np.broadcast_to(x1_mask[None, None, :], rho.shape)

        beta = _compute_beta(press, n, b_mag, None)
        if domain_mask is not None:
            beta = beta[domain_mask]
        beta = beta[np.isfinite(beta)]
        if beta.size == 0:
            continue
        thr = float(np.quantile(beta, beta_ref_quantile))
        return thr, float(time_val) if time_val is not None else float("nan"), float(frac)

    if files:
        first = files[0]
        (
            _x_faces,
            _y_faces,
            _z_faces,
            _xc,
            _yc,
            rho,
            press,
            bx,
            by,
            bz,
            _,
        ) = _athena_vtk_fields(first, fold_r)
        b_mag = np.sqrt(bx * bx + by * by + bz * bz)
        n = rho / (amu * M_U)
        beta = _compute_beta(press, n, b_mag, None)
        beta = beta[np.isfinite(beta)]
        if beta.size > 0:
            thr = float(np.quantile(beta, beta_ref_quantile))
            time_val = _get_vtk_time(first)
            return thr, float(time_val) if time_val is not None else float("nan"), 0.0

    return float("nan"), float("nan"), float("nan")


def _beta_ref_threshold_warpx(
    diags: list[Path],
    bext_waveform: Optional[dict],
    bext_waveform_kind: str,
    bext_waveform_scale: float,
    bext_waveform_bias: float,
    bext_sign_source: str,
    bext_sign_frac_threshold: float,
    beta_ref_quantile: float,
    beta_ref_progress_max: float,
    amu: float,
    t_const_K: Optional[float],
    excl_component: str,
):
    try:
        import yt
    except Exception as exc:
        raise SystemExit(f"yt not available: {exc}")

    for diag in diags:
        ds = yt.load(str(diag))
        left = np.asarray(ds.domain_left_edge.to_value(), dtype=np.float64)
        right = np.asarray(ds.domain_right_edge.to_value(), dtype=np.float64)
        dims = ds.domain_dimensions
        cg = ds.covering_grid(level=0, left_edge=left, dims=dims)
        rho = cg[("boxlib", "rho")].to_ndarray().squeeze()
        br = cg[("boxlib", "Br")].to_ndarray().squeeze()
        bz = cg[("boxlib", "Bz")].to_ndarray().squeeze()
        bt = cg[("boxlib", "Bt")].to_ndarray().squeeze()

        b_mag = np.sqrt(br * br + bz * bz + bt * bt)
        n = rho / (amu * M_U)
        if excl_component == "b1":
            b_excl = br
        elif excl_component == "b2":
            b_excl = bt
        else:
            b_excl = bz
        _, bext_mag, bext_sign_boundary = _boundary_bz_stats(b_excl, 1)
        time_val = float(ds.current_time.to_value())
        wf_vals = _waveform_bext_values(
            time_val,
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
        )
        _, bext_sign_valid = _stable_bext_sign(
            bext_sign_boundary,
            bext_sign_source,
            bext_waveform_bias,
            wf_vals,
            bext_sign_frac_threshold,
        )
        frac = wf_vals.get("frac", float("nan"))
        if not np.isfinite(frac):
            frac = 0.0
        if not bext_sign_valid or frac > beta_ref_progress_max:
            continue

        beta = _compute_beta(press=None, n=n, b_mag=b_mag, t_const_K=t_const_K)
        beta = beta[np.isfinite(beta)]
        if beta.size == 0:
            continue
        thr = float(np.quantile(beta, beta_ref_quantile))
        return thr, float(time_val), float(frac)

    return float("nan"), float("nan"), float("nan")


def _athena_series(
    run_dir: Path,
    vtk_pattern: str,
    threshold: float,
    amu: float,
    fold_r: bool,
    mask_mode: str,
    excl_bz_frac: float,
    excl_ncell: int,
    excl_component: str,
    bext_waveform: Optional[dict],
    bext_waveform_kind: str,
    bext_waveform_scale: float,
    bext_waveform_bias: float,
    bext_sign_source: str,
    bext_sign_frac_threshold: float,
    bext_progress_source: str,
    wall_depth_max: Optional[float],
    r_max_core: Optional[float],
    beta_ref_quantile: float,
    beta_ref_progress_max: float,
    beta_ref_mask_min: float,
    beta_ref_mask_max: float,
    rho_quantile: float,
    press_quantile: float,
    mass_fraction: float,
):
    pattern_path = Path(vtk_pattern)
    pattern = str(run_dir / vtk_pattern) if not pattern_path.is_absolute() else vtk_pattern
    import glob
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No VTK files matched: {pattern}")

    beta_ref_thr = float("nan")
    beta_ref_time = float("nan")
    beta_ref_progress = float("nan")
    if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq"):
        beta_ref_thr, beta_ref_time, beta_ref_progress = _beta_ref_threshold_athena(
            files,
            fold_r,
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
            bext_sign_source,
            bext_sign_frac_threshold,
            beta_ref_quantile,
            beta_ref_progress_max,
            wall_depth_max,
            amu,
            excl_component,
        )
        if not np.isfinite(beta_ref_thr):
            print("[kirtley] beta_ref: failed to find reference threshold; masks may be empty.")

    series = []
    for f in files:
        (
            x_faces,
            y_faces,
            z_faces,
            xc,
            yc,
            rho,
            press,
            bx,
            by,
            bz,
            is_3d,
        ) = _athena_vtk_fields(f, fold_r)

        if is_3d:
            dx = np.diff(x_faces)
            dy = np.diff(y_faces)
            dz = np.diff(z_faces)
            vol = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
        else:
            dr = float(y_faces[1] - y_faces[0])
            dz = float(x_faces[1] - x_faces[0])
            r = yc[None, :, None]
            vol = 2.0 * np.pi * r * dr * dz
            vol = np.broadcast_to(vol, rho.shape)

        b_mag = np.sqrt(bx * bx + by * by + bz * bz)
        n = rho / (amu * M_U)
        if excl_component == "b1":
            b_excl = bx
        elif excl_component == "b2":
            b_excl = by
        else:
            b_excl = bz
        bext_bz_median, bext_mag, bext_sign_boundary = _boundary_bz_stats(b_excl, excl_ncell)
        time_val = _get_vtk_time(f)
        wf_vals = _waveform_bext_values(
            time_val if time_val is not None else float("nan"),
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
        )
        bext_sign_mask, bext_sign_valid = _stable_bext_sign(
            bext_sign_boundary,
            bext_sign_source,
            bext_waveform_bias,
            wf_vals,
            bext_sign_frac_threshold,
        )
        if bext_progress_source == "waveform":
            bext_progress = wf_vals["frac"]
        else:
            bext_progress = bext_mag
        wall_depth = None
        domain_mask = None
        if wall_depth_max is not None and np.isfinite(wf_vals.get("frac", float("nan"))):
            x1min = float(x_faces[0])
            x1max = float(x_faces[-1])
            max_depth = 0.5 * (x1max - x1min) - 1.0e-12
            wall_depth = max(0.0, wall_depth_max * wf_vals["frac"])
            wall_depth = min(wall_depth, max_depth)
            x1_mask = (xc >= x1min + wall_depth) & (xc <= x1max - wall_depth)
            domain_mask = np.broadcast_to(x1_mask[None, None, :], rho.shape)
        if r_max_core is not None and np.isfinite(r_max_core) and r_max_core > 0.0:
            if is_3d:
                zc = 0.5 * (z_faces[:-1] + z_faces[1:])
                rr = np.sqrt(yc[None, :, None] ** 2 + zc[:, None, None] ** 2)
                r_mask = np.broadcast_to(rr <= r_max_core, rho.shape)
                domain_mask = r_mask if domain_mask is None else (domain_mask & r_mask)
        beta_mask = None
        rho_mask = None
        press_mask = None
        rho_thr = float("nan")
        press_thr = float("nan")
        rho_q_used = float("nan")
        press_q_used = float("nan")
        mask_mass = float("nan")
        mask_mass_fraction = float("nan")
        rho_avg_core = float("nan")
        mask_mass = float("nan")
        mask_mass_fraction = float("nan")
        rho_avg_core = float("nan")
        rho_q_used = float("nan")
        press_q_used = float("nan")
        rho_q_used = float("nan")
        press_q_used = float("nan")

        if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq"):
            if np.isfinite(beta_ref_thr):
                beta = _compute_beta(press, n, b_mag, t_const_K=None)
                beta_mask = beta >= beta_ref_thr
            else:
                beta_mask = np.zeros_like(rho, dtype=bool)

        if mask_mode == "mass_core":
            f_mass = threshold if np.isfinite(threshold) else mass_fraction
            mask, mask_mass, total_mass = _mass_core_mask(rho, vol, f_mass, domain_mask)
            if total_mass > 0.0:
                mask_mass_fraction = mask_mass / total_mass
        elif mask_mode == "beta_ref_rhoq":
            q = threshold if np.isfinite(threshold) else rho_quantile
            rho_q_used = q
            rho_vals = rho
            if domain_mask is not None:
                rho_vals = rho_vals[domain_mask]
            rho_vals = rho_vals[np.isfinite(rho_vals)]
            if rho_vals.size == 0:
                rho_mask = np.zeros_like(rho, dtype=bool)
            else:
                rho_thr = float(np.quantile(rho_vals, q))
                rho_mask = rho >= rho_thr
            mask = beta_mask & rho_mask
        elif mask_mode == "beta_ref_pq":
            q = threshold if np.isfinite(threshold) else press_quantile
            press_q_used = q
            if press is None:
                raise SystemExit("beta_ref_pq requires pressure data.")
            press_vals = press
            if domain_mask is not None:
                press_vals = press_vals[domain_mask]
            press_vals = press_vals[np.isfinite(press_vals)]
            if press_vals.size == 0:
                press_mask = np.zeros_like(rho, dtype=bool)
            else:
                press_thr = float(np.quantile(press_vals, q))
                press_mask = press >= press_thr
            mask = beta_mask & press_mask
        elif mask_mode == "beta_ref":
            mask = beta_mask
        elif mask_mode == "excl":
            mask = _excluded_flux_mask(b_excl, bext_sign_mask, bext_mag, excl_bz_frac)
        elif mask_mode == "excl_rho":
            rho_max = float(np.max(rho))
            rho_cut = threshold * rho_max if rho_max > 0.0 else float("inf")
            mask = _excluded_flux_mask(b_excl, bext_sign_mask, bext_mag, excl_bz_frac)
            mask &= rho > rho_cut
        else:
            mask = _mask_from_threshold(
                rho, b_mag, press, n, threshold, mask_mode, t_const_K=None
            )
        if domain_mask is not None:
            mask &= domain_mask
            if beta_mask is not None:
                beta_mask &= domain_mask
            if rho_mask is not None:
                rho_mask &= domain_mask
            if press_mask is not None:
                press_mask &= domain_mask
            domain_vol = float(np.sum(vol * domain_mask))
        else:
            domain_vol = float(np.sum(vol))
        vol_mask = vol * mask
        mask_vol = float(np.sum(vol_mask))
        mask_count = int(np.count_nonzero(mask))
        vol_sum = mask_vol
        mask_empty = False
        mask_warn = False
        if vol_sum <= 0.0:
            if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq", "mass_core"):
                mask_empty = True
            else:
                vol_sum = domain_vol
                vol_mask = vol * domain_mask if domain_mask is not None else vol
                mask_empty = True
        mask_fraction = mask_vol / domain_vol if domain_vol > 0.0 else float("nan")
        if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq") and np.isfinite(mask_fraction):
            if mask_fraction < beta_ref_mask_min:
                mask_empty = True
            if mask_fraction > beta_ref_mask_max:
                mask_warn = True

        n_avg = float("nan")
        b_avg = float("nan")
        t_avg = float("nan")
        if not mask_empty and vol_sum > 0.0:
            n_avg = float(np.sum(n * vol_mask) / vol_sum)
            b_avg = float(np.sum(b_mag * vol_mask) / vol_sum)
            if press is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    temp = press / (n * K_B)
                t_avg = float(np.sum(temp * vol_mask) / vol_sum)
        if mask_empty and mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq", "mass_core"):
            vol_sum = float("nan")
        if mask_mode == "mass_core" and vol_sum > 0.0 and mask_mass > 0.0:
            rho_avg_core = mask_mass / vol_sum

        beta_only_frac = float("nan")
        rho_only_frac = float("nan")
        press_only_frac = float("nan")
        if beta_mask is not None and domain_vol > 0.0:
            beta_only_frac = float(np.sum(vol * beta_mask) / domain_vol)
        if rho_mask is not None and domain_vol > 0.0:
            rho_only_frac = float(np.sum(vol * rho_mask) / domain_vol)
        if press_mask is not None and domain_vol > 0.0:
            press_only_frac = float(np.sum(vol * press_mask) / domain_vol)

        series.append(
            {
                "time_s": float(time_val) if time_val is not None else float("nan"),
                "volume_m3": vol_sum,
                "mask_volume_m3": mask_vol,
                "mask_volume_fraction": mask_fraction,
                "mask_cell_count": mask_count,
                "mask_empty": mask_empty,
                "mask_empty_combined": mask_empty,
                "mask_warn": mask_warn,
                "mask_fraction_beta_only": beta_only_frac,
                "mask_fraction_rho_only": rho_only_frac,
                "mask_fraction_press_only": press_only_frac,
                "mask_fraction_combined": mask_fraction,
                "mask_mass": mask_mass,
                "mask_mass_fraction": mask_mass_fraction,
                "rho_avg_core": rho_avg_core,
                "n_avg_m3": n_avg,
                "T_avg_K": t_avg,
                "B_avg_T": b_avg,
                "Bext_mag_T": bext_mag,
                "Bext_bz_median_T": bext_bz_median,
                "Bext_sign": bext_sign_mask,
                "Bext_sign_boundary": bext_sign_boundary,
                "Bext_sign_valid": bext_sign_valid,
                "Bext_frac_waveform": wf_vals["frac"],
                "Bext_delta_waveform_T": wf_vals["delta"],
                "Bext_total_waveform_T": wf_vals["total"],
                "Bext_progress": bext_progress,
                "Bext_progress_source": bext_progress_source,
                "wall_depth_m": wall_depth if wall_depth is not None else float("nan"),
                "beta_ref_threshold": beta_ref_thr,
                "beta_ref_quantile": beta_ref_quantile,
                "beta_ref_time_s": beta_ref_time,
                "beta_ref_progress": beta_ref_progress,
                "beta_ref_progress_max": beta_ref_progress_max,
                "beta_ref_mask_min": beta_ref_mask_min,
                "beta_ref_mask_max": beta_ref_mask_max,
                "rho_quantile": rho_q_used,
                "rho_threshold": rho_thr,
                "press_quantile": press_q_used,
                "press_threshold": press_thr,
            }
        )
    return series


def _infer_amu_from_metadata(diag_root: Path) -> Optional[float]:
    candidates = sorted(diag_root.glob("warpx_run_*.json"))
    if not candidates:
        return None
    try:
        with candidates[-1].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        args = data.get("args", {}) if isinstance(data, dict) else {}
        return float(args.get("ion_amu", 1.0))
    except Exception:
        return None


def _warpx_series(
    diag_root: Path,
    threshold: float,
    amu: float,
    t_const_K: Optional[float],
    mask_mode: str,
    excl_bz_frac: float,
    excl_ncell: int,
    excl_component: str,
    bext_waveform: Optional[dict],
    bext_waveform_kind: str,
    bext_waveform_scale: float,
    bext_waveform_bias: float,
    bext_sign_source: str,
    bext_sign_frac_threshold: float,
    bext_progress_source: str,
    beta_ref_quantile: float,
    beta_ref_progress_max: float,
    beta_ref_mask_min: float,
    beta_ref_mask_max: float,
    rho_quantile: float,
    press_quantile: float,
    mass_fraction: float,
):
    try:
        import yt
    except Exception as exc:
        raise SystemExit(f"yt not available: {exc}")

    diags = sorted([p for p in diag_root.iterdir() if p.is_dir() and p.name.startswith("diag")])
    if not diags:
        raise SystemExit(f"No diag* under {diag_root}")

    beta_ref_thr = float("nan")
    beta_ref_time = float("nan")
    beta_ref_progress = float("nan")
    if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq"):
        beta_ref_thr, beta_ref_time, beta_ref_progress = _beta_ref_threshold_warpx(
            diags,
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
            bext_sign_source,
            bext_sign_frac_threshold,
            beta_ref_quantile,
            beta_ref_progress_max,
            amu,
            t_const_K,
            excl_component,
        )
        if not np.isfinite(beta_ref_thr):
            print("[kirtley] beta_ref: failed to find reference threshold; masks may be empty.")

    series = []
    for diag in diags:
        ds = yt.load(str(diag))
        left = np.asarray(ds.domain_left_edge.to_value(), dtype=np.float64)
        right = np.asarray(ds.domain_right_edge.to_value(), dtype=np.float64)
        dims = ds.domain_dimensions
        cg = ds.covering_grid(level=0, left_edge=left, dims=dims)
        rho = cg[("boxlib", "rho")].to_ndarray().squeeze()
        br = cg[("boxlib", "Br")].to_ndarray().squeeze()
        bz = cg[("boxlib", "Bz")].to_ndarray().squeeze()
        bt = cg[("boxlib", "Bt")].to_ndarray().squeeze()

        nr, nz = rho.shape
        dr = float((right[0] - left[0]) / nr)
        dz = float((right[1] - left[1]) / nz)
        r = np.linspace(left[0] + dr / 2, right[0] - dr / 2, nr)
        vol = 2.0 * np.pi * r[:, None] * dr * dz
        vol = np.broadcast_to(vol, rho.shape)

        b_mag = np.sqrt(br * br + bz * bz + bt * bt)
        n = rho / (amu * M_U)
        if excl_component == "b1":
            b_excl = br
        elif excl_component == "b2":
            b_excl = bt
        else:
            b_excl = bz
        bext_bz_median, bext_mag, bext_sign_boundary = _boundary_bz_stats(b_excl, excl_ncell)
        time_val = float(ds.current_time.to_value())
        wf_vals = _waveform_bext_values(
            time_val,
            bext_waveform,
            bext_waveform_kind,
            bext_waveform_scale,
            bext_waveform_bias,
        )
        bext_sign_mask, bext_sign_valid = _stable_bext_sign(
            bext_sign_boundary,
            bext_sign_source,
            bext_waveform_bias,
            wf_vals,
            bext_sign_frac_threshold,
        )
        if bext_progress_source == "waveform":
            bext_progress = wf_vals["frac"]
        else:
            bext_progress = bext_mag
        beta_mask = None
        rho_mask = None
        press_mask = None
        rho_thr = float("nan")
        press_thr = float("nan")

        if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq"):
            if np.isfinite(beta_ref_thr):
                beta = _compute_beta(press=None, n=n, b_mag=b_mag, t_const_K=t_const_K)
                beta_mask = beta >= beta_ref_thr
            else:
                beta_mask = np.zeros_like(rho, dtype=bool)

        if mask_mode == "mass_core":
            f_mass = threshold if np.isfinite(threshold) else mass_fraction
            mask, mask_mass, total_mass = _mass_core_mask(rho, vol, f_mass, None)
            if total_mass > 0.0:
                mask_mass_fraction = mask_mass / total_mass
        elif mask_mode == "beta_ref_rhoq":
            q = threshold if np.isfinite(threshold) else rho_quantile
            rho_q_used = q
            rho_vals = rho[np.isfinite(rho)]
            if rho_vals.size == 0:
                rho_mask = np.zeros_like(rho, dtype=bool)
            else:
                rho_thr = float(np.quantile(rho_vals, q))
                rho_mask = rho >= rho_thr
            mask = beta_mask & rho_mask
        elif mask_mode == "beta_ref_pq":
            q = threshold if np.isfinite(threshold) else press_quantile
            press_q_used = q
            if t_const_K is None:
                raise SystemExit("beta_ref_pq requires pressure or --T-const-eV.")
            press = n * K_B * t_const_K
            press_vals = press[np.isfinite(press)]
            if press_vals.size == 0:
                press_mask = np.zeros_like(rho, dtype=bool)
            else:
                press_thr = float(np.quantile(press_vals, q))
                press_mask = press >= press_thr
            mask = beta_mask & press_mask
        elif mask_mode == "beta_ref":
            mask = beta_mask
        elif mask_mode == "excl":
            mask = _excluded_flux_mask(b_excl, bext_sign_mask, bext_mag, excl_bz_frac)
        elif mask_mode == "excl_rho":
            rho_max = float(np.max(rho))
            rho_cut = threshold * rho_max if rho_max > 0.0 else float("inf")
            mask = _excluded_flux_mask(b_excl, bext_sign_mask, bext_mag, excl_bz_frac)
            mask &= rho > rho_cut
        else:
            mask = _mask_from_threshold(
                rho, b_mag, press=None, n=n, threshold=threshold,
                mode=mask_mode, t_const_K=t_const_K
            )
        domain_vol = float(np.sum(vol))
        vol_mask = vol * mask
        mask_vol = float(np.sum(vol_mask))
        mask_count = int(np.count_nonzero(mask))
        vol_sum = mask_vol
        mask_empty = False
        mask_warn = False
        if vol_sum <= 0.0:
            if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq", "mass_core"):
                mask_empty = True
            else:
                vol_sum = domain_vol
                vol_mask = vol
                mask_empty = True
        mask_fraction = mask_vol / domain_vol if domain_vol > 0.0 else float("nan")
        if mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq") and np.isfinite(mask_fraction):
            if mask_fraction < beta_ref_mask_min:
                mask_empty = True
            if mask_fraction > beta_ref_mask_max:
                mask_warn = True

        n_avg = float("nan")
        b_avg = float("nan")
        if not mask_empty and vol_sum > 0.0:
            n_avg = float(np.sum(n * vol_mask) / vol_sum)
            b_avg = float(np.sum(b_mag * vol_mask) / vol_sum)
        if mask_empty and mask_mode in ("beta_ref", "beta_ref_rhoq", "beta_ref_pq", "mass_core"):
            vol_sum = float("nan")
        if mask_mode == "mass_core" and vol_sum > 0.0 and mask_mass > 0.0:
            rho_avg_core = mask_mass / vol_sum

        beta_only_frac = float("nan")
        rho_only_frac = float("nan")
        press_only_frac = float("nan")
        if beta_mask is not None and domain_vol > 0.0:
            beta_only_frac = float(np.sum(vol * beta_mask) / domain_vol)
        if rho_mask is not None and domain_vol > 0.0:
            rho_only_frac = float(np.sum(vol * rho_mask) / domain_vol)
        if press_mask is not None and domain_vol > 0.0:
            press_only_frac = float(np.sum(vol * press_mask) / domain_vol)

        t_avg = float("nan")
        if t_const_K is not None:
            t_avg = float(t_const_K)

        series.append(
            {
                "time_s": time_val,
                "volume_m3": vol_sum,
                "mask_volume_m3": mask_vol,
                "mask_volume_fraction": mask_fraction,
                "mask_cell_count": mask_count,
                "mask_empty": mask_empty,
                "mask_empty_combined": mask_empty,
                "mask_warn": mask_warn,
                "mask_fraction_beta_only": beta_only_frac,
                "mask_fraction_rho_only": rho_only_frac,
                "mask_fraction_press_only": press_only_frac,
                "mask_fraction_combined": mask_fraction,
                "mask_mass": mask_mass,
                "mask_mass_fraction": mask_mass_fraction,
                "rho_avg_core": rho_avg_core,
                "n_avg_m3": n_avg,
                "T_avg_K": t_avg,
                "B_avg_T": b_avg,
                "Bext_mag_T": bext_mag,
                "Bext_bz_median_T": bext_bz_median,
                "Bext_sign": bext_sign_mask,
                "Bext_sign_boundary": bext_sign_boundary,
                "Bext_sign_valid": bext_sign_valid,
                "Bext_frac_waveform": wf_vals["frac"],
                "Bext_delta_waveform_T": wf_vals["delta"],
                "Bext_total_waveform_T": wf_vals["total"],
                "Bext_progress": bext_progress,
                "Bext_progress_source": bext_progress_source,
                "beta_ref_threshold": beta_ref_thr,
                "beta_ref_quantile": beta_ref_quantile,
                "beta_ref_time_s": beta_ref_time,
                "beta_ref_progress": beta_ref_progress,
                "beta_ref_progress_max": beta_ref_progress_max,
                "beta_ref_mask_min": beta_ref_mask_min,
                "beta_ref_mask_max": beta_ref_mask_max,
                "rho_quantile": rho_q_used,
                "rho_threshold": rho_thr,
                "press_quantile": press_q_used,
                "press_threshold": press_thr,
            }
        )
    return series


def compute_scaling(series, exp_n, exp_t, exp_b, compression_power, gamma):
    if not series:
        raise SystemExit("No data points for scaling.")
    v0 = series[0]["volume_m3"]
    n0 = series[0]["n_avg_m3"]
    t0 = series[0]["T_avg_K"]
    b0 = series[0]["B_avg_T"]
    pvg0 = float("nan")
    if np.isfinite(t0):
        pvg0 = n0 * K_B * t0 * (v0 ** gamma)
    out = []
    for row in series:
        v = row["volume_m3"]
        comp = v0 / v if v > 0.0 else float("nan")
        c = comp ** compression_power if comp > 0.0 else float("nan")
        n_pred = n0 * (c ** exp_n) if n0 > 0.0 else float("nan")
        t_pred = t0 * (c ** exp_t) if np.isfinite(t0) else float("nan")
        b_pred = b0 * (c ** exp_b) if b0 > 0.0 else float("nan")

        def log_err(meas, pred):
            if meas > 0.0 and pred > 0.0 and np.isfinite(meas) and np.isfinite(pred):
                return float(np.log10(meas / pred))
            return float("nan")

        n_err = log_err(row["n_avg_m3"], n_pred)
        t_err = log_err(row["T_avg_K"], t_pred)
        b_err = log_err(row["B_avg_T"], b_pred)

        pvg = float("nan")
        if np.isfinite(row["T_avg_K"]):
            p = row["n_avg_m3"] * K_B * row["T_avg_K"]
            pvg = p * (v ** gamma)
        pvg_ratio = float("nan")
        if np.isfinite(pvg) and np.isfinite(pvg0) and pvg0 != 0.0:
            pvg_ratio = pvg / pvg0

        out.append(
            {
                **row,
                "compression_ratio": comp,
                "compression_c": c,
                "n_pred_m3": n_pred,
                "T_pred_K": t_pred,
                "B_pred_T": b_pred,
                "log10_err_n": n_err,
                "log10_err_T": t_err,
                "log10_err_B": b_err,
                "pV_gamma": pvg,
                "pV_gamma_ratio": pvg_ratio,
            }
        )
    return out


def write_outputs(rows, output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return csv_path, json_path


def _median_smooth(values: np.ndarray, window: int):
    if window <= 1:
        return np.array(values, dtype=float)
    if window % 2 == 0:
        raise SystemExit("fit_smooth_window must be odd.")
    n = len(values)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def _filter_fit_rows(rows, fit_start: Optional[float], fit_end: Optional[float],
                     comp_min: Optional[float], comp_max: Optional[float],
                     progress_min: Optional[float], progress_max: Optional[float]):
    filtered = []
    for r in rows:
        if r.get("mask_empty"):
            continue
        t = r.get("time_s", float("nan"))
        comp = r.get("compression_ratio", float("nan"))
        prog = r.get("Bext_progress", float("nan"))
        if fit_start is not None and np.isfinite(t) and t < fit_start:
            continue
        if fit_end is not None and np.isfinite(t) and t > fit_end:
            continue
        if comp_min is not None and np.isfinite(comp) and comp < comp_min:
            continue
        if comp_max is not None and np.isfinite(comp) and comp > comp_max:
            continue
        if progress_min is not None:
            if not np.isfinite(prog) or prog < progress_min:
                continue
        if progress_max is not None:
            if not np.isfinite(prog) or prog > progress_max:
                continue
        filtered.append(r)
    return filtered


def _select_monotonic_segment(rows, vol_key: str, smooth_window: int,
                              allow_inc_frac: float, rebound_frac: float,
                              min_points: int):
    valid_rows = []
    vols = []
    for r in rows:
        v = r.get(vol_key, float("nan"))
        if np.isfinite(v) and v > 0.0:
            valid_rows.append(r)
            vols.append(float(v))
    vols = np.array(vols, dtype=float)
    if vols.size == 0:
        return [], np.array([]), {"fit_segment_selected": False}
    if min_points > vols.size:
        return valid_rows, vols, {
            "fit_segment_selected": False,
            "fit_segment_reason": "min_points_gt_available",
        }
    smooth = _median_smooth(vols, smooth_window)

    best = None
    n = len(smooth)
    for i in range(n - 1):
        pos_count = 0
        pos_sum = 0.0
        for j in range(i + 1, n):
            dv = smooth[j] - smooth[j - 1]
            if dv > 0.0:
                pos_count += 1
                pos_sum += dv
            length = j - i + 1
            if length < min_points:
                continue
            total_drop = smooth[i] - smooth[j]
            if total_drop <= 0.0:
                continue
            if pos_count <= allow_inc_frac * (length - 1) and pos_sum <= rebound_frac * total_drop:
                if best is None or length > best["length"] or (
                    length == best["length"] and total_drop > best["drop"]
                ):
                    best = {
                        "start": i,
                        "end": j,
                        "length": length,
                        "drop": float(total_drop),
                        "pos_count": int(pos_count),
                        "pos_sum": float(pos_sum),
                    }
    if best is None:
        return valid_rows, smooth, {
            "fit_segment_selected": False,
            "fit_segment_reason": "no_segment",
        }

    seg_rows = valid_rows[best["start"] : best["end"] + 1]
    seg_vols = smooth[best["start"] : best["end"] + 1]
    info = {
        "fit_segment_selected": True,
        "fit_segment_start_index": best["start"],
        "fit_segment_end_index": best["end"],
        "fit_segment_length": best["length"],
        "fit_segment_drop": best["drop"],
        "fit_segment_pos_count": best["pos_count"],
        "fit_segment_pos_sum": best["pos_sum"],
    }
    if seg_rows:
        info.update(
            {
                "fit_segment_start_time_s": seg_rows[0].get("time_s"),
                "fit_segment_end_time_s": seg_rows[-1].get("time_s"),
                "fit_segment_start_progress": seg_rows[0].get("Bext_progress"),
                "fit_segment_end_progress": seg_rows[-1].get("Bext_progress"),
            }
        )
    return seg_rows, seg_vols, info


def fit_exponents(rows, fit_start: Optional[float], fit_end: Optional[float],
                  comp_min: Optional[float], comp_max: Optional[float],
                  vol_override: Optional[np.ndarray] = None,
                  prefiltered: bool = False):
    xs = []
    ys_n = []
    ys_b = []
    ys_t = []
    vols = []
    vol_idx = 0
    for r in rows:
        if r.get("mask_empty") and not prefiltered:
            continue
        if not prefiltered:
            t = r.get("time_s", float("nan"))
            comp = r.get("compression_ratio", float("nan"))
            if fit_start is not None and np.isfinite(t) and t < fit_start:
                continue
            if fit_end is not None and np.isfinite(t) and t > fit_end:
                continue
            if comp_min is not None and np.isfinite(comp) and comp < comp_min:
                continue
            if comp_max is not None and np.isfinite(comp) and comp > comp_max:
                continue
        if vol_override is not None:
            if vol_idx >= len(vol_override):
                break
            v = float(vol_override[vol_idx])
        else:
            v = r.get("volume_m3", float("nan"))
        n = r.get("n_avg_m3", float("nan"))
        b = r.get("B_avg_T", float("nan"))
        tt = r.get("T_avg_K", float("nan"))
        if v <= 0.0:
            vol_idx += 1
            continue
        logv = np.log(v)
        if n > 0.0:
            xs.append(logv)
            ys_n.append(np.log(n))
            ys_b.append(np.log(b) if b > 0.0 else np.nan)
            ys_t.append(np.log(tt) if tt > 0.0 else np.nan)
            vols.append(v)
        vol_idx += 1
    xs = np.array(xs, dtype=float)
    if xs.size < 2:
        return None

    def _fit(y_vals):
        y = np.array(y_vals, dtype=float)
        mask = np.isfinite(xs) & np.isfinite(y)
        if np.sum(mask) < 2:
            return None, None
        coeff = np.polyfit(xs[mask], y[mask], 1)
        slope = float(coeff[0])
        y_hat = coeff[0] * xs[mask] + coeff[1]
        ss_res = float(np.sum((y[mask] - y_hat) ** 2))
        ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        return slope, r2

    inc = 0
    dec = 0
    for i in range(1, len(vols)):
        if vols[i] > vols[i - 1]:
            inc += 1
        elif vols[i] < vols[i - 1]:
            dec += 1
    monotonic = inc == 0
    log_span = float(np.log(vols[-1] / vols[0])) if vols[0] > 0.0 and vols[-1] > 0.0 else float("nan")

    alpha_n, r2_n = _fit(ys_n)
    alpha_B, r2_B = _fit(ys_b)
    alpha_T, r2_T = _fit(ys_t)

    return {
        "alpha_n": alpha_n,
        "alpha_B": alpha_B,
        "alpha_T": alpha_T,
        "r2_n": r2_n,
        "r2_B": r2_B,
        "r2_T": r2_T,
        "n_points": int(xs.size),
        "vol_increases": int(inc),
        "vol_decreases": int(dec),
        "vol_monotonic": bool(monotonic),
        "vol_log_span": log_span,
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Kirtley 0D scaling check")
    ap.add_argument("--mode", choices=["athena", "warpx"], required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="Athena run directory.")
    ap.add_argument("--vtk-pattern", default="*.vtk")
    ap.add_argument("--diag-path", type=Path, default=None, help="WarpX diag root.")
    ap.add_argument(
        "--mask-mode",
        choices=["rho", "beta", "beta_rel", "beta_ref", "beta_ref_rhoq", "beta_ref_pq", "mass_core",
                 "excl", "excl_rho"],
        default="rho",
    )
    ap.add_argument("--mask-threshold", type=float, default=0.1)
    ap.add_argument("--mask-thresholds", type=str, default=None, help="Comma-separated thresholds to scan.")
    ap.add_argument("--excl-bz-frac", type=float, default=0.05, help="Excluded-flux Bz fraction of boundary median.")
    ap.add_argument("--excl-ncell", type=int, default=1, help="Boundary cells to sample for Bext.")
    ap.add_argument("--rho-frac", type=float, default=0.15, help="rho threshold fraction for excl_rho mode.")
    ap.add_argument("--excl-component", choices=["b1", "b2", "b3"], default="b3",
                    help="Field component for excluded-flux mask.")
    ap.add_argument("--fold-r", action="store_true", help="Fold negative r for Athena VTK.")
    ap.add_argument("--amu", type=float, default=None, help="Ion mass number A.")
    ap.add_argument("--T-const-eV", type=float, default=None, help="Use constant T (eV) for WarpX.")
    ap.add_argument("--bext-waveform", type=Path, default=None,
                    help="CSV waveform for external ramp progress (t,value).")
    ap.add_argument("--bext-waveform-column", type=int, default=1,
                    help="Value column index in waveform CSV (1-based, after time).")
    ap.add_argument("--bext-waveform-kind", choices=["fraction", "delta", "total"],
                    default="fraction", help="Waveform values represent frac, delta, or total B.")
    ap.add_argument("--bext-waveform-scale", type=float, default=1.0,
                    help="Scale applied to fraction waveform (delta = scale*frac).")
    ap.add_argument("--bext-waveform-bias", type=float, default=0.0,
                    help="Bias field added to waveform delta.")
    ap.add_argument("--bext-progress-source", choices=["auto", "waveform", "boundary"],
                    default="auto", help="Progress source for output diagnostics.")
    ap.add_argument("--bext-sign-source", choices=["auto", "waveform", "boundary", "bias"],
                    default="auto", help="Sign source for excluded-flux mask.")
    ap.add_argument("--bext-sign-frac-threshold", type=float, default=0.02,
                    help="Minimum |waveform frac| before trusting waveform sign.")
    ap.add_argument("--wall-depth-max", type=float, default=None,
                    help="Optional moving-wall depth max; applied as depth=frac*wall-depth-max.")
    ap.add_argument("--r-max-core", type=float, default=None,
                    help="Optional cylindrical r cutoff for 3D core stats (same units as x2/x3).")
    ap.add_argument("--beta-ref-quantile", type=float, default=0.70,
                    help="Reference beta quantile for beta_ref mask.")
    ap.add_argument("--beta-ref-progress-max", type=float, default=0.02,
                    help="Max waveform progress for beta_ref reference snapshot.")
    ap.add_argument("--beta-ref-mask-min", type=float, default=0.02,
                    help="Minimum mask fraction before marking beta_ref mask empty.")
    ap.add_argument("--beta-ref-mask-max", type=float, default=0.90,
                    help="Mask fraction threshold to warn for beta_ref masks.")
    ap.add_argument("--rho-quantile", type=float, default=0.80,
                    help="Quantile for rho core gate in beta_ref_rhoq.")
    ap.add_argument("--press-quantile", type=float, default=0.80,
                    help="Quantile for pressure core gate in beta_ref_pq.")
    ap.add_argument("--mass-fraction", type=float, default=0.30,
                    help="Mass fraction for mass_core gate.")
    ap.add_argument("--exp-n", type=float, default=3.0)
    ap.add_argument("--exp-T", type=float, default=2.0)
    ap.add_argument("--exp-B", type=float, default=2.0)
    ap.add_argument("--compression-power", type=float, default=1.0 / 3.0)
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument(
        "--output-prefix",
        type=Path,
        default=repo_root / "outputs" / "analysis" / "kirtley_scaling",
    )
    ap.add_argument("--fit-start", type=float, default=None, help="Fit start time (s).")
    ap.add_argument("--fit-end", type=float, default=None, help="Fit end time (s).")
    ap.add_argument("--fit-comp-min", type=float, default=None, help="Fit min compression ratio.")
    ap.add_argument("--fit-comp-max", type=float, default=None, help="Fit max compression ratio.")
    ap.add_argument("--fit-progress-min", type=float, default=None, help="Fit min ramp progress.")
    ap.add_argument("--fit-progress-max", type=float, default=None, help="Fit max ramp progress.")
    ap.add_argument("--fit-segment", choices=["none", "monotonic"], default="none",
                    help="Select a monotonic compression segment before fitting.")
    ap.add_argument("--fit-smooth-window", type=int, default=3,
                    help="Odd window size for median smoothing (volume).")
    ap.add_argument("--fit-allow-increase-frac", type=float, default=0.1,
                    help="Max fraction of increases allowed in the fit segment.")
    ap.add_argument("--fit-rebound-frac", type=float, default=0.05,
                    help="Max rebound fraction of total drop in the fit segment.")
    ap.add_argument("--fit-min-points", type=int, default=10,
                    help="Minimum points required for fit segment selection.")
    args = ap.parse_args()

    if args.mask_thresholds:
        thresholds = [float(x) for x in args.mask_thresholds.split(",") if x.strip()]
    else:
        if args.mask_mode == "beta_ref":
            thresholds = [args.beta_ref_quantile]
        elif args.mask_mode == "beta_ref_rhoq":
            thresholds = [args.rho_quantile]
        elif args.mask_mode == "beta_ref_pq":
            thresholds = [args.press_quantile]
        elif args.mask_mode == "mass_core":
            thresholds = [args.mass_fraction]
        elif args.mask_mode == "excl":
            thresholds = [args.excl_bz_frac]
        elif args.mask_mode == "excl_rho":
            thresholds = [args.rho_frac]
        else:
            thresholds = [args.mask_threshold]

    bext_waveform = _read_waveform_csv(args.bext_waveform, args.bext_waveform_column)
    bext_progress_source = args.bext_progress_source
    if bext_progress_source == "auto":
        bext_progress_source = "waveform" if bext_waveform is not None else "boundary"
    if bext_progress_source == "waveform" and bext_waveform is None:
        print("[kirtley] waveform progress requested but no waveform file provided; using boundary.")
        bext_progress_source = "boundary"
    bext_sign_source = args.bext_sign_source
    if bext_sign_source == "auto":
        bext_sign_source = "waveform" if bext_waveform is not None else "boundary"
    if bext_sign_source == "waveform" and bext_waveform is None:
        print("[kirtley] waveform sign requested but no waveform file provided; using boundary.")
        bext_sign_source = "boundary"

    for thr in thresholds:
        beta_ref_quantile = args.beta_ref_quantile
        if args.mask_mode == "beta_ref":
            beta_ref_quantile = thr

        if args.mode == "athena":
            if args.run_dir is None:
                raise SystemExit("--run-dir required for athena mode")
            amu = args.amu if args.amu is not None else 2.0
            series = _athena_series(
                args.run_dir, args.vtk_pattern, thr, amu, args.fold_r, args.mask_mode,
                args.excl_bz_frac, args.excl_ncell, args.excl_component,
                bext_waveform, args.bext_waveform_kind, args.bext_waveform_scale,
                args.bext_waveform_bias, bext_sign_source, args.bext_sign_frac_threshold,
                bext_progress_source, args.wall_depth_max, args.r_max_core,
                beta_ref_quantile, args.beta_ref_progress_max,
                args.beta_ref_mask_min, args.beta_ref_mask_max,
                args.rho_quantile, args.press_quantile, args.mass_fraction,
            )
            t_const = None
        else:
            if args.diag_path is None:
                raise SystemExit("--diag-path required for warpx mode")
            amu = args.amu if args.amu is not None else (_infer_amu_from_metadata(args.diag_path) or 1.0)
            t_const = None
            if args.T_const_eV is not None:
                t_const = args.T_const_eV * 11604.518
            series = _warpx_series(
                args.diag_path, thr, amu, t_const, args.mask_mode,
                args.excl_bz_frac, args.excl_ncell, args.excl_component,
                bext_waveform, args.bext_waveform_kind, args.bext_waveform_scale,
                args.bext_waveform_bias, bext_sign_source, args.bext_sign_frac_threshold,
                bext_progress_source,
                beta_ref_quantile, args.beta_ref_progress_max,
                args.beta_ref_mask_min, args.beta_ref_mask_max,
                args.rho_quantile, args.press_quantile, args.mass_fraction,
            )

        rows = compute_scaling(series, args.exp_n, args.exp_T, args.exp_B, args.compression_power, args.gamma)
        suffix = f"_{args.mask_mode}{str(thr).replace('.', 'p')}"
        out_prefix = args.output_prefix if len(thresholds) == 1 else args.output_prefix.with_name(
            args.output_prefix.name + suffix
        )
        csv_path, json_path = write_outputs(rows, out_prefix)
        fit_rows = _filter_fit_rows(
            rows,
            args.fit_start,
            args.fit_end,
            args.fit_comp_min,
            args.fit_comp_max,
            args.fit_progress_min,
            args.fit_progress_max,
        )
        segment_info = {
            "fit_progress_min": args.fit_progress_min,
            "fit_progress_max": args.fit_progress_max,
            "fit_segment": args.fit_segment,
            "fit_smooth_window": args.fit_smooth_window,
            "fit_allow_increase_frac": args.fit_allow_increase_frac,
            "fit_rebound_frac": args.fit_rebound_frac,
            "fit_min_points": args.fit_min_points,
            "fit_rows_total": int(len(rows)),
            "fit_rows_filtered": int(len(fit_rows)),
        }
        vol_override = None
        if args.fit_segment == "monotonic":
            fit_rows, vol_override, seg = _select_monotonic_segment(
                fit_rows,
                "volume_m3",
                args.fit_smooth_window,
                args.fit_allow_increase_frac,
                args.fit_rebound_frac,
                args.fit_min_points,
            )
            segment_info.update(seg)
        elif args.fit_smooth_window > 1 and fit_rows:
            vols = np.array([r.get("volume_m3", float("nan")) for r in fit_rows], dtype=float)
            vol_override = _median_smooth(vols, args.fit_smooth_window)

        fit = None
        if fit_rows:
            fit = fit_exponents(
                fit_rows,
                None,
                None,
                None,
                None,
                vol_override=vol_override,
                prefiltered=True,
            )
        if fit is not None:
            fit_path = out_prefix.with_suffix(".fit.json")
            with fit_path.open("w", encoding="utf-8") as fh:
                fit.update(segment_info)
                json.dump(fit, fh, indent=2)
            print(f"[kirtley] wrote {csv_path}, {json_path}, {fit_path}")
        else:
            print(f"[kirtley] wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
