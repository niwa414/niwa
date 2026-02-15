#!/usr/bin/env python3
"""
WarpX External Field Driver (RZ Geometry)
-----------------------------------------
This script now supports multiple modes:
  - field-only:     load an external B field, no particles (sanity on field I/O)
  - const-b-plasma: uniform cold plasma + analytic constant Bz
  - bfile-plasma:   uniform cold plasma + B field from openPMD file
  - full-driver:    original LCR-waveform-driven external field with particles
  - fluid-init:     seed plasma from an openPMD fluid file (rho/vr/vz/vphi/T)

Examples:
  python3 warpx_driver.py --mode field-only --max-steps 5
  python3 warpx_driver.py --mode const-b-plasma --const-B 0.05 --max-steps 20
  python3 warpx_driver.py --mode bfile-plasma --b-file warpx-driver/B_ext.h5
  python3 warpx_driver.py --mode full-driver --waveform outputs/analysis/lcr_waveform.csv
  python3 warpx_driver.py --mode fluid-init --fluid-file warpx-driver/fluid_init.h5 --b-file warpx-driver/B_ext_from_frc.h5
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pywarpx import callbacks, fields, libwarpx, my_constants, particle_containers, picmi, warpx

# ----------------------------------------------------------------------------- 
# Defaults
# -----------------------------------------------------------------------------

# ==============================================================================
# 1. Configuration & Waveform Loading
# ==============================================================================

# Simulation Parameters
MAX_STEPS = 100
# Pick a conservative default dt from CFL (dr~3.1e-3, dz~3.1e-3 -> dt ~7e-12 for Yee)
DT = 5.0e-12  # Time step (s) - overridden to respect CFL if larger
R_MAX = 0.1  # Domain radius (m)
Z_MAX = 0.2  # Domain length (m)
NZ = 64
NR = 32
CMDLINE = " ".join(sys.argv)

# Resolve repo root (one level above this script)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Waveform File (Example CSV format: t, ..., B_est, ...)
# Allow override via env var for quick coupling to LCR output.
DEFAULT_WAVEFORM = REPO_ROOT / "outputs" / "analysis" / "lcr_waveform.csv"
WAVEFORM_FILE = os.environ.get("WAVEFORM_FILE", str(DEFAULT_WAVEFORM))
# External B field file (openPMD) to load via WarpX.
DEFAULT_B_FILE = SCRIPT_DIR / "B_ext.h5"
B_FIELD_FILE = os.environ.get("B_FIELD_FILE", str(DEFAULT_B_FILE))
# Fluid file (openPMD) for fluid-init mode.
DEFAULT_FLUID_FILE = SCRIPT_DIR / "fluid_init.h5"


def git_info(repo_root: Path):
    """Return (hash, dirty) if the directory is a git repo; otherwise (None, False)."""
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            )
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(status)
        return head, dirty
    except Exception:
        return None, False

# -----------------------------------------------------------------------------
# LCR helper
# -----------------------------------------------------------------------------


class LCRModel:
    """
    Minimal LCR plasma/coil coupling used to drive B(t) self-consistently.
    This matches the standalone lcr_coupling.py logic with a linear R(t) ramp.
    """

    def __init__(self, args):
        self.V0 = args.lcr_V0
        self.C = args.lcr_C
        self.R_line = args.lcr_R_line
        self.L0 = args.lcr_L0
        self.L_alpha = args.lcr_L_alpha
        self.R_plasma0 = args.lcr_R_plasma0
        self.R_min = args.lcr_R_min
        self.v_ramp = args.lcr_v_ramp
        self.turns = args.lcr_turns
        self.R_coil = args.lcr_R_coil
        self.dt = args.dt
        self.kB = args.lcr_kB if args.lcr_kB is not None else (4e-7 * math.pi * self.turns / self.R_coil)
        self.feedback_mode = getattr(args, "lcr_feedback", "off")
        self.feedback_alpha = getattr(args, "lcr_feedback_alpha", 0.5)
        self.feedback_min = getattr(args, "lcr_feedback_min", None)
        self.feedback_max = getattr(args, "lcr_feedback_max", None)

        self.time = 0.0
        self.I = 0.0
        self.Q = self.C * self.V0
        self.history = []
        self.feedback_updates = 0
        self.feedback_signal = None
        self.feedback_time = None
        self.R_plasma_feedback = None

        if self.v_ramp > 0.0:
            self.t_compress_end = max(1e-30, (self.R_plasma0 - self.R_min) / self.v_ramp)
        else:
            self.t_compress_end = None

    def _ramp_radius(self, t):
        if self.t_compress_end is None:
            return self.R_plasma0
        if t < 0.0:
            return self.R_plasma0
        if t <= self.t_compress_end:
            return self.R_plasma0 + (self.R_min - self.R_plasma0) * (t / self.t_compress_end)
        return self.R_min

    def _feedback_bounds(self):
        r_min = min(self.R_plasma0, self.R_min)
        r_max = max(self.R_plasma0, self.R_min)
        if self.feedback_min is not None:
            r_min = float(self.feedback_min)
        if self.feedback_max is not None:
            r_max = float(self.feedback_max)
        return r_min, r_max

    def update_feedback(self, radius, t):
        if radius is None or not np.isfinite(radius):
            return False
        alpha = 0.0 if self.feedback_alpha is None else float(self.feedback_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        if self.R_plasma_feedback is None:
            blended = float(radius)
        else:
            blended = (1.0 - alpha) * self.R_plasma_feedback + alpha * float(radius)
        r_min, r_max = self._feedback_bounds()
        blended = max(r_min, min(r_max, blended))
        self.R_plasma_feedback = blended
        self.feedback_signal = float(radius)
        self.feedback_time = float(t)
        self.feedback_updates += 1
        return True

    def plasma_radius(self, t):
        if self.feedback_mode != "off" and self.R_plasma_feedback is not None:
            return self.R_plasma_feedback, "feedback"
        return self._ramp_radius(t), "ramp"

    def step(self):
        t = self.time
        R_plasma, r_source = self.plasma_radius(t)
        dR_dt = 0.0
        if r_source == "ramp" and self.t_compress_end is not None and t <= self.t_compress_end:
            dR_dt = (self.R_min - self.R_plasma0) / self.t_compress_end

        L = self.L0 + self.L_alpha * R_plasma
        dL_dt = self.L_alpha * dR_dt
        V_cap = self.Q / self.C

        dI_dt = (V_cap - self.I * (self.R_line + dL_dt)) / max(1e-30, L)
        I_new = self.I + dI_dt * self.dt
        Q_new = self.Q - self.I * self.dt

        B_est = self.kB * self.I
        E_cap = 0.5 * self.Q * self.Q / self.C
        E_ind = 0.5 * L * self.I * self.I

        self.history.append(
            {
                "t": t,
                "I": self.I,
                "V_cap": V_cap,
                "L": L,
                "dL_dt": dL_dt,
                "R_plasma": R_plasma,
                "R_plasma_source": r_source,
                "feedback_signal": self.feedback_signal,
                "feedback_time": self.feedback_time,
                "feedback_used": r_source == "feedback",
                "B_est": B_est,
                "E_cap": E_cap,
                "E_ind": E_ind,
            }
        )

        self.I = I_new
        self.Q = Q_new
        self.time += self.dt
        return B_est


class RunMonitor:
    """Lightweight per-step monitor for species totals, drops, modal energy, and centroid."""

    def __init__(self, species_names, interval, drop_threshold=None, abort_on_drop=False, grid=None, args=None):
        self.species_names = species_names or []
        self.interval = max(1, interval) if interval else None
        self.records = []
        self.last_dropped = 0
        self.drop_threshold = drop_threshold
        self.abort_on_drop = abort_on_drop
        self.drop_breach = False
        self.args = args
        self.grid = grid
        self._mode_diag_error = False
        self._rho_error = False
        self._centroid_error = False
        self._radius_error = False
        self.last_radius_rms = None
        self.last_radius_rms_time = None
        if args:
            self.dr = args.r_max / args.nr
            self.dz = args.z_max / args.nz
            self.r_centers = (np.arange(args.nr) + 0.5) * self.dr
            z0 = -0.5 * args.z_max
            self.z_centers = z0 + (np.arange(args.nz) + 0.5) * self.dz
            self.cell_volume = 2.0 * np.pi * self.r_centers[:, None] * self.dr * self.dz
            max_modes = getattr(args, "n_azimuthal_modes", 0) or 0
            self.modes = list(range(max_modes))
        else:
            self.dr = self.dz = None
            self.r_centers = self.z_centers = None
            self.cell_volume = None
            self.modes = []

    def maybe_record(self, step_idx, t_current):
        if self.interval is None or (step_idx % self.interval) != 0:
            return
        entry = {"step": step_idx, "time": t_current}
        dropped_total = None
        try:
            wx_instance = libwarpx.warpx.get_instance()
            dropped_total = wx_instance.dropped_particles_total
        except Exception:
            dropped_total = None
        if dropped_total is not None:
            entry["dropped_total"] = dropped_total
            entry["dropped_delta"] = dropped_total - self.last_dropped
            self.last_dropped = dropped_total
            if (
                self.drop_threshold is not None
                and entry["dropped_delta"] > self.drop_threshold
            ):
                self.drop_breach = True
                print(
                    f"[monitor] drop spike detected: delta={entry['dropped_delta']} (> {self.drop_threshold})"
                )

        species = gather_species_stats(self.species_names)
        if species:
            entry["species"] = species
        mode_energy = self._mode_energy_snapshot()
        if mode_energy:
            entry["mode_energy"] = mode_energy
        mode_rms = self._mode_rms_snapshot()
        if mode_rms:
            entry["mode_rms"] = mode_rms
        centroid = self._charge_centroid()
        if centroid:
            entry["charge_centroid"] = centroid
        radius_rms = self._radius_rms()
        if radius_rms is not None:
            entry["radius_rms"] = radius_rms
            self.last_radius_rms = radius_rms
            self.last_radius_rms_time = t_current
        self.records.append(entry)

    def should_abort(self):
        return self.abort_on_drop and self.drop_breach

    def as_dict(self):
        return {
            "interval": self.interval,
            "records": self.records,
            "drop_threshold": self.drop_threshold,
            "drop_breach": self.drop_breach,
            "modes_tracked": self.modes,
        }

    def _abs2(self, arr):
        try:
            arr = np.asarray(arr[:])
        except Exception:
            arr = np.asarray(arr)
        if arr.ndim >= 1 and arr.shape[-1] == 2:
            return arr[..., 0] * arr[..., 0] + arr[..., 1] * arr[..., 1]
        return arr * arr

    def _mode_energy_snapshot(self):
        if not self.args or self.cell_volume is None:
            return None
        if len(self.modes) > 1:
            if not self._mode_diag_error:
                print("[monitor] skipping per-step mode-energy for n_azimuthal_modes>1; use diag postprocess.")
                self._mode_diag_error = True
            return None
        modes = self.modes
        if not modes:
            return None
        energies = {}
        try:
            for m in modes:
                comps = [
                    self._abs2(fields.BxFPWrapper(m)),
                    self._abs2(fields.ByFPWrapper(m)),
                    self._abs2(fields.BzFPWrapper(m)),
                    self._abs2(fields.ExFPWrapper(m)),
                    self._abs2(fields.EyFPWrapper(m)),
                    self._abs2(fields.EzFPWrapper(m)),
                ]
                nr = min(a.shape[0] for a in comps + [self.cell_volume])
                nz = min(a.shape[1] for a in comps + [self.cell_volume])
                comps = [a[:nr, :nz] for a in comps]
                total = comps[0] + comps[1] + comps[2] + comps[3] + comps[4] + comps[5]
                energy = 0.5 * float(np.sum(total * self.cell_volume[:nr, :nz]))
                energies[f"m{m}"] = energy
        except Exception as exc:
            if not self._mode_diag_error:
                print(f"[monitor] mode-energy unavailable: {exc}")
                self._mode_diag_error = True
            return None
        return energies

    def _rho_components(self):
        if not self.args or self.cell_volume is None:
            return None
        rho_raw = None
        last_exc = None
        for wrapper in (fields.RhoCPWrapper, fields.RhoFPWrapper):
            try:
                rho_raw = np.asarray(wrapper()[:])
                break
            except Exception as exc:
                last_exc = exc
        if rho_raw is None:
            if not self._rho_error:
                print(f"[monitor] rho field unavailable: {last_exc}")
                self._rho_error = True
            return None

        nr = self.cell_volume.shape[0]
        nz = self.cell_volume.shape[1]
        shape = rho_raw.shape
        r_axes = [i for i, s in enumerate(shape) if s == nr]
        z_axes = [i for i, s in enumerate(shape) if s == nz]
        if not r_axes or not z_axes:
            if rho_raw.size == nr * nz:
                return rho_raw.reshape(nr, nz, 1)
            return None

        r_axis = r_axes[0]
        z_axis = z_axes[0] if z_axes[0] != r_axis else (z_axes[1] if len(z_axes) > 1 else None)
        if z_axis is None:
            return None

        rest_axes = [i for i in range(rho_raw.ndim) if i not in (r_axis, z_axis)]
        src_axes = (r_axis, z_axis, *rest_axes)
        dst_axes = (0, 1, *range(2, 2 + len(rest_axes)))
        rho = np.moveaxis(rho_raw, src_axes, dst_axes)
        if rest_axes:
            rho = rho.reshape(nr, nz, -1)
        else:
            rho = rho.reshape(nr, nz, 1)
        return rho

    def _mode_rms_snapshot(self):
        if not self.args or self.cell_volume is None:
            return None
        if not self.modes:
            return None
        metrics = {}
        try:
            rho = self._rho_components()
            if rho is None:
                return self._particle_mode_snapshot()
            ncomp = rho.shape[2]
            if ncomp >= 1:
                amp0 = np.abs(rho[:, :, 0])
                metrics["rho_m0_rms"] = float(np.sqrt(np.mean(amp0 * amp0)))
                metrics["rho_m0_max"] = float(np.max(amp0))
            if ncomp >= 3 and len(self.modes) > 1:
                real = rho[:, :, 1]
                imag = rho[:, :, 2]
                amp1 = np.sqrt(real * real + imag * imag)
                metrics["rho_m1_rms"] = float(np.sqrt(np.mean(amp1 * amp1)))
                metrics["rho_m1_max"] = float(np.max(amp1))
        except Exception as exc:
            if not self._mode_diag_error:
                print(f"[monitor] mode-rms unavailable: {exc}")
                self._mode_diag_error = True
            return None
        return metrics

    def _particle_mode_snapshot(self):
        if not self.species_names:
            return None
        species = "ions" if "ions" in self.species_names else self.species_names[0]
        try:
            pc = particle_containers.ParticleContainerWrapper(species)
            theta_tiles = pc.get_particle_theta(level=0, copy_to_host=True)
            weight_tiles = pc.get_particle_weight(level=0, copy_to_host=True)
        except Exception as exc:
            if not self._mode_diag_error:
                print(f"[monitor] particle-mode unavailable: {exc}")
                self._mode_diag_error = True
            return None

        total_w = 0.0
        sum_cos = 0.0
        sum_sin = 0.0
        for theta, weight in zip(theta_tiles, weight_tiles):
            theta = np.asarray(theta)
            weight = np.asarray(weight)
            if theta.size == 0:
                continue
            total_w += float(np.sum(weight))
            sum_cos += float(np.sum(weight * np.cos(theta)))
            sum_sin += float(np.sum(weight * np.sin(theta)))

        if total_w <= 0.0:
            return None
        amp = math.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / total_w
        return {
            "particle_m1_amp": float(amp),
            "particle_species": species,
        }

    def _charge_centroid(self):
        if not self.args or self.cell_volume is None:
            return None
        try:
            rho = self._rho_components()
            if rho is None:
                return None
            charge = rho[:, :, 0] * self.cell_volume
            q_tot = float(np.sum(charge))
            if abs(q_tot) < 1.0e-30:
                return None
            r_cent = float(np.sum(charge * self.r_centers[:, None]) / q_tot)
            z_cent = float(np.sum(charge * self.z_centers[None, :]) / q_tot)
            return {"r": r_cent, "z": z_cent, "q": q_tot}
        except Exception as exc:
            if not self._centroid_error:
                print(f"[monitor] centroid unavailable: {exc}")
                self._centroid_error = True
            return None

    def _radius_rms(self):
        if not self.args or self.cell_volume is None:
            return None
        try:
            rho = self._rho_components()
            if rho is not None:
                rho_abs = np.abs(rho[:, :, 0])
                weight = rho_abs * self.cell_volume
                total = float(np.sum(weight))
                if total <= 0.0:
                    return None
                r2 = (self.r_centers[:, None] ** 2) * weight
                return float(math.sqrt(np.sum(r2) / total))
        except Exception as exc:
            if not self._radius_error:
                print(f"[monitor] radius_rms unavailable: {exc}")
                self._radius_error = True
            rho = None

        # Particle fallback when rho diagnostics are unavailable.
        if not self.species_names:
            return None
        species = "ions" if "ions" in self.species_names else self.species_names[0]
        try:
            pc = particle_containers.ParticleContainerWrapper(species)
            if libwarpx.geometry_dim == "rz":
                r_tiles = pc.get_particle_real_arrays("r", level=0, copy_to_host=True)
            else:
                r_tiles = pc.get_particle_r(level=0, copy_to_host=True)
            w_tiles = pc.get_particle_weight(level=0, copy_to_host=True)
        except Exception as exc:
            if not self._radius_error:
                print(f"[monitor] radius_rms particle fallback unavailable: {exc}")
                self._radius_error = True
            return None

        total_w = 0.0
        sum_r2 = 0.0
        for r_arr, w_arr in zip(r_tiles, w_tiles):
            r_arr = np.asarray(r_arr)
            if r_arr.size == 0:
                continue
            w_arr = np.asarray(w_arr)
            if w_arr.size == 1:
                w_arr = np.full_like(r_arr, float(w_arr))
            elif w_arr.shape != r_arr.shape:
                try:
                    w_arr = np.broadcast_to(w_arr, r_arr.shape)
                except ValueError:
                    continue
            total_w += float(np.sum(w_arr))
            sum_r2 += float(np.sum(w_arr * r_arr * r_arr))
        if total_w <= 0.0:
            return None
        return float(math.sqrt(sum_r2 / total_w))

    def update_radius_rms(self, t_current):
        radius = self._radius_rms()
        if radius is not None:
            self.last_radius_rms = radius
            self.last_radius_rms_time = t_current
        return radius

# Load Waveform
def load_waveform(csv_path: Path, max_steps: int, dt: float):
    """Load B(t) waveform; fall back to a half-sine if unavailable."""
    t_wave = []
    B_wave = []

    if csv_path.exists():
        print(f"Loading waveform from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, comment="#")
            if "t" in df.columns and "B_est" in df.columns:
                t_wave = df["t"].values
                B_wave = df["B_est"].values
                print(
                    f"Loaded {len(t_wave)} points. B range: "
                    f"[{np.min(B_wave):.3f}, {np.max(B_wave):.3f}] T"
                )
            else:
                print("Warning: CSV missing columns 't' and 'B_est'; disabling waveform drive.")
        except Exception as exc:
            print(f"Error reading waveform: {exc}")
    else:
        print(f"Waveform {csv_path} not found. Using synthetic half-sine for smoke tests.")
        t_wave = np.linspace(0, max_steps * dt, max_steps + 1)
        B_wave = 1.0 * np.sin(2 * np.pi * t_wave / (max_steps * dt * 2))

    def interp_fn(t):
        if len(t_wave) == 0:
            return 0.0
        return np.interp(t, t_wave, B_wave)

    return t_wave, B_wave, interp_fn


def _split_waveform_spec(spec: str) -> tuple[Path, str | int | None]:
    """
    Parse a waveform spec of the form:
      - /path/to/file.csv
      - /path/to/file.csv:colname
      - /path/to/file.csv:1   (0-based column index)

    If the full string is an existing path, it is treated as a path (no column override).
    """
    raw = spec.strip()
    direct = Path(raw)
    if direct.exists():
        return direct, None
    if ":" not in raw:
        return Path(raw), None
    left, right = raw.rsplit(":", 1)
    path = Path(left)
    if not path.exists():
        return Path(raw), None
    col: str | int | None
    if right.isdigit():
        col = int(right)
    else:
        col = right
    return path, col


def load_scalar_waveform(spec: str, default_value: float = 0.0):
    """
    Load a scalar coefficient waveform from CSV.

    The CSV must contain a time column (prefer 't', else first column) and a value column.
    If no explicit column is given via 'path:col', we choose the first match in:
      coeff, value, I, B_est, else the second column.

    Returns: (t, values, interp_fn, info_dict)
    """
    path, col = _split_waveform_spec(spec)
    info: dict[str, object] = {"path": str(path), "col": col}
    if not path.exists():
        print(f"Warning: waveform {path} not found; using constant {default_value}.")
        t_arr = np.array([0.0], dtype=np.float64)
        v_arr = np.array([default_value], dtype=np.float64)
        return t_arr, v_arr, (lambda t: float(default_value)), {**info, "fallback": True}

    df = pd.read_csv(path, comment="#")
    if df.empty:
        raise ValueError(f"Waveform CSV is empty: {path}")

    time_col = "t" if "t" in df.columns else df.columns[0]
    if col is None:
        for candidate in ("coeff", "value", "I", "B_est"):
            if candidate in df.columns:
                value_col = candidate
                break
        else:
            if len(df.columns) < 2:
                raise ValueError(f"Waveform CSV needs at least 2 columns: {path}")
            value_col = df.columns[1]
    elif isinstance(col, int):
        if col < 0 or col >= len(df.columns):
            raise ValueError(
                f"Waveform column index {col} out of range for {path} (ncol={len(df.columns)})"
            )
        value_col = df.columns[col]
    else:
        if col not in df.columns:
            raise ValueError(
                f"Waveform column '{col}' not found in {path}. Available: {list(df.columns)}"
            )
        value_col = col

    t_arr = np.asarray(df[time_col].values, dtype=np.float64)
    v_arr = np.asarray(df[value_col].values, dtype=np.float64)
    if t_arr.size != v_arr.size or t_arr.size < 1:
        raise ValueError(f"Waveform columns have incompatible sizes in {path}")

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    v_arr = v_arr[order]
    info.update({"time_col": str(time_col), "value_col": str(value_col), "points": int(t_arr.size)})

    def interp_fn(t):
        return float(np.interp(t, t_arr, v_arr))

    return t_arr, v_arr, interp_fn, info


def _load_openpmd_B_components(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load (Br, Bt, Bz) arrays from an openPMD thetaMode B file."""
    meta = _read_b_field_metadata(path)
    with h5py.File(path, "r") as h5f:
        base = h5f["/data/0/meshes/B"]
        comps = {}
        for comp in ("r", "t", "z"):
            if comp not in base:
                raise ValueError(f"Missing B/{comp} dataset in {path}")
            data = np.asarray(base[comp], dtype=np.float64)
            if data.ndim == 3:
                data = data[0, :, :]
            elif data.ndim != 2:
                raise ValueError(f"Unexpected B/{comp} dataset shape {data.shape} in {path}")
            comps[comp] = data
    return meta, comps["r"], comps["t"], comps["z"]


def _compute_a_theta_from_bz(bz: np.ndarray, dr: float, r0: float = 0.0) -> np.ndarray:
    """
    Reconstruct A_theta(r,z) from Bz(r,z) for axisymmetric fields:
      Bz = (1/r) d(r A_theta)/dr  =>  A_theta = (1/r) * integral_0^r r' Bz(r',z) dr'
    """
    nr, nz = bz.shape
    r = r0 + (np.arange(nr, dtype=np.float64) + 0.5) * dr
    integrand = r[:, None] * bz
    psi = np.zeros_like(bz, dtype=np.float64)
    if nr > 0:
        psi[0, :] = 0.5 * integrand[0, :] * dr
        for i in range(1, nr):
            psi[i, :] = psi[i - 1, :] + 0.5 * (integrand[i - 1, :] + integrand[i, :]) * dr
    a_theta = np.zeros_like(bz, dtype=np.float64)
    if nr > 1:
        a_theta[1:, :] = psi[1:, :] / r[1:, None]
    return a_theta


def _match_cell_to_staggered_faces(arr: np.ndarray, target_n0: int, target_n1: int) -> np.ndarray:
    """
    Map a cell-centered (nr,nz) array onto a simple staggered (face) grid by averaging.

    Supports common Yee-like shapes:
      - (nr, nz)   : cell-centered (no-op)
      - (nr+1, nz) : faces in r (pad by averaging in r)
      - (nr, nz+1) : faces in z (pad by averaging in z)
    """
    nr, nz = arr.shape
    if (target_n0, target_n1) == (nr, nz):
        return arr
    if (target_n0, target_n1) == (nr + 1, nz):
        out = np.empty((nr + 1, nz), dtype=np.float64)
        out[1:-1, :] = 0.5 * (arr[:-1, :] + arr[1:, :])
        out[0, :] = arr[0, :]
        out[-1, :] = arr[-1, :]
        return out
    if (target_n0, target_n1) == (nr, nz + 1):
        out = np.empty((nr, nz + 1), dtype=np.float64)
        out[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:])
        out[:, 0] = arr[:, 0]
        out[:, -1] = arr[:, -1]
        return out
    if (target_n0, target_n1) == (nr + 1, nz + 1):
        # Two-step averaging: cell -> r-faces, then r-faces -> (r,z) edges.
        tmp = np.empty((nr + 1, nz), dtype=np.float64)
        tmp[1:-1, :] = 0.5 * (arr[:-1, :] + arr[1:, :])
        tmp[0, :] = arr[0, :]
        tmp[-1, :] = arr[-1, :]
        out = np.empty((nr + 1, nz + 1), dtype=np.float64)
        out[:, 1:-1] = 0.5 * (tmp[:, :-1] + tmp[:, 1:])
        out[:, 0] = tmp[:, 0]
        out[:, -1] = tmp[:, -1]
        return out
    raise ValueError(
        f"Cannot map cell array (nr={nr}, nz={nz}) to target shape ({target_n0}, {target_n1})"
    )


def _get_external_field_wrappers(hybrid: bool, mode: int = 0):
    """
    Prefer external-field wrappers when available (so we add coil fields without
    overwriting self-consistent fields). Fall back to total field wrappers.
    """
    def _grab(candidates: list[str]):
        for name in candidates:
            if hasattr(fields, name):
                try:
                    wrapper = getattr(fields, name)(mode)
                    # Some wrapper types exist but their underlying MultiFab is not registered
                    # unless external fields were enabled at runtime. Probe with a lightweight read.
                    _ = wrapper[:]
                    return wrapper
                except Exception:
                    continue
        return None

    if hybrid:
        bx = _grab(["BxHybridExternalWrapper", "BxFPExternalWrapper", "BxFPWrapper"])
        by = _grab(["ByHybridExternalWrapper", "ByFPExternalWrapper", "ByFPWrapper"])
        bz = _grab(["BzHybridExternalWrapper", "BzFPExternalWrapper", "BzFPWrapper"])
        ex = _grab(["ExHybridExternalWrapper", "ExFPExternalWrapper", "ExFPWrapper"])
        ey = _grab(["EyHybridExternalWrapper", "EyFPExternalWrapper", "EyFPWrapper"])
        ez = _grab(["EzHybridExternalWrapper", "EzFPExternalWrapper", "EzFPWrapper"])
    else:
        bx = _grab(["BxFPExternalWrapper", "BxFPWrapper"])
        by = _grab(["ByFPExternalWrapper", "ByFPWrapper"])
        bz = _grab(["BzFPExternalWrapper", "BzFPWrapper"])
        ex = _grab(["ExFPExternalWrapper", "ExFPWrapper"])
        ey = _grab(["EyFPExternalWrapper", "EyFPWrapper"])
        ez = _grab(["EzFPExternalWrapper", "EzFPWrapper"])

    if bx is None or by is None or bz is None:
        raise RuntimeError("Unable to access B field wrappers (external or total).")
    if ex is None or ey is None or ez is None:
        raise RuntimeError("Unable to access E field wrappers (external or total).")
    return {"Bx": bx, "By": by, "Bz": bz, "Ex": ex, "Ey": ey, "Ez": ez}


class ExternalFieldDriver:
    """
    Apply an axisymmetric external field as a linear combination of basis fields:
      B_ext(t) = sum_k c_k(t) * B_k
    If induced_E is enabled, also set an azimuthal E (Ey) consistent with A_theta:
      Ey_ext(t) = - d/dt [ sum_k c_k(t) * A_theta,k ]
    """

    def __init__(self, wrappers, basis, dt: float, induced_E: bool):
        self.w = wrappers
        self.basis = basis
        self.dt = float(dt)
        self.induced_E = bool(induced_E)
        self.prev_coeffs = None

    def set_coefficients(self, coeffs: np.ndarray):
        coeffs = np.asarray(coeffs, dtype=np.float64)
        if coeffs.ndim != 1 or coeffs.size != len(self.basis):
            raise ValueError(
                f"coeffs must be shape ({len(self.basis)},), got {coeffs.shape}"
            )

        if self.prev_coeffs is None or self.dt <= 0.0:
            dcoeff_dt = np.zeros_like(coeffs)
        else:
            dcoeff_dt = (coeffs - self.prev_coeffs) / self.dt

        br = np.zeros_like(self.basis[0]["Br"])
        bt = np.zeros_like(self.basis[0]["Bt"])
        bz = np.zeros_like(self.basis[0]["Bz"])
        a_theta_dt = np.zeros_like(self.basis[0]["A_theta"])
        for c, b, dc in zip(coeffs, self.basis, dcoeff_dt):
            if c != 0.0:
                br += c * b["Br"]
                bt += c * b["Bt"]
                bz += c * b["Bz"]
            if self.induced_E and dc != 0.0:
                a_theta_dt += dc * b["A_theta"]

        # In RZ thetaMode, x/y/z correspond to r/theta/z components.
        self.w["Bx"][:] = br
        self.w["By"][:] = bt
        self.w["Bz"][:] = bz

        if self.induced_E:
            # E_theta = - dA_theta/dt; map to Ey.
            self.w["Ex"][:] = 0.0
            self.w["Ey"][:] = -a_theta_dt
            self.w["Ez"][:] = 0.0

        self.prev_coeffs = coeffs
        return dcoeff_dt

def ensure_b_file(path: Path, nr: int, nz: int, r_max: float, z_max: float, Bz: float = 0.05) -> Path:
    """
    Ensure an openPMD B field file exists. If missing, create a simple uniform Bz file.
    """
    if path.exists():
        return path

    print(f"{path} not found. Creating a uniform Bz openPMD file for smoke tests.")
    try:
        import openpmd_api as io
    except Exception as exc:
        raise SystemExit(
            f"Cannot create fallback B file (openpmd_api missing): {exc}"
        )

    dr = r_max / nr
    dz = z_max / nz
    r0 = 0.0
    z0 = -0.5 * z_max
    shape = (1, nr, nz)

    path.parent.mkdir(parents=True, exist_ok=True)
    series = io.Series(str(path), io.Access.create)
    iteration = series.iterations[0]
    mesh_B = iteration.meshes["B"]
    mesh_B.set_geometry(io.Geometry.thetaMode)
    mesh_B.set_attribute("dataOrder", "C")
    mesh_B.set_axis_labels(["r", "z"])
    mesh_B.set_grid_spacing([dr, dz])
    mesh_B.set_grid_global_offset([r0, z0])
    mesh_B.unit_dimension = {io.Unit_Dimension.L: -1, io.Unit_Dimension.T: 0, io.Unit_Dimension.M: 0}

    zeros = np.zeros(shape, dtype=np.float64)
    Bz_arr = np.full(shape, Bz, dtype=np.float64)
    for comp, arr in (("r", zeros), ("t", zeros), ("z", Bz_arr)):
        rc = mesh_B[comp]
        rc.reset_dataset(io.Dataset(arr.dtype, arr.shape))
        rc.set_unit_SI(1.0)
        rc.store_chunk(np.ascontiguousarray(arr))

    series.flush()
    series.close()
    return path


def _read_b_field_metadata(path: Path):
    """Read basic metadata (nr, nz, spacing, offsets) from an openPMD B field file."""
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5f:
        # Default openPMD location for B components
        base = h5f["/data/0/meshes/B"]
        # Try common component names
        for comp in ("z", "t", "r", "x", "y"):
            if comp in base:
                data = base[comp]
                break
        else:
            raise ValueError(f"No B component dataset found in {path}")

        shape = data.shape
        if len(shape) < 2:
            raise ValueError(f"B dataset shape too small: {shape}")

        # openPMD thetaMode: (theta, r, z)
        nr = shape[-2]
        nz = shape[-1]

        spacing_attr = data.attrs.get("gridSpacing", None)
        if spacing_attr is None:
            spacing_attr = base.attrs.get("gridSpacing", None)
        if spacing_attr is not None and len(spacing_attr) >= 2:
            # thetaMode often stores (dtheta, dr, dz); take the last two entries for (dr, dz)
            spacing = (float(spacing_attr[-2]), float(spacing_attr[-1]))
        else:
            spacing = None

        offset_attr = data.attrs.get("gridGlobalOffset", None)
        if offset_attr is None:
            offset_attr = base.attrs.get("gridGlobalOffset", None)
        if offset_attr is not None and len(offset_attr) >= 2:
            # thetaMode often stores (theta0, r0, z0); take the last two entries for (r0, z0)
            offset = (float(offset_attr[-2]), float(offset_attr[-1]))
        else:
            offset = None

    return {
        "nr": nr,
        "nz": nz,
        "spacing": spacing,
        "offset": offset,
    }


def validate_b_field_file(path: Path, nr_sim: int, nz_sim: int, r_max: float, z_max: float):
    """
    Raises ValueError with a clear message if the B file grid is incompatible
    with the current RZ grid.
    """
    meta = _read_b_field_metadata(path)
    nr_file = meta["nr"]
    nz_file = meta["nz"]
    spacing = meta["spacing"]
    offset = meta["offset"]

    if nr_file != nr_sim or nz_file != nz_sim:
        raise ValueError(
            f"B file grid mismatch: file (nr={nr_file}, nz={nz_file}) "
            f"vs driver (nr={nr_sim}, nz={nz_sim}). Regenerate the B file."
        )

    # If spacing present, check against expected dr,dz
    if spacing is not None:
        dr_file, dz_file = spacing
        dr_exp = r_max / nr_sim
        dz_exp = z_max / nz_sim
        if abs(dr_file - dr_exp) > max(1e-12, 1e-3 * dr_exp) or abs(dz_file - dz_exp) > max(1e-12, 1e-3 * dz_exp):
            raise ValueError(
                f"B file spacing mismatch: file (dr={dr_file:g}, dz={dz_file:g}) "
                f"vs driver (dr={dr_exp:g}, dz={dz_exp:g}). Regenerate the B file."
            )
    else:
        raise ValueError("B file missing gridSpacing; regenerate with openPMD metadata.")

    # If offsets present, check that domain overlap is plausible
    if offset is not None:
        r0_file, z0_file = offset
        dr = r_max / nr_sim
        dz = z_max / nz_sim
        r_max_file = r0_file + dr * nr_sim
        z_min_file = z0_file
        z_max_file = z0_file + dz * nz_sim
        z_min_exp = -0.5 * z_max
        z_max_exp = 0.5 * z_max
        if r0_file < -1e-9 or r_max_file > r_max + 1e-6:
            raise ValueError(
                f"B file r-range [{r0_file:g}, {r_max_file:g}] exceeds driver r_max={r_max:g}. "
                f"Regenerate the B file with matching grid."
            )
        if z_min_file > z_min_exp + 1e-6 or z_max_file < z_max_exp - 1e-6:
            raise ValueError(
                f"B file z-range [{z_min_file:g}, {z_max_file:g}] does not cover driver "
                f"[{z_min_exp:g}, {z_max_exp:g}]. Regenerate the B file."
            )
    else:
        raise ValueError("B file missing gridGlobalOffset; regenerate with openPMD metadata.")

    return meta


def _read_fluid_metadata(path: Path):
    """Read basic metadata (nr, nz, spacing, offsets) from an openPMD fluid file."""
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5f:
        base = h5f["/data/0/meshes/fluid"]
        if "rho" not in base:
            raise ValueError(f"No 'rho' dataset found in fluid file {path}")
        data = base["rho"]
        shape = data.shape
        if len(shape) < 2:
            raise ValueError(f"Fluid dataset shape too small: {shape}")
        nr = shape[-2]
        nz = shape[-1]

        spacing_attr = data.attrs.get("gridSpacing", None)
        if spacing_attr is None:
            spacing_attr = base.attrs.get("gridSpacing", None)
        if spacing_attr is not None and len(spacing_attr) >= 2:
            spacing = (float(spacing_attr[-2]), float(spacing_attr[-1]))
        else:
            spacing = None

        offset_attr = data.attrs.get("gridGlobalOffset", None)
        if offset_attr is None:
            offset_attr = base.attrs.get("gridGlobalOffset", None)
        if offset_attr is not None and len(offset_attr) >= 2:
            offset = (float(offset_attr[-2]), float(offset_attr[-1]))
        else:
            offset = None

    return {
        "nr": nr,
        "nz": nz,
        "spacing": spacing,
        "offset": offset,
    }


def validate_fluid_file(path: Path, nr_sim: int, nz_sim: int, r_max: float, z_max: float):
    """Validate that the fluid file grid matches the simulation RZ grid."""
    meta = _read_fluid_metadata(path)
    nr_file = meta["nr"]
    nz_file = meta["nz"]
    spacing = meta["spacing"]
    offset = meta["offset"]

    if nr_file != nr_sim or nz_file != nz_sim:
        raise ValueError(
            f"Fluid file grid mismatch: file (nr={nr_file}, nz={nz_file}) "
            f"vs driver (nr={nr_sim}, nz={nz_sim}). Regenerate the fluid file."
        )

    if spacing is None:
        raise ValueError("Fluid file missing gridSpacing; regenerate with openPMD metadata.")
    dr_file, dz_file = spacing
    dr_exp = r_max / nr_sim
    dz_exp = z_max / nz_sim
    if abs(dr_file - dr_exp) > max(1e-12, 1e-3 * dr_exp) or abs(dz_file - dz_exp) > max(1e-12, 1e-3 * dz_exp):
        raise ValueError(
            f"Fluid file spacing mismatch: file (dr={dr_file:g}, dz={dz_file:g}) "
            f"vs driver (dr={dr_exp:g}, dz={dz_exp:g}). Regenerate the fluid file."
        )

    if offset is None:
        raise ValueError("Fluid file missing gridGlobalOffset; regenerate with openPMD metadata.")
    r0_file, z0_file = offset
    dr = r_max / nr_sim
    dz = z_max / nz_sim
    r_max_file = r0_file + dr * nr_sim
    z_min_file = z0_file
    z_max_file = z0_file + dz * nz_sim
    z_min_exp = -0.5 * z_max
    z_max_exp = 0.5 * z_max
    if r0_file < -1e-9 or r_max_file > r_max + 1e-6:
        raise ValueError(
            f"Fluid file r-range [{r0_file:g}, {r_max_file:g}] exceeds driver r_max={r_max:g}. "
            f"Regenerate the fluid file with matching grid."
        )
    if z_min_file > z_min_exp + 1e-6 or z_max_file < z_max_exp - 1e-6:
        raise ValueError(
            f"Fluid file z-range [{z_min_file:g}, {z_max_file:g}] does not cover driver "
            f"[{z_min_exp:g}, {z_max_exp:g}]. Regenerate the fluid file."
        )

    return meta


def load_fluid_fields(path: Path):
    """Load fluid fields from openPMD fluid file as numpy arrays."""
    with h5py.File(path, "r") as h5f:
        base = h5f["/data/0/meshes/fluid"]
        def _grab(name):
            if name in base:
                return np.array(base[name][0, :, :], dtype=np.float64)
            return None

        rho = _grab("rho")
        if rho is None:
            raise ValueError(f"'rho' missing in {path}")
        vr = _grab("vr")
        vz = _grab("vz")
        vphi = _grab("vphi")
        Ti = _grab("Ti")
        Te = _grab("Te")

        # Metadata
        any_dataset = base["rho"]
        spacing = any_dataset.attrs.get("gridSpacing", base.attrs.get("gridSpacing", None))
        offset = any_dataset.attrs.get("gridGlobalOffset", base.attrs.get("gridGlobalOffset", None))
        if spacing is None or len(spacing) < 2 or offset is None or len(offset) < 2:
            raise ValueError("Fluid file missing gridSpacing/gridGlobalOffset metadata.")
        spacing = (float(spacing[-2]), float(spacing[-1]))
        offset = (float(offset[-2]), float(offset[-1]))

    return {
        "rho": rho,
        "vr": vr,
        "vz": vz,
        "vphi": vphi,
        "Ti": Ti,
        "Te": Te,
        "spacing": spacing,
        "offset": offset,
        "path": path,
    }

def collect_diag_mode_metrics(diag_root: Path, max_mode: int, fields=("rho", "Bz")):
    """Return mode RMS/max from the latest diag (thetaMode) if available."""
    try:
        import yt
    except Exception as exc:
        return {"error": f"yt unavailable: {exc}"}

    if not diag_root.exists():
        return {"error": f"diag root {diag_root} missing"}
    diags = sorted([p for p in diag_root.iterdir() if p.is_dir() and p.name.startswith("diag")])
    if not diags:
        return {"error": f"no diag* under {diag_root}"}

    diag = diags[-1]
    try:
        ds = yt.load(str(diag))
        ad = ds.all_data()
        metrics = {}
        for field in fields:
            arr = ad["boxlib", field].to_ndarray()
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim < 2:
                continue
            nmodes = min(arr.shape[0], max_mode + 1)
            for m in range(nmodes):
                mode_arr = np.asarray(arr[m])
                amp = np.abs(mode_arr)
                metrics[f"{field}_m{m}_rms"] = float(np.sqrt(np.mean(amp * amp)))
                metrics[f"{field}_m{m}_max"] = float(np.max(amp))
        return {
            "diag": diag.name,
            "time_s": float(ds.current_time.to_value()),
            "metrics": metrics,
        }
    except Exception as exc:
        return {"error": f"diag parse failed for {diag}: {exc}", "diag": diag.name}


def sample_particles_from_fluid(fluid: dict, args, tilt_eps: float = 0.0):
    """
    Sample macro-particles from fluid fields.
    Positions are sampled uniformly in each cell in cylindrical coordinates, then mapped to Cartesian.
    Bulk velocities are optional; thermal spread can be enabled via --sample-thermal.
    tilt_eps optionally applies an m=1 density perturbation: w -> w*(1 + tilt_eps*cos(theta)).
    If --fast-ion-fraction > 0, add a separate fast-ion population using the same spatial sampling.
    """
    rng = np.random.default_rng(args.rng_seed)
    rho = fluid["rho"]
    vr = fluid.get("vr")
    vz = fluid.get("vz")
    vphi = fluid.get("vphi")
    Ti = fluid.get("Ti")
    Te = fluid.get("Te")
    dr, dz = fluid["spacing"]
    r_min, z_min = fluid["offset"]

    nr, nz = rho.shape
    ppc = max(1, args.ppc)
    amu = args.ion_amu
    Z = args.ion_charge
    use_bulk_v = args.use_fluid_velocity
    fast_fraction = max(0.0, float(getattr(args, "fast_ion_fraction", 0.0) or 0.0))
    fast_amu = (
        float(args.fast_ion_amu)
        if getattr(args, "fast_ion_amu", None) is not None
        else float(amu)
    )
    fast_Z = (
        float(args.fast_ion_charge)
        if getattr(args, "fast_ion_charge", None) is not None
        else float(Z)
    )
    fast_Ti_eV = getattr(args, "fast_ion_Ti_eV", None)
    fast_vphi = float(getattr(args, "fast_ion_vphi", 0.0) or 0.0)

    n = rho / (amu * M_U)
    x_list = []
    y_list = []
    z_list = []
    ux_list = []
    uy_list = []
    uz_list = []
    w_list = []
    ex_list = []
    ey_list = []
    ez_list = []
    ew_list = []
    fx_list = []
    fy_list = []
    fz_list = []
    fux_list = []
    fuy_list = []
    fuz_list = []
    fw_list = []

    base_seq = None
    if getattr(args, "quiet_start", False):
        base_seq = (np.arange(ppc) + 0.5) / ppc

    def velocities_to_momenta(vx_arr, vy_arr, vz_arr, max_beta):
        """Clip velocities to max_beta*c and return normalized momenta (gamma*beta)."""
        vx_arr = np.asarray(vx_arr, dtype=float)
        vy_arr = np.asarray(vy_arr, dtype=float)
        vz_arr = np.asarray(vz_arr, dtype=float)

        vmag = np.sqrt(vx_arr * vx_arr + vy_arr * vy_arr + vz_arr * vz_arr) + 1.0e-50
        beta = vmag / c
        clip = np.minimum(1.0, max_beta / np.maximum(beta, 1.0e-30))
        vx_arr *= clip
        vy_arr *= clip
        vz_arr *= clip

        beta_x = vx_arr / c
        beta_y = vy_arr / c
        beta_z = vz_arr / c
        beta2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z
        beta2 = np.minimum(beta2, 1.0 - 1.0e-12)
        gamma = 1.0 / np.sqrt(1.0 - beta2)

        ux_arr = gamma * beta_x
        uy_arr = gamma * beta_y
        uz_arr = gamma * beta_z
        return ux_arr, uy_arr, uz_arr, beta

    for ir in range(nr):
        r0 = r_min + ir * dr
        r1 = r0 + dr
        for iz in range(nz):
            n_cell = n[ir, iz]
            if n_cell <= 0.0:
                continue
            z0 = z_min + iz * dz
            volume = math.pi * (r1 * r1 - r0 * r0) * dz
            weight = n_cell * volume / ppc

            vr_cell = vr[ir, iz] if vr is not None else 0.0
            vz_cell = vz[ir, iz] if vz is not None else 0.0
            vphi_cell = vphi[ir, iz] if vphi is not None else 0.0
            Ti_cell = Ti[ir, iz] if Ti is not None else 0.0
            Te_cell = Te[ir, iz] if Te is not None else 0.0

            # Sample ppc particles inside the annular cell; r sampled with area weighting
            if base_seq is not None:
                rand_r = base_seq
                rand_theta = (base_seq + 0.5) % 1.0
                rand_z = base_seq[::-1]
            else:
                rand_r = rng.random(ppc)
                rand_theta = rng.random(ppc)
                rand_z = rng.random(ppc)
            r_samples = np.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * rand_r)
            z_samples = z0 + dz * rand_z

            # For RZ geometry the radial coordinate lives in x; y is unused.
            theta_angles = 2.0 * np.pi * rand_theta
            x_samples = r_samples  # cylindrical r -> x
            y_samples = np.zeros_like(r_samples)

            if use_bulk_v:
                vx_bulk = vr_cell * np.cos(rand_theta) - vphi_cell * np.sin(rand_theta)
                vy_bulk = vr_cell * np.sin(rand_theta) + vphi_cell * np.cos(rand_theta)
                vz_bulk = np.full_like(rand_r, vz_cell)
            else:
                vx_bulk = np.zeros_like(rand_r)
                vy_bulk = np.zeros_like(rand_r)
                vz_bulk = np.zeros_like(rand_r)

            if args.sample_thermal and Ti_cell > 0.0:
                sigma_i = np.sqrt(K_B * Ti_cell / (amu * M_U))
                vx_th = rng.normal(0.0, sigma_i, size=ppc)
                vy_th = rng.normal(0.0, sigma_i, size=ppc)
                vz_th = rng.normal(0.0, sigma_i, size=ppc)
            else:
                vx_th = np.zeros(ppc)
                vy_th = np.zeros(ppc)
                vz_th = np.zeros(ppc)

            vx = vx_bulk + vx_th
            vy = vy_bulk + vy_th
            vz_vec = vz_bulk + vz_th

            ux_vec, uy_vec, uz_vec, beta_vec = velocities_to_momenta(
                vx, vy, vz_vec, args.max_beta
            )

            weights = np.full_like(r_samples, weight)
            if tilt_eps != 0.0:
                weights = weights * (1.0 + tilt_eps * np.cos(theta_angles))

            x_list.append(x_samples)
            y_list.append(y_samples)
            z_list.append(z_samples)
            ux_list.append(ux_vec)
            uy_list.append(uy_vec)
            uz_list.append(uz_vec)
            w_list.append(weights)

            # Electrons: reuse positions, thermal spread from Te if available
            if args.sample_thermal and Te_cell > 0.0:
                sigma_e = np.sqrt(K_B * Te_cell / m_e)
                ex_th = rng.normal(0.0, sigma_e, size=ppc)
                ey_th = rng.normal(0.0, sigma_e, size=ppc)
                ez_th = rng.normal(0.0, sigma_e, size=ppc)
            else:
                ex_th = np.zeros(ppc)
                ey_th = np.zeros(ppc)
                ez_th = np.zeros(ppc)

            if args.electron_bulk_from_ions:
                ex_bulk = vx_bulk
                ey_bulk = vy_bulk
                ez_bulk = vz_vec
            else:
                ex_bulk = np.zeros(ppc)
                ey_bulk = np.zeros(ppc)
                ez_bulk = np.zeros(ppc)

            evx = ex_bulk + ex_th
            evy = ey_bulk + ey_th
            evz = ez_bulk + ez_th
            eux, euy, euz, _ = velocities_to_momenta(evx, evy, evz, args.max_beta)

            ex_list.append(eux)
            ey_list.append(euy)
            ez_list.append(euz)
            ew_list.append(weights * Z)

            if fast_fraction > 0.0:
                fast_weight = weight * fast_fraction
                fast_use_bulk = use_bulk_v or (fast_vphi != 0.0)
                if fast_use_bulk:
                    f_vr = vr_cell if use_bulk_v else 0.0
                    f_vz = vz_cell if use_bulk_v else 0.0
                    f_vphi = (vphi_cell if use_bulk_v else 0.0) + fast_vphi
                    fvx_bulk = f_vr * np.cos(rand_theta) - f_vphi * np.sin(rand_theta)
                    fvy_bulk = f_vr * np.sin(rand_theta) + f_vphi * np.cos(rand_theta)
                    fvz_bulk = np.full_like(rand_r, f_vz)
                else:
                    fvx_bulk = np.zeros_like(rand_r)
                    fvy_bulk = np.zeros_like(rand_r)
                    fvz_bulk = np.zeros_like(rand_r)

                if fast_Ti_eV is not None:
                    fast_Ti_K = float(fast_Ti_eV) * 11604.518
                else:
                    fast_Ti_K = Ti_cell if Ti_cell is not None else 0.0

                if args.sample_thermal and fast_Ti_K > 0.0:
                    sigma_f = np.sqrt(K_B * fast_Ti_K / (fast_amu * M_U))
                    fvx_th = rng.normal(0.0, sigma_f, size=ppc)
                    fvy_th = rng.normal(0.0, sigma_f, size=ppc)
                    fvz_th = rng.normal(0.0, sigma_f, size=ppc)
                else:
                    fvx_th = np.zeros(ppc)
                    fvy_th = np.zeros(ppc)
                    fvz_th = np.zeros(ppc)

                fvx = fvx_bulk + fvx_th
                fvy = fvy_bulk + fvy_th
                fvz = fvz_bulk + fvz_th
                fux, fuy, fuz, _ = velocities_to_momenta(
                    fvx, fvy, fvz, args.max_beta
                )
                fweights = np.full_like(r_samples, fast_weight)
                if tilt_eps != 0.0:
                    fweights = fweights * (1.0 + tilt_eps * np.cos(theta_angles))

                fx_list.append(x_samples)
                fy_list.append(y_samples)
                fz_list.append(z_samples)
                fux_list.append(fux)
                fuy_list.append(fuy)
                fuz_list.append(fuz)
                fw_list.append(fweights)

    if not x_list:
        raise ValueError("No particles sampled (all fluid densities non-positive).")

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    ux = np.concatenate(ux_list)
    uy = np.concatenate(uy_list)
    uz = np.concatenate(uz_list)
    w = np.concatenate(w_list)

    ex = np.concatenate(ex_list)
    ey = np.concatenate(ey_list)
    ez = np.concatenate(ez_list)
    ew = np.concatenate(ew_list)

    have_fast = bool(fx_list)
    if have_fast:
        fx = np.concatenate(fx_list)
        fy = np.concatenate(fy_list)
        fz = np.concatenate(fz_list)
        fux = np.concatenate(fux_list)
        fuy = np.concatenate(fuy_list)
        fuz = np.concatenate(fuz_list)
        fw = np.concatenate(fw_list)

    total_particles = len(x)
    if args.max_particles and total_particles > args.max_particles:
        keep = args.max_particles
        idx = np.random.choice(total_particles, keep, replace=False)
        x, y, z, ux, uy, uz, w = (arr[idx] for arr in (x, y, z, ux, uy, uz, w))
        ex, ey, ez, ew = (arr[idx] for arr in (ex, ey, ez, ew))
        if have_fast:
            fx, fy, fz, fux, fuy, fuz, fw = (
                arr[idx] for arr in (fx, fy, fz, fux, fuy, fuz, fw)
            )
        print(f"Downsampled particles to {keep} (from {total_particles}) to respect max-particles limit.")
        total_particles = keep

    def summarize_beta(ux_arr, uy_arr, uz_arr, label):
        beta2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        beta_mag = np.sqrt(beta2 / (1.0 + beta2))
        return f"{label} beta: min={beta_mag.min():.3e}, max={beta_mag.max():.3e}, mean={beta_mag.mean():.3e}"

    species_label = "ions+electrons"
    if have_fast:
        species_label = "ions+electrons+fast"
    print(
        f"[fluid-init] sampled {total_particles} macroparticles per species ({species_label}), "
        f"ppc={ppc}, max_beta cap={args.max_beta}"
    )
    print("  " + summarize_beta(ux, uy, uz, "ion"))
    print("  " + summarize_beta(ex, ey, ez, "elec"))
    if have_fast:
        print("  " + summarize_beta(fux, fuy, fuz, "fast-ion"))

    ion = {
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "w": w,
        "charge": Z * q_e,
        "mass": amu * m_p,
    }

    electrons = {
        "x": x,
        "y": y,
        "z": z,
        "ux": ex,
        "uy": ey,
        "uz": ez,
        "w": ew,
        "charge": -q_e,
        "mass": m_e,
    }
    fast_ions = None
    if have_fast:
        fast_ions = {
            "x": fx,
            "y": fy,
            "z": fz,
            "ux": fux,
            "uy": fuy,
            "uz": fuz,
            "w": fw,
            "charge": fast_Z * q_e,
            "mass": fast_amu * m_p,
        }
    return ion, electrons, fast_ions


def build_uniform_fluid_from_args(args):
    """Construct a uniform fluid dict (rho only) matching the default plasma region."""
    nr = args.nr
    nz = args.nz
    dr = args.r_max / nr
    dz = args.z_max / nz
    r0 = 0.0
    z0 = -0.5 * args.z_max

    rho = np.zeros((nr, nz), dtype=np.float64)
    # default uniform region: r < r_max/2, |z| < z_max/4
    r_cut = args.r_max * 0.5
    z_min_region = -0.25 * args.z_max
    z_max_region = 0.25 * args.z_max
    density = args.n0
    amu = args.ion_amu
    rho_val = density * amu * M_U

    for ir in range(nr):
        r_center = r0 + (ir + 0.5) * dr
        if r_center >= r_cut:
            continue
        for iz in range(nz):
            z_center = z0 + (iz + 0.5) * dz
            if z_center < z_min_region or z_center > z_max_region:
                continue
            rho[ir, iz] = rho_val

    return {
        "rho": rho,
        "vr": None,
        "vz": None,
        "vphi": None,
        "Ti": None,
        "Te": None,
        "spacing": (dr, dz),
        "offset": (r0, z0),
        "path": "<uniform>",
    }


def gather_species_stats(species_names):
    """Return particle counts/charge/energy summaries for the given species list."""
    stats = {}
    if not species_names:
        return stats
    try:
        mpc = libwarpx.warpx.multi_particle_container()
    except Exception as exc:
        print(f"Warning: cannot access particle containers for stats: {exc}")
        return stats

    for name in species_names:
        try:
            pc = mpc.get_particle_container_from_name(name)
        except Exception:
            continue
        try:
            stats[name] = {
                "num_particles": int(pc.total_number_of_particles(True, False)),
                "charge_C": float(pc.sum_particle_charge(False)),
                "weight_sum": float(pc.sum_particle_weight(False)),
                "energy_J": float(pc.sum_particle_energy(False)),
            }
        except Exception as exc:
            print(f"Warning: failed to collect stats for species '{name}': {exc}")

    return stats

# ==============================================================================
# 2. WarpX Simulation Setup (PICMI)
# ==============================================================================

# Constants
c = picmi.constants.c
q_e = picmi.constants.q_e
m_p = picmi.constants.m_p
m_e = picmi.constants.m_e
M_U = 1.66053906660e-27
K_B = picmi.constants.kb

# Grid (RZ Geometry)
# Note: WarpX RZ grid is defined as CylindricalGrid in newer versions or via geometry string
def build_simulation(args, add_default_plasma: bool = True):
    use_spectral = args.solver == "psatd"
    z_field_bc = "periodic" if getattr(args, "periodic_z", False) else "dirichlet"
    z_particle_bc = "periodic" if getattr(args, "periodic_z", False) else "absorbing"
    r_field_bc_hi = "dirichlet"
    if use_spectral:
        if not getattr(args, "periodic_z", False):
            z_field_bc = "open"
            print("Note: PSATD selected; switching z-field BC to open/PML (PEC/PMC unsupported).")
        r_field_bc_hi = "open"
        print("Note: PSATD selected; switching r_max field BC to open/PML (PEC/PMC unsupported).")
    grid = picmi.CylindricalGrid(
        number_of_cells=[args.nr, args.nz],
        n_azimuthal_modes=args.n_azimuthal_modes,
        warpx_max_grid_size=32,
        lower_bound=[0, -args.z_max / 2.0],
        upper_bound=[args.r_max, args.z_max / 2.0],
        lower_boundary_conditions=["none", z_field_bc],  # r_min (axis), z_min
        upper_boundary_conditions=[r_field_bc_hi, z_field_bc],  # r_max, z_max
        lower_boundary_conditions_particles=["none", z_particle_bc],  # r=0 must be none in RZ
        upper_boundary_conditions_particles=["absorbing", z_particle_bc],
        warpx_blocking_factor=8,
    )

    if getattr(args, "hybrid", False):
        n0_val = args.hybrid_n0 if hasattr(args, "hybrid_n0") and args.hybrid_n0 else args.hybrid_n0_fallback
        Te_val = args.hybrid_Te if hasattr(args, "hybrid_Te") and args.hybrid_Te is not None else args.hybrid_Te_fallback
        n_floor = args.hybrid_nfloor_scale * n0_val if n0_val else None
        eta_param = args.hybrid_eta_expr if args.hybrid_eta_expr else args.hybrid_eta
        eta_h_param = args.hybrid_eta_h_expr if args.hybrid_eta_h_expr else args.hybrid_eta_h
        solver = picmi.HybridPICSolver(
            grid=grid,
            Te=Te_val if Te_val is not None else 0.0,
            n0=n0_val if n0_val is not None else 0.0,
            plasma_resistivity=eta_param,
            plasma_resistivity_scale=args.hybrid_eta_scale,
            plasma_hyper_resistivity=eta_h_param,
            plasma_hyper_resistivity_scale=args.hybrid_eta_h_scale,
            substeps=args.hybrid_substeps,
            n_floor=n_floor,
            gamma=args.hybrid_gamma,
        )
    else:
        if use_spectral:
            # Some pywarpx builds expose PSATD via ElectromagneticSolver(method="PSATD")
            solver = picmi.ElectromagneticSolver(grid=grid, method="PSATD")
        else:
            solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=getattr(args, "cfl", 0.9))

    if args.n_azimuthal_modes > 1 and args.solver != "psatd":
        print("Note: n_azimuthal_modes>1 requested; PSATD is recommended but solver='yee' is in use.")

    sim = picmi.Simulation(
        solver=solver,
        max_steps=args.max_steps,
        time_step_size=args.dt,
        verbose=1,
    )

    species_names = []

    # Particles: only when not in field-only mode
    if args.mode != "field-only" and add_default_plasma:
        ion_mass = args.ion_amu * M_U
        ion_charge = args.ion_charge * q_e

        electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            warpx_do_not_deposit=False,
            initial_distribution=picmi.UniformDistribution(
                density=1e16,
                lower_bound=[0, -args.z_max / 4.0, 0],
                upper_bound=[args.r_max / 2.0, args.z_max / 4.0, 2 * np.pi],
            ),
        )
        sim.add_species(
            electrons,
                layout=picmi.PseudoRandomLayout(
                    n_macroparticles_per_cell=args.ppc, grid=grid
                ),
            )
        species_names.append("electrons")
        if args.neutralize:
            ions = picmi.Species(
                particle_type="He",
                name="ions",
                charge=ion_charge,
                mass=ion_mass,
                warpx_do_not_deposit=False,
                initial_distribution=picmi.UniformDistribution(
                    density=1e16,
                    lower_bound=[0, -args.z_max / 4.0, 0],
                    upper_bound=[args.r_max / 2.0, args.z_max / 4.0, 2 * np.pi],
                ),
            )
            sim.add_species(
                ions,
                layout=picmi.PseudoRandomLayout(
                    n_macroparticles_per_cell=args.ppc, grid=grid
                ),
            )
            species_names.append("ions")

    applied_field = None
    b_path = None
    b_meta = None
    use_external_basis = bool(getattr(args, "external_basis", None))
    if use_external_basis:
        # Basis mode: we will set external-field arrays directly in Python after initialization.
        # This avoids openPMD external-field loading limitations (e.g., with n_azimuthal_modes>1).
        basis_files = list(args.external_basis)
        basis0 = Path(basis_files[0]) if basis_files else None
        if basis0 is not None and basis0.exists() and args.n_azimuthal_modes <= 1:
            # Prefer loading one basis via openPMD so WarpX allocates external-field buffers
            # (and we avoid overwriting self-consistent fields in Python).
            try:
                b_meta = validate_b_field_file(basis0, args.nr, args.nz, args.r_max, args.z_max)
                b_path = basis0
                applied_field = picmi.LoadInitialField(
                    read_fields_from_path=str(basis0),
                    load_E=False,
                    load_B=True,
                )
                print(
                    f"[external] basis mode enabled ({len(basis_files)} basis files); "
                    f"loaded '{basis0}' to allocate external-field buffers."
                )
            except Exception as exc:
                print(
                    f"Warning: failed to load basis0 '{basis0}' as applied field ({exc}); "
                    "falling back to direct total-field injection."
                )
                applied_field = picmi.AnalyticAppliedField(Bx_expression="0", By_expression="0", Bz_expression="0")
        else:
            print(
                f"[external] basis mode enabled ({len(basis_files)} basis files); "
                "skipping openPMD applied-field load (n_azimuthal_modes>1 or missing basis0)."
            )
            applied_field = picmi.AnalyticAppliedField(Bx_expression="0", By_expression="0", Bz_expression="0")
    elif args.mode == "const-b-plasma":
        if args.const_B == 0.0:
            print("Warning: const-b-plasma mode requested but const-B=0; running with zero field.")
        applied_field = picmi.AnalyticAppliedField(
            Bx_expression="0", By_expression="0", Bz_expression=str(args.const_B)
        )
    else:
        b_path = ensure_b_file(Path(args.b_file), args.nr, args.nz, args.r_max, args.z_max)
        try:
            b_meta = validate_b_field_file(b_path, args.nr, args.nz, args.r_max, args.z_max)
        except Exception as exc:
            raise SystemExit(f"B field validation failed for {b_path}: {exc}")
        print(
            f"[B file] {b_path} nr={b_meta['nr']} nz={b_meta['nz']} "
            f"spacing={b_meta['spacing']} offset={b_meta['offset']}"
        )
        applied_field = picmi.LoadInitialField(
            read_fields_from_path=str(b_path),
            load_E=False,
            load_B=True,
        )

    if applied_field:
        sim.add_applied_field(applied_field)

    diag_dir = getattr(args, "diag_dir", None)
    field_diag = picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=args.diag_period,
        data_list=["Br", "Bt", "Bz", "Er", "Et", "Ez", "rho"],
        write_dir=diag_dir,
    )
    diag_display = diag_dir or "diags"
    print(
        f"[diagnostics] writing diags every {args.diag_period} steps to '{diag_display}/' "
        f"with fields {field_diag.data_list}"
    )
    sim.add_diagnostic(field_diag)

    return sim, grid, species_names, b_path, b_meta

# ==============================================================================
# 3. Python Callback for Time-Varying B Field
# ==============================================================================

# We use the 'beforestep' callback to update the external field value before the PIC push.
# However, 'LoadAppliedField' in PICMI typically sets a static parser expression at init.
# To update it dynamically, we need to access the WarpX C++ layer via pywarpx.

# Let's use the "Python Source" approach mentioned in the prompt:
# "在 RZ-IM 模式中用 Python 源实时更新场" -> Python Source (Current/Field Source).
# Maybe the user meant "Python controlling the source term" or "Python controlling the B-field".

# Given the constraints and lack of deep documentation access right now, 
# I will implement the 'step-by-step' drive loop which is robust and easy to understand.
# instead of 'sim.step(MAX_STEPS)', we loop.

# ==============================================================================
# 4. Execution Loop
# ==============================================================================

def run_simulation(args):
    print(f"Running mode={args.mode}, steps={args.max_steps}, dt={args.dt:.2e}")
    git_head, git_dirty = git_info(REPO_ROOT)
    print(f"[git] hash={git_head or 'NA'} dirty={git_dirty}")
    # Enforce CFL stability for Yee (RZ 2D)
    dr = args.r_max / args.nr
    dz = args.z_max / args.nz
    dt_cfl = args.cfl * min(dr, dz) / (c * math.sqrt(2.0))
    if args.dt > dt_cfl:
        print(f"[cfl] dt={args.dt:.2e} exceeds limit {dt_cfl:.2e} (dr={dr:.3e}, dz={dz:.3e}); clamping.")
        args.dt = dt_cfl

    # Resolve diagnostics write directory under metadata-dir unless explicitly overridden.
    diag_write_dir = None
    if getattr(args, "diag_dir", None):
        diag_write_dir = Path(args.diag_dir)
    elif getattr(args, "run_tag", None):
        diag_write_dir = Path(args.metadata_dir) / f"diag_{args.run_tag}"
    else:
        diag_write_dir = Path(args.metadata_dir) / "diag"
    if diag_write_dir is not None:
        diag_write_dir.mkdir(parents=True, exist_ok=True)
        args.diag_dir = str(diag_write_dir)
    use_synthetic_plasma = args.tilt_eps != 0.0 and args.mode != "field-only"
    add_default = args.mode != "fluid-init" and not use_synthetic_plasma
    apply_induced_E = bool(args.induced_E)
    if apply_induced_E and args.mode != "field-only":
        print("Warning: induced-E enabled for particle mode; ensure stability.")
        # apply_induced_E = False  # User requested to allow this
    lcr_model = LCRModel(args) if args.use_lcr else None

    fluid = None
    fluid_path = None
    fluid_meta = None
    if args.mode == "fluid-init":
        fluid_path = Path(args.fluid_file)
        try:
            fluid_meta = validate_fluid_file(fluid_path, args.nr, args.nz, args.r_max, args.z_max)
        except Exception as exc:
            raise SystemExit(f"Fluid file validation failed for {fluid_path}: {exc}")
        print(
            f"[fluid] {fluid_path} nr={fluid_meta['nr']} nz={fluid_meta['nz']} "
            f"spacing={fluid_meta['spacing']} offset={fluid_meta['offset']}"
        )
        fluid = load_fluid_fields(fluid_path)
        # Estimate n0/Te for hybrid solver if enabled
        if args.hybrid:
            rho = fluid["rho"]
            n_local = rho / (args.ion_amu * M_U)
            weights = np.maximum(n_local, 0.0)
            n0_val = float(np.average(n_local, weights=weights))
            args.hybrid_n0 = n0_val
            Te_field = fluid.get("Te")
            if Te_field is not None:
                Te_mean_K = float(np.average(Te_field, weights=weights))
            else:
                Te_mean_K = args.hybrid_Te_fallback if args.hybrid_Te_fallback is not None else 0.0
            # Convert K -> eV for WarpX
            Te_mean_eV = Te_mean_K / 11604.518
            args.hybrid_Te = Te_mean_eV
            n_floor = args.hybrid_nfloor_scale * n0_val if n0_val else 0.0
            print(
                f"Hybrid mode: using density-weighted n0={args.hybrid_n0:.3e} m^-3, "
                f"Te={args.hybrid_Te:.3e} eV (from Te_K={Te_mean_K:.3e}), n_floor={n_floor:.3e}"
            )
    elif use_synthetic_plasma:
        fluid = build_uniform_fluid_from_args(args)

    sim, grid, species_names, b_path, b_meta = build_simulation(args, add_default_plasma=add_default)

    # Fluid-init: seed particles before initialization
    if fluid is not None:
        ion_part, elec_part, fast_part = sample_particles_from_fluid(
            fluid, args, tilt_eps=args.tilt_eps
        )

        ions_dist = picmi.ParticleListDistribution(
            x=ion_part["x"],
            y=ion_part["y"],
            z=ion_part["z"],
            ux=ion_part["ux"],
            uy=ion_part["uy"],
            uz=ion_part["uz"],
            weight=ion_part["w"],
        )
        ions = picmi.Species(
            particle_type="He",
            name="ions",
            charge=ion_part["charge"],
            mass=ion_part["mass"],
            warpx_do_not_deposit=False,
            initial_distribution=ions_dist,
        )
        # ParticleListDistribution carries positions/weights; layout can be None.
        sim.add_species(ions, layout=None)
        if "ions" not in species_names:
            species_names.append("ions")

        if not args.hybrid:
            elec_dist = picmi.ParticleListDistribution(
                x=elec_part["x"],
                y=elec_part["y"],
                z=elec_part["z"],
                ux=elec_part["ux"],
                uy=elec_part["uy"],
                uz=elec_part["uz"],
                weight=elec_part["w"],
            )
            electrons = picmi.Species(
                particle_type="electron",
                name="electrons",
                charge=elec_part["charge"],
                mass=elec_part["mass"],
                warpx_do_not_deposit=False,
                initial_distribution=elec_dist,
            )
            sim.add_species(electrons, layout=None)
            if "electrons" not in species_names:
                species_names.append("electrons")

        if fast_part is not None:
            fast_dist = picmi.ParticleListDistribution(
                x=fast_part["x"],
                y=fast_part["y"],
                z=fast_part["z"],
                ux=fast_part["ux"],
                uy=fast_part["uy"],
                uz=fast_part["uz"],
                weight=fast_part["w"],
            )
            fast_ions = picmi.Species(
                particle_type="He",
                name="fast_ions",
                charge=fast_part["charge"],
                mass=fast_part["mass"],
                warpx_do_not_deposit=False,
                initial_distribution=fast_dist,
            )
            sim.add_species(fast_ions, layout=None)
            if "fast_ions" not in species_names:
                species_names.append("fast_ions")
    elif getattr(args, "fast_ion_fraction", 0.0) and args.fast_ion_fraction > 0.0:
        print("Warning: --fast-ion-fraction set but no fluid-init plasma; ignoring fast ions.")

    species_names = list(dict.fromkeys(species_names))

    # Waveform only matters in full-driver mode
    use_waveform = args.mode == "full-driver" and not args.use_lcr
    if use_waveform:
        t_wave, B_wave, get_B_ext = load_waveform(Path(args.waveform), args.max_steps, args.dt)
        if len(t_wave) == 0 or len(B_wave) == 0:
            print("Waveform unavailable; disabling time-dependent scaling.")
            use_waveform = False
    else:
        t_wave = []
        B_wave = []
        get_B_ext = lambda t: 0.0

    # Initialize
    sim.initialize_inputs()
    sim.initialize_warpx()
    initial_species = gather_species_stats(species_names)
    if initial_species:
        print("[init] species counts after loading:")
        for name, stats in initial_species.items():
            print(
                f"  {name}: N={stats['num_particles']}, q={stats['charge_C']:.3e} C, "
                f"w={stats['weight_sum']:.3e}, E={stats['energy_J']:.3e} J"
            )
    elif args.mode != "field-only":
        print("Warning: no particles detected after initialization (check density/ppc).")
    try:
        libwarpx.warpx.reset_dropped_particles_total()
    except Exception as exc:
        print(f"Warning: unable to reset dropped-particle counter: {exc}")

    # External field driver
    use_external_basis = bool(getattr(args, "external_basis", None))
    external_basis_cfg = None
    external_history = None
    external_driver = None
    base_norm = None
    b_scale_ref = None
    record_period = max(1, args.diag_period)

    if use_external_basis:
        basis_paths = [Path(p) for p in args.external_basis]
        wrappers = _get_external_field_wrappers(hybrid=bool(args.hybrid), mode=0)
        bx_shape = wrappers["Bx"][:].shape
        by_shape = wrappers["By"][:].shape
        bz_shape = wrappers["Bz"][:].shape
        ey_shape = wrappers["Ey"][:].shape
        bx_n0, bx_n1 = int(bx_shape[0]), int(bx_shape[1])
        by_n0, by_n1 = int(by_shape[0]), int(by_shape[1])
        bz_n0, bz_n1 = int(bz_shape[0]), int(bz_shape[1])
        ey_n0, ey_n1 = int(ey_shape[0]), int(ey_shape[1])

        basis_fields = []
        coeff_fns = []
        coeff_cfg = []
        const_cursor = 0
        for i, p in enumerate(basis_paths):
            if not p.exists():
                raise SystemExit(f"external basis file not found: {p}")
            try:
                validate_b_field_file(p, args.nr, args.nz, args.r_max, args.z_max)
            except Exception as exc:
                raise SystemExit(f"External basis B file validation failed for {p}: {exc}")
            meta, br, bt, bz = _load_openpmd_B_components(p)
            dr_file, _dz_file = meta["spacing"]
            r0_file, _z0_file = meta["offset"]
            a_theta_cell = _compute_a_theta_from_bz(bz, dr=dr_file, r0=r0_file)

            br_use = _match_cell_to_staggered_faces(br, bx_n0, bx_n1)
            bt_use = _match_cell_to_staggered_faces(bt, by_n0, by_n1)
            bz_use = _match_cell_to_staggered_faces(bz, bz_n0, bz_n1)
            a_theta_use = _match_cell_to_staggered_faces(a_theta_cell, ey_n0, ey_n1)
            basis_fields.append(
                {"Br": br_use, "Bt": bt_use, "Bz": bz_use, "A_theta": a_theta_use}
            )

            wf_spec = args.external_basis_waveform[i] if i < len(args.external_basis_waveform) else None
            if wf_spec:
                _t, _v, wf_fn, wf_info = load_scalar_waveform(wf_spec, default_value=0.0)
                scale = (
                    float(args.external_basis_waveform_scale[i])
                    if i < len(args.external_basis_waveform_scale)
                    else 1.0
                )
                coeff_fns.append(lambda t, fn=wf_fn, sc=scale: sc * fn(t))
                coeff_cfg.append({"type": "waveform", "spec": wf_spec, "scale": scale, "info": wf_info})
            else:
                if const_cursor < len(args.external_basis_const):
                    const = float(args.external_basis_const[const_cursor])
                    const_cursor += 1
                else:
                    const = 0.0
                coeff_fns.append(lambda _t, c=const: float(c))
                coeff_cfg.append({"type": "const", "value": const})
        external_driver = ExternalFieldDriver(
            wrappers=wrappers, basis=basis_fields, dt=args.dt, induced_E=apply_induced_E
        )
        external_history = []
        external_basis_cfg = {
            "mode": "basis",
            "basis_files": [str(p) for p in basis_paths],
            "coefficients": coeff_cfg,
        }

        def get_coeffs(t_current: float) -> np.ndarray:
            return np.array([fn(t_current) for fn in coeff_fns], dtype=np.float64)

    # Scale mode (legacy waveform/LCR): scale the currently loaded external field
    enable_dynamic_B = (use_waveform or lcr_model is not None) and not use_external_basis
    if enable_dynamic_B:
        wrappers = _get_external_field_wrappers(hybrid=bool(args.hybrid), mode=0)
        base_br = np.asarray(wrappers["Bx"][:], dtype=np.float64)
        base_bt = np.asarray(wrappers["By"][:], dtype=np.float64)
        base_bz = np.asarray(wrappers["Bz"][:], dtype=np.float64)
        base_norm = float(
            max(1.0e-30, np.max(np.abs(base_br)), np.max(np.abs(base_bt)), np.max(np.abs(base_bz)))
        )
        a_theta = _compute_a_theta_from_bz(base_bz, dr=args.r_max / args.nr, r0=0.0)
        external_driver = ExternalFieldDriver(
            wrappers=wrappers,
            basis=[{"Br": base_br, "Bt": base_bt, "Bz": base_bz, "A_theta": a_theta}],
            dt=args.dt,
            induced_E=apply_induced_E,
        )
        b_scale_ref = {"base_norm": base_norm}

    print("Starting simulation loop...")
    b_scale_history = [] if enable_dynamic_B else None
    monitor = (
        RunMonitor(
            species_names,
            args.monitor_interval,
            drop_threshold=args.drop_threshold,
            abort_on_drop=args.abort_on_drop,
            grid=grid,
            args=args,
        )
        if args.monitor_interval
        else None
    )
    if lcr_model is not None and args.lcr_feedback != "off" and monitor is None:
        print("[lcr] feedback requested without monitor_interval; enabling lightweight monitor.")
        monitor = RunMonitor(
            species_names,
            None,
            drop_threshold=None,
            abort_on_drop=False,
            grid=grid,
            args=args,
        )
    coupling_stride = max(1, int(getattr(args, "lcr_coupling_stride", 1) or 1))
    strong_coupling = str(getattr(args, "lcr_coupling_mode", "weak")).lower() == "strong"
    driver_writeback_count = 0
    circuit_update_count = 0
    for i in range(args.max_steps):
        t_current = i * args.dt
        if external_driver is not None:
            if use_external_basis:
                coeffs = get_coeffs(t_current)
                external_driver.set_coefficients(coeffs)
                driver_writeback_count += 1
                if external_history is not None and i % record_period == 0:
                    external_history.append(
                        {"step": i, "time": t_current, "coeffs": [float(x) for x in coeffs]}
                    )
                if i % record_period == 0:
                    preview = ", ".join(f"{c:.4g}" for c in coeffs[:3])
                    more = "" if coeffs.size <= 3 else f", ... (n={coeffs.size})"
                    print(f"Step {i}: t={t_current:.3e}, external coeffs=[{preview}{more}]")
            elif enable_dynamic_B and base_norm is not None:
                if lcr_model is not None and args.lcr_feedback == "radius_rms" and monitor:
                    if strong_coupling and (i % coupling_stride == 0):
                        feedback_val = monitor.update_radius_rms(t_current)
                        if feedback_val is not None:
                            lcr_model.update_feedback(feedback_val, t_current)
                    elif not strong_coupling:
                        feedback_val = monitor.last_radius_rms
                        if feedback_val is not None:
                            fb_time = monitor.last_radius_rms_time
                            lcr_model.update_feedback(
                                feedback_val, fb_time if fb_time is not None else t_current
                            )
                if use_waveform:
                    B_target = float(get_B_ext(t_current))
                else:
                    B_target = float(lcr_model.step())
                    circuit_update_count += 1
                scale = B_target / base_norm
                external_driver.set_coefficients(np.array([scale], dtype=np.float64))
                driver_writeback_count += 1
                if b_scale_history is not None and i % record_period == 0:
                    b_scale_history.append(
                        {"step": i, "time": t_current, "B_target": B_target, "scale": float(scale)}
                    )
                if i % record_period == 0:
                    print(
                        f"Step {i}: t={t_current:.3e}, B_target={B_target:.4f}, scale={scale:.4f}"
                    )
        sim.step(1)
        if monitor:
            monitor.maybe_record(i, t_current)
            if monitor.should_abort():
                print("[monitor] aborting simulation due to drop threshold breach.")
                break

    if monitor and monitor.drop_breach and not monitor.should_abort():
        print("[monitor] drop threshold breached but abort disabled; inspect metadata.")

    dropped_total = None
    try:
        wx_instance = libwarpx.warpx.get_instance()
        dropped_total = wx_instance.dropped_particles_total
    except Exception as exc:
        print(f"Warning: unable to read dropped-particle counter: {exc}")

    species_summary = gather_species_stats(species_names)
    diag_metrics = None
    try:
        max_mode = max(0, args.n_azimuthal_modes - 1)
        diag_root = Path(args.diag_dir) if getattr(args, "diag_dir", None) else Path(args.metadata_dir)
        diag_metrics = collect_diag_mode_metrics(diag_root, max_mode)
    except Exception as exc:
        print(f"Warning: diag mode metrics unavailable: {exc}")

    print("Simulation complete.")
    if dropped_total is not None:
        print(f"[run-summary] dropped_particles_total={dropped_total}")
    if species_summary:
        print("[run-summary] species (global):")
        for name, stats in species_summary.items():
            print(
                f"  {name}: N={stats['num_particles']}, q={stats['charge_C']:.3e} C, "
                f"w={stats['weight_sum']:.3e}, E={stats['energy_J']:.3e} J"
            )
    lcr_stats = None
    lcr_history_path = None
    if lcr_model is not None and lcr_model.history:
        h = lcr_model.history
        lcr_stats = {
            "E_cap_initial": h[0]["E_cap"],
            "E_cap_final": h[-1]["E_cap"],
            "E_ind_initial": h[0]["E_ind"],
            "E_ind_final": h[-1]["E_ind"],
            "E_total_initial": h[0]["E_cap"] + h[0]["E_ind"],
            "E_total_final": h[-1]["E_cap"] + h[-1]["E_ind"],
            "I_final": h[-1]["I"],
            "B_est_final": h[-1]["B_est"],
            "feedback_mode": lcr_model.feedback_mode,
            "feedback_updates": lcr_model.feedback_updates,
            "feedback_alpha": lcr_model.feedback_alpha,
            "feedback_min": lcr_model.feedback_min,
            "feedback_max": lcr_model.feedback_max,
            "feedback_last": lcr_model.feedback_signal,
            "feedback_last_time": lcr_model.feedback_time,
        }
        print(
            f"[run-summary] LCR: final I={lcr_model.I:.3e} A, "
            f"B_est={lcr_stats['B_est_final']:.3e} T, "
            f"E_total={lcr_stats['E_total_final']:.3e} J "
            f"(steps={len(lcr_model.history)})"
        )

    metadata_path = None
    if args.metadata_dir:
        metadata_dir = Path(args.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        tag = args.run_tag or time.strftime("%Y%m%d-%H%M%S")
        metadata_path = metadata_dir / f"warpx_run_{tag}.json"
        args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        node = os.uname().nodename if hasattr(os, "uname") else None
        resistivity_meta = None
        if getattr(args, "hybrid", False):
            eta_param = args.hybrid_eta_expr if args.hybrid_eta_expr else args.hybrid_eta
            eta_h_param = args.hybrid_eta_h_expr if args.hybrid_eta_h_expr else args.hybrid_eta_h
            resistivity_meta = {
                "plasma_resistivity_expr": eta_param,
                "plasma_resistivity_scale": args.hybrid_eta_scale,
                "plasma_hyper_resistivity_expr": eta_h_param,
                "plasma_hyper_resistivity_scale": args.hybrid_eta_h_scale,
                "eta_source": "input_expr_scale",
            }
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": args.mode,
            "command": CMDLINE,
            "cwd": str(Path.cwd()),
            "hostname": node,
            "git_hash": git_head,
            "git_dirty": git_dirty,
            "run_tag": tag,
            "species_names": species_names,
            "species_stats": species_summary,
            "species_stats_init": initial_species,
            "dropped_particles_total": dropped_total,
            "drop_threshold": args.drop_threshold,
            "diag_period": args.diag_period,
            "args": args_dict,
            "resistivity": resistivity_meta,
            "lcr_stats": lcr_stats,
            "lcr_circuit_update_count": circuit_update_count if lcr_model is not None else None,
            "lcr_driver_writeback_count": driver_writeback_count if lcr_model is not None else None,
            "lcr_history": lcr_model.history if lcr_model is not None else None,
            "monitor": monitor.as_dict() if monitor else None,
            "diag_mode_metrics": diag_metrics,
            "inputs": {
                "b_file": str(b_path) if b_path else None,
                "b_meta": b_meta,
                "waveform": str(args.waveform) if args.waveform else None,
                "fluid_file": str(fluid_path) if fluid_path else None,
                "fluid_meta": fluid_meta,
                "use_lcr": args.use_lcr,
                "external_basis": list(getattr(args, "external_basis", []) or []),
                "external_basis_cfg": external_basis_cfg,
                "diag_dir": str(args.diag_dir) if getattr(args, "diag_dir", None) else None,
            },
            "b_scaling": {
                "enabled": enable_dynamic_B,
                "reference": b_scale_ref if enable_dynamic_B else None,
                "history": b_scale_history,
                "record_period": record_period if enable_dynamic_B else None,
            },
            "external_field": {
                "mode": external_basis_cfg["mode"] if external_basis_cfg else ("scale" if enable_dynamic_B else "static"),
                "basis": external_basis_cfg,
                "history": external_history,
                "record_period": record_period if external_history is not None else None,
            },
            "artifacts": {
                "lcr_history": None,
                "diag_mode_csv": None,
                "diag_mode_plot": None,
            },
        }
        try:
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            print(f"[run-summary] metadata saved to {metadata_path}")
        except Exception as exc:
            print(f"Warning: failed to write metadata to {metadata_path}: {exc}")

    if args.lcr_out and lcr_model is not None and lcr_model.history:
        out_path = Path(args.lcr_out)
        try:
            header = "t,I,V_cap,L,dL_dt,R_plasma,B_est,E_cap,E_ind"
            import csv
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header.split(","))
                for row in lcr_model.history:
                    writer.writerow(
                        [
                            row["t"],
                            row["I"],
                            row["V_cap"],
                            row["L"],
                            row["dL_dt"],
                            row["R_plasma"],
                            row["B_est"],
                            row["E_cap"],
                            row["E_ind"],
                        ]
                    )
            print(f"[run-summary] LCR history saved to {out_path}")
            lcr_history_path = str(out_path)
        except Exception as exc:
            print(f"Warning: failed to write LCR history to {out_path}: {exc}")
    # Backfill artifact path if available
    try:
        if metadata_path and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                meta_loaded = json.load(fh)
            artifacts = meta_loaded.get("artifacts", {})
            if lcr_history_path:
                artifacts["lcr_history"] = lcr_history_path
            meta_loaded["artifacts"] = artifacts
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(meta_loaded, fh, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"Warning: failed to update artifacts in metadata: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="WarpX RZ driver & smoke tests")
    
    # Pre-parse for --from-json to load defaults
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--from-json", type=str, help="Load arguments from a JSON run metadata file (under 'args' key).")
    known, _ = pre_parser.parse_known_args()
    
    defaults = {}
    if known.from_json:
        p = Path(known.from_json)
        if p.exists():
            print(f"[config] loading defaults from {p}...")
            try:
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # If the JSON structure is a run metadata file, args are in "args"
                if "args" in data and isinstance(data["args"], dict):
                    defaults = data["args"]
                else:
                    # Fallback: assume the root is the args dict
                    defaults = data
            except Exception as exc:
                print(f"Warning: failed to load JSON config {p}: {exc}")
        else:
            print(f"Warning: JSON config {p} not found.")

    parser.add_argument("--from-json", type=str, help="Load arguments from a JSON run metadata file.")
    parser.add_argument(
        "--mode",
        choices=["field-only", "const-b-plasma", "bfile-plasma", "full-driver", "fluid-init"],
        default="full-driver",
        help="Select smoke test or full driver mode.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--nr", type=int, default=NR)
    parser.add_argument("--nz", type=int, default=NZ)
    parser.add_argument("--r-max", type=float, default=R_MAX, dest="r_max")
    parser.add_argument("--z-max", type=float, default=Z_MAX, dest="z_max")
    parser.add_argument(
        "--periodic-z",
        action="store_true",
        help="Use periodic boundary conditions in z (fields+particles) to reduce end losses.",
    )
    parser.add_argument("--ppc", type=int, default=10, help="Particles per cell for uniform plasma.")
    parser.add_argument(
        "--b-file",
        default=str(B_FIELD_FILE),
        help="openPMD B field file to load (auto-generated if missing).",
    )
    parser.add_argument(
        "--external-basis",
        action="append",
        default=[],
        help=(
            "Optional: one or more openPMD thetaMode B files to superpose as external-field bases. "
            "If provided, the driver skips PICMI applied-field loading and sets external-field arrays "
            "directly each step (compatible with n_azimuthal_modes>1)."
        ),
    )
    parser.add_argument(
        "--external-basis-const",
        action="append",
        default=[],
        type=float,
        help="Constant coefficient for each --external-basis (order-matched). Missing entries default to 0.",
    )
    parser.add_argument(
        "--external-basis-waveform",
        action="append",
        default=[],
        help=(
            "CSV coefficient waveform for each --external-basis (order-matched). "
            "Format: 'path[:col]'. If col omitted, auto-picks from {coeff,value,I,B_est}."
        ),
    )
    parser.add_argument(
        "--external-basis-waveform-scale",
        action="append",
        default=[],
        type=float,
        help="Optional scale factor applied to each external basis waveform value (order-matched).",
    )
    parser.add_argument(
        "--waveform",
        default=WAVEFORM_FILE,
        help="CSV waveform with columns t,B_est (only used in full-driver mode).",
    )
    parser.add_argument(
        "--const-B",
        type=float,
        default=0.05,
        dest="const_B",
        help="Constant Bz (Tesla) for const-b-plasma mode.",
    )
    parser.add_argument(
        "--diag-period",
        type=int,
        default=5,
        help="Diagnostic output period (steps).",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.9,
        help="CFL factor for Yee solver; dt will be clamped to satisfy cfl*min(dr,dz)/c/sqrt(2).",
    )
    parser.add_argument(
        "--fluid-file",
        default=str(DEFAULT_FLUID_FILE),
        help="openPMD fluid file (rho/vr/vz/vphi/Ti/Te) for fluid-init mode.",
    )
    parser.add_argument("--ion-amu", type=float, default=1.0, help="Ion mass number A (used for default plasma and fluid-init).")
    parser.add_argument("--ion-charge", type=float, default=1.0, help="Ion charge state Z (used for default plasma and fluid-init).")
    parser.add_argument("--n0", type=float, default=1.0e16, help="Uniform plasma density (m^-3) for default/tilt synthetic loading.")
    parser.add_argument(
        "--fast-ion-fraction",
        type=float,
        default=0.0,
        help="Extra fast-ion density fraction relative to base ions (fluid-init only).",
    )
    parser.add_argument(
        "--fast-ion-amu",
        type=float,
        default=None,
        help="Fast-ion mass number A (defaults to ion_amu).",
    )
    parser.add_argument(
        "--fast-ion-charge",
        type=float,
        default=None,
        help="Fast-ion charge state Z (defaults to ion_charge).",
    )
    parser.add_argument(
        "--fast-ion-Ti-eV",
        type=float,
        default=None,
        help="Fast-ion temperature in eV (defaults to fluid Ti if present).",
    )
    parser.add_argument(
        "--fast-ion-vphi",
        type=float,
        default=0.0,
        help="Additional fast-ion toroidal bulk velocity (m/s).",
    )
    parser.add_argument(
        "--neutralize",
        action="store_true",
        default=True,
        help="Add a neutralizing ion species to the default plasma (on by default).",
    )
    parser.add_argument(
        "--no-neutralize",
        action="store_false",
        dest="neutralize",
        help="Disable the neutralizing ion species in default plasma modes.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag to include in run metadata filename.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "warpx"),
        help="Directory to store run metadata JSON (created if missing).",
    )
    parser.add_argument(
        "--diag-dir",
        type=str,
        default=None,
        help=(
            "Directory for WarpX field diagnostics output. If not set, a per-run subdirectory "
            "<metadata-dir>/diag_<run-tag>/ is used when --run-tag is provided, otherwise "
            "<metadata-dir>/diag/."
        ),
    )
    parser.add_argument("--max-particles", type=int, default=200000, help="Cap total macro-particles for fluid-init mode.")
    parser.add_argument(
        "--max-beta",
        type=float,
        default=0.2,
        help="Cap |v|/c for sampled particles (applied to fluid-init bulk/thermal); helps avoid superluminal artifacts.",
    )
    parser.add_argument(
        "--use-fluid-velocity",
        action="store_true",
        help="Seed particle velocities from fluid vr/vphi/vz (default: zero velocities).",
    )
    parser.add_argument(
        "--electron-bulk-from-ions",
        action="store_true",
        help="Give electrons the same bulk velocity as ions in fluid-init to reduce current noise.",
    )
    parser.add_argument(
        "--sample-thermal",
        action="store_true",
        help="Sample Maxwellian thermal velocities from Ti/Te in the fluid file.",
    )
    parser.add_argument("--rng-seed", type=int, default=None, help="Optional RNG seed for particle sampling.")
    parser.add_argument(
        "--quiet-start",
        action="store_true",
        help="Use deterministic stratified sampling per cell to reduce shot noise in fluid-init.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use WarpX Hybrid-PIC solver (ions kinetic, electrons fluid). Only supported in fluid-init for now.",
    )
    parser.add_argument("--hybrid-eta", type=float, default=0.0, help="Plasma resistivity eta (Ohm-m) for Hybrid-PIC.")
    parser.add_argument("--hybrid-eta-expr", type=str, default=None, help="Plasma resistivity expression for Hybrid-PIC (overrides --hybrid-eta).")
    parser.add_argument("--hybrid-eta-scale", type=float, default=1.0, help="Scale factor applied to hybrid plasma resistivity.")
    parser.add_argument("--hybrid-eta-h", type=float, default=0.0, help="Plasma hyper-resistivity for Hybrid-PIC.")
    parser.add_argument("--hybrid-eta-h-expr", type=str, default=None, help="Plasma hyper-resistivity expression for Hybrid-PIC (overrides --hybrid-eta-h).")
    parser.add_argument("--hybrid-eta-h-scale", type=float, default=1.0, help="Scale factor applied to hybrid plasma hyper-resistivity.")
    parser.add_argument("--hybrid-substeps", type=int, default=1, help="Hybrid-PIC B-field substeps.")
    parser.add_argument("--hybrid-n0", type=float, dest="hybrid_n0_fallback", default=0.0, help="Fallback n0 (m^-3) if not inferred from fluid.")
    parser.add_argument("--hybrid-Te", type=float, dest="hybrid_Te_fallback", default=None, help="Fallback Te (K) if not inferred from fluid.")
    parser.add_argument("--hybrid-nfloor-scale", type=float, default=0.05, help="n_floor = scale * n0 for Hybrid-PIC.")
    parser.add_argument("--hybrid-gamma", type=float, default=5.0 / 3.0, help="Gamma exponent for electron pressure (Hybrid-PIC).")
    parser.add_argument(
        "--induced-E",
        action="store_true",
        dest="induced_E",
        help="When using waveform scaling, inject analytic Et = -0.5*r*dB/dt (Faraday) to reduce inconsistency.",
    )
    parser.add_argument(
        "--solver",
        choices=["yee", "psatd"],
        default="yee",
        help="Field solver (PSATD required for n_azimuthal_modes>1 tilt-mode runs).",
    )
    parser.add_argument(
        "--n-azimuthal-modes",
        type=int,
        default=1,
        help="Number of azimuthal modes in RZ (set to 2 for m=1 tilt-mode tracking).",
    )
    parser.add_argument(
        "--use-lcr",
        action="store_true",
        help="Drive B(t) with a built-in LCR circuit instead of a precomputed waveform.",
    )
    parser.add_argument("--lcr-V0", type=float, default=2.0e4, help="Initial capacitor voltage (V).")
    parser.add_argument("--lcr-C", type=float, default=1.0e-4, help="Capacitance (F).")
    parser.add_argument("--lcr-R-line", type=float, default=0.01, help="Series line resistance (Ohm).")
    parser.add_argument("--lcr-L0", type=float, default=2.5e-6, help="Base inductance (H).")
    parser.add_argument("--lcr-L-alpha", type=float, default=1.0e-6, help="Inductance slope vs radius dL/dR (H/m).")
    parser.add_argument("--lcr-R-plasma0", type=float, default=0.1, help="Initial plasma radius (m) used in L(t).")
    parser.add_argument("--lcr-R-min", type=float, default=0.03, help="Minimum radius after compression (m).")
    parser.add_argument("--lcr-v-ramp", type=float, default=5e3, help="Radial ramp speed (m/s) for compression.")
    parser.add_argument("--lcr-turns", type=float, default=10.0, help="Coil turns for B estimate.")
    parser.add_argument("--lcr-R-coil", type=float, default=0.1, help="Coil radius for B estimate (m).")
    parser.add_argument("--lcr-kB", type=float, default=None, help="Override B= kB * I if provided.")
    parser.add_argument(
        "--lcr-feedback",
        choices=["off", "radius_rms"],
        default="off",
        help="Enable closed-loop feedback for LCR (requires --monitor-interval).",
    )
    parser.add_argument(
        "--lcr-coupling-mode",
        choices=["weak", "strong"],
        default="weak",
        help="Coupling mode for LCR feedback (strong updates every stride).",
    )
    parser.add_argument(
        "--lcr-coupling-stride",
        type=int,
        default=1,
        help="Stride (steps) for strong coupling feedback updates.",
    )
    parser.add_argument(
        "--lcr-feedback-alpha",
        type=float,
        default=0.5,
        help="Smoothing factor for feedback radius (0=hold, 1=raw).",
    )
    parser.add_argument(
        "--lcr-feedback-min",
        type=float,
        default=None,
        help="Clamp minimum feedback radius (m).",
    )
    parser.add_argument(
        "--lcr-feedback-max",
        type=float,
        default=None,
        help="Clamp maximum feedback radius (m).",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=None,
        help="If set, record per-step species totals and dropped counts every N steps into metadata.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=int,
        default=None,
        help="Warn when dropped_particles delta exceeds this value (requires --monitor-interval).",
    )
    parser.add_argument(
        "--abort-on-drop",
        action="store_true",
        help="Abort the run if drop-threshold is exceeded during monitoring.",
    )
    parser.add_argument(
        "--lcr-out",
        type=str,
        default=None,
        help="Optional CSV path to dump LCR history (t,I,V_cap,L,dL_dt,R_plasma,B_est,E_cap,E_ind).",
    )
    parser.add_argument(
        "--tilt-eps",
        type=float,
        default=0.0,
        help="Amplitude for m=1 density perturbation (weights scaled by 1+eps*cos(theta)); triggers synthetic loading.",
    )
    
    if defaults:
        # Filter defaults to only known args if necessary, or just let set_defaults handle it.
        # argparse set_defaults ignores extra keys? No, it adds them to the namespace.
        # We should try to only set defaults for keys that exist in the parser?
        # Actually, having extra keys in the namespace is usually fine.
        parser.set_defaults(**defaults)
        
    return parser.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
