#!/usr/bin/env python3
"""
Minimal WarpX 3D tilt smoke driver (non-hybrid).
Creates a small 3D particle cloud with an initial offset and drift to
produce a measurable centroid shift for tilt diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pywarpx
import pywarpx.Diagnostics as pw_diag
import pywarpx.Collisions as pw_collisions
from pywarpx import libwarpx, particle_containers, picmi
try:
    from pywarpx.LoadThirdParty import load_cupy
except Exception:
    load_cupy = None


DEFAULTS = {
    "nx": 32,
    "ny": 32,
    "nz": 64,
    "x_min": None,
    "x_max": 0.1,
    "y_min": None,
    "y_max": 0.1,
    "z_min": None,
    "z_max": 0.2,
    "ppc": 1,
    "blob_sigma": 0.04,
    "blob_center": [0.02, 0.0, 0.0],
    "drift_beta": [0.15, 0.0, 0.0],
    "particle_weight": 1.0e-8,
    "background_ppc": 0,
    "background_weight_scale": 0.05,
    "max_steps": 20,
    "dt": 5.0e-12,
    "diag_period": 4,
    "enable_field_diag": True,
    "enable_m1rho_diag": True,
    "enable_m1mom_diag": True,
    "enable_coil_diag": False,
    "coil_diag_name": "COIL",
    "coil_diag_interval": None,
    "enable_energy_diag": False,
    "energy_diag_name": "ENERGY0D",
    "energy_diag_interval": None,
    "enable_particle_number_diag": False,
    "particle_number_diag_name": "PNUM",
    "particle_number_diag_interval": None,
    "enable_rho_max_diag": False,
    "rho_max_diag_name": "RHOMAX",
    "rho_max_diag_interval": None,
    "enable_particle_energy_diag": False,
    "particle_energy_diag_name": "PENERGY",
    "particle_energy_diag_interval": None,
    "enable_u2_stats_diag": False,
    "u2_diag_name": "U2",
    "u2_species": "ions",
    "u2_interval": None,
    "enable_u2_direct_stats": False,
    "u2_direct_species": None,
    "enable_energy_hist_diag": False,
    "energy_hist_diag_name": "EHist",
    "energy_hist_species": "ions",
    "energy_hist_bin_number": 128,
    "energy_hist_bin_min": 0.0,
    "energy_hist_bin_max": 20.0,
    "energy_hist_interval": 4,
    "energy_hist_normalization": "",
    "energy_hist_function": "sqrt(1+ux*ux+uy*uy+uz*uz)-1",
    "enable_u2_hist_diag": False,
    "u2_hist_diag_name": "U2Hist",
    "u2_hist_species": "ions",
    "u2_hist_bin_number": 256,
    "u2_hist_bin_min": 0.0,
    "u2_hist_bin_max": 1.0e14,
    "u2_hist_interval": 4,
    "u2_hist_normalization": "",
    "u2_hist_function": "ux*ux+uy*uy+uz*uz",
    "enable_u2_hist_custom": False,
    "enable_collisions": False,
    "collision_nu_scale": 1.0,
    "enable_energy_drag": False,
    "energy_drag_nu_scale": 1.0,
    "enable_velocity_reset": True,
    "velocity_reset_end_step": None,
    "velocity_reset_interval": 1,
    "velocity_reset_species": None,
    "enable_energy_diffusion": False,
    "energy_diffusion_scale": 0.0,
    "energy_diffusion_mode": "u_kick",
    "energy_diffusion_seed": None,
    "enable_inject": True,
    "inject_end_step": None,
    "inject_end_istep": None,
    "inject_end_call": None,
    "inject_repeat_nsteps": None,
    "inject_stride_steps": None,
    "coil_axis": "x",
    "coil_center": None,
    "coil_rmax": None,
    "coil_plane_pos": None,
    "coil_turns": 1,
    "monitor_interval": 1,
    "drop_threshold": 100,
    "monitor_split_axis": None,
    "monitor_split_value": 0.0,
    "monitor_species": "ions",
    "cfl": 0.9,
    "seed": 42,
    "add_electrons": False,
    "applied_field_enabled": False,
    "applied_Bz_T": 0.0,
    "init_mode": "blob",
    "opmd_fluid_path": None,
    "opmd_b_path": None,
    "opmd_b_scale": 1.0,
    "ext_drive_start_step": 0,
    "opmd_ppc": None,
    "opmd_use_fluid_velocity": True,
    "opmd_weight_scale": 1.0,
    "opmd_vel_scale": 1.0,
    "opmd_max_beta": 0.95,
    "drive_envelope_enable": False,
    "drive_envelope_method": "step_rampdown",
    "drive_envelope_off_step": 0,
    "drive_envelope_ramp_steps": 0,
    "drive_envelope_floor": 0.0,
    "seed_config_path": None,
    "metadata_heartbeat_s": 60.0,
    "metadata_heartbeat_steps": None,
    "warpx_amr_restart": None,
    "checkpoint": {
        "enabled": False,
        "period": None,
        "write_dir": None,
        "file_prefix": None,
        "file_min_digits": None,
        "name": "chkpoint",
        "verbose": None,
    },
    "opmd_double_seed_shift": [0.0, 0.0, 0.0],
    "opmd_double_seed_drift": [0.0, 0.0, 0.0],
    "drift_mag_scale": 1.0,
    "opmd_double_seed_symmetric": True,
    "opmd_double_seed_drift_is_beta": False,
    "opmd_double_seed_drift_dynamic": False,
    "opmd_double_seed_group_ids": False,
    "opmd_double_seed_drift_axis": None,
    "opmd_double_seed_drift_vector": None,
    "opmd_double_seed_common_drift": [0.0, 0.0, 0.0],
    "opmd_double_seed_common_drift_is_beta": False,
    "drift": None,
    "drift_axis": None,
    "drift_vector": None,
    "tilt_seed_mode": "none",
    "tilt_seed_vkick_frac": 0.0,
    "tilt_seed_vkick_abs": 0.0,
    "tilt_seed_vkick_is_beta": False,
    "tilt_seed_bins_z": 16,
    "tilt_seed_function": "sin(pi*(z-zmin)/Lz)",
    "tilt_seed_sign": "A:+,B:-",
    "tilt_seed_y_offset_amp": 0.0,
    "tilt_seed_y_offset_profile": "sin",
    "tilt_seed_y_offset_z0": None,
    "tilt_seed_y_offset_z_halfwidth": None,
    "tilt_seed_y_offset_k": None,
    "m1_inject_mode": "none",
    "m1_inject_eps": 0.0,
    "m1_inject_axis": "y",
    "m1_inject_r_ref": None,
    "m1_inject_center": "domain",
    "m1_inject_phase": 0.0,
    "m1_inject_rho_min": 0.0,
    "m1_inject_dry_run": False,
    "m1_drive_repeat": False,
    "m1_drive_nsteps": 1,
    "m1_drive_stride": 1,
    "m1_rho_cos_repeat": False,
    "m1_rho_cos_nsteps": 1,
    "m1_rho_cos_stride": 1,
    "m1_rho_cos_eps": None,
    "particle_vel_stats_enabled": False,
    "circuit_mvp": {},
    "hybrid": {
        "enabled": False,
        "n0": 1.0e19,
        "Te_eV": 10.0,
        "eta": 1.0e-6,
        "eta_scale": 1.0,
        "eta_scale_low": None,
        "eta_scale_high": None,
        "eta_switch_step": None,
        "eta_h": 0.0,
        "eta_h_scale": 1.0,
        "etaJ2_diag_enabled": False,
        "etaJ2_diag_stride": 1,
        "substeps": 4,
        "nfloor_scale": 0.05,
        "gamma": 5.0 / 3.0,
    },
    "energy_spectrum": {
        "enabled": False,
        "species": "ions",
        "bins": 64,
        "min_eV": 1.0e-3,
        "max_eV": 1.0e5,
        "log_bins": True,
        "time_fractions": [0.0, 0.5, 1.0],
    },
}


def git_info(repo_root: Path):
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
        return head, bool(status)
    except Exception:
        return None, None


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            cfg.update(json.load(handle))
    seed_path = cfg.get("seed_config_path")
    if seed_path:
        seed_path = Path(seed_path)
        if not seed_path.is_absolute():
            seed_path = path.parent / seed_path
        if seed_path.exists():
            with seed_path.open("r", encoding="utf-8") as handle:
                cfg.update(json.load(handle))
    return cfg


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


_XP_CACHE = None


def array_module():
    global _XP_CACHE
    if _XP_CACHE is None:
        if load_cupy is None:
            _XP_CACHE = np
        else:
            xp, status = load_cupy()
            if status:
                try:
                    libwarpx.amr.Print(status)
                except Exception:
                    pass
            _XP_CACHE = xp
    return _XP_CACHE


def latest_diag_dir(diag_dir: Path | None) -> str | None:
    if diag_dir is None or not diag_dir.exists():
        return None
    latest = None
    latest_idx = None
    for item in diag_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith("diag"):
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        if latest_idx is None or idx > latest_idx:
            latest_idx = idx
            latest = item
    return str(latest) if latest is not None else None


def build_meta_base(
    cfg: dict,
    git_head: str | None,
    git_dirty: bool | None,
    hybrid_enabled: bool,
    species_names: list[str],
    initial_stats: dict,
    init_mode: str,
    opmd_fluid_data: dict | None,
    opmd_b_data: dict | None,
    particle_weight_hist: dict | None,
    b_apply: dict | None,
    double_seed_meta: dict | None,
    ee_cfg: dict,
) -> dict:
    base = {
        "args": cfg,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "git_hash": git_head,
        "git_dirty": git_dirty,
        "solver": "hybrid" if hybrid_enabled else "em_pic",
        "species_names": species_names,
        "species_stats_init": initial_stats,
    }
    if hybrid_enabled:
        hybrid_cfg = cfg.get("hybrid") or {}
        eta_scale = hybrid_cfg.get("eta_scale", 1.0)
        eta_h_scale = hybrid_cfg.get("eta_h_scale", 1.0)
        eta_scale_low = hybrid_cfg.get("eta_scale_low")
        eta_scale_high = hybrid_cfg.get("eta_scale_high")
        eta_switch_step = hybrid_cfg.get("eta_switch_step")
        etaJ2_diag_enabled = bool(hybrid_cfg.get("etaJ2_diag_enabled", False))
        etaJ2_diag_stride = int(hybrid_cfg.get("etaJ2_diag_stride", 1))
        try:
            eta_scale = float(eta_scale)
        except (TypeError, ValueError):
            pass
        try:
            eta_h_scale = float(eta_h_scale)
        except (TypeError, ValueError):
            pass
        eta_profile_enabled = False
        if eta_scale_low is not None and eta_scale_high is not None and eta_switch_step is not None:
            try:
                eta_scale_low = float(eta_scale_low)
                eta_scale_high = float(eta_scale_high)
                eta_switch_step = int(eta_switch_step)
                eta_profile_enabled = True
            except (TypeError, ValueError):
                eta_profile_enabled = False
        base["resistivity"] = {
            "plasma_resistivity_expr": hybrid_cfg.get("eta"),
            "plasma_resistivity_scale": eta_scale,
            "plasma_hyper_resistivity_expr": hybrid_cfg.get("eta_h"),
            "plasma_hyper_resistivity_scale": eta_h_scale,
            "eta_source": "input_expr_scale",
            "requested_eta_scale": eta_scale,
            "effective_eta_scale": eta_scale,
            "eta_clip_fraction": 0.0,
        }
        if eta_profile_enabled:
            base["resistivity"]["eta_profile"] = "two_segment"
            base["resistivity"]["eta_scale_low"] = eta_scale_low
            base["resistivity"]["eta_scale_high"] = eta_scale_high
            base["resistivity"]["eta_switch_step"] = eta_switch_step
        eta_min = hybrid_cfg.get("eta_min")
        eta_max = hybrid_cfg.get("eta_max")
        if eta_min is not None:
            base["resistivity"]["eta_min"] = eta_min
        if eta_max is not None:
            base["resistivity"]["eta_max"] = eta_max
        base["etaJ2"] = {
            "enabled": etaJ2_diag_enabled,
            "stride": etaJ2_diag_stride,
        }
    collisions_meta = cfg.get("collisions_meta")
    if isinstance(collisions_meta, dict):
        base["collisions"] = collisions_meta
    if init_mode in ("opmd", "opmd_double_seed"):
        base["opmd"] = {
            "fluid_path": opmd_fluid_data.get("path") if opmd_fluid_data else None,
            "b_path": opmd_b_data.get("path") if opmd_b_data else None,
            "shape": list(opmd_fluid_data["rho"].shape) if opmd_fluid_data else None,
            "spacing": opmd_fluid_data.get("spacing").tolist() if opmd_fluid_data else None,
            "offset": opmd_fluid_data.get("offset").tolist() if opmd_fluid_data else None,
            "unitSI": opmd_fluid_data.get("unitSI") if opmd_fluid_data else None,
            "init_mode": init_mode,
        }
        if particle_weight_hist is not None:
            base["particle_weight_hist"] = particle_weight_hist
        if b_apply is not None:
            base["bfield_apply"] = b_apply
        if init_mode == "opmd_double_seed":
            if double_seed_meta is None:
                double_seed_meta = {}
            base["opmd_double_seed"] = double_seed_meta
    if ee_cfg:
        base["electron_energy"] = {"config": ee_cfg, "records": [], "updates": 0, "feedback_updates": 0}
    return base


def read_etaJ2_proxy() -> dict | None:
    try:
        wx_instance = libwarpx.warpx.get_instance()
    except Exception:
        return None
    if not hasattr(wx_instance, "get_hybrid_pic_etaJ2_mean"):
        return None
    try:
        etaJ2_mean = float(wx_instance.get_hybrid_pic_etaJ2_mean())
        J2_mean = float(wx_instance.get_hybrid_pic_J2_mean())
        samples = int(wx_instance.get_hybrid_pic_etaJ2_samples())
        updates = int(wx_instance.get_hybrid_pic_etaJ2_updates())
    except Exception:
        return None
    return {
        "etaJ2_mean": etaJ2_mean,
        "J2_mean": J2_mean,
        "samples": samples,
        "updates": updates,
    }


def read_restart_sanity() -> dict | None:
    try:
        wx_instance = libwarpx.warpx.get_instance()
    except Exception:
        return None
    if not hasattr(wx_instance, "restart_efield_checked"):
        return None
    try:
        checked = bool(wx_instance.restart_efield_checked)
        finite = bool(wx_instance.restart_efield_finite)
        maxabs = float(wx_instance.restart_efield_maxabs)
    except Exception:
        return None
    return {
        "Efield_fp_checked": checked,
        "Efield_fp_finite": finite,
        "Efield_fp_maxabs": maxabs,
    }


def build_heartbeat(
    status: str,
    step: int | None,
    t_current: float | None,
    diag_dir: Path | None,
    monitor: RunMonitor | None,
    etaJ2_meta: dict | None = None,
) -> dict:
    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "last_step": step,
        "last_time": t_current,
    }
    last_diag = latest_diag_dir(diag_dir)
    if last_diag is not None:
        heartbeat["last_diag_dir"] = last_diag
    if monitor is not None:
        heartbeat["monitor_records"] = len(monitor.records)
        heartbeat["monitor_drop_breach"] = monitor.drop_breach
    if etaJ2_meta:
        heartbeat["etaJ2"] = etaJ2_meta
    return heartbeat


def write_metadata_snapshot(
    meta_base: dict,
    metadata_path: Path,
    heartbeat_path: Path,
    monitor: RunMonitor | None,
    status: str,
    step: int | None,
    t_current: float | None,
    diag_dir: Path | None,
    dropped_total: int | None = None,
    final_stats: dict | None = None,
    electron_energy_model: ElectronEnergyModel | None = None,
    energy_spectrum_model: ParticleEnergySpectrum | None = None,
    ee_cfg: dict | None = None,
) -> None:
    etaJ2_meta = read_etaJ2_proxy()
    heartbeat = build_heartbeat(status, step, t_current, diag_dir, monitor, etaJ2_meta)
    resistivity_meta = meta_base.get("resistivity")
    if resistivity_meta:
        heartbeat["resistivity"] = resistivity_meta
    restart_meta = meta_base.get("restart_sanity")
    if restart_meta:
        heartbeat["restart_sanity"] = restart_meta
    write_json_atomic(heartbeat_path, heartbeat)
    meta = dict(meta_base)
    meta["timestamp"] = heartbeat["timestamp"]
    meta["run_status"] = status
    meta["heartbeat"] = heartbeat
    meta["heartbeat_updated_at"] = heartbeat["timestamp"]
    meta["heartbeat_last_step"] = heartbeat.get("last_step")
    meta["heartbeat_last_time"] = heartbeat.get("last_time")
    meta["heartbeat_last_diag_dir"] = heartbeat.get("last_diag_dir")
    if "monitor_records" in heartbeat:
        meta["heartbeat_monitor_records"] = heartbeat.get("monitor_records")
    if "monitor_drop_breach" in heartbeat:
        meta["heartbeat_monitor_drop_breach"] = heartbeat.get("monitor_drop_breach")
    meta["monitor"] = monitor.as_dict() if monitor else {}
    if etaJ2_meta:
        etaJ2_cfg = meta.get("etaJ2") or {}
        etaJ2_cfg.update(etaJ2_meta)
        meta["etaJ2"] = etaJ2_cfg
    if energy_spectrum_model is not None and energy_spectrum_model.enabled:
        meta["energy_spectrum"] = energy_spectrum_model.as_dict()
    if dropped_total is not None:
        meta["dropped_particles_total"] = dropped_total
    if final_stats is not None:
        meta["species_stats"] = final_stats
    if electron_energy_model is not None:
        meta["electron_energy"] = {
            "config": ee_cfg or {},
            "records": electron_energy_model.records,
            "updates": electron_energy_model.updates,
            "feedback_updates": electron_energy_model.feedback_updates,
            "feedback_failures": electron_energy_model.feedback_failures,
            "feedback_target_used": electron_energy_model.feedback_target_used,
        }
    elif ee_cfg:
        meta["electron_energy"] = {"config": ee_cfg, "records": [], "updates": 0, "feedback_updates": 0}
    write_json_atomic(metadata_path, meta)


def sample_bounded_gaussian(rng, n, mean, sigma, vmin, vmax):
    samples = []
    remaining = n
    while remaining > 0:
        draw = rng.normal(mean, sigma, size=remaining * 2)
        draw = draw[(draw >= vmin) & (draw <= vmax)]
        if draw.size == 0:
            continue
        take = draw[:remaining]
        samples.append(take)
        remaining -= take.size
    return np.concatenate(samples)


def velocities_to_momenta(vx_arr, vy_arr, vz_arr, max_beta):
    c = picmi.constants.c
    vx_arr = np.asarray(vx_arr, dtype=float)
    vy_arr = np.asarray(vy_arr, dtype=float)
    vz_arr = np.asarray(vz_arr, dtype=float)
    vmag = np.sqrt(vx_arr * vx_arr + vy_arr * vy_arr + vz_arr * vz_arr) + 1.0e-50
    beta = vmag / c
    clip = np.minimum(1.0, max_beta / np.maximum(beta, 1.0e-30))
    vx_arr = vx_arr * clip
    vy_arr = vy_arr * clip
    vz_arr = vz_arr * clip
    beta_x = vx_arr / c
    beta_y = vy_arr / c
    beta_z = vz_arr / c
    beta2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z
    beta2 = np.minimum(beta2, 1.0 - 1.0e-12)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    ux_arr = gamma * beta_x
    uy_arr = gamma * beta_y
    uz_arr = gamma * beta_z
    return ux_arr, uy_arr, uz_arr


def _as_vec3(value):
    if isinstance(value, np.ndarray) and value.shape == (3,):
        return value.astype(float)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.array(value, dtype=float)
    return None


def resolve_drift_vector(cfg, shift_vec, centers=None):
    eps = 1.0e-12
    drift_axis = cfg.get("opmd_double_seed_drift_axis")
    if drift_axis is None:
        drift_axis = cfg.get("drift_axis")
    drift_axis = str(drift_axis).strip().lower() if drift_axis is not None else None

    drift_vector_cfg = cfg.get("opmd_double_seed_drift_vector")
    if drift_vector_cfg is None:
        drift_vector_cfg = cfg.get("drift_vector")
    drift_vector = _as_vec3(drift_vector_cfg) if drift_vector_cfg is not None else None

    drift_raw = cfg.get("opmd_double_seed_drift")
    if drift_raw is None:
        drift_raw = cfg.get("drift", [0.0, 0.0, 0.0])
    drift_raw_vec = _as_vec3(drift_raw)
    drift_mag = None
    if drift_raw_vec is not None:
        drift_mag = float(np.linalg.norm(drift_raw_vec))
    elif isinstance(drift_raw, (int, float)):
        drift_mag = float(drift_raw)
    drift_mag_scale = cfg.get("opmd_double_seed_drift_mag_scale")
    if drift_mag_scale is None:
        drift_mag_scale = cfg.get("drift_mag_scale", 1.0)
    try:
        drift_mag_scale = float(drift_mag_scale)
    except (TypeError, ValueError):
        drift_mag_scale = 1.0

    direction = None
    source = "legacy"
    fallback_used = False
    fallback_reason = None
    com_a = None
    com_b = None
    com_sep_vec = None
    com_sep_norm = None
    if centers is not None:
        com_a = np.array(centers[0], dtype=float)
        com_b = np.array(centers[1], dtype=float)
        com_sep = com_b - com_a
        com_sep_vec = com_sep.tolist()
        com_sep_norm = float(np.linalg.norm(com_sep))
    if drift_vector is not None and float(np.linalg.norm(drift_vector)) > 0.0:
        direction = drift_vector
        source = "vector"
    elif drift_axis:
        if drift_axis == "auto":
            fallback_reasons = []
            if centers is None:
                fallback_reasons.append("centers_missing")
            else:
                direction = -(np.array(com_b) - np.array(com_a))
                if float(np.linalg.norm(direction)) <= eps:
                    direction = None
                    fallback_reasons.append("com_sep_norm<eps")
            if direction is None:
                if shift_vec is None:
                    fallback_reasons.append("shift_missing")
                else:
                    shift_norm = float(np.linalg.norm(shift_vec))
                    if shift_norm <= eps:
                        fallback_reasons.append("shift_norm<eps")
                    else:
                        direction = shift_vec
                        fallback_reasons.append("use_shift")
            if direction is None:
                direction = np.array([1.0, 0.0, 0.0])
                fallback_reasons.append("use_x_axis")
            if fallback_reasons:
                fallback_used = True
                fallback_reason = ";".join(fallback_reasons)
            source = "auto"
        elif drift_axis in ("x", "y", "z"):
            axis_map = {"x": np.array([1.0, 0.0, 0.0]),
                        "y": np.array([0.0, 1.0, 0.0]),
                        "z": np.array([0.0, 0.0, 1.0])}
            direction = axis_map[drift_axis]
            source = f"axis:{drift_axis}"

    drift_unit = None
    if direction is not None:
        norm = float(np.linalg.norm(direction))
        if drift_mag is None:
            drift_mag = float(np.linalg.norm(direction)) if source == "vector" else 0.0
        if norm > 0.0:
            drift_unit = (direction / norm)
            drift_mag = drift_mag * drift_mag_scale
            drift_vec = drift_unit * drift_mag
        else:
            drift_vec = np.zeros(3)
    elif drift_raw_vec is not None:
        drift_vec = drift_raw_vec * drift_mag_scale
        drift_mag = float(np.linalg.norm(drift_vec))
    else:
        drift_vec = np.zeros(3)

    meta = {
        "drift_axis": drift_axis,
        "drift_mag": drift_mag,
        "drift_mag_scale": drift_mag_scale,
        "drift_source": source,
        "method": source,
        "drift_dir": direction.tolist() if direction is not None else None,
        "drift_unit": drift_unit.tolist() if drift_unit is not None else None,
        "com_a": com_a.tolist() if com_a is not None else None,
        "com_b": com_b.tolist() if com_b is not None else None,
        "com_sep_vec": com_sep_vec,
        "sep_vec": com_sep_vec,
        "com_sep_norm": com_sep_norm,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }
    return drift_vec, meta


def compute_global_centroid_from_tiles(x_tiles, y_tiles, z_tiles, w_tiles):
    total_mass = 0.0
    sum_xyz = np.zeros(3)
    for x_arr, y_arr, z_arr, w_arr in zip(x_tiles, y_tiles, z_tiles, w_tiles):
        if x_arr is None or len(x_arr) == 0:
            continue
        w_sum = w_arr.sum()
        w_sum_val = float(w_sum)
        if w_sum_val <= 0.0:
            continue
        total_mass += w_sum_val
        sum_xyz[0] += float((w_arr * x_arr).sum())
        sum_xyz[1] += float((w_arr * y_arr).sum())
        sum_xyz[2] += float((w_arr * z_arr).sum())
    if total_mass <= 0.0:
        return None
    return sum_xyz / total_mass


def compute_plane_split_centroids_from_tiles(x_tiles, y_tiles, z_tiles, w_tiles, point, normal):
    left_mass = right_mass = 0.0
    left_sum = np.zeros(3)
    right_sum = np.zeros(3)
    for x_arr, y_arr, z_arr, w_arr in zip(x_tiles, y_tiles, z_tiles, w_tiles):
        if x_arr is None or len(x_arr) == 0:
            continue
        plane = (
            (x_arr - point[0]) * normal[0]
            + (y_arr - point[1]) * normal[1]
            + (z_arr - point[2]) * normal[2]
        )
        left_mask = plane <= 0.0
        right_mask = plane > 0.0
        if int(left_mask.sum()) > 0:
            w_left = w_arr[left_mask]
            left_mass += float(w_left.sum())
            left_sum[0] += float((w_left * x_arr[left_mask]).sum())
            left_sum[1] += float((w_left * y_arr[left_mask]).sum())
            left_sum[2] += float((w_left * z_arr[left_mask]).sum())
        if int(right_mask.sum()) > 0:
            w_right = w_arr[right_mask]
            right_mass += float(w_right.sum())
            right_sum[0] += float((w_right * x_arr[right_mask]).sum())
            right_sum[1] += float((w_right * y_arr[right_mask]).sum())
            right_sum[2] += float((w_right * z_arr[right_mask]).sum())
    if left_mass <= 0.0 or right_mass <= 0.0:
        return None, None
    left_centroid = left_sum / left_mass
    right_centroid = right_sum / right_mass
    return left_centroid, right_centroid


def select_group_ids_by_plane(pc, point, normal):
    x_tiles = pc.get_particle_x(copy_to_host=True)
    y_tiles = pc.get_particle_y(copy_to_host=True)
    z_tiles = pc.get_particle_z(copy_to_host=True)
    id_tiles = pc.get_particle_id(copy_to_host=True)
    ids_left = []
    ids_right = []
    for x_arr, y_arr, z_arr, id_arr in zip(x_tiles, y_tiles, z_tiles, id_tiles):
        if x_arr is None or len(x_arr) == 0:
            continue
        plane = (
            (x_arr - point[0]) * normal[0]
            + (y_arr - point[1]) * normal[1]
            + (z_arr - point[2]) * normal[2]
        )
        left_mask = plane <= 0.0
        right_mask = plane > 0.0
        if int(left_mask.sum()) > 0:
            ids_left.append(id_arr[left_mask])
        if int(right_mask.sum()) > 0:
            ids_right.append(id_arr[right_mask])
    if not ids_left or not ids_right:
        return None, None
    return np.concatenate(ids_left), np.concatenate(ids_right)


def compute_group_centroids_from_tiles_by_ids(
    x_tiles,
    y_tiles,
    z_tiles,
    w_tiles,
    id_tiles,
    ids_left,
    ids_right,
):
    if ids_left is None or ids_right is None:
        return None, None
    if len(ids_left) == 0 or len(ids_right) == 0:
        return None, None
    xp = array_module()
    ids_left_xp = xp.asarray(ids_left)
    ids_right_xp = xp.asarray(ids_right)
    left_mass = right_mass = 0.0
    left_sum = np.zeros(3)
    right_sum = np.zeros(3)
    for x_arr, y_arr, z_arr, w_arr, id_arr in zip(
        x_tiles, y_tiles, z_tiles, w_tiles, id_tiles
    ):
        if x_arr is None or len(x_arr) == 0:
            continue
        mask_left = xp.isin(id_arr, ids_left_xp)
        mask_right = xp.isin(id_arr, ids_right_xp)
        if int(mask_left.sum()) > 0:
            w_left = w_arr[mask_left]
            left_mass += float(w_left.sum())
            left_sum[0] += float((w_left * x_arr[mask_left]).sum())
            left_sum[1] += float((w_left * y_arr[mask_left]).sum())
            left_sum[2] += float((w_left * z_arr[mask_left]).sum())
        if int(mask_right.sum()) > 0:
            w_right = w_arr[mask_right]
            right_mass += float(w_right.sum())
            right_sum[0] += float((w_right * x_arr[mask_right]).sum())
            right_sum[1] += float((w_right * y_arr[mask_right]).sum())
            right_sum[2] += float((w_right * z_arr[mask_right]).sum())
    if left_mass <= 0.0 or right_mass <= 0.0:
        return None, None
    left_centroid = left_sum / left_mass
    right_centroid = right_sum / right_mass
    return left_centroid, right_centroid


def apply_velocity_delta_by_ids(
    ux_tiles,
    uy_tiles,
    uz_tiles,
    id_tiles,
    ids_left,
    ids_right,
    delta_u,
):
    if ids_left is None or ids_right is None:
        return
    if len(ids_left) == 0 or len(ids_right) == 0:
        return
    xp = array_module()
    ids_left_xp = xp.asarray(ids_left)
    ids_right_xp = xp.asarray(ids_right)
    for ux_arr, uy_arr, uz_arr, id_arr in zip(ux_tiles, uy_tiles, uz_tiles, id_tiles):
        if ux_arr is None or len(ux_arr) == 0:
            continue
        mask_left = xp.isin(id_arr, ids_left_xp)
        mask_right = xp.isin(id_arr, ids_right_xp)
        if int(mask_left.sum()) > 0:
            ux_arr[mask_left] -= delta_u[0]
            uy_arr[mask_left] -= delta_u[1]
            uz_arr[mask_left] -= delta_u[2]
        if int(mask_right.sum()) > 0:
            ux_arr[mask_right] += delta_u[0]
            uy_arr[mask_right] += delta_u[1]
            uz_arr[mask_right] += delta_u[2]


def apply_velocity_delta_by_plane(
    x_tiles,
    y_tiles,
    z_tiles,
    ux_tiles,
    uy_tiles,
    uz_tiles,
    point,
    normal,
    delta_u,
):
    for x_arr, y_arr, z_arr, ux_arr, uy_arr, uz_arr in zip(
        x_tiles, y_tiles, z_tiles, ux_tiles, uy_tiles, uz_tiles
    ):
        if x_arr is None or len(x_arr) == 0:
            continue
        plane = (
            (x_arr - point[0]) * normal[0]
            + (y_arr - point[1]) * normal[1]
            + (z_arr - point[2]) * normal[2]
        )
        left_mask = plane <= 0.0
        right_mask = plane > 0.0
        if int(left_mask.sum()) > 0:
            ux_arr[left_mask] -= delta_u[0]
            uy_arr[left_mask] -= delta_u[1]
            uz_arr[left_mask] -= delta_u[2]
        if int(right_mask.sum()) > 0:
            ux_arr[right_mask] += delta_u[0]
            uy_arr[right_mask] += delta_u[1]
            uz_arr[right_mask] += delta_u[2]


def update_dynamic_drift(
    pc,
    drift_unit,
    drift_mag,
    drift_u_current,
    max_beta,
    smooth_alpha=0.2,
    group_ids=None,
):
    x_tiles = pc.get_particle_x(copy_to_host=False)
    y_tiles = pc.get_particle_y(copy_to_host=False)
    z_tiles = pc.get_particle_z(copy_to_host=False)
    w_tiles = pc.get_particle_weight(copy_to_host=False)
    id_tiles = pc.get_particle_id(copy_to_host=False)
    point = None
    left_centroid = None
    right_centroid = None
    if group_ids is not None:
        left_centroid, right_centroid = compute_group_centroids_from_tiles_by_ids(
            x_tiles, y_tiles, z_tiles, w_tiles, id_tiles, group_ids[0], group_ids[1]
        )
    if left_centroid is None or right_centroid is None:
        point = compute_global_centroid_from_tiles(x_tiles, y_tiles, z_tiles, w_tiles)
        if point is None:
            return drift_unit, drift_u_current, None
        left_centroid, right_centroid = compute_plane_split_centroids_from_tiles(
            x_tiles, y_tiles, z_tiles, w_tiles, point, drift_unit
        )
    if left_centroid is None or right_centroid is None:
        return drift_unit, drift_u_current, None
    sep_vec = right_centroid - left_centroid
    sep_norm = float(np.linalg.norm(sep_vec))
    if sep_norm <= 0.0:
        return drift_unit, drift_u_current, None
    new_unit = -sep_vec / sep_norm
    if drift_unit is not None:
        new_unit = (1.0 - smooth_alpha) * drift_unit + smooth_alpha * new_unit
        new_norm = float(np.linalg.norm(new_unit))
        if new_norm > 0.0:
            new_unit = new_unit / new_norm
        else:
            new_unit = drift_unit
    target = drift_mag * new_unit
    ux_t, uy_t, uz_t = velocities_to_momenta(
        np.array([target[0]]), np.array([target[1]]), np.array([target[2]]), max_beta
    )
    u_target = np.array([float(ux_t[0]), float(uy_t[0]), float(uz_t[0])])
    delta_u = u_target - drift_u_current
    if np.linalg.norm(delta_u) <= 0.0:
        return new_unit, drift_u_current, None
    ux_tiles = pc.get_particle_ux(copy_to_host=False)
    uy_tiles = pc.get_particle_uy(copy_to_host=False)
    uz_tiles = pc.get_particle_uz(copy_to_host=False)
    if group_ids is not None and left_centroid is not None and right_centroid is not None:
        apply_velocity_delta_by_ids(
            ux_tiles, uy_tiles, uz_tiles, id_tiles, group_ids[0], group_ids[1], delta_u
        )
    else:
        apply_velocity_delta_by_plane(
            x_tiles,
            y_tiles,
            z_tiles,
            ux_tiles,
            uy_tiles,
            uz_tiles,
            point,
            drift_unit,
            delta_u,
        )
    info = {
        "drift_unit": new_unit.tolist(),
        "drift_point": point.tolist() if point is not None else None,
        "drift_u": u_target.tolist(),
    }
    return new_unit, u_target, info


def load_openpmd_cartesian(path: Path, mesh: str, components: list[str]) -> dict:
    try:
        import h5py
    except Exception as exc:
        raise SystemExit(f"h5py required to read openPMD files: {exc}")

    with h5py.File(path, "r") as h5f:
        base = h5f[f"/data/0/meshes/{mesh}"]
        data = {}
        unit_si = {}
        for comp in components:
            if comp not in base:
                raise SystemExit(f"Missing component '{comp}' in openPMD file {path}")
            arr = np.asarray(base[comp], dtype=float)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            data[comp] = arr
            unit_val = base[comp].attrs.get("unitSI", None)
            if unit_val is not None:
                unit_si[comp] = float(unit_val)
        spacing = base[components[0]].attrs.get("gridSpacing", base.attrs.get("gridSpacing", None))
        offset = base[components[0]].attrs.get("gridGlobalOffset", base.attrs.get("gridGlobalOffset", None))
        if spacing is None or offset is None:
            raise SystemExit(f"gridSpacing/gridGlobalOffset missing in {path}")

    spacing = np.asarray(spacing, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if spacing.size != 3 or offset.size != 3:
        raise SystemExit(f"Expected 3D spacing/offset in {path}, got {spacing}, {offset}")
    data["spacing"] = spacing
    data["offset"] = offset
    data["path"] = str(path)
    if unit_si:
        data["unitSI"] = unit_si
    return data


def _cell_to_face_3d(arr: np.ndarray, target_shape: tuple[int, int, int], axis: int) -> np.ndarray:
    nx, ny, nz = arr.shape
    tx, ty, tz = target_shape
    if (tx, ty, tz) == (nx, ny, nz):
        return arr
    if axis == 0 and (tx, ty, tz) == (nx + 1, ny, nz):
        out = np.empty((nx + 1, ny, nz), dtype=arr.dtype)
        out[1:-1, :, :] = 0.5 * (arr[:-1, :, :] + arr[1:, :, :])
        out[0, :, :] = arr[0, :, :]
        out[-1, :, :] = arr[-1, :, :]
        return out
    if axis == 1 and (tx, ty, tz) == (nx, ny + 1, nz):
        out = np.empty((nx, ny + 1, nz), dtype=arr.dtype)
        out[:, 1:-1, :] = 0.5 * (arr[:, :-1, :] + arr[:, 1:, :])
        out[:, 0, :] = arr[:, 0, :]
        out[:, -1, :] = arr[:, -1, :]
        return out
    if axis == 2 and (tx, ty, tz) == (nx, ny, nz + 1):
        out = np.empty((nx, ny, nz + 1), dtype=arr.dtype)
        out[:, :, 1:-1] = 0.5 * (arr[:, :, :-1] + arr[:, :, 1:])
        out[:, :, 0] = arr[:, :, 0]
        out[:, :, -1] = arr[:, :, -1]
        return out
    raise ValueError(f"Cannot map cell array {arr.shape} to target {target_shape} on axis {axis}")


def _resolve_m1_center(
    cfg: dict, axis: str, spacing: np.ndarray, offset: np.ndarray, shape: tuple[int, int, int]
) -> tuple[tuple[float, float, float] | None, str | None]:
    nx, ny, nz = shape
    dx, dy, dz = spacing
    x0, y0, z0 = offset
    x_center = x0 + 0.5 * nx * dx
    y_center = y0 + 0.5 * ny * dy
    z_center = z0 + 0.5 * nz * dz
    center_cfg = cfg.get("m1_inject_center", "domain")
    if isinstance(center_cfg, (list, tuple)) and len(center_cfg) >= 2:
        try:
            if axis == "x":
                y_center = float(center_cfg[0])
                z_center = float(center_cfg[1])
            elif axis == "y":
                x_center = float(center_cfg[0])
                z_center = float(center_cfg[1])
            else:
                x_center = float(center_cfg[0])
                y_center = float(center_cfg[1])
        except Exception:
            return None, "center_parse_failed"
    elif center_cfg not in (None, "domain", "auto"):
        return None, "center_invalid"
    return (float(x_center), float(y_center), float(z_center)), None


def apply_m1_rho_cos_modulation(
    rho: np.ndarray, spacing: np.ndarray, offset: np.ndarray, cfg: dict
) -> tuple[np.ndarray, dict | None]:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("rho_cos", "rho_cosine", "rho_cosphi", "rho_cos_phi"):
        return rho, None
    eps = float(cfg.get("m1_inject_eps", 0.0))
    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
    phase = float(cfg.get("m1_inject_phase", 0.0))
    rho_min = float(cfg.get("m1_inject_rho_min", 0.0))
    if eps == 0.0:
        return rho, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "eps_zero",
        }
    if axis not in ("x", "y", "z"):
        return rho, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "axis_invalid",
        }
    center, center_err = _resolve_m1_center(cfg, axis, spacing, offset, rho.shape)
    if center_err is not None:
        return rho, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": center_err,
        }
    x_center, y_center, z_center = center
    nx, ny, nz = rho.shape
    dx, dy, dz = spacing
    x0, y0, z0 = offset

    rho_in = rho
    rho = np.array(rho, copy=True)
    stats_before = {
        "min": float(np.min(rho_in)),
        "max": float(np.max(rho_in)),
        "mean": float(np.mean(rho_in)),
    }

    if axis == "x":
        y_centers = y0 + (np.arange(ny) + 0.5) * dy
        z_centers = z0 + (np.arange(nz) + 0.5) * dz
        phi = np.arctan2(z_centers[None, :] - z_center, y_centers[:, None] - y_center)
        mod = 1.0 + eps * np.cos(phi - phase)
        rho *= mod[None, :, :]
    elif axis == "y":
        x_centers = x0 + (np.arange(nx) + 0.5) * dx
        z_centers = z0 + (np.arange(nz) + 0.5) * dz
        phi = np.arctan2(z_centers[None, :] - z_center, x_centers[:, None] - x_center)
        mod = 1.0 + eps * np.cos(phi - phase)
        rho *= mod[:, None, :]
    else:
        x_centers = x0 + (np.arange(nx) + 0.5) * dx
        y_centers = y0 + (np.arange(ny) + 0.5) * dy
        phi = np.arctan2(y_centers[None, :] - y_center, x_centers[:, None] - x_center)
        mod = 1.0 + eps * np.cos(phi - phase)
        rho *= mod[:, :, None]

    clipped = None
    if rho_min > 0.0:
        clipped = int(np.sum(rho < rho_min))
        rho = np.maximum(rho, rho_min)

    stats_after = {
        "min": float(np.min(rho)),
        "max": float(np.max(rho)),
        "mean": float(np.mean(rho)),
    }
    meta = {
        "applied": True,
        "mode": mode,
        "eps": eps,
        "axis": axis,
        "phase": phase,
        "center_used": [float(x_center), float(y_center), float(z_center)],
        "rho_min": rho_min,
        "rho_stats_before": stats_before,
        "rho_stats_after": stats_after,
    }
    if clipped is not None:
        meta["rho_clip_count"] = clipped
    return rho, meta


def apply_initial_bfield_from_opmd(b_field: dict, hybrid: bool, scale: float = 1.0) -> dict:
    try:
        from pywarpx import fields
    except Exception as exc:
        return {"applied": False, "error": f"pywarpx.fields unavailable: {exc}"}

    def _grab(candidates: list[str]):
        for name in candidates:
            if hasattr(fields, name):
                try:
                    wrapper = getattr(fields, name)(0)
                    _ = wrapper[:]
                    return wrapper, name
                except Exception:
                    continue
        return None, None

    if hybrid:
        bx, bx_name = _grab(
            ["BxFPWrapper", "BxFPExternalWrapper", "BxHybridExternalWrapper", "BxWrapper"]
        )
        by, by_name = _grab(
            ["ByFPWrapper", "ByFPExternalWrapper", "ByHybridExternalWrapper", "ByWrapper"]
        )
        bz, bz_name = _grab(
            ["BzFPWrapper", "BzFPExternalWrapper", "BzHybridExternalWrapper", "BzWrapper"]
        )
    else:
        bx, bx_name = _grab(["BxFPWrapper", "BxFPExternalWrapper", "BxWrapper"])
        by, by_name = _grab(["ByFPWrapper", "ByFPExternalWrapper", "ByWrapper"])
        bz, bz_name = _grab(["BzFPWrapper", "BzFPExternalWrapper", "BzWrapper"])

    if bx is None or by is None or bz is None:
        return {"applied": False, "error": "B field wrappers unavailable"}

    try:
        scale_val = float(scale)
    except (TypeError, ValueError):
        scale_val = 1.0
    Bx = b_field["x"] * scale_val
    By = b_field["y"] * scale_val
    Bz = b_field["z"] * scale_val
    try:
        bx[:] = _cell_to_face_3d(Bx, bx[:].shape, axis=0)
        by[:] = _cell_to_face_3d(By, by[:].shape, axis=1)
        bz[:] = _cell_to_face_3d(Bz, bz[:].shape, axis=2)
    except Exception as exc:
        return {"applied": False, "error": f"field_write_failed: {exc}"}
    return {
        "applied": True,
        "wrapper": {"Bx": bx_name, "By": by_name, "Bz": bz_name},
        "scale": scale_val,
    }


def apply_m1_vel_kick_modulation(
    vx: np.ndarray | None,
    vy: np.ndarray | None,
    vz: np.ndarray | None,
    spacing: np.ndarray,
    offset: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("vel_kick", "velocity_kick", "v_kick"):
        return vx, vy, vz, None
    eps = float(cfg.get("m1_inject_eps", 0.0))
    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
    phase = float(cfg.get("m1_inject_phase", 0.0))
    if eps == 0.0:
        return vx, vy, vz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "eps_zero",
        }
    if axis not in ("x", "y", "z"):
        return vx, vy, vz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "axis_invalid",
        }
    if vx is None or vy is None or vz is None:
        return vx, vy, vz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "missing_bulk_velocity",
        }

    center, center_err = _resolve_m1_center(cfg, axis, spacing, offset, vx.shape)
    if center_err is not None:
        return vx, vy, vz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": center_err,
        }
    x_center, y_center, z_center = center
    nx, ny, nz = vx.shape
    dx, dy, dz = spacing
    x0, y0, z0 = offset

    if axis == "x":
        v_cos = vy
        v_sin = vz
        y_centers = y0 + (np.arange(ny) + 0.5) * dy
        z_centers = z0 + (np.arange(nz) + 0.5) * dz
        phi = np.arctan2(z_centers[None, :] - z_center, y_centers[:, None] - y_center)
        mod_cos = 1.0 + eps * np.cos(phi - phase)
        mod_sin = 1.0 + eps * np.sin(phi - phase)
        v_cos_min_before = float(np.min(v_cos))
        v_cos_max_before = float(np.max(v_cos))
        v_sin_min_before = float(np.min(v_sin))
        v_sin_max_before = float(np.max(v_sin))
        vy = v_cos * mod_cos[None, :, :]
        vz = v_sin * mod_sin[None, :, :]
        delta_cos = vy - v_cos
        delta_sin = vz - v_sin
        v_cos_min_after = float(np.min(vy))
        v_cos_max_after = float(np.max(vy))
        v_sin_min_after = float(np.min(vz))
        v_sin_max_after = float(np.max(vz))
        comp_cos = "vy"
        comp_sin = "vz"
    elif axis == "y":
        v_cos = vx
        v_sin = vz
        x_centers = x0 + (np.arange(nx) + 0.5) * dx
        z_centers = z0 + (np.arange(nz) + 0.5) * dz
        phi = np.arctan2(z_centers[None, :] - z_center, x_centers[:, None] - x_center)
        mod_cos = 1.0 + eps * np.cos(phi - phase)
        mod_sin = 1.0 + eps * np.sin(phi - phase)
        v_cos_min_before = float(np.min(v_cos))
        v_cos_max_before = float(np.max(v_cos))
        v_sin_min_before = float(np.min(v_sin))
        v_sin_max_before = float(np.max(v_sin))
        vx = v_cos * mod_cos[:, None, :]
        vz = v_sin * mod_sin[:, None, :]
        delta_cos = vx - v_cos
        delta_sin = vz - v_sin
        v_cos_min_after = float(np.min(vx))
        v_cos_max_after = float(np.max(vx))
        v_sin_min_after = float(np.min(vz))
        v_sin_max_after = float(np.max(vz))
        comp_cos = "vx"
        comp_sin = "vz"
    else:
        v_cos = vx
        v_sin = vy
        x_centers = x0 + (np.arange(nx) + 0.5) * dx
        y_centers = y0 + (np.arange(ny) + 0.5) * dy
        phi = np.arctan2(y_centers[None, :] - y_center, x_centers[:, None] - x_center)
        mod_cos = 1.0 + eps * np.cos(phi - phase)
        mod_sin = 1.0 + eps * np.sin(phi - phase)
        v_cos_min_before = float(np.min(v_cos))
        v_cos_max_before = float(np.max(v_cos))
        v_sin_min_before = float(np.min(v_sin))
        v_sin_max_before = float(np.max(v_sin))
        vx = v_cos * mod_cos[:, :, None]
        vy = v_sin * mod_sin[:, :, None]
        delta_cos = vx - v_cos
        delta_sin = vy - v_sin
        v_cos_min_after = float(np.min(vx))
        v_cos_max_after = float(np.max(vx))
        v_sin_min_after = float(np.min(vy))
        v_sin_max_after = float(np.max(vy))
        comp_cos = "vx"
        comp_sin = "vy"

    delta_cos_linf = float(np.max(np.abs(delta_cos)))
    delta_sin_linf = float(np.max(np.abs(delta_sin)))
    num_modified = int(np.count_nonzero((np.abs(delta_cos) > 0.0) | (np.abs(delta_sin) > 0.0)))

    meta = {
        "applied": True,
        "mode": mode,
        "eps": eps,
        "axis": axis,
        "phase": phase,
        "center_used": [float(x_center), float(y_center), float(z_center)],
        "component_cos": comp_cos,
        "component_sin": comp_sin,
        "num_modified": num_modified,
        f"{comp_cos}_delta_linf": delta_cos_linf,
        f"{comp_sin}_delta_linf": delta_sin_linf,
        f"{comp_cos}_min_before": v_cos_min_before,
        f"{comp_cos}_max_before": v_cos_max_before,
        f"{comp_sin}_min_before": v_sin_min_before,
        f"{comp_sin}_max_before": v_sin_max_before,
        f"{comp_cos}_min_after": v_cos_min_after,
        f"{comp_cos}_max_after": v_cos_max_after,
        f"{comp_sin}_min_after": v_sin_min_after,
        f"{comp_sin}_max_after": v_sin_max_after,
    }
    return vx, vy, vz, meta


def apply_m1_particle_vel_kick(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("particle_vel_kick", "particle_v_kick", "particle_kick"):
        return ux, uy, uz, None
    eps = float(cfg.get("m1_inject_eps", 0.0))
    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
    phase = float(cfg.get("m1_inject_phase", 0.0))
    do_kick = eps != 0.0
    if axis not in ("x", "y", "z"):
        return ux, uy, uz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "axis_invalid",
        }
    if x.size == 0 or ux.size == 0:
        return ux, uy, uz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "no_particles",
        }

    nx = int(cfg.get("nx") or 0)
    ny = int(cfg.get("ny") or 0)
    nz = int(cfg.get("nz") or 0)
    x_min = float(cfg.get("x_min", 0.0))
    x_max = float(cfg.get("x_max", 0.0))
    y_min = float(cfg.get("y_min", 0.0))
    y_max = float(cfg.get("y_max", 0.0))
    z_min = float(cfg.get("z_min", 0.0))
    z_max = float(cfg.get("z_max", 0.0))
    if nx <= 0 or ny <= 0 or nz <= 0 or x_max == x_min or y_max == y_min or z_max == z_min:
        center = (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
    else:
        spacing = np.array(
            [(x_max - x_min) / nx, (y_max - y_min) / ny, (z_max - z_min) / nz],
            dtype=float,
        )
        offset = np.array([x_min, y_min, z_min], dtype=float)
        center, center_err = _resolve_m1_center(cfg, axis, spacing, offset, (nx, ny, nz))
        if center_err is not None or center is None:
            center = (float(np.mean(x)), float(np.mean(y)), float(np.mean(z)))
    x_center, y_center, z_center = center

    c = picmi.constants.c
    u2 = ux * ux + uy * uy + uz * uz
    if axis == "x":
        uperp2 = uy * uy + uz * uz
    elif axis == "y":
        uperp2 = ux * ux + uz * uz
    else:
        uperp2 = ux * ux + uy * uy
    uperp_rms = float(np.sqrt(np.mean(uperp2))) if uperp2.size else 0.0
    uy_rms = float(np.sqrt(np.mean(uy * uy))) if uy.size else 0.0
    uz_rms = float(np.sqrt(np.mean(uz * uz))) if uz.size else 0.0

    gamma_before = np.sqrt(1.0 + u2 / (c * c))
    vx = ux / gamma_before
    vy = uy / gamma_before
    vz = uz / gamma_before

    vy_rms = float(np.sqrt(np.mean(vy * vy))) if vy.size else 0.0
    vz_rms = float(np.sqrt(np.mean(vz * vz))) if vz.size else 0.0
    vperp_rms_check = float(np.sqrt(vy_rms * vy_rms + vz_rms * vz_rms))
    if axis == "x":
        vperp2 = vy * vy + vz * vz
    elif axis == "y":
        vperp2 = vx * vx + vz * vz
    else:
        vperp2 = vx * vx + vy * vy
    vperp_rms = float(np.sqrt(np.mean(vperp2))) if vperp2.size else 0.0
    if not np.isfinite(vperp_rms) or vperp_rms <= 0.0:
        return ux, uy, uz, {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "vperp_rms_nonpositive",
        }
    delta_v = eps * vperp_rms if do_kick else 0.0

    if axis == "x":
        d1 = y - y_center
        d2 = z - z_center
        comp_cos = "uy"
        comp_sin = "uz"
        v_cos_before = vy
        v_sin_before = vz
        v_para = vx
        u_cos_before = uy.copy()
        u_sin_before = uz.copy()
    elif axis == "y":
        d1 = x - x_center
        d2 = z - z_center
        comp_cos = "ux"
        comp_sin = "uz"
        v_cos_before = vx
        v_sin_before = vz
        v_para = vy
        u_cos_before = ux.copy()
        u_sin_before = uz.copy()
    else:
        d1 = x - x_center
        d2 = y - y_center
        comp_cos = "ux"
        comp_sin = "uy"
        v_cos_before = vx
        v_sin_before = vy
        v_para = vz
        u_cos_before = ux.copy()
        u_sin_before = uy.copy()

    phi = np.arctan2(d2, d1)
    phi_mean = float(np.mean(phi)) if phi.size else 0.0
    phi_std = float(np.std(phi)) if phi.size else 0.0
    delta_cos = delta_v * np.cos(phi - phase) if do_kick else 0.0
    delta_sin = delta_v * np.sin(phi - phase) if do_kick else 0.0

    if axis == "x":
        v_cos_after = v_cos_before + delta_cos
        v_sin_after = v_sin_before + delta_sin
        vx_new = v_para
        vy_new = v_cos_after
        vz_new = v_sin_after
    elif axis == "y":
        v_cos_after = v_cos_before + delta_cos
        v_sin_after = v_sin_before + delta_sin
        vx_new = v_cos_after
        vy_new = v_para
        vz_new = v_sin_after
    else:
        v_cos_after = v_cos_before + delta_cos
        v_sin_after = v_sin_before + delta_sin
        vx_new = v_cos_after
        vy_new = v_sin_after
        vz_new = v_para

    beta2 = (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new) / (c * c)
    max_beta = float(cfg.get("opmd_max_beta", 0.95))
    beta = np.sqrt(beta2)
    clip = np.minimum(1.0, max_beta / np.maximum(beta, 1.0e-30))
    clip_mask = clip < 1.0
    clip_fraction = float(np.mean(clip_mask)) if clip_mask.size else 0.0
    if np.any(clip_mask):
        vx_new = vx_new * clip
        vy_new = vy_new * clip
        vz_new = vz_new * clip
        v_cos_after = v_cos_after * clip
        v_sin_after = v_sin_after * clip
    beta2 = (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new) / (c * c)
    beta2 = np.minimum(beta2, 1.0 - 1.0e-12)
    gamma_after = 1.0 / np.sqrt(1.0 - beta2)
    ux = gamma_after * vx_new
    uy = gamma_after * vy_new
    uz = gamma_after * vz_new

    if axis == "x":
        u_cos_after = uy
        u_sin_after = uz
    elif axis == "y":
        u_cos_after = ux
        u_sin_after = uz
    else:
        u_cos_after = ux
        u_sin_after = uy

    delta_v_cos_linf = float(np.max(np.abs(v_cos_after - v_cos_before)))
    delta_v_sin_linf = float(np.max(np.abs(v_sin_after - v_sin_before)))
    delta_u_cos_linf = float(np.max(np.abs(u_cos_after - u_cos_before)))
    delta_u_sin_linf = float(np.max(np.abs(u_sin_after - u_sin_before)))
    num_modified = int(x.size) if do_kick else 0

    ratio_uy_to_vy = float(uy_rms / (vy_rms + 1.0e-30))

    vel_unit = float(cfg.get("opmd_velocity_unitSI") or 1.0)
    meta = {
        "applied": bool(do_kick),
        "mode": mode,
        "eps": eps,
        "axis": axis,
        "phase": phase,
        "center_used": [float(x_center), float(y_center), float(z_center)],
        "kick_stage": "post_seed_pre_picmi",
        "dv_mode": "eps_times_vrms",
        "vperp_rms_at_inject": vperp_rms,
        "uperp_rms_at_inject": uperp_rms,
        "uy_rms_at_inject": uy_rms,
        "uz_rms_at_inject": uz_rms,
        "vy_rms_at_inject": vy_rms,
        "vz_rms_at_inject": vz_rms,
        "vperp_rms_at_inject_check": vperp_rms_check,
        "vperp_rms_at_inject_SI": vperp_rms * vel_unit,
        "velocity_unitSI": vel_unit,
        "ratio_uy_to_vy": ratio_uy_to_vy,
        "units_hint": "u=gamma*v (m/s); v=m/s",
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "dv_abs_applied": float(delta_v),
        "dv_abs_linf": max(delta_v_cos_linf, delta_v_sin_linf),
        "beta_clip_applied": bool(clip_fraction > 0.0),
        "beta_clip_fraction": clip_fraction,
        "num_modified": num_modified,
        f"{comp_cos}_delta_v_linf": delta_v_cos_linf,
        f"{comp_sin}_delta_v_linf": delta_v_sin_linf,
        f"{comp_cos}_delta_u_linf": delta_u_cos_linf,
        f"{comp_sin}_delta_u_linf": delta_u_sin_linf,
        f"{comp_cos}_min_before": float(np.min(v_cos_before)),
        f"{comp_cos}_max_before": float(np.max(v_cos_before)),
        f"{comp_sin}_min_before": float(np.min(v_sin_before)),
        f"{comp_sin}_max_before": float(np.max(v_sin_before)),
        f"{comp_cos}_min_after": float(np.min(v_cos_after)),
        f"{comp_cos}_max_after": float(np.max(v_cos_after)),
        f"{comp_sin}_min_after": float(np.min(v_sin_after)),
        f"{comp_sin}_max_after": float(np.max(v_sin_after)),
    }
    if not do_kick:
        meta["reason"] = "eps_zero"
    return ux, uy, uz, meta


def apply_energy_drag(pc, nu_scale: float, dt: float) -> dict:
    if nu_scale <= 0.0:
        return {"applied": False, "reason": "nu_scale_non_positive"}
    try:
        scale = math.exp(-nu_scale * dt)
    except Exception:
        scale = 1.0
    ux_tiles = pc.get_particle_ux(copy_to_host=False)
    uy_tiles = pc.get_particle_uy(copy_to_host=False)
    uz_tiles = pc.get_particle_uz(copy_to_host=False)
    if not ux_tiles:
        return {"applied": False, "reason": "no_particles"}
    xp = array_module()
    count = 0
    sum_u2_before = 0.0
    sum_u2_after = 0.0
    for ux_arr, uy_arr, uz_arr in zip(ux_tiles, uy_tiles, uz_tiles):
        if ux_arr is None or len(ux_arr) == 0:
            continue
        u2_before = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        sum_u2_before += float(xp.sum(u2_before))
        ux_arr *= scale
        uy_arr *= scale
        uz_arr *= scale
        u2_after = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        sum_u2_after += float(xp.sum(u2_after))
        count += int(ux_arr.size)
    delta_u2_sum = sum_u2_after - sum_u2_before
    return {
        "applied": True,
        "nu_scale": float(nu_scale),
        "dt": float(dt),
        "scale": float(scale),
        "num_modified": count,
        "u2_mean_before": float(sum_u2_before / count) if count > 0 else None,
        "u2_mean_after": float(sum_u2_after / count) if count > 0 else None,
        "delta_u2_sum": float(delta_u2_sum),
    }


def apply_energy_diffusion(pc, scale: float, dt: float, seed: int | None, step: int | None) -> dict:
    if scale <= 0.0:
        return {"applied": False, "reason": "scale_non_positive"}
    if dt <= 0.0:
        return {"applied": False, "reason": "dt_non_positive"}
    ux_tiles = pc.get_particle_ux(copy_to_host=False)
    uy_tiles = pc.get_particle_uy(copy_to_host=False)
    uz_tiles = pc.get_particle_uz(copy_to_host=False)
    if not ux_tiles:
        return {"applied": False, "reason": "no_particles"}
    xp = array_module()
    sigma = math.sqrt(scale * dt)
    seed_val = None
    if seed is not None:
        seed_val = int(seed) + (int(step) if step is not None else 0)
        try:
            if xp is np:
                rng = np.random.default_rng(seed_val)
            else:
                if hasattr(xp.random, "seed"):
                    xp.random.seed(seed_val)
                rng = xp.random
        except Exception:
            rng = None
    else:
        rng = None

    count = 0
    sum_u2_before = 0.0
    sum_u2_after = 0.0
    for ux_arr, uy_arr, uz_arr in zip(ux_tiles, uy_tiles, uz_tiles):
        if ux_arr is None or len(ux_arr) == 0:
            continue
        u2_before = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        sum_u2_before += float(xp.sum(u2_before))
        if rng is None:
            noise_x = xp.random.standard_normal(ux_arr.shape)
            noise_y = xp.random.standard_normal(uy_arr.shape)
            noise_z = xp.random.standard_normal(uz_arr.shape)
        else:
            if rng is np.random:
                noise_x = rng.standard_normal(ux_arr.shape)
                noise_y = rng.standard_normal(uy_arr.shape)
                noise_z = rng.standard_normal(uz_arr.shape)
            else:
                noise_x = rng.standard_normal(ux_arr.shape)
                noise_y = rng.standard_normal(uy_arr.shape)
                noise_z = rng.standard_normal(uz_arr.shape)
        ux_arr += sigma * noise_x
        uy_arr += sigma * noise_y
        uz_arr += sigma * noise_z
        u2_after = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        sum_u2_after += float(xp.sum(u2_after))
        count += int(ux_arr.size)
    delta_u2_sum = sum_u2_after - sum_u2_before
    return {
        "applied": True,
        "scale": float(scale),
        "dt": float(dt),
        "sigma": float(sigma),
        "num_modified": count,
        "u2_mean_before": float(sum_u2_before / count) if count > 0 else None,
        "u2_mean_after": float(sum_u2_after / count) if count > 0 else None,
        "delta_u2_sum": float(delta_u2_sum),
        "seed_used": seed_val,
    }


def read_u2_stats_snapshot(u2_path: Path, off_step: int | None) -> dict:
    if not u2_path.exists():
        return {}
    steps = []
    u2_mean = []
    u2_max = []
    try:
        for line in u2_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("step"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                step = int(float(parts[0]))
                mean_val = float(parts[2])
                max_val = float(parts[3])
            except Exception:
                continue
            steps.append(step)
            u2_mean.append(mean_val)
            u2_max.append(max_val)
    except Exception:
        return {}
    if not steps:
        return {}

    def value_at(target_step: int | None):
        if target_step is None:
            return None, None, None
        if target_step in steps:
            idx = steps.index(target_step)
            return u2_mean[idx], u2_max[idx], target_step
        nearest = min(steps, key=lambda s: abs(s - target_step))
        idx = steps.index(nearest)
        return u2_mean[idx], u2_max[idx], nearest

    step_end = max(steps)
    mean0, max0, step0_used = value_at(0)
    mean_off, max_off, step_off_used = value_at(off_step)
    mean_end, max_end, step_end_used = value_at(step_end)
    return {
        "u2_path": str(u2_path),
        "u2_step0_used": step0_used,
        "u2_stepOff_used": step_off_used,
        "u2_stepEnd_used": step_end_used,
        "u2_mean_step0": mean0,
        "u2_mean_stepOff": mean_off,
        "u2_mean_stepEnd": mean_end,
        "u2_max_step0": max0,
        "u2_max_stepOff": max_off,
        "u2_max_stepEnd": max_end,
    }


def compute_u2_direct(pc) -> dict:
    ux_tiles = pc.get_particle_ux(copy_to_host=False)
    uy_tiles = pc.get_particle_uy(copy_to_host=False)
    uz_tiles = pc.get_particle_uz(copy_to_host=False)
    w_tiles = pc.get_particle_weight(copy_to_host=False)
    if not ux_tiles:
        return {
            "u2_direct_count_end": 0,
            "u2_direct_weight_sum_end": 0.0,
            "u2_direct_u2_sum_end": 0.0,
            "u2_mean_end_direct": None,
        }
    xp = array_module()
    sum_w = 0.0
    sum_u2w = 0.0
    count = 0
    for idx, (ux_arr, uy_arr, uz_arr) in enumerate(zip(ux_tiles, uy_tiles, uz_tiles)):
        if ux_arr is None or len(ux_arr) == 0:
            continue
        u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        if w_tiles and idx < len(w_tiles) and w_tiles[idx] is not None and len(w_tiles[idx]) == len(ux_arr):
            w_arr = w_tiles[idx]
            sum_u2w += float(xp.sum(u2 * w_arr))
            sum_w += float(xp.sum(w_arr))
        else:
            sum_u2w += float(xp.sum(u2))
            sum_w += float(u2.size)
        count += int(u2.size)
    mean_direct = (sum_u2w / sum_w) if sum_w > 0.0 else None
    return {
        "u2_direct_count_end": count,
        "u2_direct_weight_sum_end": float(sum_w),
        "u2_direct_u2_sum_end": float(sum_u2w),
        "u2_mean_end_direct": float(mean_direct) if mean_direct is not None else None,
    }


def apply_m1_particle_vel_kick_container(pc, cfg: dict) -> dict | None:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("particle_vel_kick", "particle_v_kick", "particle_kick"):
        return None
    eps = float(cfg.get("m1_inject_eps", 0.0))
    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
    phase = float(cfg.get("m1_inject_phase", 0.0))
    if axis not in ("x", "y", "z"):
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "axis_invalid",
        }

    x_tiles = pc.get_particle_x(copy_to_host=False)
    y_tiles = pc.get_particle_y(copy_to_host=False)
    z_tiles = pc.get_particle_z(copy_to_host=False)
    ux_tiles = pc.get_particle_ux(copy_to_host=False)
    uy_tiles = pc.get_particle_uy(copy_to_host=False)
    uz_tiles = pc.get_particle_uz(copy_to_host=False)

    if not ux_tiles:
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "no_particles",
        }

    c = picmi.constants.c
    sum_uy2 = 0.0
    sum_uz2 = 0.0
    sum_vy2 = 0.0
    sum_vz2 = 0.0
    count = 0
    sum_phi = 0.0
    sum_phi2 = 0.0

    for x_arr, y_arr, z_arr, ux_arr, uy_arr, uz_arr in zip(
        x_tiles, y_tiles, z_tiles, ux_tiles, uy_tiles, uz_tiles
    ):
        if ux_arr is None or len(ux_arr) == 0:
            continue
        x_arr = np.asarray(x_arr, dtype=float)
        y_arr = np.asarray(y_arr, dtype=float)
        z_arr = np.asarray(z_arr, dtype=float)
        ux_arr = np.asarray(ux_arr, dtype=float)
        uy_arr = np.asarray(uy_arr, dtype=float)
        uz_arr = np.asarray(uz_arr, dtype=float)

        u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
        gamma = np.sqrt(1.0 + u2 / (c * c))
        vy = uy_arr / gamma
        vz = uz_arr / gamma
        sum_uy2 += float(np.sum(uy_arr * uy_arr))
        sum_uz2 += float(np.sum(uz_arr * uz_arr))
        sum_vy2 += float(np.sum(vy * vy))
        sum_vz2 += float(np.sum(vz * vz))
        count += int(uy_arr.size)

        if axis == "x":
            d1 = y_arr
            d2 = z_arr
        elif axis == "y":
            d1 = x_arr
            d2 = z_arr
        else:
            d1 = x_arr
            d2 = y_arr
        phi = np.arctan2(d2, d1)
        sum_phi += float(np.sum(phi))
        sum_phi2 += float(np.sum(phi * phi))

    if count <= 0:
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "no_particles",
        }

    uy_rms = math.sqrt(sum_uy2 / count)
    uz_rms = math.sqrt(sum_uz2 / count)
    uperp_rms = math.sqrt(uy_rms * uy_rms + uz_rms * uz_rms)
    vy_rms = math.sqrt(sum_vy2 / count)
    vz_rms = math.sqrt(sum_vz2 / count)
    vperp_rms = math.sqrt(vy_rms * vy_rms + vz_rms * vz_rms)
    phi_mean = sum_phi / count
    phi_var = max(0.0, (sum_phi2 / count) - phi_mean * phi_mean)
    phi_std = math.sqrt(phi_var)

    vel_unit = float(cfg.get("opmd_velocity_unitSI") or 1.0)

    do_kick = eps != 0.0
    delta_v = eps * vperp_rms if do_kick else 0.0
    delta_v_cos_linf = 0.0
    delta_v_sin_linf = 0.0
    delta_u_cos_linf = 0.0
    delta_u_sin_linf = 0.0
    clip_fraction = 0.0
    num_modified = 0

    if do_kick:
        total_tiles = 0
        clip_accum = 0.0
        for x_arr, y_arr, z_arr, ux_arr, uy_arr, uz_arr in zip(
            x_tiles, y_tiles, z_tiles, ux_tiles, uy_tiles, uz_tiles
        ):
            if ux_arr is None or len(ux_arr) == 0:
                continue
            x_arr = np.asarray(x_arr, dtype=float)
            y_arr = np.asarray(y_arr, dtype=float)
            z_arr = np.asarray(z_arr, dtype=float)
            ux_arr = np.asarray(ux_arr, dtype=float)
            uy_arr = np.asarray(uy_arr, dtype=float)
            uz_arr = np.asarray(uz_arr, dtype=float)

            u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
            gamma = np.sqrt(1.0 + u2 / (c * c))
            vx = ux_arr / gamma
            vy = uy_arr / gamma
            vz = uz_arr / gamma

            if axis == "x":
                d1 = y_arr
                d2 = z_arr
                v_cos_before = vy
                v_sin_before = vz
                v_para = vx
                u_cos_before = uy_arr
                u_sin_before = uz_arr
            elif axis == "y":
                d1 = x_arr
                d2 = z_arr
                v_cos_before = vx
                v_sin_before = vz
                v_para = vy
                u_cos_before = ux_arr
                u_sin_before = uz_arr
            else:
                d1 = x_arr
                d2 = y_arr
                v_cos_before = vx
                v_sin_before = vy
                v_para = vz
                u_cos_before = ux_arr
                u_sin_before = uy_arr

            phi = np.arctan2(d2, d1)
            delta_cos = delta_v * np.cos(phi - phase)
            delta_sin = delta_v * np.sin(phi - phase)

            v_cos_after = v_cos_before + delta_cos
            v_sin_after = v_sin_before + delta_sin

            if axis == "x":
                vx_new = v_para
                vy_new = v_cos_after
                vz_new = v_sin_after
            elif axis == "y":
                vx_new = v_cos_after
                vy_new = v_para
                vz_new = v_sin_after
            else:
                vx_new = v_cos_after
                vy_new = v_sin_after
                vz_new = v_para

            beta2 = (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new) / (c * c)
            max_beta = float(cfg.get("opmd_max_beta", 0.95))
            beta = np.sqrt(beta2)
            clip = np.minimum(1.0, max_beta / np.maximum(beta, 1.0e-30))
            clip_mask = clip < 1.0
            if np.any(clip_mask):
                vx_new = vx_new * clip
                vy_new = vy_new * clip
                vz_new = vz_new * clip
                v_cos_after = v_cos_after * clip
                v_sin_after = v_sin_after * clip
            clip_accum += float(np.mean(clip_mask)) if clip_mask.size else 0.0
            total_tiles += 1

            beta2 = (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new) / (c * c)
            beta2 = np.minimum(beta2, 1.0 - 1.0e-12)
            gamma_after = 1.0 / np.sqrt(1.0 - beta2)
            ux_new = gamma_after * vx_new
            uy_new = gamma_after * vy_new
            uz_new = gamma_after * vz_new

            if axis == "x":
                u_cos_after = uy_new
                u_sin_after = uz_new
            elif axis == "y":
                u_cos_after = ux_new
                u_sin_after = uz_new
            else:
                u_cos_after = ux_new
                u_sin_after = uy_new

            delta_v_cos_linf = max(delta_v_cos_linf, float(np.max(np.abs(v_cos_after - v_cos_before))))
            delta_v_sin_linf = max(delta_v_sin_linf, float(np.max(np.abs(v_sin_after - v_sin_before))))
            delta_u_cos_linf = max(delta_u_cos_linf, float(np.max(np.abs(u_cos_after - u_cos_before))))
            delta_u_sin_linf = max(delta_u_sin_linf, float(np.max(np.abs(u_sin_after - u_sin_before))))

            # Write back to tile arrays
            ux_arr[:] = ux_new
            uy_arr[:] = uy_new
            uz_arr[:] = uz_new
            num_modified += int(ux_arr.size)

        clip_fraction = (clip_accum / total_tiles) if total_tiles > 0 else 0.0

    meta = {
        "applied": bool(do_kick),
        "mode": mode,
        "eps": eps,
        "axis": axis,
        "phase": phase,
        "kick_stage": "postmap_container",
        "vy_rms_postmap": vy_rms,
        "vz_rms_postmap": vz_rms,
        "vperp_rms_postmap": vperp_rms,
        "vy_rms_postmap_SI": vy_rms * vel_unit,
        "vz_rms_postmap_SI": vz_rms * vel_unit,
        "vperp_rms_postmap_SI": vperp_rms * vel_unit,
        "uy_rms_postmap": uy_rms,
        "uz_rms_postmap": uz_rms,
        "uperp_rms_postmap": uperp_rms,
        "ratio_uy_to_vy_postmap": uy_rms / (vy_rms + 1.0e-30),
        "velocity_unitSI": vel_unit,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "dv_mode": "eps_times_vrms",
        "dv_abs_applied": float(delta_v),
        "dv_abs_linf": max(delta_v_cos_linf, delta_v_sin_linf),
        "beta_clip_applied": bool(clip_fraction > 0.0),
        "beta_clip_fraction": clip_fraction,
        "num_modified": num_modified,
        "reason": "eps_zero" if not do_kick else None,
        "units_hint": "u=gamma*v (m/s); v=m/s",
    }
    return meta


def snapshot_particle_weights(pc):
    w_tiles = pc.get_particle_weight(copy_to_host=False)
    base_weights = []
    for w_arr in w_tiles:
        if w_arr is None or len(w_arr) == 0:
            base_weights.append(None)
        else:
            base_weights.append(np.array(w_arr, dtype=float, copy=True))
    return base_weights


def apply_m1_rho_cos_weight_container(
    pc,
    cfg: dict,
    center: tuple[float, float, float] | None,
    base_weights: list[np.ndarray | None],
) -> dict | None:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("rho_cos", "rho_cosine", "rho_cosphi", "rho_cos_phi"):
        return None
    eps_raw = cfg.get("m1_rho_cos_eps", cfg.get("m1_inject_eps", 0.0))
    try:
        eps = float(eps_raw)
    except (TypeError, ValueError):
        eps = 0.0
    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
    phase = float(cfg.get("m1_inject_phase", 0.0))
    if axis not in ("x", "y", "z"):
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "axis_invalid",
        }
    if center is None:
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "center_missing",
        }
    if eps == 0.0:
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "eps_zero",
        }

    x_tiles = pc.get_particle_x(copy_to_host=False)
    y_tiles = pc.get_particle_y(copy_to_host=False)
    z_tiles = pc.get_particle_z(copy_to_host=False)
    w_tiles = pc.get_particle_weight(copy_to_host=False)
    if not w_tiles:
        return {
            "applied": False,
            "mode": mode,
            "eps": eps,
            "axis": axis,
            "phase": phase,
            "reason": "no_particles",
        }

    x_center, y_center, z_center = center
    total = 0
    clip_count = 0
    for idx, (x_arr, y_arr, z_arr, w_arr) in enumerate(zip(x_tiles, y_tiles, z_tiles, w_tiles)):
        if w_arr is None or len(w_arr) == 0:
            continue
        if idx >= len(base_weights):
            base_weights.append(None)
        if base_weights[idx] is None or len(base_weights[idx]) != len(w_arr):
            base_weights[idx] = np.array(w_arr, dtype=float, copy=True)
        base = base_weights[idx]

        x_arr = np.asarray(x_arr, dtype=float)
        y_arr = np.asarray(y_arr, dtype=float)
        z_arr = np.asarray(z_arr, dtype=float)
        if axis == "x":
            d1 = y_arr - y_center
            d2 = z_arr - z_center
        elif axis == "y":
            d1 = x_arr - x_center
            d2 = z_arr - z_center
        else:
            d1 = x_arr - x_center
            d2 = y_arr - y_center
        phi = np.arctan2(d2, d1)
        mod = 1.0 + eps * np.cos(phi - phase)
        clip_mask = mod < 0.0
        if np.any(clip_mask):
            mod = np.maximum(mod, 0.0)
        w_new = base * mod
        w_arr[:] = w_new
        total += int(w_arr.size)
        clip_count += int(np.sum(clip_mask))

    clip_fraction = (clip_count / total) if total > 0 else 0.0
    return {
        "applied": True,
        "mode": mode,
        "eps": eps,
        "axis": axis,
        "phase": phase,
        "num_particles_modified": total,
        "rho_clip_fraction": clip_fraction,
    }

def enforce_vel_kick(cfg: dict, meta: dict | None) -> None:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("vel_kick", "velocity_kick", "v_kick"):
        return
    eps = float(cfg.get("m1_inject_eps", 0.0))
    if eps == 0.0:
        return
    if meta is None:
        raise RuntimeError("vel_kick requested but bulk velocity missing.")
    if not bool(meta.get("applied", False)):
        reason = meta.get("reason") or meta.get("fail_reason") or "unknown_failure"
        raise RuntimeError(f"vel_kick requested but failed: {reason}")
    num_modified = int(meta.get("num_modified") or 0)
    if num_modified <= 0:
        raise RuntimeError("vel_kick applied to 0 elements.")


def enforce_particle_vel_kick(cfg: dict, meta: dict | None) -> None:
    mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if mode not in ("particle_vel_kick", "particle_v_kick", "particle_kick"):
        return
    eps = float(cfg.get("m1_inject_eps", 0.0))
    if eps == 0.0:
        return
    if meta is None:
        raise RuntimeError("particle_vel_kick requested but missing metadata.")
    if not bool(meta.get("applied", False)):
        reason = meta.get("reason") or meta.get("fail_reason") or "unknown_failure"
        if reason == "deferred_post_init":
            return
        raise RuntimeError(f"particle_vel_kick requested but failed: {reason}")
    num_modified = int(meta.get("num_modified") or 0)
    if num_modified <= 0:
        raise RuntimeError("particle_vel_kick applied to 0 elements.")


def sample_particles_from_cartesian(fluid: dict, cfg: dict, rng):
    rho = fluid["rho"]
    vx = fluid.get("vx")
    vy = fluid.get("vy")
    vz = fluid.get("vz")
    spacing = fluid["spacing"]
    offset = fluid["offset"]

    m1_rho_meta = None
    rho, m1_rho_meta = apply_m1_rho_cos_modulation(rho, spacing, offset, cfg)
    m1_vel_meta = None
    vx, vy, vz, m1_vel_meta = apply_m1_vel_kick_modulation(
        vx, vy, vz, spacing, offset, cfg
    )

    nx, ny, nz = rho.shape
    dx, dy, dz = spacing
    x0, y0, z0 = offset
    ppc = int(cfg.get("opmd_ppc") or cfg.get("ppc") or 1)
    if ppc <= 0:
        raise SystemExit("opmd_ppc/ppc must be >= 1 for openPMD initialization.")
    amu = float(cfg.get("ion_amu", 1.0))
    use_bulk_v = bool(cfg.get("opmd_use_fluid_velocity", True))
    weight_scale = float(cfg.get("opmd_weight_scale", 1.0))
    vel_scale = float(cfg.get("opmd_vel_scale", 1.0))
    max_beta = float(cfg.get("opmd_max_beta", 0.95))

    c = picmi.constants.c
    n = rho / (amu * 1.66053906660e-27)
    x_list = []
    y_list = []
    z_list = []
    ux_list = []
    uy_list = []
    uz_list = []
    w_list = []
    clip_weight = 0.0
    clip_weighted_sum = 0.0
    clip_weighted_clipped = 0.0
    clip_min = 1.0
    clip_max = 1.0

    for ix in range(nx):
        x_cell_min = x0 + ix * dx
        for iy in range(ny):
            y_cell_min = y0 + iy * dy
            for iz in range(nz):
                n_cell = n[ix, iy, iz]
                if n_cell <= 0.0:
                    continue
                z_cell_min = z0 + iz * dz
                volume = dx * dy * dz
                weight = n_cell * volume / ppc * weight_scale
                clip_factor = 1.0

                rand_x = rng.random(ppc)
                rand_y = rng.random(ppc)
                rand_z = rng.random(ppc)
                x_samples = x_cell_min + dx * rand_x
                y_samples = y_cell_min + dy * rand_y
                z_samples = z_cell_min + dz * rand_z

                if use_bulk_v and vx is not None and vy is not None and vz is not None:
                    vx_cell = float(vx[ix, iy, iz]) * vel_scale
                    vy_cell = float(vy[ix, iy, iz]) * vel_scale
                    vz_cell = float(vz[ix, iy, iz]) * vel_scale
                    vmag = math.sqrt(vx_cell * vx_cell + vy_cell * vy_cell + vz_cell * vz_cell)
                    if vmag > 0.0:
                        beta = vmag / c
                        if beta > 0.0:
                            clip_factor = min(1.0, max_beta / beta)
                    vx_samples = np.full(ppc, vx_cell)
                    vy_samples = np.full(ppc, vy_cell)
                    vz_samples = np.full(ppc, vz_cell)
                else:
                    vx_samples = np.zeros(ppc)
                    vy_samples = np.zeros(ppc)
                    vz_samples = np.zeros(ppc)

                ux_samples, uy_samples, uz_samples = velocities_to_momenta(
                    vx_samples, vy_samples, vz_samples, max_beta
                )
                clip_weight += n_cell * volume
                clip_weighted_sum += clip_factor * n_cell * volume
                if clip_factor < 1.0:
                    clip_weighted_clipped += n_cell * volume
                clip_min = min(clip_min, clip_factor)
                clip_max = max(clip_max, clip_factor)
                x_list.append(x_samples)
                y_list.append(y_samples)
                z_list.append(z_samples)
                ux_list.append(ux_samples)
                uy_list.append(uy_samples)
                uz_list.append(uz_samples)
                w_list.append(np.full(ppc, weight))

    if not x_list:
        raise SystemExit("No particles sampled from openPMD fluid (rho <= 0 everywhere).")

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    ux = np.concatenate(ux_list)
    uy = np.concatenate(uy_list)
    uz = np.concatenate(uz_list)
    w = np.concatenate(w_list)
    if clip_weight > 0.0:
        clip_mean = clip_weighted_sum / clip_weight
        clip_fraction = clip_weighted_clipped / clip_weight
    else:
        clip_mean = 1.0
        clip_fraction = 0.0
    vel_scale_meta = {
        "requested_merge_strength": vel_scale,
        "effective_merge_strength": vel_scale * clip_mean,
        "merge_strength_clip_fraction": clip_fraction,
        "merge_strength_clip_min": clip_min,
        "merge_strength_clip_max": clip_max,
        "merge_strength_clip_mean": clip_mean,
        "merge_strength_beta_cap": max_beta,
    }
    return x, y, z, ux, uy, uz, w, m1_rho_meta, vel_scale_meta, m1_vel_meta


def gather_species_stats(species_names):
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


class RunMonitor:
    def __init__(
        self,
        species_names,
        interval,
        drop_threshold=None,
        split_axis=None,
        split_value=0.0,
        split_species=None,
        electron_energy_model=None,
    ):
        self.species_names = species_names or []
        self.interval = max(1, interval) if interval else None
        self.records = []
        self.drop_threshold = drop_threshold
        self.drop_breach = False
        self.last_dropped = 0
        self.split_axis = split_axis
        self.split_value = split_value
        self.split_species = split_species
        self.electron_energy_model = electron_energy_model
        self.group_ids = None
        self.group_ids_meta = None

    def set_group_ids(self, ids_left, ids_right, meta=None):
        if ids_left is None or ids_right is None:
            return
        self.group_ids = (np.array(ids_left), np.array(ids_right))
        self.group_ids_meta = meta

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
        split_centroids = self._split_centroids()
        if split_centroids:
            entry["split_centroids"] = split_centroids
        group_centroids = self._group_centroids()
        if group_centroids:
            entry["group_centroids"] = group_centroids
        global_centroid = self._global_centroid()
        if global_centroid:
            entry["global_centroid"] = global_centroid
        if self.electron_energy_model is not None:
            ee_record = self.electron_energy_model.update(step_idx, t_current)
            if ee_record:
                entry["electron_energy"] = ee_record
        self.records.append(entry)

    def as_dict(self):
        payload = {
            "interval": self.interval,
            "records": self.records,
            "drop_threshold": self.drop_threshold,
            "drop_breach": self.drop_breach,
        }
        if self.group_ids_meta is not None:
            payload["group_ids_meta"] = self.group_ids_meta
        return payload

    def _split_centroids(self):
        if not self.split_axis or self.split_species is None:
            return None
        axis = str(self.split_axis).lower()
        if axis not in ("x", "y", "z"):
            return None
        try:
            pc = particle_containers.ParticleContainerWrapper(self.split_species)
        except Exception as exc:
            return {"error": f"split_centroids_container_failed: {exc}", "axis": axis}

        try:
            x_tiles = pc.get_particle_x(copy_to_host=True)
            y_tiles = pc.get_particle_y(copy_to_host=True)
            z_tiles = pc.get_particle_z(copy_to_host=True)
            w_tiles = pc.get_particle_weight(copy_to_host=True)
        except Exception as exc:
            return {"error": f"split_centroids_read_failed: {exc}", "axis": axis}

        left_mass = right_mass = 0.0
        left_sum = np.zeros(3)
        right_sum = np.zeros(3)

        for x_arr, y_arr, z_arr, w_arr in zip(x_tiles, y_tiles, z_tiles, w_tiles):
            if x_arr is None or len(x_arr) == 0:
                continue
            if axis == "x":
                left_mask = x_arr < self.split_value
                right_mask = x_arr > self.split_value
            elif axis == "y":
                left_mask = y_arr < self.split_value
                right_mask = y_arr > self.split_value
            else:
                left_mask = z_arr < self.split_value
                right_mask = z_arr > self.split_value

            if np.any(left_mask):
                w_left = w_arr[left_mask]
                left_mass += float(np.sum(w_left))
                left_sum[0] += float(np.sum(w_left * x_arr[left_mask]))
                left_sum[1] += float(np.sum(w_left * y_arr[left_mask]))
                left_sum[2] += float(np.sum(w_left * z_arr[left_mask]))
            if np.any(right_mask):
                w_right = w_arr[right_mask]
                right_mass += float(np.sum(w_right))
                right_sum[0] += float(np.sum(w_right * x_arr[right_mask]))
                right_sum[1] += float(np.sum(w_right * y_arr[right_mask]))
                right_sum[2] += float(np.sum(w_right * z_arr[right_mask]))

        if left_mass <= 0.0 or right_mass <= 0.0:
            return {"error": "split_centroids_empty", "axis": axis}

        left_centroid = (left_sum / left_mass).tolist()
        right_centroid = (right_sum / right_mass).tolist()
        sep_x = abs(left_centroid[0] - right_centroid[0])
        sep_xy = float(np.hypot(left_centroid[0] - right_centroid[0], left_centroid[1] - right_centroid[1]))
        sep_xyz = float(np.linalg.norm(np.array(left_centroid) - np.array(right_centroid)))
        return {
            "axis": axis,
            "split_value": self.split_value,
            "left": {"x": left_centroid[0], "y": left_centroid[1], "z": left_centroid[2]},
            "right": {"x": right_centroid[0], "y": right_centroid[1], "z": right_centroid[2]},
            "sep_x": sep_x,
            "sep_xy": sep_xy,
            "sep_xyz": sep_xyz,
        }

    def _group_centroids(self):
        if self.group_ids is None or self.split_species is None:
            return None
        ids_left, ids_right = self.group_ids
        if ids_left is None or ids_right is None:
            return None
        try:
            pc = particle_containers.ParticleContainerWrapper(self.split_species)
        except Exception as exc:
            return {"error": f"group_centroids_container_failed: {exc}"}

        try:
            x_tiles = pc.get_particle_x(copy_to_host=False)
            y_tiles = pc.get_particle_y(copy_to_host=False)
            z_tiles = pc.get_particle_z(copy_to_host=False)
            w_tiles = pc.get_particle_weight(copy_to_host=False)
            id_tiles = pc.get_particle_id(copy_to_host=False)
        except Exception as exc:
            return {"error": f"group_centroids_read_failed: {exc}"}

        left_centroid, right_centroid = compute_group_centroids_from_tiles_by_ids(
            x_tiles, y_tiles, z_tiles, w_tiles, id_tiles, ids_left, ids_right
        )
        if left_centroid is None or right_centroid is None:
            return {"error": "group_centroids_empty"}

        sep_x = abs(left_centroid[0] - right_centroid[0])
        sep_xy = float(
            np.hypot(left_centroid[0] - right_centroid[0], left_centroid[1] - right_centroid[1])
        )
        sep_xyz = float(np.linalg.norm(np.array(left_centroid) - np.array(right_centroid)))
        return {
            "left": {"x": left_centroid[0], "y": left_centroid[1], "z": left_centroid[2]},
            "right": {"x": right_centroid[0], "y": right_centroid[1], "z": right_centroid[2]},
            "sep_x": sep_x,
            "sep_xy": sep_xy,
            "sep_xyz": sep_xyz,
            "source": "id_groups",
        }

    def _global_centroid(self):
        if self.split_species is None:
            return None
        try:
            pc = particle_containers.ParticleContainerWrapper(self.split_species)
            x_tiles = pc.get_particle_x(copy_to_host=True)
            y_tiles = pc.get_particle_y(copy_to_host=True)
            z_tiles = pc.get_particle_z(copy_to_host=True)
            w_tiles = pc.get_particle_weight(copy_to_host=True)
        except Exception as exc:
            return {"error": f"global_centroid_read_failed: {exc}"}

        total_mass = 0.0
        sum_xyz = np.zeros(3)
        for x_arr, y_arr, z_arr, w_arr in zip(x_tiles, y_tiles, z_tiles, w_tiles):
            if x_arr is None or len(x_arr) == 0:
                continue
            w_sum = float(np.sum(w_arr))
            total_mass += w_sum
            sum_xyz[0] += float(np.sum(w_arr * x_arr))
            sum_xyz[1] += float(np.sum(w_arr * y_arr))
            sum_xyz[2] += float(np.sum(w_arr * z_arr))
        if total_mass <= 0.0:
            return {"error": "global_centroid_empty"}
        centroid = (sum_xyz / total_mass).tolist()
        amp_xy = float(np.hypot(centroid[0], centroid[1]))
        return {"x": centroid[0], "y": centroid[1], "z": centroid[2], "amp_xy": amp_xy}


class ElectronEnergyModel:
    def __init__(self, cfg: dict, species_names: list[str], hybrid_cfg: dict | None):
        self.enabled = bool(cfg.get("enabled", False))
        self.model = str(cfg.get("model", "radius_rms"))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.gamma = float(cfg.get("gamma", 5.0 / 3.0))
        self.update_interval = max(1, int(cfg.get("update_interval", 1)))
        self.species = str(cfg.get("species", "ions"))
        self.Te0_eV = float(
            cfg.get("Te0_eV", (hybrid_cfg or {}).get("Te_eV", 10.0))
        )
        self.Te_min_eV = float(cfg.get("Te_min_eV", 0.1))
        self.Te_max_eV = float(cfg.get("Te_max_eV", 1.0e3))
        self.feedback_target = str(cfg.get("feedback_target", "none")).lower()
        self.feedback_scale = float(cfg.get("feedback_scale", 1.0))
        self.floor_alpha = float(cfg.get("floor_alpha", 1.0))
        self.floor_fmin = float(cfg.get("floor_fmin", 0.0))
        self.floor_fmax = float(cfg.get("floor_fmax", 0.0))
        self.floor_max_abs = float(cfg.get("floor_max_abs", 0.0))
        self.eta_alpha = float(cfg.get("eta_alpha", 1.0))
        self.eta_min = float(cfg.get("eta_min", 0.0))
        self.eta_max = float(cfg.get("eta_max", 0.0))
        self.Te_ref_eV = float(cfg.get("Te_ref_eV", self.Te0_eV))
        self.Te_eps_eV = float(cfg.get("Te_eps_eV", 1.0e-6))
        self.feedback_updates = 0
        self.feedback_failures = 0
        self.feedback_target_used = None
        self.updates = 0
        self.records = []
        self._r0 = None
        self._hybrid_runtime = None
        self._hybrid_n0 = None
        self._hybrid_floor_scale = None
        self._eta0 = None
        if hybrid_cfg:
            self._hybrid_n0 = float(hybrid_cfg.get("n0", 0.0))
            self._hybrid_floor_scale = float(hybrid_cfg.get("nfloor_scale", 0.0))
            eta_raw = hybrid_cfg.get("eta", 0.0)
            if isinstance(eta_raw, (int, float)):
                self._eta0 = float(cfg.get("eta0", eta_raw))
            else:
                eta_override = cfg.get("eta0")
                if isinstance(eta_override, (int, float)):
                    self._eta0 = float(eta_override)
        if self.species not in species_names and species_names:
            self.species = species_names[0]

    def _particle_radius_rms(self) -> float | None:
        try:
            pc = particle_containers.ParticleContainerWrapper(self.species)
            x_tiles = pc.get_particle_x(copy_to_host=True)
            y_tiles = pc.get_particle_y(copy_to_host=True)
            z_tiles = pc.get_particle_z(copy_to_host=True)
            w_tiles = pc.get_particle_weight(copy_to_host=True)
        except Exception:
            return None
        total_w = 0.0
        sum_r2 = 0.0
        for x_arr, y_arr, z_arr, w_arr in zip(x_tiles, y_tiles, z_tiles, w_tiles):
            if x_arr is None or len(x_arr) == 0:
                continue
            x_arr = np.asarray(x_arr)
            y_arr = np.asarray(y_arr)
            z_arr = np.asarray(z_arr)
            w_arr = np.asarray(w_arr)
            if w_arr.size == 1:
                w_arr = np.full_like(x_arr, float(w_arr))
            elif w_arr.shape != x_arr.shape:
                try:
                    w_arr = np.broadcast_to(w_arr, x_arr.shape)
                except ValueError:
                    continue
            total_w += float(np.sum(w_arr))
            sum_r2 += float(np.sum(w_arr * (x_arr * x_arr + y_arr * y_arr + z_arr * z_arr)))
        if total_w <= 0.0:
            return None
        return float(math.sqrt(sum_r2 / total_w))

    def _update_density_floor(self, te_eV: float) -> float | None:
        if self.feedback_target != "density_floor":
            return None
        if not self._hybrid_n0 or self._hybrid_floor_scale is None:
            return None
        if self._hybrid_runtime is None:
            from pywarpx.HybridPICModel import HybridPICModel as HybridPICRuntime

            self._hybrid_runtime = HybridPICRuntime()
        base_floor = self._hybrid_n0 * self._hybrid_floor_scale
        te_ref = self.Te_ref_eV if self.Te_ref_eV > 0.0 else self.Te0_eV
        te_safe = max(te_eV, self.Te_eps_eV)
        scale = (te_ref / te_safe) ** self.floor_alpha if te_ref > 0.0 else 1.0
        if self.floor_fmin > 0.0:
            scale = max(scale, self.floor_fmin)
        if self.floor_fmax > 0.0:
            scale = min(scale, self.floor_fmax)
        n_floor = float(base_floor * scale * self.feedback_scale)
        if self.floor_max_abs > 0.0:
            n_floor = min(n_floor, self.floor_max_abs)
        self._hybrid_runtime.density_floor = n_floor
        self.feedback_updates += 1
        self.feedback_target_used = "density_floor"
        return n_floor

    def _update_resistivity(self, te_eV: float) -> float | None:
        if self.feedback_target != "resistivity":
            return None
        if self._eta0 is None or self._eta0 <= 0.0:
            self.feedback_failures += 1
            return None
        if self._hybrid_runtime is None:
            from pywarpx.HybridPICModel import HybridPICModel as HybridPICRuntime

            self._hybrid_runtime = HybridPICRuntime()
        te_ref = self.Te_ref_eV if self.Te_ref_eV > 0.0 else self.Te0_eV
        if te_ref <= 0.0:
            self.feedback_failures += 1
            return None
        scale = (te_ref / te_eV) ** self.eta_alpha if te_eV > 0.0 else 1.0
        eta_val = float(self._eta0 * scale * self.feedback_scale)
        eta_val = max(self.eta_min, eta_val)
        if self.eta_max > 0.0:
            eta_val = min(self.eta_max, eta_val)
        for attr in ("plasma_resistivity", "eta", "resistivity"):
            if hasattr(self._hybrid_runtime, attr):
                setattr(self._hybrid_runtime, attr, float(eta_val))
                self.feedback_updates += 1
                self.feedback_target_used = attr
                return eta_val
        for method in ("set_plasma_resistivity", "set_resistivity", "set_eta"):
            if hasattr(self._hybrid_runtime, method):
                getattr(self._hybrid_runtime, method)(float(eta_val))
                self.feedback_updates += 1
                self.feedback_target_used = method
                return eta_val
        self.feedback_failures += 1
        return None

    def update(self, step_idx: int, t_current: float) -> dict | None:
        if not self.enabled:
            return None
        if step_idx % self.update_interval != 0:
            return None
        if self.model != "radius_rms":
            return None
        r_rms = self._particle_radius_rms()
        if r_rms is None or not np.isfinite(r_rms):
            return None
        if self._r0 is None:
            self._r0 = r_rms
        ratio = self._r0 / r_rms if r_rms > 0.0 else 1.0
        te_model = self.Te0_eV * (ratio ** self.gamma)
        te_eV = (1.0 - self.alpha) * self.Te0_eV + self.alpha * te_model
        te_eV = max(self.Te_min_eV, min(self.Te_max_eV, te_eV))
        n_floor = self._update_density_floor(te_eV)
        eta_val = self._update_resistivity(te_eV)
        record = {
            "time": float(t_current),
            "r_rms": float(r_rms),
            "te_eV": float(te_eV),
            "te_model_eV": float(te_model),
            "te_ratio": float(ratio),
            "n_floor": n_floor,
            "eta": eta_val,
        }
        self.records.append(record)
        self.updates += 1
        return record


class ParticleEnergySpectrum:
    def __init__(self, cfg: dict, species_names: list[str], ion_amu: float):
        self.enabled = bool(cfg.get("enabled", False))
        self.species = str(cfg.get("species", "ions"))
        if self.species not in species_names and species_names:
            self.species = species_names[0]
        self.bins = max(4, int(cfg.get("bins", 64)))
        self.min_eV = float(cfg.get("min_eV", 1.0e-3))
        self.max_eV = float(cfg.get("max_eV", 1.0e5))
        self.log_bins = bool(cfg.get("log_bins", True))
        self.time_fractions = cfg.get("time_fractions", [0.0, 0.5, 1.0])
        self.records = []
        self.edges_eV = None
        self._steps = []
        self._recorded = set()
        if self.species == "electrons":
            self._mass_kg = picmi.constants.m_e
        else:
            self._mass_kg = ion_amu * 1.66053906660e-27

    def initialize(self, max_steps: int) -> None:
        if not self.enabled:
            return
        if max_steps <= 0:
            self._steps = []
            return
        steps = set()
        for frac in self.time_fractions:
            try:
                frac_val = float(frac)
            except (TypeError, ValueError):
                continue
            frac_val = max(0.0, min(1.0, frac_val))
            step = int(round(frac_val * (max_steps - 1)))
            steps.add(step)
        self._steps = sorted(steps)
        if self.min_eV <= 0.0:
            self.min_eV = 1.0e-6
        if self.log_bins:
            self.edges_eV = np.logspace(
                np.log10(self.min_eV), np.log10(self.max_eV), self.bins + 1
            ).tolist()
        else:
            self.edges_eV = np.linspace(self.min_eV, self.max_eV, self.bins + 1).tolist()

    def maybe_record(self, step_idx: int, t_current: float) -> dict | None:
        if not self.enabled or step_idx not in self._steps or step_idx in self._recorded:
            return None
        if self.edges_eV is None:
            return None
        try:
            pc = particle_containers.ParticleContainerWrapper(self.species)
            ux_tiles = pc.get_particle_ux(copy_to_host=True)
            uy_tiles = pc.get_particle_uy(copy_to_host=True)
            uz_tiles = pc.get_particle_uz(copy_to_host=True)
            w_tiles = pc.get_particle_weight(copy_to_host=True)
        except Exception:
            return None
        edges = np.asarray(self.edges_eV, dtype=float)
        min_edge = float(edges[0])
        max_edge = float(edges[-1])
        hist = np.zeros(len(edges) - 1, dtype=float)
        underflow_weight = 0.0
        overflow_weight = 0.0
        sample_count = 0
        e_min = None
        e_max = None
        gamma_min = None
        gamma_max = None
        beta_min = None
        beta_max = None
        c = picmi.constants.c
        q_e = picmi.constants.q_e
        mc2_eV = self._mass_kg * c * c / q_e
        for ux_arr, uy_arr, uz_arr, w_arr in zip(ux_tiles, uy_tiles, uz_tiles, w_tiles):
            if ux_arr is None or len(ux_arr) == 0:
                continue
            ux_arr = np.asarray(ux_arr, dtype=float)
            uy_arr = np.asarray(uy_arr, dtype=float)
            uz_arr = np.asarray(uz_arr, dtype=float)
            w_arr = np.asarray(w_arr, dtype=float)
            if w_arr.size == 1:
                w_arr = np.full_like(ux_arr, float(w_arr))
            elif w_arr.shape != ux_arr.shape:
                try:
                    w_arr = np.broadcast_to(w_arr, ux_arr.shape)
                except ValueError:
                    continue
            u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
            gamma = np.sqrt(1.0 + u2)
            energy_eV = (gamma - 1.0) * mc2_eV
            mask = np.isfinite(energy_eV)
            if not np.all(mask):
                energy_eV = energy_eV[mask]
                w_arr = w_arr[mask]
                gamma = gamma[mask]
            if energy_eV.size == 0:
                continue
            sample_count += int(energy_eV.size)
            tile_min = float(np.min(energy_eV))
            tile_max = float(np.max(energy_eV))
            e_min = tile_min if e_min is None else min(e_min, tile_min)
            e_max = tile_max if e_max is None else max(e_max, tile_max)
            g_min = float(np.min(gamma))
            g_max = float(np.max(gamma))
            gamma_min = g_min if gamma_min is None else min(gamma_min, g_min)
            gamma_max = g_max if gamma_max is None else max(gamma_max, g_max)
            u = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
            beta = u / np.maximum(gamma, 1.0e-30)
            b_min = float(np.min(beta))
            b_max = float(np.max(beta))
            beta_min = b_min if beta_min is None else min(beta_min, b_min)
            beta_max = b_max if beta_max is None else max(beta_max, b_max)
            under_mask = energy_eV < min_edge
            over_mask = energy_eV >= max_edge
            if np.any(under_mask):
                underflow_weight += float(np.sum(w_arr[under_mask]))
            if np.any(over_mask):
                overflow_weight += float(np.sum(w_arr[over_mask]))
            inrange_mask = ~(under_mask | over_mask)
            if np.any(inrange_mask):
                hist += np.histogram(energy_eV[inrange_mask], bins=edges, weights=w_arr[inrange_mask])[0]
        total = float(np.sum(hist))
        counts_norm = (hist / total).tolist() if total > 0.0 else None
        record = {
            "step": int(step_idx),
            "time": float(t_current),
            "counts": hist.tolist(),
            "counts_norm": counts_norm,
            "total_weight": total,
            "inrange_weight": total,
            "underflow_weight": float(underflow_weight),
            "overflow_weight": float(overflow_weight),
            "num_particles_sampled": int(sample_count),
            "species_name": self.species,
            "min_eV": float(self.min_eV),
            "max_eV": float(self.max_eV),
            "bins": int(self.bins),
            "E_min_eV": None if e_min is None else float(e_min),
            "E_max_eV": None if e_max is None else float(e_max),
            "gamma_min": None if gamma_min is None else float(gamma_min),
            "gamma_max": None if gamma_max is None else float(gamma_max),
            "v_over_c_min": None if beta_min is None else float(beta_min),
            "v_over_c_max": None if beta_max is None else float(beta_max),
            "mc2_eV": float(mc2_eV),
            "E_from_gamma_eV_min": None if gamma_min is None else float((gamma_min - 1.0) * mc2_eV),
            "E_from_gamma_eV_max": None if gamma_max is None else float((gamma_max - 1.0) * mc2_eV),
        }
        self.records.append(record)
        self._recorded.add(step_idx)
        return record

    def as_dict(self) -> dict:
        return {
            "config": {
                "species": self.species,
                "bins": self.bins,
                "min_eV": self.min_eV,
                "max_eV": self.max_eV,
                "log_bins": self.log_bins,
                "time_fractions": self.time_fractions,
            },
            "edges_eV": self.edges_eV,
            "records": self.records,
        }


class ParticleVelStats:
    def __init__(self, species_names: list[str], cfg: dict, output_path: Path):
        self.enabled = True
        self.species = str(cfg.get("monitor_species") or (species_names[0] if species_names else "ions"))
        self.interval = max(1, int(cfg.get("diag_period", 1)))
        self.output_path = output_path
        self._header_written = False
        self.axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
        self.center = self._resolve_center(cfg)

    def _resolve_center(self, cfg: dict) -> tuple[float, float, float]:
        x_min = float(cfg.get("x_min", 0.0))
        x_max = float(cfg.get("x_max", 0.0))
        y_min = float(cfg.get("y_min", 0.0))
        y_max = float(cfg.get("y_max", 0.0))
        z_min = float(cfg.get("z_min", 0.0))
        z_max = float(cfg.get("z_max", 0.0))
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        z_center = 0.5 * (z_min + z_max)
        center_cfg = cfg.get("m1_inject_center", "domain")
        if isinstance(center_cfg, (list, tuple)) and len(center_cfg) >= 2:
            try:
                if self.axis == "x":
                    y_center = float(center_cfg[0])
                    z_center = float(center_cfg[1])
                elif self.axis == "y":
                    x_center = float(center_cfg[0])
                    z_center = float(center_cfg[1])
                else:
                    x_center = float(center_cfg[0])
                    y_center = float(center_cfg[1])
            except Exception:
                pass
        return (float(x_center), float(y_center), float(z_center))

    def _write_header(self):
        if self._header_written:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.write_text(
                "step,time,uy_rms,uz_rms,uperp_rms,vy_mean,vy_rms,vy_max,"
                "vz_mean,vz_rms,vz_max,vperp_rms,m1_vperp_amp,m1_vperp_ratio,samples\n",
                encoding="utf-8",
            )
        self._header_written = True

    def maybe_record(self, step_idx: int, t_current: float) -> None:
        if not self.enabled or (step_idx % self.interval) != 0:
            return
        try:
            pc = particle_containers.ParticleContainerWrapper(self.species)
            x_tiles = pc.get_particle_x(copy_to_host=True)
            y_tiles = pc.get_particle_y(copy_to_host=True)
            z_tiles = pc.get_particle_z(copy_to_host=True)
            ux_tiles = pc.get_particle_ux(copy_to_host=True)
            uy_tiles = pc.get_particle_uy(copy_to_host=True)
            uz_tiles = pc.get_particle_uz(copy_to_host=True)
        except Exception as exc:
            print(f"Warning: particle vel stats failed for '{self.species}': {exc}")
            return

        c = picmi.constants.c
        sum_uy2 = 0.0
        sum_uz2 = 0.0
        sum_vy = 0.0
        sum_vy2 = 0.0
        sum_vz = 0.0
        sum_vz2 = 0.0
        sum_vperp = 0.0
        sum_vperp_cos = 0.0
        sum_vperp_sin = 0.0
        max_vy = None
        max_vz = None
        count = 0
        x_center, y_center, z_center = self.center
        for x_arr, y_arr, z_arr, ux_arr, uy_arr, uz_arr in zip(
            x_tiles, y_tiles, z_tiles, ux_tiles, uy_tiles, uz_tiles
        ):
            if ux_arr is None or len(ux_arr) == 0:
                continue
            ux_arr = np.asarray(ux_arr, dtype=float)
            uy_arr = np.asarray(uy_arr, dtype=float)
            uz_arr = np.asarray(uz_arr, dtype=float)
            x_arr = np.asarray(x_arr, dtype=float)
            y_arr = np.asarray(y_arr, dtype=float)
            z_arr = np.asarray(z_arr, dtype=float)
            u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
            gamma = np.sqrt(1.0 + u2 / (c * c))
            vy = uy_arr / gamma
            vz = uz_arr / gamma
            vx = ux_arr / gamma
            sum_uy2 += float(np.sum(uy_arr * uy_arr))
            sum_uz2 += float(np.sum(uz_arr * uz_arr))
            sum_vy += float(np.sum(vy))
            sum_vy2 += float(np.sum(vy * vy))
            sum_vz += float(np.sum(vz))
            sum_vz2 += float(np.sum(vz * vz))
            if self.axis == "x":
                d1 = y_arr - y_center
                d2 = z_arr - z_center
                vperp = np.sqrt(vy * vy + vz * vz)
            elif self.axis == "y":
                d1 = x_arr - x_center
                d2 = z_arr - z_center
                vperp = np.sqrt(vx * vx + vz * vz)
            else:
                d1 = x_arr - x_center
                d2 = y_arr - y_center
                vperp = np.sqrt(vx * vx + vy * vy)
            phi = np.arctan2(d2, d1)
            sum_vperp += float(np.sum(vperp))
            sum_vperp_cos += float(np.sum(vperp * np.cos(phi)))
            sum_vperp_sin += float(np.sum(vperp * np.sin(phi)))
            tile_max_vy = float(np.max(vy))
            tile_max_vz = float(np.max(vz))
            max_vy = tile_max_vy if max_vy is None else max(max_vy, tile_max_vy)
            max_vz = tile_max_vz if max_vz is None else max(max_vz, tile_max_vz)
            count += int(vy.size)

        if count <= 0:
            return
        vy_mean = sum_vy / count
        vy_rms = math.sqrt(sum_vy2 / count)
        vz_mean = sum_vz / count
        vz_rms = math.sqrt(sum_vz2 / count)
        uy_rms = math.sqrt(sum_uy2 / count)
        uz_rms = math.sqrt(sum_uz2 / count)
        uperp_rms = math.sqrt(uy_rms * uy_rms + uz_rms * uz_rms)
        vperp_rms = math.sqrt(vy_rms * vy_rms + vz_rms * vz_rms)
        m1_vperp_amp = math.sqrt(sum_vperp_cos * sum_vperp_cos + sum_vperp_sin * sum_vperp_sin)
        m1_vperp_ratio = m1_vperp_amp / (sum_vperp + 1.0e-30)
        if max_vy is None:
            max_vy = 0.0
        if max_vz is None:
            max_vz = 0.0

        self._write_header()
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{int(step_idx)},{t_current:.6e},{uy_rms:.6e},{uz_rms:.6e},{uperp_rms:.6e},"
                f"{vy_mean:.6e},{vy_rms:.6e},{max_vy:.6e},{vz_mean:.6e},{vz_rms:.6e},"
                f"{max_vz:.6e},{vperp_rms:.6e},{m1_vperp_amp:.6e},{m1_vperp_ratio:.6e},{count}\n"
            )


class U2StatsDiag:
    def __init__(self, species_names: list[str], cfg: dict, diag_dir: Path | None):
        self.enabled = bool(cfg.get("enable_u2_stats_diag", False)) and diag_dir is not None
        self.species = str(cfg.get("u2_species") or (species_names[0] if species_names else "ions"))
        interval = cfg.get("u2_interval")
        if interval is None:
            interval = cfg.get("diag_period", 1)
        self.interval = max(1, int(interval))
        self.diag_name = str(cfg.get("u2_diag_name", "U2")).strip() or "U2"
        self.output_path = None
        if diag_dir is not None:
            self.output_path = Path(diag_dir) / "reducedfiles" / f"{self.diag_name}.txt"
        self._header_written = False
        self.hist_enabled = bool(cfg.get("enable_u2_hist_custom", False)) and diag_dir is not None
        hist_interval = cfg.get("u2_hist_interval")
        if hist_interval is None:
            hist_interval = interval
        self.hist_interval = max(1, int(hist_interval))
        self.hist_bin_number = int(cfg.get("u2_hist_bin_number") or 256)
        self.hist_bin_min = float(cfg.get("u2_hist_bin_min") or 0.0)
        self.hist_bin_max = float(cfg.get("u2_hist_bin_max") or 1.0e14)
        self.hist_diag_name = str(cfg.get("u2_hist_diag_name") or "U2Hist").strip() or "U2Hist"
        self.hist_output_path = None
        if diag_dir is not None:
            self.hist_output_path = Path(diag_dir) / "reducedfiles" / f"{self.hist_diag_name}.txt"
        self._hist_header_written = False

    def _write_header(self) -> None:
        if self._header_written or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.write_text(
                "step,time,u2_mean,u2_max,sum_w,count\n",
                encoding="utf-8",
            )
        self._header_written = True

    def _write_hist_header(self) -> None:
        if self._hist_header_written or self.hist_output_path is None:
            return
        self.hist_output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.hist_output_path.exists():
            self.hist_output_path.write_text(
                (
                    f"# u2_hist_custom\n"
                    f"# bin_min={self.hist_bin_min} bin_max={self.hist_bin_max} bin_number={self.hist_bin_number}\n"
                ),
                encoding="utf-8",
            )
        self._hist_header_written = True

    def maybe_record(self, step_idx: int, t_current: float) -> None:
        if not self.enabled or (step_idx % self.interval) != 0:
            return
        try:
            pc = particle_containers.ParticleContainerWrapper(self.species)
            ux_tiles = pc.get_particle_ux(copy_to_host=True)
            uy_tiles = pc.get_particle_uy(copy_to_host=True)
            uz_tiles = pc.get_particle_uz(copy_to_host=True)
            try:
                w_tiles = pc.get_particle_weight(copy_to_host=True)
            except Exception:
                w_tiles = None
        except Exception as exc:
            print(f"Warning: u2 stats failed for '{self.species}': {exc}")
            return

        if w_tiles is None:
            w_tiles = [None] * len(ux_tiles)

        sum_w = 0.0
        sum_u2_w = 0.0
        max_u2 = None
        count = 0
        hist_counts = None
        bin_width = None
        if self.hist_enabled and (step_idx % self.hist_interval) == 0:
            hist_counts = np.zeros(self.hist_bin_number, dtype=float)
            bin_width = (self.hist_bin_max - self.hist_bin_min) / float(self.hist_bin_number)
            if bin_width <= 0.0:
                bin_width = None
        for ux_arr, uy_arr, uz_arr, w_arr in zip(ux_tiles, uy_tiles, uz_tiles, w_tiles):
            if ux_arr is None or len(ux_arr) == 0:
                continue
            ux_arr = np.asarray(ux_arr, dtype=float)
            uy_arr = np.asarray(uy_arr, dtype=float)
            uz_arr = np.asarray(uz_arr, dtype=float)
            u2 = ux_arr * ux_arr + uy_arr * uy_arr + uz_arr * uz_arr
            if w_arr is not None:
                w_arr = np.asarray(w_arr, dtype=float)
                if w_arr.size == u2.size:
                    sum_w += float(np.sum(w_arr))
                    sum_u2_w += float(np.sum(u2 * w_arr))
                    if hist_counts is not None and bin_width:
                        idx = np.floor((u2 - self.hist_bin_min) / bin_width).astype(int)
                        idx = np.clip(idx, 0, self.hist_bin_number - 1)
                        np.add.at(hist_counts, idx, w_arr)
                else:
                    sum_u2_w += float(np.sum(u2))
                    if hist_counts is not None and bin_width:
                        idx = np.floor((u2 - self.hist_bin_min) / bin_width).astype(int)
                        idx = np.clip(idx, 0, self.hist_bin_number - 1)
                        np.add.at(hist_counts, idx, 1.0)
            else:
                sum_u2_w += float(np.sum(u2))
                if hist_counts is not None and bin_width:
                    idx = np.floor((u2 - self.hist_bin_min) / bin_width).astype(int)
                    idx = np.clip(idx, 0, self.hist_bin_number - 1)
                    np.add.at(hist_counts, idx, 1.0)
            tile_max = float(np.max(u2)) if u2.size else None
            if tile_max is not None:
                max_u2 = tile_max if max_u2 is None else max(max_u2, tile_max)
            count += int(u2.size)

        if count <= 0:
            return
        if sum_w > 0.0:
            u2_mean = sum_u2_w / sum_w
        else:
            u2_mean = sum_u2_w / count
        if max_u2 is None:
            max_u2 = 0.0

        self._write_header()
        if self.output_path is None:
            return
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{int(step_idx)},{t_current:.6e},{u2_mean:.6e},{max_u2:.6e},{sum_w:.6e},{count}\n"
            )

        if hist_counts is not None and self.hist_output_path is not None:
            self._write_hist_header()
            with self.hist_output_path.open("a", encoding="utf-8") as handle:
                counts_str = " ".join(f"{val:.6e}" for val in hist_counts.tolist())
                handle.write(f"{int(step_idx)} {t_current:.6e} {counts_str}\n")


class CoilFluxDiag:
    def __init__(self, cfg: dict, diag_dir: Path | None):
        self.enabled = bool(cfg.get("enable_coil_diag", False)) and diag_dir is not None
        interval = cfg.get("coil_diag_interval")
        if interval is None:
            interval = cfg.get("diag_period", 1)
        self.interval = max(1, int(interval))
        self.axis = str(cfg.get("coil_axis", "x")).strip().lower()
        self.turns = int(cfg.get("coil_turns", 1))
        self.diag_name = str(cfg.get("coil_diag_name", "COIL")).strip() or "COIL"
        self.output_path = None
        if diag_dir is not None:
            self.output_path = Path(diag_dir) / "reducedfiles" / f"{self.diag_name}.txt"
        self._header_written = False
        self._field_error = False
        self._wrapper = None

        x_min = float(cfg.get("x_min", 0.0))
        x_max = float(cfg.get("x_max", 0.0))
        y_min = float(cfg.get("y_min", 0.0))
        y_max = float(cfg.get("y_max", 0.0))
        z_min = float(cfg.get("z_min", 0.0))
        z_max = float(cfg.get("z_max", 0.0))
        nx = int(cfg.get("nx", 0))
        ny = int(cfg.get("ny", 0))
        nz = int(cfg.get("nz", 0))
        self._spacing = (
            (x_max - x_min) / nx if nx > 0 else 0.0,
            (y_max - y_min) / ny if ny > 0 else 0.0,
            (z_max - z_min) / nz if nz > 0 else 0.0,
        )
        self._mins = (x_min, y_min, z_min)
        self._maxs = (x_max, y_max, z_max)

        self._center = self._resolve_center(cfg)
        self._rmax = self._resolve_rmax(cfg)
        self._plane_pos = self._resolve_plane_pos(cfg)
        self._mask = None
        self._dA = None
        self._plane_index = None
        if self.enabled:
            self._prepare_mask()

    def _resolve_center(self, cfg: dict) -> tuple[float, float, float]:
        x_min, y_min, z_min = self._mins
        x_max, y_max, z_max = self._maxs
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        z_center = 0.5 * (z_min + z_max)
        center_cfg = cfg.get("coil_center")
        if isinstance(center_cfg, (list, tuple)):
            try:
                if len(center_cfg) == 3:
                    x_center, y_center, z_center = (
                        float(center_cfg[0]),
                        float(center_cfg[1]),
                        float(center_cfg[2]),
                    )
                elif len(center_cfg) >= 2:
                    if self.axis == "x":
                        y_center, z_center = float(center_cfg[0]), float(center_cfg[1])
                    elif self.axis == "y":
                        x_center, z_center = float(center_cfg[0]), float(center_cfg[1])
                    else:
                        x_center, y_center = float(center_cfg[0]), float(center_cfg[1])
            except Exception:
                pass
        return (float(x_center), float(y_center), float(z_center))

    def _resolve_rmax(self, cfg: dict) -> float:
        rmax = cfg.get("coil_rmax")
        if rmax is not None:
            try:
                return float(rmax)
            except (TypeError, ValueError):
                return 0.0
        x_min, y_min, z_min = self._mins
        x_max, y_max, z_max = self._maxs
        if self.axis == "x":
            span = min(abs(y_max - y_min), abs(z_max - z_min))
        elif self.axis == "y":
            span = min(abs(x_max - x_min), abs(z_max - z_min))
        else:
            span = min(abs(x_max - x_min), abs(y_max - y_min))
        return 0.5 * span

    def _resolve_plane_pos(self, cfg: dict) -> float:
        plane_pos = cfg.get("coil_plane_pos")
        if plane_pos is not None:
            try:
                return float(plane_pos)
            except (TypeError, ValueError):
                pass
        x_min, y_min, z_min = self._mins
        x_max, y_max, z_max = self._maxs
        if self.axis == "x":
            return 0.5 * (x_min + x_max)
        if self.axis == "y":
            return 0.5 * (y_min + y_max)
        return 0.5 * (z_min + z_max)

    def _prepare_mask(self):
        dx, dy, dz = self._spacing
        x_min, y_min, z_min = self._mins
        nx = int(round((self._maxs[0] - x_min) / dx)) if dx > 0 else 0
        ny = int(round((self._maxs[1] - y_min) / dy)) if dy > 0 else 0
        nz = int(round((self._maxs[2] - z_min) / dz)) if dz > 0 else 0
        if self.axis == "x":
            y_centers = y_min + (np.arange(ny) + 0.5) * dy
            z_centers = z_min + (np.arange(nz) + 0.5) * dz
            yy, zz = np.meshgrid(y_centers, z_centers, indexing="ij")
            self._dA = dy * dz
            y0, z0 = self._center[1], self._center[2]
            self._mask = (yy - y0) ** 2 + (zz - z0) ** 2 <= self._rmax**2
            plane_index = int(round((self._plane_pos - x_min) / dx)) if dx > 0 else 0
            self._plane_index = max(0, min(nx, plane_index))
        elif self.axis == "y":
            x_centers = x_min + (np.arange(nx) + 0.5) * dx
            z_centers = z_min + (np.arange(nz) + 0.5) * dz
            xx, zz = np.meshgrid(x_centers, z_centers, indexing="ij")
            self._dA = dx * dz
            x0, z0 = self._center[0], self._center[2]
            self._mask = (xx - x0) ** 2 + (zz - z0) ** 2 <= self._rmax**2
            plane_index = int(round((self._plane_pos - y_min) / dy)) if dy > 0 else 0
            self._plane_index = max(0, min(ny, plane_index))
        else:
            x_centers = x_min + (np.arange(nx) + 0.5) * dx
            y_centers = y_min + (np.arange(ny) + 0.5) * dy
            xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
            self._dA = dx * dy
            x0, y0 = self._center[0], self._center[1]
            self._mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= self._rmax**2
            plane_index = int(round((self._plane_pos - z_min) / dz)) if dz > 0 else 0
            self._plane_index = max(0, min(nz, plane_index))

    def _write_header(self):
        if self._header_written or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.write_text("step,time,phi,area,bn_avg\n", encoding="utf-8")
        self._header_written = True

    def _get_wrapper(self):
        if self._wrapper is not None:
            return self._wrapper
        try:
            from pywarpx import fields
        except Exception as exc:
            if not self._field_error:
                print(f"[coil_diag] pywarpx.fields unavailable: {exc}")
                self._field_error = True
            return None
        candidates = {
            "x": ["BxFPWrapper", "BxFPExternalWrapper", "BxWrapper"],
            "y": ["ByFPWrapper", "ByFPExternalWrapper", "ByWrapper"],
            "z": ["BzFPWrapper", "BzFPExternalWrapper", "BzWrapper"],
        }.get(self.axis, [])
        for name in candidates:
            if not hasattr(fields, name):
                continue
            try:
                wrapper = getattr(fields, name)()
                _ = wrapper[:]
                self._wrapper = wrapper
                return wrapper
            except Exception:
                continue
        if not self._field_error:
            print(f"[coil_diag] unable to access B field wrapper for axis={self.axis}")
            self._field_error = True
        return None

    def maybe_record(self, step_idx: int, t_current: float) -> None:
        if not self.enabled or self.output_path is None:
            return
        if (step_idx % self.interval) != 0:
            return
        wrapper = self._get_wrapper()
        if wrapper is None or self._mask is None or self._dA is None or self._plane_index is None:
            return
        try:
            data = np.asarray(wrapper[:])
        except Exception as exc:
            if not self._field_error:
                print(f"[coil_diag] field access failed: {exc}")
                self._field_error = True
            return
        if self.axis == "x":
            if data.ndim != 3:
                return
            idx = min(self._plane_index, data.shape[0] - 1)
            plane = data[idx, :, :]
        elif self.axis == "y":
            if data.ndim != 3:
                return
            idx = min(self._plane_index, data.shape[1] - 1)
            plane = data[:, idx, :]
        else:
            if data.ndim != 3:
                return
            idx = min(self._plane_index, data.shape[2] - 1)
            plane = data[:, :, idx]
        if plane.shape != self._mask.shape:
            try:
                plane = plane[: self._mask.shape[0], : self._mask.shape[1]]
            except Exception:
                return
        mask = self._mask
        area = float(np.sum(mask)) * float(self._dA)
        if area <= 0.0:
            return
        phi = float(np.sum(plane[mask]) * float(self._dA))
        bn_avg = phi / area
        self._write_header()
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{int(step_idx)},{t_current:.6e},{phi:.6e},{area:.6e},{bn_avg:.6e}\n"
            )

    def as_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "interval": int(self.interval),
            "axis": self.axis,
            "center": [float(v) for v in self._center],
            "rmax": float(self._rmax),
            "plane_pos": float(self._plane_pos),
            "turns": int(self.turns),
            "path": str(self.output_path) if self.output_path is not None else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="WarpX 3D tilt smoke driver.")
    parser.add_argument("--from-json", required=True, help="Path to JSON config.")
    parser.add_argument("--metadata-dir", default="outputs/warpx", help="Directory for metadata output.")
    parser.add_argument("--diag-dir", default=None, help="Directory for field diagnostics output.")
    parser.add_argument("--run-tag", default="tilt_smoke_3d", help="Run tag for metadata.")
    parser.add_argument(
        "--metadata-heartbeat-s",
        type=float,
        default=None,
        help="Seconds between metadata heartbeats (override config).",
    )
    parser.add_argument(
        "--metadata-heartbeat-steps",
        type=int,
        default=None,
        help="Steps between metadata heartbeats (override config).",
    )
    parser.add_argument(
        "--init-mode",
        choices=["blob", "opmd", "opmd_double_seed"],
        help="Override init_mode from config.",
    )
    parser.add_argument("--opmd-fluid", help="Override opmd_fluid_path from config.")
    parser.add_argument("--opmd-b", help="Override opmd_b_path from config.")
    args = parser.parse_args()

    config_path = Path(args.from_json)
    cfg = load_config(config_path)
    cfg.setdefault("electron_energy", {"enabled": False})
    cfg.setdefault("checkpoint", {"enabled": False})

    repo_root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(cfg["seed"])

    if args.init_mode:
        cfg["init_mode"] = args.init_mode
    if args.opmd_fluid:
        cfg["opmd_fluid_path"] = args.opmd_fluid
    if args.opmd_b:
        cfg["opmd_b_path"] = args.opmd_b
    if args.metadata_heartbeat_s is not None:
        cfg["metadata_heartbeat_s"] = args.metadata_heartbeat_s
    if args.metadata_heartbeat_steps is not None:
        cfg["metadata_heartbeat_steps"] = args.metadata_heartbeat_steps

    init_mode = str(cfg.get("init_mode", "blob")).lower()
    opmd_fluid = cfg.get("opmd_fluid_path")
    opmd_b = cfg.get("opmd_b_path")
    opmd_fluid_data = None
    opmd_b_data = None
    dynamic_drift_enabled = False
    dynamic_drift_unit = None
    dynamic_drift_u = None
    dynamic_drift_mag = 0.0
    dynamic_drift_max_beta = None
    dynamic_drift_species = None
    dynamic_group_ids = None
    group_ids_meta = None

    if init_mode in ("opmd", "opmd_double_seed"):
        if not opmd_fluid:
            raise SystemExit(
                "init_mode=opmd/opmd_double_seed requires opmd_fluid_path (or --opmd-fluid)."
            )
        opmd_fluid_data = load_openpmd_cartesian(Path(opmd_fluid), "fluid", ["rho", "vx", "vy", "vz", "Ti", "Te"])
        vel_unit = None
        if opmd_fluid_data is not None:
            vel_unit = (opmd_fluid_data.get("unitSI") or {}).get("vx")
        if vel_unit is not None:
            cfg["opmd_velocity_unitSI"] = float(vel_unit)
        if opmd_b:
            opmd_b_data = load_openpmd_cartesian(Path(opmd_b), "B", ["x", "y", "z"])
        nx, ny, nz = opmd_fluid_data["rho"].shape
        spacing = opmd_fluid_data["spacing"]
        offset = opmd_fluid_data["offset"]
        x_min = float(offset[0])
        y_min = float(offset[1])
        z_min = float(offset[2])
        x_max = x_min + float(spacing[0]) * nx
        y_max = y_min + float(spacing[1]) * ny
        z_max = z_min + float(spacing[2]) * nz
        cfg.update(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
            }
        )
        if bool(cfg.get("m1_inject_dry_run", False)):
            rho = opmd_fluid_data.get("rho")
            vx = opmd_fluid_data.get("vx")
            vy = opmd_fluid_data.get("vy")
            vz = opmd_fluid_data.get("vz")
            m1_rho_meta = None
            if rho is not None and spacing is not None and offset is not None:
                _rho_mod, m1_rho_meta = apply_m1_rho_cos_modulation(
                    rho, spacing, offset, cfg
                )
            m1_vel_meta = None
            if spacing is not None and offset is not None:
                _vx, _vy, _vz, m1_vel_meta = apply_m1_vel_kick_modulation(
                    vx, vy, vz, spacing, offset, cfg
                )
            double_seed_meta = {}
            if m1_vel_meta is not None:
                double_seed_meta["m1_inject"] = m1_vel_meta
            elif m1_rho_meta is not None:
                double_seed_meta["m1_inject"] = m1_rho_meta
            git_head, git_dirty = git_info(repo_root)
            hybrid_enabled = bool((cfg.get("hybrid") or {}).get("enabled", False))
            meta_base = build_meta_base(
                cfg,
                git_head,
                git_dirty,
                hybrid_enabled,
                [],
                {},
                init_mode,
                opmd_fluid_data,
                opmd_b_data,
                None,
                None,
                double_seed_meta,
                {},
            )
            meta_base["run_status"] = "dry_run"
            meta_base["dry_run"] = True
            meta_base["timestamp"] = datetime.now(timezone.utc).isoformat()
            metadata_dir = Path(args.metadata_dir)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"warpx_run_{args.run_tag}.json"
            write_json_atomic(metadata_path, meta_base)
            enforce_vel_kick(cfg, m1_vel_meta)
            print(f"[dry_run] {metadata_path}")
            return
    else:
        nx = int(cfg["nx"])
        ny = int(cfg["ny"])
        nz = int(cfg["nz"])
        x_max = float(cfg["x_max"])
        y_max = float(cfg["y_max"])
        z_max = float(cfg["z_max"])
        x_min = cfg.get("x_min")
        y_min = cfg.get("y_min")
        z_min = cfg.get("z_min")
        x_min = float(x_min) if x_min is not None else -x_max
        y_min = float(y_min) if y_min is not None else -y_max
        z_min = float(z_min) if z_min is not None else -z_max
        cfg.update({"x_min": x_min, "y_min": y_min, "z_min": z_min})

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    c = picmi.constants.c
    dt = float(cfg["dt"])
    dt_cfl = float(cfg["cfl"]) * min(dx, dy, dz) / (c * math.sqrt(3.0))
    if dt > dt_cfl:
        print(f"[cfl] dt={dt:.2e} exceeds limit {dt_cfl:.2e}; clamping.")
        dt = dt_cfl
        cfg["dt"] = dt

    grid = picmi.Cartesian3DGrid(
        number_of_cells=[nx, ny, nz],
        lower_bound=[x_min, y_min, z_min],
        upper_bound=[x_max, y_max, z_max],
        lower_boundary_conditions=["periodic", "periodic", "periodic"],
        upper_boundary_conditions=["periodic", "periodic", "periodic"],
        lower_boundary_conditions_particles=["periodic", "periodic", "periodic"],
        upper_boundary_conditions_particles=["periodic", "periodic", "periodic"],
        warpx_max_grid_size=32,
        warpx_blocking_factor=8,
    )

    hybrid_cfg = cfg.get("hybrid") or {}
    hybrid_enabled = bool(hybrid_cfg.get("enabled", False))
    eta_profile_enabled = False
    eta_profile_low = None
    eta_profile_high = None
    eta_profile_switch = None
    eta_profile_base = None
    eta_profile_method = "disabled"
    eta_profile_low = None
    eta_profile_high = None
    eta_profile_switch = None
    eta_effective_inputs_lines: list[str] = []
    if hybrid_enabled:
        n0_val = float(hybrid_cfg.get("n0", 0.0))
        Te_eV = float(hybrid_cfg.get("Te_eV", 0.0))
        eta_raw = hybrid_cfg.get("eta", 0.0)
        eta = float(eta_raw) if isinstance(eta_raw, (int, float)) else str(eta_raw)
        eta_scale = float(hybrid_cfg.get("eta_scale", 1.0))
        eta_scale_low = hybrid_cfg.get("eta_scale_low")
        eta_scale_high = hybrid_cfg.get("eta_scale_high")
        eta_switch_step = hybrid_cfg.get("eta_switch_step")
        eta_scale_input = eta_scale
        if eta_scale_low is not None and eta_scale_high is not None and eta_switch_step is not None:
            try:
                eta_profile_low = float(eta_scale_low)
                eta_profile_high = float(eta_scale_high)
                eta_profile_switch = int(eta_switch_step)
                if eta_profile_low == eta_profile_high:
                    eta_profile_method = "constant"
                    eta_scale_input = float(eta_profile_low)
                    eta_effective_inputs_lines.append(f"hybrid.eta_scale = {eta_profile_low:.6g}")
                else:
                    eta_profile_method = "piecewise_time_parser"
                    eta_t_switch = float(eta_profile_switch) * float(dt)
                    eta_effective_inputs_lines.append(f"eta_low = {eta_profile_low:.6g}")
                    eta_effective_inputs_lines.append(f"eta_high = {eta_profile_high:.6g}")
                    eta_effective_inputs_lines.append(f"eta_t_switch = {eta_t_switch:.6g}")
                    eta_effective_inputs_lines.append(
                        "hybrid.eta_scale = \"if(t < eta_t_switch, eta_low, eta_high)\""
                    )
                    eta_scale_input = (
                        f"if(t < {eta_t_switch:.6g}, {eta_profile_low:.6g}, {eta_profile_high:.6g})"
                    )
            except (TypeError, ValueError):
                eta_profile_method = "disabled"
                eta_scale_input = eta_scale
        eta_h_raw = hybrid_cfg.get("eta_h", 0.0)
        eta_h = float(eta_h_raw) if isinstance(eta_h_raw, (int, float)) else str(eta_h_raw)
        eta_h_scale = float(hybrid_cfg.get("eta_h_scale", 1.0))
        etaJ2_diag_enabled = bool(hybrid_cfg.get("etaJ2_diag_enabled", False))
        etaJ2_diag_stride = int(hybrid_cfg.get("etaJ2_diag_stride", 1))
        substeps = int(hybrid_cfg.get("substeps", 1))
        nfloor_scale = float(hybrid_cfg.get("nfloor_scale", 0.0))
        gamma = float(hybrid_cfg.get("gamma", 5.0 / 3.0))
        n_floor = nfloor_scale * n0_val if n0_val else None
        solver = picmi.HybridPICSolver(
            grid=grid,
            Te=Te_eV,
            n0=n0_val,
            plasma_resistivity=eta,
            plasma_resistivity_scale=eta_scale_input,
            plasma_hyper_resistivity=eta_h,
            plasma_hyper_resistivity_scale=eta_h_scale,
            etaJ2_diag_enabled=etaJ2_diag_enabled,
            etaJ2_diag_stride=etaJ2_diag_stride,
            substeps=substeps,
            n_floor=n_floor,
            gamma=gamma,
        )
    else:
        solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=float(cfg["cfl"]))
    sim_kwargs = {}
    reduced_diags_path = None
    if args.metadata_dir:
        reduced_diags_path = str(Path(args.metadata_dir) / "diags" / "reducedfiles")
        if not reduced_diags_path.endswith(os.sep):
            reduced_diags_path += os.sep
        sim_kwargs["warpx_reduced_diags_path"] = reduced_diags_path
    amr_restart = cfg.get("warpx_amr_restart")
    if amr_restart:
        sim_kwargs["warpx_amr_restart"] = str(amr_restart)
    sim = picmi.Simulation(
        solver=solver,
        max_steps=int(cfg["max_steps"]),
        time_step_size=dt,
        verbose=1,
        **sim_kwargs,
    )

    if bool(cfg.get("applied_field_enabled", False)):
        bz_val = float(cfg.get("applied_Bz_T", 0.0))
        bz_expr = str(cfg.get("applied_Bz_expr", "")).strip()
        if bz_expr or bz_val != 0.0:
            if bz_expr:
                bz_expr_use = bz_expr
            else:
                bz_expr_use = str(bz_val)
            applied_field = picmi.AnalyticInitialField(
                Bx_expression="0.0",
                By_expression="0.0",
                Bz_expression=bz_expr_use,
            )
            sim.add_applied_field(applied_field)
            if bz_expr:
                print(f"[config] applied initial Bz enabled (expr='{bz_expr_use}').")
            else:
                print(f"[config] applied initial Bz enabled (Bz={bz_val}).")
        else:
            print("[config] applied_field_enabled true but applied_Bz_T=0; skipping.")

    def build_blob(
        center_vec,
        sigma_val,
        drift_beta,
        ppc_val,
        weight_scale,
        particle_count=None,
    ):
        if particle_count is None:
            particle_count = int(ppc_val * nx * ny * nz)
        if particle_count <= 0:
            return None
        x0, y0, z0 = float(center_vec[0]), float(center_vec[1]), float(center_vec[2])
        xb = sample_bounded_gaussian(rng, particle_count, x0, sigma_val, x_min, x_max)
        yb = sample_bounded_gaussian(rng, particle_count, y0, sigma_val, y_min, y_max)
        zb = sample_bounded_gaussian(rng, particle_count, z0, sigma_val, z_min, z_max)
        bx, by, bz = float(drift_beta[0]), float(drift_beta[1]), float(drift_beta[2])
        beta2 = bx * bx + by * by + bz * bz
        beta2 = min(beta2, 0.95)
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        # PICMI expects ux,uy,uz in units of u=gamma*v (m/s), not beta.
        u_scale = gamma * c
        ux_b = np.full(particle_count, u_scale * bx)
        uy_b = np.full(particle_count, u_scale * by)
        uz_b = np.full(particle_count, u_scale * bz)
        w_b = np.full(particle_count, float(cfg["particle_weight"]) * weight_scale)
        return xb, yb, zb, ux_b, uy_b, uz_b, w_b

    particle_weight_hist = None
    double_seed_meta = None
    if init_mode in ("opmd", "opmd_double_seed"):
        x, y, z, ux, uy, uz, weight, m1_rho_meta, vel_scale_meta, m1_vel_meta = sample_particles_from_cartesian(
            opmd_fluid_data, cfg, rng
        )
        if init_mode == "opmd_double_seed":
            shift = np.array(cfg.get("opmd_double_seed_shift", [0.0, 0.0, 0.0]), dtype=float)
            symmetric = bool(cfg.get("opmd_double_seed_symmetric", True))
            max_beta = float(cfg.get("opmd_max_beta", 0.95))
            drift_is_beta = bool(cfg.get("opmd_double_seed_drift_is_beta", False))
            common_drift = np.array(cfg.get("opmd_double_seed_common_drift", [0.0, 0.0, 0.0]), dtype=float)
            common_drift_is_beta = bool(cfg.get("opmd_double_seed_common_drift_is_beta", False))
            if common_drift_is_beta:
                common_drift = common_drift * picmi.constants.c
            if symmetric:
                shift_a = -0.5 * shift
                shift_b = 0.5 * shift
            else:
                shift_a = np.zeros(3)
                shift_b = shift

            x_a = x + shift_a[0]
            y_a = y + shift_a[1]
            z_a = z + shift_a[2]
            x_b = x + shift_b[0]
            y_b = y + shift_b[1]
            z_b = z + shift_b[2]
            com_a = np.array([float(np.mean(x_a)), float(np.mean(y_a)), float(np.mean(z_a))])
            com_b = np.array([float(np.mean(x_b)), float(np.mean(y_b)), float(np.mean(z_b))])
            drift, drift_meta = resolve_drift_vector(cfg, shift, centers=(com_a, com_b))
            if drift_is_beta:
                drift = drift * picmi.constants.c
            drift_meta["drift_is_beta"] = drift_is_beta
            requested_beta = None
            if drift_is_beta:
                try:
                    requested_beta = float(drift_meta.get("drift_mag"))
                except (TypeError, ValueError):
                    requested_beta = None
            drift_meta["requested_beta"] = requested_beta
            drift_meta["beta_cap"] = max_beta
            drift_meta["drift"] = drift.tolist()
            drift_meta["drift_applied_a"] = (-drift).tolist()
            drift_meta["drift_applied_b"] = drift.tolist()
            drift_meta["mask_source"] = "opmd_double_seed_groups"
            dot_check = None
            sep_vec = drift_meta.get("com_sep_vec")
            if sep_vec is not None:
                try:
                    sep_vec = np.array(sep_vec, dtype=float)
                    v_rel = np.array(drift_meta["drift_applied_b"]) - np.array(
                        drift_meta["drift_applied_a"]
                    )
                    dot_check = float(np.dot(sep_vec, v_rel))
                except Exception:
                    dot_check = None
            drift_meta["dot_check"] = dot_check
            try:
                drift_meta_out = dict(drift_meta)
                drift_meta_out["run_tag"] = args.run_tag
                drift_meta_path = Path(args.metadata_dir) / "drift_meta.json"
                drift_meta_path.parent.mkdir(parents=True, exist_ok=True)
                drift_meta_path.write_text(
                    json.dumps(drift_meta_out, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"[drift_meta] {drift_meta_path}")
            except Exception as exc:
                print(f"Warning: failed to write drift meta: {exc}")
            ux_d, uy_d, uz_d = velocities_to_momenta(
                np.array([drift[0]]), np.array([drift[1]]), np.array([drift[2]]), max_beta
            )
            ux_d = float(ux_d[0])
            uy_d = float(uy_d[0])
            uz_d = float(uz_d[0])
            dynamic_drift_enabled = bool(cfg.get("opmd_double_seed_drift_dynamic", False))
            dynamic_drift_mag = float(np.linalg.norm(drift))
            if dynamic_drift_mag > 0.0:
                dynamic_drift_unit = drift / dynamic_drift_mag
                dynamic_drift_u = np.array([ux_d, uy_d, uz_d], dtype=float)
            dynamic_drift_max_beta = max_beta
            dynamic_drift_species = cfg.get("monitor_species") or "ions"
            ux_c, uy_c, uz_c = velocities_to_momenta(
                np.array([common_drift[0]]),
                np.array([common_drift[1]]),
                np.array([common_drift[2]]),
                max_beta,
            )
            ux_c = float(ux_c[0])
            uy_c = float(uy_c[0])
            uz_c = float(uz_c[0])

            tilt_seed_meta = None
            tilt_mode = str(cfg.get("tilt_seed_mode", "none")).strip().lower()
            tilt_vkick_frac = float(cfg.get("tilt_seed_vkick_frac", 0.0))
            tilt_vkick_abs = float(cfg.get("tilt_seed_vkick_abs", 0.0))
            tilt_vkick_is_beta = bool(cfg.get("tilt_seed_vkick_is_beta", False))
            tilt_func = str(cfg.get("tilt_seed_function", "sin(pi*(z-zmin)/Lz)"))
            tilt_sign = str(cfg.get("tilt_seed_sign", "A:+,B:-"))
            if tilt_mode in ("pos_tilt", "position_tilt", "pos_tilt_y"):
                amp = float(cfg.get("tilt_seed_y_offset_amp", 0.0))
                profile = str(cfg.get("tilt_seed_y_offset_profile", "sin")).strip().lower()
                z0_raw = cfg.get("tilt_seed_y_offset_z0")
                z0 = 0.5 * (z_min + z_max) if z0_raw is None else float(z0_raw)
                k_raw = cfg.get("tilt_seed_y_offset_k")
                k_val = float(k_raw) if k_raw is not None else None
                halfwidth_raw = cfg.get("tilt_seed_y_offset_z_halfwidth")
                halfwidth = float(halfwidth_raw) if halfwidth_raw is not None else None
                lz = None
                reason = None
                offset_a = None
                offset_b = None
                if amp == 0.0:
                    reason = "y_offset_amp_zero"
                elif profile == "sin":
                    if k_val is None:
                        lz = float(0.5 * (z_max - z_min))
                        if lz <= 0.0:
                            reason = "invalid_domain_length"
                        else:
                            k_val = math.pi / lz
                    elif k_val <= 0.0:
                        reason = "y_offset_k_nonpositive"
                    else:
                        lz = math.pi / k_val
                    if reason is None:
                        offset_a = amp * np.sin(k_val * (z_a - z0))
                        offset_b = -amp * np.sin(k_val * (z_b - z0))
                elif profile == "tanh":
                    if halfwidth is None:
                        halfwidth = float(0.25 * (z_max - z_min))
                    if halfwidth <= 0.0:
                        reason = "y_offset_halfwidth_nonpositive"
                    else:
                        offset_a = amp * np.tanh((z_a - z0) / halfwidth)
                        offset_b = -amp * np.tanh((z_b - z0) / halfwidth)
                elif profile == "linear":
                    if halfwidth is None:
                        halfwidth = float(0.25 * (z_max - z_min))
                    if halfwidth <= 0.0:
                        reason = "y_offset_halfwidth_nonpositive"
                    else:
                        phase_a = np.clip((z_a - z0) / halfwidth, -1.0, 1.0)
                        phase_b = np.clip((z_b - z0) / halfwidth, -1.0, 1.0)
                        offset_a = amp * phase_a
                        offset_b = -amp * phase_b
                else:
                    reason = "y_offset_profile_unknown"

                tilt_params = {"z0": z0, "k": k_val, "z_halfwidth": halfwidth}
                if lz is not None:
                    tilt_params["lz"] = lz
                tilt_seed_meta = {
                    "applied": False,
                    "mode": tilt_mode,
                    "reason": reason or "y_offset_not_applied",
                    "y_offset_amp": amp,
                    "y_offset_profile": profile,
                    "y_offset_params": tilt_params,
                    "vkick_frac": tilt_vkick_frac,
                    "vkick_abs": tilt_vkick_abs,
                    "vkick_is_beta": tilt_vkick_is_beta,
                    "bins_z": int(cfg.get("tilt_seed_bins_z", 16)),
                }
                if offset_a is not None and offset_b is not None:
                    y_a = y_a + offset_a
                    y_b = y_b + offset_b
                    tilt_seed_meta["applied"] = True
                    tilt_seed_meta.pop("reason", None)

            m1_inject_meta = None
            m1_mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
            m1_eps = float(cfg.get("m1_inject_eps", 0.0))
            m1_axis = str(cfg.get("m1_inject_axis", "y")).strip().lower()
            m1_r_ref_cfg = cfg.get("m1_inject_r_ref")
            if m1_mode in ("shift", "pos_shift", "global_shift"):
                reason = None
                if m1_eps == 0.0:
                    reason = "eps_zero"
                if m1_axis not in ("y", "z"):
                    reason = "axis_invalid"
                if reason is None:
                    if m1_r_ref_cfg is not None:
                        r_ref = float(m1_r_ref_cfg)
                    else:
                        r_ref = 0.5 * min((y_max - y_min), (z_max - z_min))
                    if r_ref <= 0.0:
                        reason = "r_ref_nonpositive"
                if reason is None:
                    delta = m1_eps * r_ref
                    if m1_axis == "y":
                        y_a = y_a + delta
                        y_b = y_b + delta
                    else:
                        z_a = z_a + delta
                        z_b = z_b + delta
                    m1_inject_meta = {
                        "applied": True,
                        "mode": m1_mode,
                        "axis": m1_axis,
                        "eps": m1_eps,
                        "r_ref": r_ref,
                        "delta": delta,
                    }
                else:
                    m1_inject_meta = {
                        "applied": False,
                        "mode": m1_mode,
                        "axis": m1_axis,
                        "eps": m1_eps,
                        "reason": reason,
                    }

            def _out_of_bounds_fraction(xv, yv, zv):
                mask = (
                    (xv < x_min)
                    | (xv > x_max)
                    | (yv < y_min)
                    | (yv > y_max)
                    | (zv < z_min)
                    | (zv > z_max)
                )
                return float(np.mean(mask))

            out_of_bounds_frac_a = _out_of_bounds_fraction(x_a, y_a, z_a)
            out_of_bounds_frac_b = _out_of_bounds_fraction(x_b, y_b, z_b)

            ux_a = ux - ux_d
            uy_a = uy - uy_d
            uz_a = uz - uz_d
            ux_b = ux + ux_d
            uy_b = uy + uy_d
            uz_b = uz + uz_d

            ux_a = ux_a + ux_c
            uy_a = uy_a + uy_c
            uz_a = uz_a + uz_c
            ux_b = ux_b + ux_c
            uy_b = uy_b + uy_c
            uz_b = uz_b + uz_c

            if tilt_mode == "z_shear_vy":
                lz = float(z_max - z_min)
                v_ref = float(np.linalg.norm(drift))
                if v_ref <= 0.0:
                    v_ref = float(np.linalg.norm(common_drift))
                if tilt_vkick_abs > 0.0:
                    v_kick = tilt_vkick_abs * (c if tilt_vkick_is_beta else 1.0)
                else:
                    v_kick = tilt_vkick_frac * v_ref
                if lz <= 0.0:
                    tilt_seed_meta = {
                        "applied": False,
                        "reason": "invalid_domain_length",
                        "mode": tilt_mode,
                        "vkick_abs": v_kick,
                    }
                elif v_kick <= 0.0:
                    tilt_seed_meta = {
                        "applied": False,
                        "reason": "vkick_abs_nonpositive",
                        "mode": tilt_mode,
                        "vkick_abs": v_kick,
                        "vkick_frac": tilt_vkick_frac,
                        "vkick_ref": v_ref,
                    }
                else:
                    phase_a = np.sin(np.pi * (z_a - z_min) / lz)
                    phase_b = np.sin(np.pi * (z_b - z_min) / lz)
                    beta_a = np.clip(v_kick * phase_a / c, -max_beta, max_beta)
                    beta_b = np.clip(-v_kick * phase_b / c, -max_beta, max_beta)
                    gamma_a = 1.0 / np.sqrt(1.0 - beta_a * beta_a)
                    gamma_b = 1.0 / np.sqrt(1.0 - beta_b * beta_b)
                    uy_a = uy_a + gamma_a * beta_a * c
                    uy_b = uy_b + gamma_b * beta_b * c
                    tilt_seed_meta = {
                        "applied": True,
                        "mode": tilt_mode,
                        "vkick_abs": v_kick,
                        "vkick_frac": tilt_vkick_frac,
                        "vkick_ref": v_ref,
                        "vkick_is_beta": tilt_vkick_is_beta,
                        "function": tilt_func,
                        "sign": tilt_sign,
                        "bins_z": int(cfg.get("tilt_seed_bins_z", 16)),
                    }

            x = np.concatenate([x_a, x_b])
            y = np.concatenate([y_a, y_b])
            z = np.concatenate([z_a, z_b])
            ux = np.concatenate([ux_a, ux_b])
            uy = np.concatenate([uy_a, uy_b])
            uz = np.concatenate([uz_a, uz_b])
            weight = np.concatenate([weight, weight])
            m1_particle_meta = None
            if m1_mode in ("particle_vel_kick", "particle_v_kick", "particle_kick"):
                m1_particle_meta = {
                    "applied": False,
                    "mode": m1_mode,
                    "eps": float(cfg.get("m1_inject_eps", 0.0)),
                    "axis": str(cfg.get("m1_inject_axis", "x")).strip().lower(),
                    "phase": float(cfg.get("m1_inject_phase", 0.0)),
                    "reason": "deferred_post_init",
                }
            double_seed_meta = {
                "shift": shift.tolist(),
                "drift": drift.tolist(),
                "symmetric": symmetric,
                "drift_is_beta": drift_is_beta,
                "drift_meta": drift_meta,
                "common_drift": common_drift.tolist(),
                "common_drift_is_beta": common_drift_is_beta,
                "copies": 2,
                "out_of_bounds_frac_a": out_of_bounds_frac_a,
                "out_of_bounds_frac_b": out_of_bounds_frac_b,
            }
            if tilt_seed_meta is not None:
                double_seed_meta["tilt_seed"] = tilt_seed_meta
            if m1_rho_meta is not None:
                double_seed_meta["m1_inject"] = m1_rho_meta
            if vel_scale_meta is not None:
                double_seed_meta.update(vel_scale_meta)
            if m1_vel_meta is not None:
                double_seed_meta["m1_inject"] = m1_vel_meta
            if m1_inject_meta is not None:
                double_seed_meta["m1_inject"] = m1_inject_meta
            if m1_particle_meta is not None:
                double_seed_meta["m1_inject"] = m1_particle_meta
            enforce_vel_kick(cfg, m1_vel_meta)
            enforce_particle_vel_kick(cfg, m1_particle_meta)
        n_particles = int(x.size)
        hist_bins = int(cfg.get("opmd_weight_hist_bins", 50))
        counts, bins = np.histogram(weight, bins=hist_bins)
        particle_weight_hist = {"bins": bins.tolist(), "counts": counts.tolist()}
    else:
        blobs_cfg = cfg.get("blobs")
        x_parts = []
        y_parts = []
        z_parts = []
        ux_parts = []
        uy_parts = []
        uz_parts = []
        w_parts = []
        if blobs_cfg:
            for blob in blobs_cfg:
                ppc_val = float(blob.get("ppc", cfg["ppc"]))
                weight_scale = float(blob.get("weight_scale", 1.0))
                center_vec = blob.get("center", cfg["blob_center"])
                sigma_val = float(blob.get("sigma", cfg["blob_sigma"]))
                drift_beta = blob.get("drift_beta", cfg["drift_beta"])
                particle_count = blob.get("particles")
                part = build_blob(
                    center_vec,
                    sigma_val,
                    drift_beta,
                    ppc_val,
                    weight_scale,
                    particle_count=particle_count,
                )
                if part is None:
                    continue
                xb, yb, zb, ux_b, uy_b, uz_b, w_b = part
                x_parts.append(xb)
                y_parts.append(yb)
                z_parts.append(zb)
                ux_parts.append(ux_b)
                uy_parts.append(uy_b)
                uz_parts.append(uz_b)
                w_parts.append(w_b)
        else:
            part = build_blob(
                cfg["blob_center"],
                float(cfg["blob_sigma"]),
                cfg["drift_beta"],
                float(cfg["ppc"]),
                1.0,
            )
            if part is not None:
                xb, yb, zb, ux_b, uy_b, uz_b, w_b = part
                x_parts.append(xb)
                y_parts.append(yb)
                z_parts.append(zb)
                ux_parts.append(ux_b)
                uy_parts.append(uy_b)
                uz_parts.append(uz_b)
                w_parts.append(w_b)

        background_ppc = int(cfg.get("background_ppc", 0))
        background_weight_scale = float(cfg.get("background_weight_scale", 0.0))
        n_bg = 0
        if background_ppc > 0 and background_weight_scale > 0.0:
            n_bg = int(background_ppc * nx * ny * nz)
            if n_bg > 0:
                x_bg = rng.uniform(x_min, x_max, size=n_bg)
                y_bg = rng.uniform(y_min, y_max, size=n_bg)
                z_bg = rng.uniform(z_min, z_max, size=n_bg)
                x_parts.append(x_bg)
                y_parts.append(y_bg)
                z_parts.append(z_bg)
                ux_parts.append(np.zeros(n_bg))
                uy_parts.append(np.zeros(n_bg))
                uz_parts.append(np.zeros(n_bg))
                w_parts.append(np.full(n_bg, float(cfg["particle_weight"]) * background_weight_scale))

        if not x_parts:
            raise SystemExit("No particles configured: check ppc/blobs/background settings.")

        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)
        z = np.concatenate(z_parts)
        ux = np.concatenate(ux_parts)
        uy = np.concatenate(uy_parts)
        uz = np.concatenate(uz_parts)
        weight = np.concatenate(w_parts)
        n_particles = int(x.size)

    ions_dist = picmi.ParticleListDistribution(
        x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, weight=weight
    )
    ions = picmi.Species(
        particle_type="proton",
        name="ions",
        warpx_do_not_deposit=False,
        initial_distribution=ions_dist,
    )
    sim.add_species(ions, layout=None)
    species_names = ["ions"]

    if hybrid_enabled and cfg.get("add_electrons", False):
        print("[config] hybrid enabled; ignoring add_electrons to avoid double counting.")

    if cfg.get("add_electrons", False) and not hybrid_enabled:
        electrons_dist = picmi.ParticleListDistribution(
            x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, weight=weight
        )
        electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            warpx_do_not_deposit=False,
            initial_distribution=electrons_dist,
        )
        sim.add_species(electrons, layout=None)
        species_names.append("electrons")

    diag_dir = Path(args.diag_dir) if args.diag_dir else None
    if bool(cfg.get("enable_field_diag", True)):
        diag_fields = cfg.get("diag_fields")
        if diag_fields is None:
            diag_fields = cfg.get("diag_data_list")
        if diag_fields is None:
            diag_fields = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho"]
        elif isinstance(diag_fields, str):
            diag_fields = [field.strip() for field in diag_fields.split(",") if field.strip()]
        elif isinstance(diag_fields, (list, tuple)):
            diag_fields = [str(field).strip() for field in diag_fields if str(field).strip()]
        else:
            diag_fields = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho"]
        if not diag_fields:
            diag_fields = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho"]
        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=grid,
            period=int(cfg["diag_period"]),
            data_list=diag_fields,
            write_dir=str(diag_dir) if diag_dir else None,
        )
        sim.add_diagnostic(field_diag)
    checkpoint_cfg = cfg.get("checkpoint") or {}
    if bool(checkpoint_cfg.get("enabled", False)):
        period = int(checkpoint_cfg.get("period") or 0)
        if period > 0:
            write_dir = checkpoint_cfg.get("write_dir")
            if write_dir:
                Path(write_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_diag = picmi.Checkpoint(
                period=period,
                write_dir=write_dir,
                name=checkpoint_cfg.get("name") or "chkpoint",
                warpx_file_prefix=checkpoint_cfg.get("file_prefix"),
                warpx_file_min_digits=checkpoint_cfg.get("file_min_digits"),
                warpx_verbose=checkpoint_cfg.get("verbose"),
            )
            sim.add_diagnostic(checkpoint_diag)

    if bool(cfg.get("enable_m1mom_diag", True)):
        try:
            diag_name = "M1MOM"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "MomModeM1"
            diag.intervals = int(cfg["diag_period"])
            diag.axis = "x"
        except Exception as exc:
            print(f"[config] unable to enable M1MOM reduced diag: {exc}")

    if bool(cfg.get("enable_m1rho_diag", True)):
        try:
            diag_name = "M1RHO"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "RhoModeM1"
            diag.intervals = int(cfg["diag_period"])
            diag.axis = "x"
        except Exception as exc:
            print(f"[config] unable to enable M1RHO reduced diag: {exc}")

    if bool(cfg.get("enable_energy_diag", False)):
        try:
            diag_name = str(cfg.get("energy_diag_name") or "ENERGY0D").strip() or "ENERGY0D"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "FieldEnergy"
            interval = cfg.get("energy_diag_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
        except Exception as exc:
            print(f"[config] unable to enable ENERGY0D reduced diag: {exc}")

    if bool(cfg.get("enable_particle_number_diag", False)):
        try:
            diag_name = str(cfg.get("particle_number_diag_name") or "PNUM").strip() or "PNUM"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "ParticleNumber"
            interval = cfg.get("particle_number_diag_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
        except Exception as exc:
            print(f"[config] unable to enable ParticleNumber reduced diag: {exc}")

    if bool(cfg.get("enable_rho_max_diag", False)):
        try:
            diag_name = str(cfg.get("rho_max_diag_name") or "RHOMAX").strip() or "RHOMAX"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "RhoMaximum"
            interval = cfg.get("rho_max_diag_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
        except Exception as exc:
            print(f"[config] unable to enable RhoMaximum reduced diag: {exc}")

    if bool(cfg.get("enable_particle_energy_diag", False)):
        try:
            diag_name = str(cfg.get("particle_energy_diag_name") or "PENERGY").strip() or "PENERGY"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "ParticleEnergy"
            interval = cfg.get("particle_energy_diag_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
        except Exception as exc:
            print(f"[config] unable to enable ParticleEnergy reduced diag: {exc}")

    if bool(cfg.get("enable_energy_hist_diag", False)):
        try:
            diag_name = str(cfg.get("energy_hist_diag_name") or "EHist").strip() or "EHist"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "ParticleHistogram"
            diag.species = str(cfg.get("energy_hist_species") or "ions")
            diag.set_or_replace_attr(
                "histogram_function(t,x,y,z,ux,uy,uz)",
                str(cfg.get("energy_hist_function") or "sqrt(1+ux*ux+uy*uy+uz*uz)-1"),
            )
            diag.bin_number = int(cfg.get("energy_hist_bin_number") or 128)
            diag.bin_min = float(cfg.get("energy_hist_bin_min") or 0.0)
            diag.bin_max = float(cfg.get("energy_hist_bin_max") or 20.0)
            interval = cfg.get("energy_hist_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
            normalization = cfg.get("energy_hist_normalization")
            if normalization:
                diag.normalization = str(normalization)
        except Exception as exc:
            print(f"[config] unable to enable Energy histogram reduced diag: {exc}")

    if bool(cfg.get("enable_u2_hist_diag", False)) and not bool(cfg.get("enable_u2_hist_custom", False)):
        try:
            diag_name = str(cfg.get("u2_hist_diag_name") or "U2Hist").strip() or "U2Hist"
            bucket = pywarpx.reduced_diagnostics
            diag = bucket._diagnostics_dict.get(diag_name)
            if diag is None:
                diag = pw_diag.Diagnostic(diag_name, _species_dict={})
                bucket._diagnostics_dict[diag_name] = diag
            diag.type = "ParticleHistogram"
            diag.species = str(cfg.get("u2_hist_species") or "ions")
            diag.set_or_replace_attr(
                "histogram_function(t,x,y,z,ux,uy,uz)",
                str(cfg.get("u2_hist_function") or "ux*ux+uy*uy+uz*uz"),
            )
            diag.bin_number = int(cfg.get("u2_hist_bin_number") or 256)
            diag.bin_min = float(cfg.get("u2_hist_bin_min") or 0.0)
            diag.bin_max = float(cfg.get("u2_hist_bin_max") or 1.0e14)
            interval = cfg.get("u2_hist_interval")
            if interval is None:
                interval = cfg.get("diag_period", 1)
            diag.intervals = int(interval)
            normalization = cfg.get("u2_hist_normalization")
            if normalization:
                diag.normalization = str(normalization)
        except Exception as exc:
            print(f"[config] unable to enable U2 histogram reduced diag: {exc}")

    collisions_meta = {"enabled": False}
    if bool(cfg.get("enable_collisions", False)):
        try:
            collision_name = "collide_bgstop"
            nu_scale = float(cfg.get("collision_nu_scale", 1.0))
            n0_val = None
            Te_val = None
            if hybrid_enabled:
                n0_val = float((cfg.get("hybrid") or {}).get("n0", 1.0e19))
                Te_val = float((cfg.get("hybrid") or {}).get("Te_eV", 10.0))
            if n0_val is None:
                n0_val = float(cfg.get("collision_background_density", 1.0e19))
            if Te_val is None:
                Te_val = float(cfg.get("collision_background_temperature", 10.0))
            bg_density = n0_val * nu_scale
            bg_temperature = Te_val

            pywarpx.collisions.collision_names = [collision_name]
            collision = pw_collisions.newcollision(collision_name)
            collision.type = "background_stopping"
            collision.species = ["ions"]
            collision.ndt = 1
            collision.background_type = "electrons"
            collision.background_density = float(bg_density)
            collision.background_temperature = float(bg_temperature)

            collisions_meta = {
                "enabled": True,
                "collision_name": collision_name,
                "collision_type": "background_stopping",
                "collision_nu_scale": float(nu_scale),
                "collision_background_density": float(bg_density),
                "collision_background_temperature": float(bg_temperature),
                "collision_species": ["ions"],
                "collision_background_type": "electrons",
                "collision_ndt": 1,
            }
        except Exception as exc:
            print(f"[config] unable to enable collisions: {exc}")
            collisions_meta = {"enabled": False, "error": str(exc)}
    cfg["collisions_meta"] = collisions_meta

    print("Starting 3D tilt smoke loop...")
    ee_cfg = cfg.get("electron_energy") or {}
    electron_energy_model = None
    if ee_cfg:
        electron_energy_model = ElectronEnergyModel(ee_cfg, species_names, hybrid_cfg)
    energy_cfg = cfg.get("energy_spectrum") or {}
    energy_spectrum_model = ParticleEnergySpectrum(
        energy_cfg, species_names, float(cfg.get("ion_amu", 1.0))
    )
    particle_vel_stats = None
    m1_mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
    if m1_mode in ("particle_vel_kick", "particle_v_kick", "particle_kick") or bool(
        cfg.get("particle_vel_stats_enabled", False)
    ):
        try:
            metadata_dir = Path(args.metadata_dir)
            stats_path = metadata_dir / "particle_vel_stats.csv"
            particle_vel_stats = ParticleVelStats(species_names, cfg, stats_path)
        except Exception as exc:
            print(f"Warning: failed to init particle_vel_stats: {exc}")
    u2_stats = None
    if bool(cfg.get("enable_u2_stats_diag", False)):
        try:
            u2_stats = U2StatsDiag(species_names, cfg, diag_dir)
        except Exception as exc:
            print(f"Warning: failed to init u2 stats diag: {exc}")
    coil_diag = None
    if bool(cfg.get("enable_coil_diag", False)):
        try:
            coil_diag = CoilFluxDiag(cfg, diag_dir)
        except Exception as exc:
            print(f"Warning: failed to init coil flux diag: {exc}")
    monitor = RunMonitor(
        species_names,
        cfg["monitor_interval"],
        drop_threshold=cfg["drop_threshold"],
        split_axis=cfg.get("monitor_split_axis"),
        split_value=float(cfg.get("monitor_split_value", 0.0)),
        split_species=cfg.get("monitor_species"),
        electron_energy_model=electron_energy_model,
    ) if cfg.get("monitor_interval") else None

    b_apply = None
    ext_drive_start_step = 0
    m1_drive_repeat = bool(cfg.get("m1_drive_repeat", False))
    try:
        m1_drive_nsteps = int(cfg.get("m1_drive_nsteps", 1))
    except (TypeError, ValueError):
        m1_drive_nsteps = 1
    try:
        m1_drive_stride = int(cfg.get("m1_drive_stride", 1))
    except (TypeError, ValueError):
        m1_drive_stride = 1
    if m1_drive_nsteps < 1:
        m1_drive_nsteps = 0
    if m1_drive_stride < 1:
        m1_drive_stride = 1
    m1_rho_cos_repeat = bool(cfg.get("m1_rho_cos_repeat", False))
    try:
        m1_rho_cos_nsteps = int(cfg.get("m1_rho_cos_nsteps", 1))
    except (TypeError, ValueError):
        m1_rho_cos_nsteps = 1
    try:
        m1_rho_cos_stride = int(cfg.get("m1_rho_cos_stride", 1))
    except (TypeError, ValueError):
        m1_rho_cos_stride = 1
    inject_repeat_nsteps = cfg.get("inject_repeat_nsteps")
    try:
        if inject_repeat_nsteps is not None:
            m1_rho_cos_nsteps = int(inject_repeat_nsteps)
    except (TypeError, ValueError):
        pass
    inject_stride_steps = cfg.get("inject_stride_steps")
    try:
        if inject_stride_steps is not None:
            m1_rho_cos_stride = int(inject_stride_steps)
    except (TypeError, ValueError):
        pass
    if m1_rho_cos_nsteps < 1:
        m1_rho_cos_nsteps = 0
    if m1_rho_cos_stride < 1:
        m1_rho_cos_stride = 1
    try:
        ext_drive_start_step = int(cfg.get("ext_drive_start_step", 0))
    except (TypeError, ValueError):
        ext_drive_start_step = 0
    try:
        opmd_b_scale = float(cfg.get("opmd_b_scale", 1.0))
    except (TypeError, ValueError):
        opmd_b_scale = 1.0
    try:
        drive_amp_scale = float(cfg.get("driveAmp_scale", 1.0))
    except (TypeError, ValueError):
        drive_amp_scale = 1.0
    opmd_b_scale_eff = float(opmd_b_scale) * float(drive_amp_scale)
    drive_envelope_enabled = bool(cfg.get("drive_envelope_enable", False))
    drive_envelope_method = str(cfg.get("drive_envelope_method", "step_rampdown")).strip().lower()
    try:
        drive_envelope_off_step = int(cfg.get("drive_envelope_off_step", 0))
    except (TypeError, ValueError):
        drive_envelope_off_step = 0
    try:
        drive_envelope_ramp_steps = int(cfg.get("drive_envelope_ramp_steps", 0))
    except (TypeError, ValueError):
        drive_envelope_ramp_steps = 0
    try:
        drive_envelope_floor = float(cfg.get("drive_envelope_floor", 0.0))
    except (TypeError, ValueError):
        drive_envelope_floor = 0.0
    enable_inject = bool(cfg.get("enable_inject", True))
    inject_end_call = cfg.get("inject_end_call")
    try:
        if inject_end_call is not None:
            inject_end_call = int(inject_end_call)
    except (TypeError, ValueError):
        inject_end_call = None
    inject_end_istep = cfg.get("inject_end_istep")
    try:
        if inject_end_istep is not None:
            inject_end_istep = int(inject_end_istep)
    except (TypeError, ValueError):
        inject_end_istep = None
    inject_end_step = cfg.get("inject_end_step")
    try:
        if inject_end_step is not None:
            inject_end_step = int(inject_end_step)
    except (TypeError, ValueError):
        inject_end_step = None

    def drive_envelope(step: int) -> float:
        if not drive_envelope_enabled or drive_envelope_method != "step_rampdown":
            return 1.0
        off = int(drive_envelope_off_step)
        ramp = int(drive_envelope_ramp_steps)
        floor = float(drive_envelope_floor)
        if step < off:
            return 1.0
        if ramp <= 0:
            return floor
        if step < off + ramp:
            frac = (step - off) / float(ramp)
            return max(floor, 1.0 - frac)
        return floor
    ext_drive_pending = False
    m1_drive_enabled = False
    m1_drive_pc = None
    m1_rho_cos_enabled = False
    m1_rho_cos_pc = None
    m1_rho_cos_base_weights = []
    m1_rho_cos_center = None
    m1_drive_meta = {
        "repeat": bool(m1_drive_repeat),
        "nsteps": int(m1_drive_nsteps),
        "stride": int(m1_drive_stride),
        "start_step": 0,
        "drive_applied_steps": [],
        "drive_num_applied": 0,
        "dv_abs_applied_each": [],
        "dv_abs_linf_each": [],
    }
    m1_rho_cos_meta = {
        "repeat": bool(m1_rho_cos_repeat),
        "nsteps": int(m1_rho_cos_nsteps),
        "stride": int(m1_rho_cos_stride),
        "start_step": 0,
        "eps": cfg.get("m1_rho_cos_eps", cfg.get("m1_inject_eps", 0.0)),
        "repeat_applied_steps": [],
        "num_applied": 0,
        "num_particles_modified_each": [],
        "rho_clip_fraction_each": [],
        "num_particles_modified_mean": None,
        "rho_clip_fraction_mean": None,
    }
    energy_drag_enabled = bool(cfg.get("enable_energy_drag", False))
    try:
        energy_drag_nu_scale = float(cfg.get("energy_drag_nu_scale", cfg.get("collision_nu_scale", 0.0)))
    except (TypeError, ValueError):
        energy_drag_nu_scale = 0.0
    velocity_reset_enabled = bool(cfg.get("enable_velocity_reset", True))
    try:
        velocity_reset_interval = int(cfg.get("velocity_reset_interval", 1))
    except (TypeError, ValueError):
        velocity_reset_interval = 1
    velocity_reset_end_step = cfg.get("velocity_reset_end_step")
    try:
        if velocity_reset_end_step is not None:
            velocity_reset_end_step = int(velocity_reset_end_step)
    except (TypeError, ValueError):
        velocity_reset_end_step = None
    velocity_reset_species = str(
        cfg.get("velocity_reset_species")
        or cfg.get("monitor_species")
        or "ions"
    ).strip() or "ions"
    energy_drag_pc = None
    energy_drag_meta = {
        "enabled": bool(energy_drag_enabled),
        "nu_scale": float(energy_drag_nu_scale),
        "velocity_reset_enabled": bool(velocity_reset_enabled),
        "velocity_reset_interval": int(velocity_reset_interval),
        "velocity_reset_end_step": velocity_reset_end_step,
        "velocity_reset_species": velocity_reset_species,
        "num_applied": 0,
        "last_step": None,
        "drag_apply_calls": 0,
        "drag_particles_touched": 0,
        "drag_delta_u2_sum": 0.0,
        "effective_drag_coeff": float(energy_drag_nu_scale * dt),
    }
    energy_diffusion_enabled = bool(cfg.get("enable_energy_diffusion", False))
    energy_diffusion_mode = str(cfg.get("energy_diffusion_mode", "u_kick")).strip().lower()
    try:
        energy_diffusion_scale = float(cfg.get("energy_diffusion_scale", 0.0))
    except (TypeError, ValueError):
        energy_diffusion_scale = 0.0
    energy_diffusion_seed = cfg.get("energy_diffusion_seed")
    try:
        energy_diffusion_start_step = int(cfg.get("energy_diffusion_start_step", 0))
    except (TypeError, ValueError):
        energy_diffusion_start_step = 0
    energy_diffusion_end_step = cfg.get("energy_diffusion_end_step")
    try:
        if energy_diffusion_end_step is not None:
            energy_diffusion_end_step = int(energy_diffusion_end_step)
    except (TypeError, ValueError):
        energy_diffusion_end_step = None
    energy_diffusion_pc = None
    energy_diffusion_meta = {
        "enabled": bool(energy_diffusion_enabled),
        "mode": energy_diffusion_mode,
        "scale": float(energy_diffusion_scale),
        "seed": energy_diffusion_seed,
        "start_step": int(energy_diffusion_start_step),
        "end_step": energy_diffusion_end_step,
        "num_applied": 0,
        "last_step": None,
        "diffusion_apply_calls": 0,
        "diffusion_particles_touched": 0,
        "diffusion_delta_u2_sum": 0.0,
        "effective_diffusion_coeff": float(energy_diffusion_scale * dt),
    }
    drive_envelope_meta = {
        "enabled": bool(drive_envelope_enabled),
        "method": drive_envelope_method,
        "off_step": int(drive_envelope_off_step),
        "ramp_steps": int(drive_envelope_ramp_steps),
        "floor": float(drive_envelope_floor),
        "last_step": None,
        "last_env": None,
    }
    runtime_guards = {
        "inject_calls": 0,
        "inject_skipped_calls": 0,
        "inject_particles_total": 0,
        "inject_step_used_min": None,
        "inject_step_used_max": None,
        "inject_step_used_first3": [],
        "inject_step_used_last3": [],
        "inject_step_used_unique_count": 0,
        "inject_istep_min": None,
        "inject_istep_max": None,
        "inject_istep_first3": [],
        "inject_istep_last3": [],
        "inject_istep_unique_count": 0,
        "inject_step_skipped_min": None,
        "inject_step_skipped_max": None,
        "inject_step_skipped_first3": [],
        "inject_step_skipped_last3": [],
        "inject_step_skipped_unique_count": 0,
        "inject_istep_skipped_min": None,
        "inject_istep_skipped_max": None,
        "inject_istep_skipped_first3": [],
        "inject_istep_skipped_last3": [],
        "inject_istep_skipped_unique_count": 0,
        "resample_calls": 0,
        "resample_particles_total": 0,
        "velocity_reset_calls": 0,
        "velocity_reset_particles_total": 0,
    }
    inject_step_unique = set()
    inject_istep_unique = set()
    inject_step_skipped_unique = set()
    inject_istep_skipped_unique = set()

    def accumulate_inject_guard(meta: dict | None, step_used: int | None, istep_used: int | None) -> None:
        if not meta or not bool(meta.get("applied", False)):
            return
        try:
            step_val = int(step_used) if step_used is not None else None
        except (TypeError, ValueError):
            step_val = None
        num_modified = meta.get("num_modified")
        if num_modified is None:
            num_modified = meta.get("num_particles_modified")
        try:
            num_modified = int(num_modified or 0)
        except (TypeError, ValueError):
            num_modified = 0
        if num_modified <= 0:
            return
        runtime_guards["inject_calls"] = int(runtime_guards.get("inject_calls", 0)) + 1
        runtime_guards["inject_particles_total"] = int(
            runtime_guards.get("inject_particles_total", 0)
        ) + num_modified
        if step_val is not None:
            inject_step_unique.add(step_val)
            if runtime_guards["inject_step_used_min"] is None or step_val < runtime_guards["inject_step_used_min"]:
                runtime_guards["inject_step_used_min"] = step_val
            if runtime_guards["inject_step_used_max"] is None or step_val > runtime_guards["inject_step_used_max"]:
                runtime_guards["inject_step_used_max"] = step_val
            if len(runtime_guards["inject_step_used_first3"]) < 3:
                runtime_guards["inject_step_used_first3"].append(step_val)
            runtime_guards["inject_step_used_last3"].append(step_val)
            if len(runtime_guards["inject_step_used_last3"]) > 3:
                runtime_guards["inject_step_used_last3"] = runtime_guards["inject_step_used_last3"][-3:]
        try:
            istep_val = int(istep_used) if istep_used is not None else None
        except (TypeError, ValueError):
            istep_val = None
        if istep_val is not None:
            inject_istep_unique.add(istep_val)
            if runtime_guards.get("inject_istep_min") is None or istep_val < runtime_guards.get("inject_istep_min"):
                runtime_guards["inject_istep_min"] = istep_val
            if runtime_guards.get("inject_istep_max") is None or istep_val > runtime_guards.get("inject_istep_max"):
                runtime_guards["inject_istep_max"] = istep_val
            if len(runtime_guards.get("inject_istep_first3", [])) < 3:
                runtime_guards.setdefault("inject_istep_first3", []).append(istep_val)
            runtime_guards.setdefault("inject_istep_last3", []).append(istep_val)
            if len(runtime_guards["inject_istep_last3"]) > 3:
                runtime_guards["inject_istep_last3"] = runtime_guards["inject_istep_last3"][-3:]

    def record_inject_skip(step_used: int | None, istep_used: int | None) -> None:
        runtime_guards["inject_skipped_calls"] = int(
            runtime_guards.get("inject_skipped_calls", 0)
        ) + 1
        try:
            step_val = int(step_used) if step_used is not None else None
        except (TypeError, ValueError):
            step_val = None
        if step_val is not None:
            inject_step_skipped_unique.add(step_val)
            if runtime_guards.get("inject_step_skipped_min") is None or step_val < runtime_guards.get("inject_step_skipped_min"):
                runtime_guards["inject_step_skipped_min"] = step_val
            if runtime_guards.get("inject_step_skipped_max") is None or step_val > runtime_guards.get("inject_step_skipped_max"):
                runtime_guards["inject_step_skipped_max"] = step_val
            if len(runtime_guards.get("inject_step_skipped_first3", [])) < 3:
                runtime_guards.setdefault("inject_step_skipped_first3", []).append(step_val)
            runtime_guards.setdefault("inject_step_skipped_last3", []).append(step_val)
            if len(runtime_guards["inject_step_skipped_last3"]) > 3:
                runtime_guards["inject_step_skipped_last3"] = runtime_guards["inject_step_skipped_last3"][-3:]
        try:
            istep_val = int(istep_used) if istep_used is not None else None
        except (TypeError, ValueError):
            istep_val = None
        if istep_val is not None:
            inject_istep_skipped_unique.add(istep_val)
            if runtime_guards.get("inject_istep_skipped_min") is None or istep_val < runtime_guards.get("inject_istep_skipped_min"):
                runtime_guards["inject_istep_skipped_min"] = istep_val
            if runtime_guards.get("inject_istep_skipped_max") is None or istep_val > runtime_guards.get("inject_istep_skipped_max"):
                runtime_guards["inject_istep_skipped_max"] = istep_val
            if len(runtime_guards.get("inject_istep_skipped_first3", [])) < 3:
                runtime_guards.setdefault("inject_istep_skipped_first3", []).append(istep_val)
            runtime_guards.setdefault("inject_istep_skipped_last3", []).append(istep_val)
            if len(runtime_guards["inject_istep_skipped_last3"]) > 3:
                runtime_guards["inject_istep_skipped_last3"] = runtime_guards["inject_istep_skipped_last3"][-3:]
    if init_mode in ("opmd", "opmd_double_seed"):
        try:
            sim.initialize_inputs()
            if hasattr(sim, "initialize_warpx"):
                sim.initialize_warpx()
            if opmd_b_data is not None:
                if ext_drive_start_step <= 0:
                    b_apply = apply_initial_bfield_from_opmd(
                        opmd_b_data, hybrid_enabled, scale=opmd_b_scale_eff
                    )
                    if b_apply is None:
                        b_apply = {"applied": False, "error": "bfield_apply_missing"}
                    b_apply["start_step"] = 0
                    b_apply["start_time"] = 0.0
                    b_apply["requested_start_step"] = ext_drive_start_step
                    b_apply["drive_amp_scale"] = float(drive_amp_scale)
                    b_apply["opmd_b_scale_eff"] = float(opmd_b_scale_eff)
                else:
                    b_apply = {
                        "applied": False,
                        "pending": True,
                        "start_step": int(ext_drive_start_step),
                        "start_time": float(ext_drive_start_step * dt),
                        "requested_start_step": int(ext_drive_start_step),
                    }
                    ext_drive_pending = True
            m1_mode = str(cfg.get("m1_inject_mode", "none")).strip().lower()
            if m1_rho_cos_repeat and m1_mode in (
                "rho_cos",
                "rho_cosine",
                "rho_cosphi",
                "rho_cos_phi",
            ):
                try:
                    m1_rho_cos_pc = particle_containers.ParticleContainerWrapper(
                        cfg.get("monitor_species") or "ions"
                    )
                    if opmd_fluid_data is not None and "rho" in opmd_fluid_data:
                        rho_shape = tuple(opmd_fluid_data["rho"].shape)
                        spacing = np.array(opmd_fluid_data["spacing"], dtype=float)
                        offset = np.array(opmd_fluid_data["offset"], dtype=float)
                    else:
                        rho_shape = (int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"]))
                        dx = (float(cfg["x_max"]) - float(cfg["x_min"])) / int(cfg["nx"])
                        dy = (float(cfg["y_max"]) - float(cfg["y_min"])) / int(cfg["ny"])
                        dz = (float(cfg["z_max"]) - float(cfg["z_min"])) / int(cfg["nz"])
                        spacing = np.array([dx, dy, dz], dtype=float)
                        offset = np.array([float(cfg["x_min"]), float(cfg["y_min"]), float(cfg["z_min"])], dtype=float)
                    axis = str(cfg.get("m1_inject_axis", "x")).strip().lower()
                    m1_rho_cos_center, center_err = _resolve_m1_center(
                        cfg, axis, spacing, offset, rho_shape
                    )
                    if center_err:
                        print(f"[m1_rho_cos] center resolve failed: {center_err}")
                        m1_rho_cos_center = None
                    m1_rho_cos_base_weights = snapshot_particle_weights(m1_rho_cos_pc)
                    m1_rho_cos_enabled = bool(m1_rho_cos_nsteps > 0)
                except Exception as exc:
                    print(f"[m1_rho_cos] setup failed: {exc}")
            if init_mode == "opmd_double_seed" and m1_mode in (
                "particle_vel_kick",
                "particle_v_kick",
                "particle_kick",
            ):
                try:
                    pc = particle_containers.ParticleContainerWrapper(
                        cfg.get("monitor_species") or "ions"
                    )
                    m1_drive_enabled = bool(m1_drive_repeat and m1_drive_nsteps > 0)
                    if not m1_drive_enabled:
                        allow_inject = enable_inject
                        if allow_inject:
                            if inject_end_call is not None:
                                allow_inject = 0 < inject_end_call
                            else:
                                allow_inject = 0 < inject_end_istep
                        if allow_inject:
                            m1_post_meta = apply_m1_particle_vel_kick_container(pc, cfg)
                            if m1_post_meta is not None:
                                if double_seed_meta is None:
                                    double_seed_meta = {}
                                double_seed_meta["m1_inject"] = m1_post_meta
                                enforce_particle_vel_kick(cfg, m1_post_meta)
                                accumulate_inject_guard(m1_post_meta, 0, 0)
                        else:
                            record_inject_skip(0, 0)
                    else:
                        m1_drive_pc = pc
                except Exception as exc:
                    print(f"[m1_inject] post-init apply failed: {exc}")
            if (
                init_mode == "opmd_double_seed"
                and bool(cfg.get("opmd_double_seed_group_ids", False))
            ):
                group_plane_point = None
                group_plane_normal = None
                com_a = drift_meta.get("com_a") if "drift_meta" in locals() else None
                com_b = drift_meta.get("com_b") if "drift_meta" in locals() else None
                if com_a is not None and com_b is not None:
                    com_a = np.array(com_a, dtype=float)
                    com_b = np.array(com_b, dtype=float)
                    sep_vec = com_b - com_a
                    sep_norm = float(np.linalg.norm(sep_vec))
                    if sep_norm > 0.0:
                        group_plane_point = 0.5 * (com_a + com_b)
                        group_plane_normal = sep_vec / sep_norm
                if group_plane_point is not None and group_plane_normal is not None:
                    try:
                        pc = particle_containers.ParticleContainerWrapper(
                            dynamic_drift_species or "ions"
                        )
                        ids_left, ids_right = select_group_ids_by_plane(
                            pc, group_plane_point, group_plane_normal
                        )
                        if ids_left is not None and ids_right is not None:
                            ids_left = np.array(ids_left)
                            ids_right = np.array(ids_right)
                            ids_left_sorted = np.sort(ids_left)
                            ids_right_sorted = np.sort(ids_right)
                            group_ids_meta = {
                                "method": "plane_split",
                                "plane_point": group_plane_point.tolist(),
                                "plane_normal": group_plane_normal.tolist(),
                                "count_left": int(ids_left_sorted.size),
                                "count_right": int(ids_right_sorted.size),
                                "hash_left": hashlib.sha256(ids_left_sorted.tobytes()).hexdigest(),
                                "hash_right": hashlib.sha256(ids_right_sorted.tobytes()).hexdigest(),
                            }
                            dynamic_group_ids = (ids_left_sorted, ids_right_sorted)
                            if double_seed_meta is not None:
                                double_seed_meta["group_ids"] = group_ids_meta
                            drift_meta["group_ids"] = group_ids_meta
                            if monitor is not None:
                                monitor.set_group_ids(
                                    ids_left_sorted, ids_right_sorted, group_ids_meta
                                )
                            if "drift_meta_path" in locals() and drift_meta_path is not None:
                                drift_meta_out = dict(drift_meta)
                                drift_meta_out["run_tag"] = args.run_tag
                                write_json_atomic(drift_meta_path, drift_meta_out)
                    except Exception as exc:
                        print(f"[group_ids] selection_failed: {exc}")
        except Exception as exc:
            b_apply = {"applied": False, "error": f"init_failed: {exc}"}

    initial_stats = gather_species_stats(species_names)
    git_head, git_dirty = git_info(repo_root)

    metadata_dir = Path(args.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"warpx_run_{args.run_tag}.json"
    heartbeat_path = metadata_dir / "warpx_heartbeat.json"
    meta_base = build_meta_base(
        cfg,
        git_head,
        git_dirty,
        hybrid_enabled,
        species_names,
        initial_stats,
        init_mode,
        opmd_fluid_data,
        opmd_b_data,
        particle_weight_hist,
        b_apply,
        double_seed_meta,
        ee_cfg,
    )
    if energy_drag_enabled:
        meta_base["energy_drag"] = dict(energy_drag_meta)
    if energy_diffusion_enabled:
        meta_base["energy_diffusion"] = dict(energy_diffusion_meta)
    if drive_envelope_enabled:
        meta_base["drive_envelope"] = dict(drive_envelope_meta)
    eta_profile_runtime = None
    eta_profile_current = None
    eta_profile_method = None
    if hybrid_enabled and eta_profile_enabled and eta_profile_low is not None and eta_profile_high is not None:
        try:
            from pywarpx.HybridPICModel import HybridPICModel as HybridPICRuntime
            eta_profile_runtime = HybridPICRuntime()
        except Exception as exc:
            print(f"[eta_profile] unable to access HybridPICRuntime: {exc}")
            eta_profile_runtime = None

    def apply_eta_scale_runtime(scale: float) -> bool:
        nonlocal eta_profile_runtime, eta_profile_method
        if eta_profile_runtime is None:
            return False
        scale_val = float(scale)
        # Prefer scaling attribute if available.
        for attr in ("plasma_resistivity_scale", "resistivity_scale", "eta_scale"):
            if hasattr(eta_profile_runtime, attr):
                try:
                    setattr(eta_profile_runtime, attr, scale_val)
                    eta_profile_method = attr
                    return True
                except Exception:
                    pass
        for method in ("set_plasma_resistivity_scale", "set_resistivity_scale", "set_eta_scale"):
            if hasattr(eta_profile_runtime, method):
                try:
                    getattr(eta_profile_runtime, method)(scale_val)
                    eta_profile_method = method
                    return True
                except Exception:
                    pass
        # Fallback: set absolute resistivity if base value is available.
        if eta_profile_base is not None:
            eta_val = float(eta_profile_base) * scale_val
            for attr in ("plasma_resistivity", "eta", "resistivity"):
                if hasattr(eta_profile_runtime, attr):
                    try:
                        setattr(eta_profile_runtime, attr, eta_val)
                        eta_profile_method = attr
                        return True
                    except Exception:
                        pass
            for method in ("set_plasma_resistivity", "set_resistivity", "set_eta"):
                if hasattr(eta_profile_runtime, method):
                    try:
                        getattr(eta_profile_runtime, method)(eta_val)
                        eta_profile_method = method
                        return True
                    except Exception:
                        pass
        return False

    if eta_profile_runtime is not None:
        if apply_eta_scale_runtime(eta_profile_low):
            eta_profile_current = eta_profile_low
            resistivity_meta = meta_base.get("resistivity") or {}
            resistivity_meta["effective_eta_scale"] = float(eta_profile_low)
            resistivity_meta["eta_profile_state"] = "low"
            if eta_profile_method:
                resistivity_meta["eta_profile_method"] = eta_profile_method
            meta_base["resistivity"] = resistivity_meta
        meta_base["eta_profile_debug"] = {
            "eta_scale_low": eta_profile_low,
            "eta_scale_high": eta_profile_high,
            "eta_switch_step": eta_profile_switch,
            "eta_profile_method": eta_profile_method,
            "eta_profile_base": eta_profile_base,
        }
    m1_drive_meta["enabled"] = bool(m1_drive_enabled)
    meta_base["m1_drive"] = m1_drive_meta
    m1_rho_cos_meta["enabled"] = bool(m1_rho_cos_enabled)
    meta_base["m1_rho_cos_drive"] = m1_rho_cos_meta
    if particle_vel_stats is not None:
        meta_base["particle_vel_stats"] = {
            "path": str(particle_vel_stats.output_path),
            "interval": particle_vel_stats.interval,
            "species": particle_vel_stats.species,
        }
    if u2_stats is not None:
        meta_base["u2_stats"] = {
            "path": str(u2_stats.output_path) if u2_stats.output_path is not None else None,
            "interval": u2_stats.interval,
            "species": u2_stats.species,
        }
    if coil_diag is not None:
        meta_base["coil_diag"] = coil_diag.as_dict()
    circuit_cfg = cfg.get("circuit_mvp")
    if isinstance(circuit_cfg, dict) and circuit_cfg:
        meta_base["circuit_mvp"] = circuit_cfg
    restart_meta = None
    if cfg.get("warpx_amr_restart"):
        restart_meta = read_restart_sanity()
        meta_base["restart"] = {"amr_restart": cfg.get("warpx_amr_restart")}
        if restart_meta:
            meta_base["restart_sanity"] = restart_meta
    max_steps = int(cfg["max_steps"])
    if inject_end_step is None or inject_end_step < 0:
        inject_end_step = max_steps
    if inject_end_istep is None or inject_end_istep < 0:
        inject_end_istep = max_steps
    if velocity_reset_end_step is None or velocity_reset_end_step < 0:
        velocity_reset_end_step = max_steps
        energy_drag_meta["velocity_reset_end_step"] = int(velocity_reset_end_step)
    if velocity_reset_interval < 1:
        velocity_reset_interval = 1
        energy_drag_meta["velocity_reset_interval"] = int(velocity_reset_interval)
    energy_spectrum_model.initialize(max_steps)
    if energy_spectrum_model.enabled:
        meta_base["energy_spectrum"] = energy_spectrum_model.as_dict()
    meta_base["run_start"] = datetime.now(timezone.utc).isoformat()
    write_metadata_snapshot(
        meta_base,
        metadata_path,
        heartbeat_path,
        monitor,
        "running",
        None,
        None,
        diag_dir,
        electron_energy_model=electron_energy_model,
        energy_spectrum_model=energy_spectrum_model,
        ee_cfg=ee_cfg,
    )

    heartbeat_s = cfg.get("metadata_heartbeat_s", 60.0)
    try:
        heartbeat_s = float(heartbeat_s) if heartbeat_s is not None else None
    except (TypeError, ValueError):
        heartbeat_s = None
    heartbeat_steps = cfg.get("metadata_heartbeat_steps")
    try:
        heartbeat_steps = int(heartbeat_steps) if heartbeat_steps is not None else None
    except (TypeError, ValueError):
        heartbeat_steps = None
    last_heartbeat = time.monotonic()
    last_heartbeat_step = None

    for step in range(max_steps):
        t_current = step * dt
        if eta_profile_runtime is not None and eta_profile_switch is not None:
            target_scale = eta_profile_low if step < eta_profile_switch else eta_profile_high
            if eta_profile_current is None or target_scale != eta_profile_current:
                if apply_eta_scale_runtime(target_scale):
                    eta_profile_current = target_scale
                    resistivity_meta = meta_base.get("resistivity") or {}
                    resistivity_meta["effective_eta_scale"] = float(target_scale)
                    resistivity_meta["eta_profile_state"] = "low" if step < eta_profile_switch else "high"
                    if eta_profile_method:
                        resistivity_meta["eta_profile_method"] = eta_profile_method
                    meta_base["resistivity"] = resistivity_meta
        if m1_drive_enabled and m1_drive_nsteps > 0:
            if step >= m1_drive_meta["start_step"] and step < (m1_drive_meta["start_step"] + m1_drive_nsteps):
                if (step - m1_drive_meta["start_step"]) % m1_drive_stride == 0:
                    allow_inject = enable_inject
                    if allow_inject:
                        if inject_end_call is not None:
                            allow_inject = int(m1_drive_meta["drive_num_applied"]) < int(inject_end_call)
                        else:
                            allow_inject = step < inject_end_istep
                    if allow_inject:
                        try:
                            if m1_drive_pc is None:
                                m1_drive_pc = particle_containers.ParticleContainerWrapper(
                                    cfg.get("monitor_species") or "ions"
                                )
                            drive_meta = apply_m1_particle_vel_kick_container(m1_drive_pc, cfg)
                            if drive_meta is not None and bool(drive_meta.get("applied", False)):
                                m1_drive_meta["drive_applied_steps"].append(int(step))
                                m1_drive_meta["drive_num_applied"] = int(
                                    len(m1_drive_meta["drive_applied_steps"])
                                )
                                m1_drive_meta["dv_abs_applied_each"].append(
                                    float(drive_meta.get("dv_abs_applied", 0.0))
                                )
                                m1_drive_meta["dv_abs_linf_each"].append(
                                    float(drive_meta.get("dv_abs_linf", 0.0))
                                )
                                accumulate_inject_guard(
                                    drive_meta,
                                    step_used=int(m1_drive_meta["drive_num_applied"]),
                                    istep_used=step,
                                )
                        except Exception as exc:
                            print(f"[m1_drive] apply failed at step {step}: {exc}")
                    else:
                        record_inject_skip(step_used=int(m1_drive_meta["drive_num_applied"]), istep_used=step)
        if m1_rho_cos_enabled and m1_rho_cos_nsteps > 0 and m1_rho_cos_pc is not None:
            if step >= m1_rho_cos_meta["start_step"] and step < (
                m1_rho_cos_meta["start_step"] + m1_rho_cos_nsteps
            ):
                if (step - m1_rho_cos_meta["start_step"]) % m1_rho_cos_stride == 0:
                    allow_inject = enable_inject
                    if allow_inject:
                        if inject_end_call is not None:
                            allow_inject = int(m1_rho_cos_meta["num_applied"]) < int(inject_end_call)
                        else:
                            allow_inject = step < inject_end_istep
                    if allow_inject:
                        try:
                            rho_meta = apply_m1_rho_cos_weight_container(
                                m1_rho_cos_pc, cfg, m1_rho_cos_center, m1_rho_cos_base_weights
                            )
                            if rho_meta is not None and bool(rho_meta.get("applied", False)):
                                m1_rho_cos_meta["repeat_applied_steps"].append(int(step))
                                m1_rho_cos_meta["num_applied"] = int(
                                    len(m1_rho_cos_meta["repeat_applied_steps"])
                                )
                                num_mod = rho_meta.get("num_particles_modified")
                                clip_frac = rho_meta.get("rho_clip_fraction")
                                if num_mod is not None:
                                    m1_rho_cos_meta["num_particles_modified_each"].append(float(num_mod))
                                if clip_frac is not None:
                                    m1_rho_cos_meta["rho_clip_fraction_each"].append(float(clip_frac))
                                if m1_rho_cos_meta["num_particles_modified_each"]:
                                    m1_rho_cos_meta["num_particles_modified_mean"] = float(
                                        np.mean(m1_rho_cos_meta["num_particles_modified_each"])
                                    )
                                if m1_rho_cos_meta["rho_clip_fraction_each"]:
                                    m1_rho_cos_meta["rho_clip_fraction_mean"] = float(
                                        np.mean(m1_rho_cos_meta["rho_clip_fraction_each"])
                                    )
                                meta_base["m1_rho_cos_drive"] = m1_rho_cos_meta
                                accumulate_inject_guard(
                                    rho_meta,
                                    step_used=int(m1_rho_cos_meta["num_applied"]),
                                    istep_used=step,
                                )
                        except Exception as exc:
                            print(f"[m1_rho_cos] apply failed at step {step}: {exc}")
                    else:
                        record_inject_skip(step_used=int(m1_rho_cos_meta["num_applied"]), istep_used=step)
        if drive_envelope_enabled and opmd_b_data is not None and step >= ext_drive_start_step:
            env = float(drive_envelope(step))
            scale_env = opmd_b_scale_eff * env
            b_apply = apply_initial_bfield_from_opmd(
                opmd_b_data, hybrid_enabled, scale=scale_env
            )
            if b_apply is None:
                b_apply = {"applied": False, "error": "bfield_apply_missing"}
            b_apply["start_step"] = int(step)
            b_apply["start_time"] = float(t_current)
            b_apply["requested_start_step"] = int(ext_drive_start_step)
            b_apply["drive_envelope_env"] = float(env)
            b_apply["drive_envelope_scale"] = float(scale_env)
            b_apply["drive_amp_scale"] = float(drive_amp_scale)
            b_apply["opmd_b_scale_eff"] = float(opmd_b_scale_eff)
            meta_base["bfield_apply"] = b_apply
            drive_envelope_meta["last_step"] = int(step)
            drive_envelope_meta["last_env"] = float(env)
            meta_base["drive_envelope"] = dict(drive_envelope_meta)
        elif ext_drive_pending and opmd_b_data is not None and step >= ext_drive_start_step:
            b_apply = apply_initial_bfield_from_opmd(
                opmd_b_data, hybrid_enabled, scale=opmd_b_scale_eff
            )
            if b_apply is None:
                b_apply = {"applied": False, "error": "bfield_apply_missing"}
            b_apply["start_step"] = int(step)
            b_apply["start_time"] = float(t_current)
            b_apply["requested_start_step"] = int(ext_drive_start_step)
            b_apply["drive_amp_scale"] = float(drive_amp_scale)
            b_apply["opmd_b_scale_eff"] = float(opmd_b_scale_eff)
            meta_base["bfield_apply"] = b_apply
            ext_drive_pending = False
        if particle_vel_stats is not None:
            particle_vel_stats.maybe_record(step, t_current)
        if u2_stats is not None:
            u2_stats.maybe_record(step, t_current)
        if coil_diag is not None:
            coil_diag.maybe_record(step, t_current)
        if (
            energy_drag_enabled
            and velocity_reset_enabled
            and step < velocity_reset_end_step
            and (step % velocity_reset_interval == 0)
        ):
            try:
                if energy_drag_pc is None:
                    energy_drag_pc = particle_containers.ParticleContainerWrapper(
                        velocity_reset_species
                    )
                drag_meta = apply_energy_drag(energy_drag_pc, energy_drag_nu_scale, dt)
                if drag_meta.get("applied", False):
                    energy_drag_meta["num_applied"] = int(energy_drag_meta.get("num_applied", 0)) + 1
                    energy_drag_meta["last_step"] = int(step)
                    energy_drag_meta["drag_apply_calls"] = int(energy_drag_meta.get("drag_apply_calls", 0)) + 1
                    energy_drag_meta["drag_particles_touched"] = int(
                        energy_drag_meta.get("drag_particles_touched", 0)
                        + int(drag_meta.get("num_modified", 0))
                    )
                    energy_drag_meta["drag_delta_u2_sum"] = float(
                        energy_drag_meta.get("drag_delta_u2_sum", 0.0)
                        + float(drag_meta.get("delta_u2_sum", 0.0))
                    )
                    energy_drag_meta["effective_drag_coeff"] = float(
                        float(drag_meta.get("nu_scale", energy_drag_nu_scale)) * float(drag_meta.get("dt", dt))
                    )
                    energy_drag_meta.update(drag_meta)
                    meta_base["energy_drag"] = energy_drag_meta
                    try:
                        num_modified = int(drag_meta.get("num_modified", 0))
                    except (TypeError, ValueError):
                        num_modified = 0
                    if num_modified > 0:
                        runtime_guards["velocity_reset_calls"] = int(
                            runtime_guards.get("velocity_reset_calls", 0)
                        ) + 1
                        runtime_guards["velocity_reset_particles_total"] = int(
                            runtime_guards.get("velocity_reset_particles_total", 0)
                        ) + num_modified
            except Exception as exc:
                print(f"[energy_drag] apply failed at step {step}: {exc}")
        if (
            energy_diffusion_enabled
            and energy_diffusion_mode == "u_kick"
            and step >= energy_diffusion_start_step
            and (energy_diffusion_end_step is None or step < energy_diffusion_end_step)
        ):
            try:
                if energy_diffusion_pc is None:
                    energy_diffusion_pc = particle_containers.ParticleContainerWrapper(
                        cfg.get("monitor_species") or "ions"
                    )
                diff_meta = apply_energy_diffusion(
                    energy_diffusion_pc,
                    energy_diffusion_scale,
                    dt,
                    energy_diffusion_seed,
                    step,
                )
                if diff_meta.get("applied", False):
                    energy_diffusion_meta["num_applied"] = int(energy_diffusion_meta.get("num_applied", 0)) + 1
                    energy_diffusion_meta["last_step"] = int(step)
                    energy_diffusion_meta["diffusion_apply_calls"] = int(
                        energy_diffusion_meta.get("diffusion_apply_calls", 0)
                    ) + 1
                    energy_diffusion_meta["diffusion_particles_touched"] = int(
                        energy_diffusion_meta.get("diffusion_particles_touched", 0)
                        + int(diff_meta.get("num_modified", 0))
                    )
                    energy_diffusion_meta["diffusion_delta_u2_sum"] = float(
                        energy_diffusion_meta.get("diffusion_delta_u2_sum", 0.0)
                        + float(diff_meta.get("delta_u2_sum", 0.0))
                    )
                    energy_diffusion_meta["effective_diffusion_coeff"] = float(
                        float(diff_meta.get("scale", energy_diffusion_scale)) * float(diff_meta.get("dt", dt))
                    )
                    energy_diffusion_meta.update(diff_meta)
                    meta_base["energy_diffusion"] = energy_diffusion_meta
            except Exception as exc:
                print(f"[energy_diffusion] apply failed at step {step}: {exc}")
        sim.step(1)
        if monitor:
            monitor.maybe_record(step, t_current)
        if energy_spectrum_model.enabled:
            energy_spectrum_model.maybe_record(step, t_current)
        heartbeat_due = False
        if heartbeat_s is not None and heartbeat_s > 0:
            now = time.monotonic()
            if (now - last_heartbeat) >= heartbeat_s:
                heartbeat_due = True
                last_heartbeat = now
        if heartbeat_steps is not None and heartbeat_steps > 0:
            if step % heartbeat_steps == 0 and step != last_heartbeat_step:
                heartbeat_due = True
                last_heartbeat_step = step
        if heartbeat_due:
            if (
                dynamic_drift_enabled
                and dynamic_drift_unit is not None
                and dynamic_drift_u is not None
                and dynamic_drift_mag > 0.0
                and dynamic_drift_max_beta is not None
                and dynamic_drift_species is not None
            ):
                try:
                    pc = particle_containers.ParticleContainerWrapper(dynamic_drift_species)
                    dynamic_drift_unit, dynamic_drift_u, _ = update_dynamic_drift(
                        pc,
                        dynamic_drift_unit,
                        dynamic_drift_mag,
                        dynamic_drift_u,
                        dynamic_drift_max_beta,
                        group_ids=dynamic_group_ids,
                    )
                except Exception as exc:
                    print(f"[drift_dynamic] update_failed: {exc}")
            write_metadata_snapshot(
                meta_base,
                metadata_path,
                heartbeat_path,
                monitor,
                "running",
                step,
                t_current,
                diag_dir,
                electron_energy_model=electron_energy_model,
                energy_spectrum_model=energy_spectrum_model,
                ee_cfg=ee_cfg,
            )

    dropped_total = None
    try:
        wx_instance = libwarpx.warpx.get_instance()
        dropped_total = wx_instance.dropped_particles_total
    except Exception as exc:
        print(f"Warning: unable to read dropped-particle counter: {exc}")

    final_stats = gather_species_stats(species_names)
    if init_mode == "opmd_double_seed" and double_seed_meta is not None:
        double_seed_meta["particle_count_total"] = int(n_particles)
    write_metadata_snapshot(
        meta_base,
        metadata_path,
        heartbeat_path,
        monitor,
        "completed",
        max_steps - 1,
        (max_steps - 1) * dt,
        diag_dir,
        dropped_total=dropped_total,
        final_stats=final_stats,
        electron_energy_model=electron_energy_model,
        energy_spectrum_model=energy_spectrum_model,
        ee_cfg=ee_cfg,
    )
    if energy_drag_enabled:
        try:
            analysis_dir = metadata_dir.parent.parent / "analysis"
            metrics_drag_path = analysis_dir / "metrics_drag.json"
            write_json_atomic(metrics_drag_path, energy_drag_meta)
        except Exception as exc:
            print(f"[energy_drag] metrics write failed: {exc}")
    if energy_diffusion_enabled:
        try:
            analysis_dir = metadata_dir.parent.parent / "analysis"
            metrics_diff_path = analysis_dir / "metrics_diffusion.json"
            write_json_atomic(metrics_diff_path, energy_diffusion_meta)
        except Exception as exc:
            print(f"[energy_diffusion] metrics write failed: {exc}")
    try:
        analysis_dir = metadata_dir.parent.parent / "analysis"
        if inject_step_unique:
            runtime_guards["inject_step_used_unique_count"] = int(len(inject_step_unique))
        if inject_istep_unique:
            runtime_guards["inject_istep_unique_count"] = int(len(inject_istep_unique))
        if inject_step_skipped_unique:
            runtime_guards["inject_step_skipped_unique_count"] = int(len(inject_step_skipped_unique))
        if inject_istep_skipped_unique:
            runtime_guards["inject_istep_skipped_unique_count"] = int(len(inject_istep_skipped_unique))
        runtime_guard = dict(runtime_guards)
        off_step = drive_envelope_off_step if drive_envelope_enabled else cfg.get("drive_envelope_off_step")
        try:
            off_step = int(off_step) if off_step is not None else None
        except (TypeError, ValueError):
            off_step = None
        runtime_guard["drive_envelope_off_step"] = off_step
        runtime_guard["velocity_reset_enabled"] = bool(velocity_reset_enabled)
        runtime_guard["velocity_reset_interval"] = int(velocity_reset_interval)
        runtime_guard["velocity_reset_end_step"] = int(velocity_reset_end_step)
        runtime_guard["velocity_reset_species"] = velocity_reset_species
        runtime_guard["enable_inject"] = bool(enable_inject)
        runtime_guard["inject_end_step"] = int(inject_end_step)
        runtime_guard["inject_end_step_effective"] = int(inject_end_step)
        runtime_guard["inject_end_call"] = inject_end_call
        runtime_guard["inject_end_istep"] = int(inject_end_istep)
        runtime_guard["inject_end_istep_effective"] = int(inject_end_istep)
        runtime_guard["inject_end_main_step_effective"] = int(inject_end_istep)
        runtime_guard["inject_gate_mode"] = "call_index" if inject_end_call is not None else "main_step"
        runtime_guard["inject_main_step_min"] = runtime_guard.get("inject_istep_min")
        runtime_guard["inject_main_step_max"] = runtime_guard.get("inject_istep_max")
        runtime_guard["inject_main_step_first3"] = runtime_guard.get("inject_istep_first3")
        runtime_guard["inject_main_step_last3"] = runtime_guard.get("inject_istep_last3")
        runtime_guard["inject_repeat_nsteps_effective"] = int(m1_rho_cos_nsteps)
        runtime_guard["inject_stride_steps_effective"] = int(m1_rho_cos_stride)
        runtime_guard["inject_call_index_min"] = runtime_guard.get("inject_step_used_min")
        runtime_guard["inject_call_index_max"] = runtime_guard.get("inject_step_used_max")
        runtime_guard["inject_call_index_first3"] = runtime_guard.get("inject_step_used_first3")
        runtime_guard["inject_call_index_last3"] = runtime_guard.get("inject_step_used_last3")
        if u2_stats is not None and u2_stats.output_path is not None:
            runtime_guard.update(read_u2_stats_snapshot(Path(u2_stats.output_path), off_step))
        if bool(cfg.get("enable_u2_direct_stats", False)):
            try:
                direct_species = str(
                    cfg.get("u2_direct_species")
                    or velocity_reset_species
                    or cfg.get("monitor_species")
                    or "ions"
                ).strip() or "ions"
                pc = particle_containers.ParticleContainerWrapper(direct_species)
                runtime_guard.update(compute_u2_direct(pc))
            except Exception as exc:
                runtime_guard["u2_direct_error"] = str(exc)
        metrics_guard_path = analysis_dir / "metrics_runtime_guards.json"
        write_json_atomic(metrics_guard_path, runtime_guard)
    except Exception as exc:
        print(f"[runtime_guard] metrics write failed: {exc}")
    print(f"[metadata] {metadata_path}")


if __name__ == "__main__":
    main()
