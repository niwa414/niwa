#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import tarfile
import shlex
from pathlib import Path
from statistics import median

import numpy as np

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_json_if_exists(path: Path | None):
    try:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _first_present(d: dict, keys: list[str]):
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _merge_time_candidates(h3a_metrics: dict, formation_metrics: dict | None):
    cands = {}
    v = _first_present(h3a_metrics, ["merge_time"])
    if v is not None:
        cands["h3a.merge_time"] = v
    v = _first_present(h3a_metrics, ["merge_time_seedmask"])
    if v is not None:
        cands["h3a.merge_time_seedmask"] = v
    v = _first_present(h3a_metrics, ["t_at_s_min", "t_s_min"])
    if v is not None:
        cands["h3a.t_at_s_min"] = v
    fm = formation_metrics or {}
    v = None
    for k in (
        "formation_kpi_phase.t_rho_max_peak",
        "formation_kpi2.t_B_energy_peak",
        "t_rho_max_peak",
        "t_B_energy_peak",
    ):
        if k in fm and fm[k] is not None:
            v = fm[k]
            break
        if "." in k:
            a, b = k.split(".", 1)
            if isinstance(fm.get(a), dict) and fm[a].get(b) is not None:
                v = fm[a][b]
                break
    if v is not None:
        cands["formation.phase_peak"] = v
    return cands


def _select_merge_time(candidates: dict, dt: float, t_end: float):
    order = [
        "h3a.merge_time",
        "h3a.merge_time_seedmask",
        "h3a.t_at_s_min",
        "formation.phase_peak",
    ]
    c = {k: float(v) for k, v in (candidates or {}).items() if v is not None}
    thr = max(2 * dt, 0.05 * t_end)
    if not c:
        return {
            "merge_time": None,
            "merge_time_source": None,
            "merge_time_conflict": False,
            "merge_time_confidence": "none",
            "merge_time_proxy_used": False,
            "merge_time_consensus_dt": thr,
        }
    vals = list(c.values())
    conflict = False
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) > thr:
                conflict = True
                break
        if conflict:
            break
    primary_key = next((k for k in order if k in c), None)
    primary_t = c.get(primary_key) if primary_key else vals[0]
    in_consensus = {k: v for k, v in c.items() if abs(v - primary_t) <= thr}
    if in_consensus:
        chosen_t = float(median(list(in_consensus.values())))
        chosen_src = primary_key if primary_key in in_consensus else next(iter(in_consensus.keys()))
        proxy_used = chosen_src != "h3a.merge_time"
        confidence = (
            "high"
            if chosen_src == "h3a.merge_time" and not conflict
            else ("medium" if not conflict else "low")
        )
        return {
            "merge_time": chosen_t,
            "merge_time_source": chosen_src,
            "merge_time_conflict": conflict,
            "merge_time_confidence": confidence,
            "merge_time_proxy_used": proxy_used,
            "merge_time_consensus_dt": thr,
        }
    chosen_src = primary_key or next(iter(c.keys()))
    chosen_t = c[chosen_src]
    proxy_used = chosen_src != "h3a.merge_time"
    return {
        "merge_time": chosen_t,
        "merge_time_source": chosen_src,
        "merge_time_conflict": True,
        "merge_time_confidence": "proxy",
        "merge_time_proxy_used": proxy_used,
        "merge_time_consensus_dt": thr,
    }



def load_warpx_meta(path: Path | None) -> dict:
    if path is None:
        return {}
    if path.exists():
        return load_json(path)
    try:
        output_root = path.parents[2]
    except IndexError:
        return {}
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError:
        return {}
    for name in ("raw_fail.tar.gz", "raw_pass.tar.gz"):
        archive_path = output_root / name
        if not archive_path.exists():
            continue
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                try:
                    member = tar.getmember(rel)
                except KeyError:
                    continue
                handle = tar.extractfile(member)
                if handle:
                    return json.loads(handle.read().decode("utf-8"))
        except Exception:
            continue
    return {}


def load_args_from_command(command: str | None) -> dict:
    if not command:
        return {}
    try:
        parts = shlex.split(command)
    except Exception:
        parts = str(command).split()
    for idx, tok in enumerate(parts):
        if tok == "--from-json" and idx + 1 < len(parts):
            return load_json(Path(parts[idx + 1]))
    return {}


def load_heartbeat(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def load_output_inputs(output_root: Path) -> tuple[dict, dict]:
    args_cfg = {}
    seed_cfg = {}
    inputs_root = output_root / "raw" / "inputs" / "inputs"
    if not inputs_root.exists():
        return args_cfg, seed_cfg
    args_path = inputs_root / "warpx_args.json"
    seed_path = inputs_root / "h5_seed_config.json"
    if args_path.exists():
        args_cfg = load_json(args_path)
    if seed_path.exists():
        seed_cfg = load_json(seed_path)
    return args_cfg, seed_cfg


def load_tilt_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    amps: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t_raw = row.get("time_s")
            a_raw = row.get("amp_xy")
            if a_raw is None:
                a_raw = row.get("amp_smooth") or row.get("amp_raw") or row.get("amp")
            if t_raw is None or a_raw is None:
                continue
            try:
                times.append(float(t_raw))
                amps.append(float(a_raw))
            except ValueError:
                continue
    return np.array(times, dtype=float), np.array(amps, dtype=float)


def _parse_m1_header(header: str) -> list[str]:
    tokens = header.strip().split()
    names = []
    for tok in tokens:
        if "]" in tok:
            names.append(tok.split("]", 1)[1])
        else:
            names.append(tok)
    return names


def _find_m1_indices(
    names: list[str], ratio_prefixes: tuple[str, ...]
) -> tuple[int | None, int | None, str | None]:
    time_idx = None
    ratio_idx = None
    ratio_name = None
    for i, name in enumerate(names):
        if time_idx is None and name.startswith("time"):
            time_idx = i
    if ratio_idx is None:
        for prefix in ratio_prefixes:
            for i, name in enumerate(names):
                if name.startswith(prefix):
                    ratio_idx = i
                    ratio_name = name
                    break
            if ratio_idx is not None:
                break
    return time_idx, ratio_idx, ratio_name


def _sha1_file(path: Path) -> str | None:
    try:
        h = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def load_m1_series(
    path: Path,
    ratio_prefixes: tuple[str, ...],
    extra_prefixes: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    times: list[float] = []
    ratios: list[float] = []
    extras: dict[str, list[float]] = {prefix: [] for prefix in extra_prefixes}
    header = None
    time_idx = None
    ratio_idx = None
    ratio_name = None
    extras_idx: dict[str, int | None] = {prefix: None for prefix in extra_prefixes}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith("#"):
                if header is None:
                    header = line
                    names = _parse_m1_header(header)
                    time_idx, ratio_idx, ratio_name = _find_m1_indices(names, ratio_prefixes)
                    for i, name in enumerate(names):
                        for prefix in extra_prefixes:
                            if extras_idx[prefix] is None and name.startswith(prefix):
                                extras_idx[prefix] = i
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if time_idx is None:
                time_idx = 1 if len(parts) > 1 else None
            if ratio_idx is None:
                ratio_idx = 5 if len(parts) > 5 else None
            if time_idx is None or ratio_idx is None:
                continue
            if len(parts) <= max(time_idx, ratio_idx):
                continue
            try:
                times.append(float(parts[time_idx]))
                ratios.append(float(parts[ratio_idx]))
            except ValueError:
                continue
            if extra_prefixes:
                for prefix, idx in extras_idx.items():
                    if idx is None or len(parts) <= idx:
                        continue
                    try:
                        extras[prefix].append(float(parts[idx]))
                    except ValueError:
                        extras[prefix].append(float("nan"))
    audit = {
        "m1_source_path": str(path.resolve()),
        "m1_sha1": _sha1_file(path),
        "m1_filesize_bytes": None,
        "m1_mtime": None,
        "m1_ratio_field": ratio_name,
    }
    try:
        stat = path.stat()
        audit["m1_filesize_bytes"] = int(stat.st_size)
        audit["m1_mtime"] = float(stat.st_mtime)
    except Exception:
        pass
    return np.array(times, dtype=float), np.array(ratios, dtype=float), audit, extras


def load_m1rho_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    times, ratios, audit, _extras = load_m1_series(path, ("m1_rho_ratio", "m1_ratio_raw", "m1_ratio"))
    return times, ratios, audit


def load_m1mom_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    return load_m1_series(
        path,
        ("m1_mom_ratio_B", "m1_mom_ratio_A", "m1_mom_ratio", "m1_ratio_raw", "m1_ratio"),
        ("m0_mom_amp", "m0_rho_abs"),
    )


def load_particle_vel_stats_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    times: list[float] = []
    ratios: list[float] = []
    extras: dict[str, list[float | None]] = {"m1_vperp_amp": [], "step": []}
    ratio_name = None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if ratio_name is None:
                ratio_name = "m1_vperp_ratio"
            t_raw = row.get("time") or row.get("t")
            r_raw = row.get("m1_vperp_ratio")
            if t_raw is None or r_raw is None:
                continue
            try:
                times.append(float(t_raw))
                ratios.append(float(r_raw))
            except ValueError:
                continue
            step_val: int | None = None
            s_raw = row.get("step") or row.get("it")
            if s_raw is not None:
                try:
                    step_val = int(float(s_raw))
                except ValueError:
                    step_val = None
            extras["step"].append(step_val)
            amp_raw = row.get("m1_vperp_amp")
            if amp_raw is not None:
                try:
                    extras["m1_vperp_amp"].append(float(amp_raw))
                except ValueError:
                    extras["m1_vperp_amp"].append(float("nan"))
    audit = {
        "m1_source_path": str(path.resolve()),
        "m1_sha1": _sha1_file(path),
        "m1_filesize_bytes": None,
        "m1_mtime": None,
        "m1_ratio_field": ratio_name,
    }
    try:
        stat = path.stat()
        audit["m1_filesize_bytes"] = int(stat.st_size)
        audit["m1_mtime"] = float(stat.st_mtime)
    except Exception:
        pass
    return np.array(times, dtype=float), np.array(ratios, dtype=float), audit, extras


def linear_fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    dt = t - t_mean
    dy = y - y_mean
    var = float(np.sum(dt * dt))
    if var <= 0.0:
        return None
    slope = float(np.sum(dt * dy) / var)
    intercept = float(y_mean - slope * t_mean)
    y_pred = slope * t + intercept
    resid = y - y_pred
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    resid_std = float(np.sqrt(ss_res / max(1, y.size)))
    return slope, intercept, r2, resid_std


def fit_quality_label(r2: float | None) -> str | None:
    if r2 is None or not np.isfinite(r2):
        return None
    if r2 >= 0.9:
        return "good"
    if r2 >= 0.8:
        return "ok"
    return "low"


def load_coil_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    times: list[float] = []
    steps: list[int] = []
    phi: list[float] = []
    area: list[float] = []
    bn_avg: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t_raw = row.get("time")
            s_raw = row.get("step")
            phi_raw = row.get("phi")
            area_raw = row.get("area")
            bn_raw = row.get("bn_avg")
            if t_raw is None or s_raw is None or phi_raw is None:
                continue
            try:
                times.append(float(t_raw))
                steps.append(int(float(s_raw)))
                phi.append(float(phi_raw))
                if area_raw is not None:
                    area.append(float(area_raw))
                if bn_raw is not None:
                    bn_avg.append(float(bn_raw))
            except ValueError:
                continue
    audit = {
        "coil_source_path": str(path.resolve()),
        "coil_sha1": _sha1_file(path),
    }
    try:
        stat = path.stat()
        audit["coil_filesize_bytes"] = int(stat.st_size)
        audit["coil_mtime"] = float(stat.st_mtime)
    except Exception:
        pass
    return (
        np.array(times, dtype=float),
        np.array(steps, dtype=int),
        np.array(phi, dtype=float),
        np.array(area, dtype=float) if area else np.array([], dtype=float),
        np.array(bn_avg, dtype=float) if bn_avg else np.array([], dtype=float),
        audit,
    )


def load_energy_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    times: list[float] = []
    steps: list[int] = []
    wtot: list[float] = []
    we: list[float] = []
    wb: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
                vals = [float(v) for v in parts[2:]]
            except ValueError:
                continue
            if len(vals) % 3 != 0:
                continue
            nlev = len(vals) // 3
            total_sum = 0.0
            e_sum = 0.0
            b_sum = 0.0
            for i in range(nlev):
                total_sum += vals[3 * i + 0]
                e_sum += vals[3 * i + 1]
                b_sum += vals[3 * i + 2]
            steps.append(step)
            times.append(time)
            wtot.append(total_sum)
            we.append(e_sum)
            wb.append(b_sum)
    audit = {
        "energy_source_path": str(path.resolve()),
        "energy_sha1": _sha1_file(path),
    }
    try:
        stat = path.stat()
        audit["energy_filesize_bytes"] = int(stat.st_size)
        audit["energy_mtime"] = float(stat.st_mtime)
    except Exception:
        pass
    return (
        np.array(times, dtype=float),
        np.array(steps, dtype=int),
        np.array(wtot, dtype=float),
        np.array(we, dtype=float),
        np.array(wb, dtype=float),
        audit,
    )


def compute_derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    if n == 0 or times.size != n:
        return np.array([], dtype=float)
    deriv = np.zeros(n, dtype=float)
    if n == 1:
        return deriv
    dt = np.diff(times)
    dv = np.diff(values)
    for i in range(1, n - 1):
        dt_mid = times[i + 1] - times[i - 1]
        if dt_mid != 0.0:
            deriv[i] = (values[i + 1] - values[i - 1]) / dt_mid
    if dt[0] != 0.0:
        deriv[0] = dv[0] / dt[0]
    if dt[-1] != 0.0:
        deriv[-1] = dv[-1] / dt[-1]
    return deriv


def simulate_circuit_mvp(
    times: np.ndarray,
    phi: np.ndarray,
    turns: float,
    L_H: float,
    R_ohm: float,
    R_load_ohm: float,
    R_switch_ohm: float,
    C_F: float,
    I0_A: float,
    Q0_C: float,
    ode_substeps: int = 1,
    ode_method: str = "rk4",
    ode_interp: str = "linear",
) -> dict:
    n = int(phi.size)
    if n == 0 or times.size != n:
        return {}
    if L_H <= 0.0 or C_F <= 0.0:
        return {}
    ode_substeps = max(1, int(ode_substeps))
    ode_method = str(ode_method or "rk4").strip().lower()
    ode_interp = str(ode_interp or "linear").strip().lower()
    R_total = float(R_ohm) + float(R_load_ohm) + float(R_switch_ohm)
    dphi_dt = compute_derivative(times, phi)
    vind = -turns * dphi_dt
    I = float(I0_A)
    Q = float(Q0_C)
    I_series = [I]
    vind_series = vind.tolist()
    e_in = 0.0
    e_R = 0.0
    e_load = 0.0
    e_switch = 0.0
    for i in range(n - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0.0:
            I_series.append(I)
            continue
        dphi_k = dphi_dt[i]
        dphi_k1 = dphi_dt[i + 1]
        dphi_const = dphi_k
        if ode_interp == "linear":
            dphi_const = dphi_k

        dt_sub = dt / float(ode_substeps)
        for sub in range(ode_substeps):
            if ode_interp == "linear":
                frac = (sub + 0.5) / float(ode_substeps)
                dphi_use = dphi_k + (dphi_k1 - dphi_k) * frac
            else:
                dphi_use = dphi_const
            vind_sub = -turns * dphi_use

            def deriv(i_val: float, q_val: float) -> tuple[float, float]:
                dI = (vind_sub - R_total * i_val - q_val / C_F) / L_H
                dQ = i_val
                return dI, dQ

            if ode_method == "euler":
                dI, dQ = deriv(I, Q)
                I_next = I + dt_sub * dI
                Q_next = Q + dt_sub * dQ
            else:
                k1_I, k1_Q = deriv(I, Q)
                k2_I, k2_Q = deriv(I + 0.5 * dt_sub * k1_I, Q + 0.5 * dt_sub * k1_Q)
                k3_I, k3_Q = deriv(I + 0.5 * dt_sub * k2_I, Q + 0.5 * dt_sub * k2_Q)
                k4_I, k4_Q = deriv(I + dt_sub * k3_I, Q + dt_sub * k3_Q)
                I_next = I + (dt_sub / 6.0) * (k1_I + 2.0 * k2_I + 2.0 * k3_I + k4_I)
                Q_next = Q + (dt_sub / 6.0) * (k1_Q + 2.0 * k2_Q + 2.0 * k3_Q + k4_Q)

            I_mid = 0.5 * (I + I_next)
            e_in += -I_mid * turns * dphi_use * dt_sub
            e_R += R_ohm * I_mid * I_mid * dt_sub
            e_load += R_load_ohm * I_mid * I_mid * dt_sub
            e_switch += R_switch_ohm * I_mid * I_mid * dt_sub
            I = I_next
            Q = Q_next
        I_series.append(I)

    I_series = np.array(I_series, dtype=float)
    e_L = 0.5 * L_H * I * I
    e_C = 0.5 * (Q * Q / C_F) if C_F > 0.0 else 0.0
    e_diss = e_R + e_load + e_switch
    e_recaptured = e_in - e_diss
    e_stored_end = e_L + e_C
    denom = max(abs(e_in), 1.0e-30)
    residual_rel = abs(e_in - (e_L + e_C + e_diss)) / denom
    vind_arr = np.array(vind_series, dtype=float)
    eta_delivered = None
    eta_losses = None
    eta_recaptured = None
    eta_stored_end = None
    if abs(e_in) > 0.0:
        eta_delivered = float(e_load / e_in)
        eta_losses = float((e_R + e_switch) / e_in)
        eta_recaptured = float(e_recaptured / e_in)
        eta_stored_end = float(e_stored_end / e_in)
    out = {
        "R_total_ohm": float(R_total),
        "vind_peak_V": float(np.max(np.abs(vind_arr))) if vind_arr.size else None,
        "vind_rms_V": float(np.sqrt(np.mean(vind_arr * vind_arr))) if vind_arr.size else None,
        "i_peak_A": float(np.max(np.abs(I_series))) if I_series.size else None,
        "e_in_J": float(e_in),
        "e_cap_end_J": float(e_C),
        "e_L_end_J": float(e_L),
        "e_R_J": float(e_R),
        "e_load_J": float(e_load),
        "e_switch_J": float(e_switch),
        "e_diss_J": float(e_diss),
        "e_stored_end_J": float(e_stored_end),
        "e_recaptured_J": float(e_recaptured),
        "eta_delivered": eta_delivered,
        "eta_losses": eta_losses,
        "eta_recaptured": eta_recaptured,
        "eta_stored_end": eta_stored_end,
        "ode_substeps": int(ode_substeps),
        "ode_method": str(ode_method),
        "ode_interp": str(ode_interp),
        "energy_residual_rel": float(residual_rel),
        "vind_series": vind_arr,
        "i_series": I_series,
    }
    return out


def fit_window_log(
    times: np.ndarray,
    amps: np.ndarray,
    start_idx: int,
    window_len: int,
) -> dict | None:
    n = int(times.size)
    window_len = int(window_len)
    if n == 0 or amps.size != times.size:
        return None
    if start_idx < 0 or start_idx + window_len > n or window_len < 2:
        return None
    a_slice = amps[start_idx : start_idx + window_len]
    if np.any(~np.isfinite(a_slice)) or np.any(a_slice <= 0.0):
        return None
    t_slice = times[start_idx : start_idx + window_len]
    fit = linear_fit(t_slice, np.log(a_slice))
    if fit is None:
        return None
    slope, _intercept, r2, resid_std = fit
    return {
        "gamma": slope,
        "r2": r2,
        "resid_std": resid_std,
        "start_idx": int(start_idx),
        "end_idx": int(start_idx + window_len - 1),
    }


def scan_best_window_log(
    times: np.ndarray,
    amps: np.ndarray,
    start_min: int,
    window_len: int,
) -> dict | None:
    n = int(times.size)
    if n == 0 or amps.size != times.size:
        return None
    window_len = int(window_len)
    max_start = n - window_len
    best = None
    for idx in range(int(start_min), max_start + 1):
        fit = fit_window_log(times, amps, idx, window_len)
        if fit is None:
            continue
        if best is None:
            best = fit
            continue
        if fit["r2"] > best["r2"] or (
            np.isfinite(fit["r2"]) and np.isfinite(best["r2"]) and fit["r2"] == best["r2"]
            and fit["start_idx"] < best["start_idx"]
        ):
            best = fit
    return best


def compute_tail_floor(values: np.ndarray, tail_count: int) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    tail = finite[-tail_count:] if finite.size >= tail_count else finite
    if tail.size == 0:
        return None, None
    return float(np.median(tail)), float(np.mean(tail))


def compute_start_idx(
    times: np.ndarray,
    merge_time: float | None,
    guard_dt: float | None,
    window_len: int,
) -> int:
    if times.size == 0:
        return 0
    if merge_time is not None:
        start_time = merge_time + (guard_dt or 0.0)
        if np.isfinite(start_time):
            idx = np.where(times >= start_time)[0]
            if idx.size:
                return int(idx[0])
    return min(8, max(0, int(times.size) - int(window_len)))


def best_window_fit(
    times: np.ndarray,
    amps: np.ndarray,
    window_len: int,
    start_idx: int,
    r2_threshold: float,
    floor_median: float | None = None,
    floor_factor: float | None = None,
    start_idx_cap: int | None = None,
) -> dict:
    result = {
        "r2_fit_len24_best": None,
        "gamma_fit_len24_best": None,
        "fit_window_start_idx": None,
        "fit_window_end_idx": None,
        "fit_window_len": None,
        "fit_window_len_forced": None,
        "fit_window_found": False,
        "num_candidate_windows_len24": 0,
        "num_windows_above_floor": None,
        "fit_window_start_idx_cap": None,
        "residual_std_best": None,
    }
    result["fit_window_len_forced"] = int(window_len)
    if start_idx_cap is not None:
        result["fit_window_start_idx_cap"] = int(start_idx_cap)
    if times.size == 0 or amps.size != times.size:
        return result
    window_len = max(2, int(window_len))
    start_idx = max(0, int(start_idx))
    n = int(times.size)
    max_start = n - window_len
    if start_idx_cap is not None:
        max_start = min(max_start, int(start_idx_cap))
    if start_idx >= n or start_idx > max_start or n - start_idx < window_len:
        return result
    best_r2 = None
    if floor_median is not None and floor_factor is not None and np.isfinite(floor_median):
        if floor_median <= 0.0 or floor_factor <= 0.0:
            floor_median = None
    if floor_median is not None and floor_factor is not None:
        result["num_windows_above_floor"] = 0
    for i in range(start_idx, max_start + 1):
        j = i + window_len - 1
        a_slice = amps[i : j + 1]
        if np.any(~np.isfinite(a_slice)) or np.any(a_slice <= 0.0):
            continue
        if floor_median is not None and floor_factor is not None:
            window_median = float(np.median(a_slice))
            if not np.isfinite(window_median):
                continue
            if window_median < float(floor_factor) * float(floor_median):
                continue
            result["num_windows_above_floor"] += 1
        result["num_candidate_windows_len24"] += 1
        t_slice = times[i : j + 1]
        y_slice = np.log(a_slice)
        fit = linear_fit(t_slice, y_slice)
        if fit is None:
            continue
        slope, _intercept, r2, resid_std = fit
        if (best_r2 is None) or (r2 > best_r2):
            best_r2 = r2
            result.update(
                {
                    "r2_fit_len24_best": r2,
                    "gamma_fit_len24_best": slope,
                    "fit_window_start_idx": int(i),
                    "fit_window_end_idx": int(j),
                    "fit_window_len": int(window_len),
                    "residual_std_best": resid_std,
                }
            )
    if best_r2 is not None and np.isfinite(best_r2):
        result["fit_window_found"] = bool(best_r2 >= float(r2_threshold))
    return result


def best_window_fit_retry(
    times: np.ndarray,
    amps: np.ndarray,
    window_len_list: list[int],
    merge_time: float | None,
    guard_dt: float | None,
    r2_threshold: float,
    min_fit_points: int,
    floor_median: float | None = None,
    floor_factor: float | None = None,
    start_idx_cap: int | None = None,
    start_idx_min: int | None = None,
) -> tuple[dict, dict]:
    meta = {
        "fit_window_len_list": [int(v) for v in window_len_list],
        "fit_window_min_points": int(min_fit_points),
        "fit_window_retry_count": 0,
        "fit_window_retry_reason": None,
    }
    last = None
    last_reason = None
    for idx, win_len in enumerate(window_len_list):
        if win_len < min_fit_points:
            continue
        start_idx = compute_start_idx(times, merge_time, guard_dt, win_len)
        if start_idx_min is not None:
            start_idx = int(start_idx_min)
        meta["fit_window_retry_count"] = int(idx)
        result = best_window_fit(
            times,
            amps,
            win_len,
            start_idx,
            r2_threshold,
            floor_median=floor_median,
            floor_factor=floor_factor,
            start_idx_cap=start_idx_cap,
        )
        last = result
        if result.get("fit_window_found"):
            meta["fit_window_retry_reason"] = None
            return result, meta
        # classify reason
        if times.size < win_len or start_idx > (times.size - win_len):
            last_reason = "INSUFFICIENT_POINTS"
        elif result.get("num_candidate_windows_len24", 0) == 0:
            last_reason = "WINDOW_OUT_OF_RANGE"
        else:
            last_reason = "LOW_R2"
        meta["fit_window_retry_reason"] = last_reason
    if last is None:
        last = best_window_fit(times, amps, window_len_list[0], 0, r2_threshold)
    if meta["fit_window_retry_reason"] is None:
        meta["fit_window_retry_reason"] = last_reason
    return last, meta


def compute_guard_dt(meta: dict, h3a: dict) -> float | None:
    args = meta.get("args") or {}
    dt = args.get("dt")
    guard_outputs = h3a.get("guard_outputs")
    diag_period = args.get("diag_period", 1)
    if dt is None or guard_outputs is None:
        return None
    try:
        return float(dt) * max(1, int(diag_period)) * max(0, int(guard_outputs))
    except (TypeError, ValueError):
        return None


def compute_drop_breach(meta: dict, start_time: float | None) -> bool | None:
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []
    drop_threshold = monitor.get("drop_threshold")
    if drop_threshold is None:
        drop_threshold = (meta.get("args") or {}).get("drop_threshold")
    if drop_threshold is None or not records:
        return None
    try:
        threshold = float(drop_threshold)
    except (TypeError, ValueError):
        return None
    for rec in records:
        delta = rec.get("dropped_delta")
        if delta is None:
            continue
        t = rec.get("time")
        if start_time is not None and t is not None and t < start_time:
            continue
        if delta > threshold:
            return True
    return False


def find_m1rho_file(warpx_meta_path: Path | None) -> Path | None:
    if warpx_meta_path is None:
        return None
    run_dir = warpx_meta_path.parent
    candidates = [
        run_dir / "diags/reducedfiles/M1RHO.txt",
        run_dir / "diags/reducedfilesM1RHO.txt",
        run_dir / "diag/diags/reducedfiles/M1RHO.txt",
        run_dir / "diag/diags/reducedfilesM1RHO.txt",
        run_dir / "diag/reducedfiles/M1RHO.txt",
        run_dir / "diag/reducedfilesM1RHO.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_m1mom_file(warpx_meta_path: Path | None) -> Path | None:
    if warpx_meta_path is None:
        return None
    run_dir = warpx_meta_path.parent
    candidates = [
        run_dir / "diags/reducedfiles/M1MOM.txt",
        run_dir / "diags/reducedfilesM1MOM.txt",
        run_dir / "diag/diags/reducedfiles/M1MOM.txt",
        run_dir / "diag/diags/reducedfilesM1MOM.txt",
        run_dir / "diag/reducedfiles/M1MOM.txt",
        run_dir / "diag/reducedfilesM1MOM.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_particle_vel_stats_file(warpx_meta_path: Path | None, warpx_meta: dict | None) -> Path | None:
    if warpx_meta:
        stats_meta = warpx_meta.get("particle_vel_stats") or {}
        stats_path = stats_meta.get("path")
        if stats_path:
            candidate = Path(stats_path)
            if candidate.exists():
                return candidate
    if warpx_meta_path is None:
        return None
    run_dir = warpx_meta_path.parent
    candidate = run_dir / "particle_vel_stats.csv"
    if candidate.exists():
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge H5 merge+tilt metrics with gamma fit metrics.")
    parser.add_argument("--h3a-metrics", required=True, help="Metrics JSON from analyze_h3a_merge_tilt_case.py")
    parser.add_argument("--h3b-metrics", required=True, help="Metrics JSON from analyze_h3b_growth_fit.py")
    parser.add_argument("--warpx-meta", default=None, help="WarpX metadata JSON (optional).")
    parser.add_argument("--metrics-out", required=True, help="Combined metrics JSON output.")
    args = parser.parse_args()

    h3a_path = Path(args.h3a_metrics)
    h3b_path = Path(args.h3b_metrics)
    case_id = None
    outputs_root = None
    try:
        case_id = h3a_path.parents[1].name
        outputs_root = h3a_path.parents[2]
    except IndexError:
        case_id = None
        outputs_root = None
    h3a = load_json(h3a_path)
    h3b = load_json(h3b_path)
    warpx_meta_path = Path(args.warpx_meta) if args.warpx_meta else None
    warpx_meta = load_warpx_meta(warpx_meta_path) if warpx_meta_path else {}
    heartbeat = {}
    heartbeat_path = None
    if warpx_meta_path is not None:
        heartbeat_path = warpx_meta_path.with_name("warpx_heartbeat.json")
    heartbeat_file = load_heartbeat(heartbeat_path)
    if warpx_meta:
        heartbeat.update(warpx_meta.get("heartbeat") or {})
    if heartbeat_file:
        heartbeat.update(heartbeat_file)

    merge_time_seedmask = h3a.get("merge_time_seedmask")
    merge_time_exists_seedmask = bool(h3a.get("merge_time_exists_seedmask"))
    merge_time_frac_seedmask = h3a.get("merge_time_frac_seedmask")
    merge_time = None
    merge_time_exists = False
    merge_time_frac = None
    merge_time_proxy = None
    merge_fallback_reason = None
    t_at_s_min = h3a.get("t_at_s_min")
    t_end = h3a.get("t_end") or h3b.get("t_end")

    formation_metrics = _load_json_if_exists(h3a_path.parent / "metrics_formation.json") or {}
    dt = 0.0
    try:
        dt = float((warpx_meta.get("args") or {}).get("dt") or 0.0)
    except (TypeError, ValueError):
        dt = 0.0
    t_end_val = float(t_end) if t_end is not None else 0.0
    merge_candidates = _merge_time_candidates(h3a, formation_metrics)
    sel = _select_merge_time(merge_candidates, dt=dt, t_end=t_end_val)
    merge_time = sel["merge_time"]
    merge_time_exists = merge_time is not None
    merge_time_source = sel["merge_time_source"]
    merge_time_conflict = bool(sel["merge_time_conflict"])
    merge_time_confidence = sel["merge_time_confidence"]
    merge_time_proxy_used = bool(sel["merge_time_proxy_used"])
    merge_time_consensus_dt = sel["merge_time_consensus_dt"]
    if merge_time_exists and t_end_val > 0.0:
        merge_time_frac = float(merge_time / t_end_val)
    if merge_time_proxy_used and merge_time is not None:
        merge_time_proxy = merge_time
        merge_fallback_reason = f"merge_time_proxy:{merge_time_source}"
    elif not merge_time_exists:
        merge_fallback_reason = "merge_time_missing"
    elif merge_time_conflict:
        merge_fallback_reason = "merge_time_conflict"

    guard_dt = compute_guard_dt(warpx_meta, h3a)
    if merge_time_exists and merge_time is not None:
        start_time = merge_time + (guard_dt or 0.0)
        window_label = "post_merge"
    elif merge_time_proxy is not None:
        start_time = merge_time_proxy + (guard_dt or 0.0)
        window_label = "proxy"
    else:
        start_time = None
        window_label = "full"
    drop_breach = compute_drop_breach(warpx_meta, start_time)
    drop_breach_source = None
    drop_breach_heartbeat_step = None
    drop_breach_heartbeat_time = None
    drop_breach_heartbeat_diag_dir = None
    drop_breach_proxy_metric_name = None
    if drop_breach is not None:
        drop_breach_source = f"monitor_{window_label}" if window_label != "full" else "monitor"
    else:
        heartbeat_step = warpx_meta.get("heartbeat_last_step", heartbeat.get("last_step"))
        heartbeat_time = warpx_meta.get("heartbeat_last_time", heartbeat.get("last_time"))
        heartbeat_diag_dir = warpx_meta.get(
            "heartbeat_last_diag_dir", heartbeat.get("last_diag_dir")
        )
        heartbeat_drop_breach = warpx_meta.get(
            "heartbeat_monitor_drop_breach", heartbeat.get("monitor_drop_breach")
        )
        if heartbeat_drop_breach is not None:
            drop_breach = bool(heartbeat_drop_breach)
            drop_breach_source = (
                f"heartbeat_{window_label}" if window_label != "full" else "heartbeat"
            )
            drop_breach_heartbeat_step = heartbeat_step
            drop_breach_heartbeat_time = heartbeat_time
            drop_breach_heartbeat_diag_dir = heartbeat_diag_dir
            drop_breach_proxy_metric_name = "monitor_drop_breach"
        else:
            fallback = h3a.get("drop_breach")
            if fallback is not None:
                drop_breach = fallback
                drop_breach_source = "h3a_monitor"
            else:
                drop_breach_source = "missing_monitor_and_heartbeat"

    initial_asymmetry_metric = None
    initial_asymmetry_source = None
    if warpx_meta:
        opmd_double_seed = warpx_meta.get("opmd_double_seed") or {}
        drift_meta = opmd_double_seed.get("drift_meta") or warpx_meta.get("drift_meta")
        if drift_meta:
            sep_vec = drift_meta.get("com_sep_vec")
            if isinstance(sep_vec, (list, tuple)) and len(sep_vec) >= 3:
                try:
                    initial_asymmetry_metric = float(
                        np.hypot(float(sep_vec[1]), float(sep_vec[2]))
                    )
                    initial_asymmetry_source = "drift_meta.com_sep_vec_transverse"
                except (TypeError, ValueError):
                    initial_asymmetry_metric = None
            if initial_asymmetry_metric is None:
                com_a = drift_meta.get("com_a")
                com_b = drift_meta.get("com_b")
                if isinstance(com_a, (list, tuple)) and isinstance(com_b, (list, tuple)):
                    if len(com_a) >= 3 and len(com_b) >= 3:
                        try:
                            dy = float(com_a[1]) - float(com_b[1])
                            dz = float(com_a[2]) - float(com_b[2])
                            initial_asymmetry_metric = float(np.hypot(dy, dz))
                            initial_asymmetry_source = "drift_meta.com_a_com_b_transverse"
                        except (TypeError, ValueError):
                            initial_asymmetry_metric = None
    if initial_asymmetry_metric is None:
        initial_asymmetry_metric = h3b.get("tilt_amp_initial")
        if initial_asymmetry_metric is not None:
            initial_asymmetry_source = "tilt_amp_initial"
    if initial_asymmetry_metric is None:
        initial_asymmetry_metric = h3a.get("tilt_post_merge_amp_max")
        if initial_asymmetry_metric is not None:
            initial_asymmetry_source = "tilt_post_merge_amp_max"
    tilt_seed_enabled = None
    dynamic_drift_enabled = None

    combined = {
        "case_id": case_id,
        "ran_to_completion": h3a.get("ran_to_completion"),
        "num_outputs": h3a.get("num_outputs"),
        "no_nan_in_metrics": bool(h3a.get("no_nan_in_metrics", True) and h3b.get("no_nan_in_metrics", True)),
        "drop_breach": drop_breach,
        "drop_breach_source": drop_breach_source,
        "drop_breach_heartbeat_step": drop_breach_heartbeat_step,
        "drop_breach_heartbeat_time": drop_breach_heartbeat_time,
        "drop_breach_heartbeat_diag_dir": drop_breach_heartbeat_diag_dir,
        "drop_breach_proxy_metric_name": drop_breach_proxy_metric_name,
        "merge_time": merge_time,
        "merge_time_exists": merge_time_exists,
        "merge_time_frac": merge_time_frac,
        "merge_time_proxy": merge_time_proxy,
        "merge_time_ok": bool(merge_time_exists or merge_time_proxy is not None),
        "merge_time_candidates": merge_candidates,
        "merge_time_source": merge_time_source,
        "merge_time_conflict": merge_time_conflict,
        "merge_time_confidence": merge_time_confidence,
        "merge_time_proxy_used": merge_time_proxy_used,
        "merge_time_consensus_dt": merge_time_consensus_dt,
        "merge_time_seedmask": merge_time_seedmask,
        "merge_time_exists_seedmask": merge_time_exists_seedmask,
        "merge_time_frac_seedmask": merge_time_frac_seedmask,
        "merge_fallback_reason": merge_fallback_reason,
        "merge_sep_thresh": h3a.get("merge_sep_thresh"),
        "merge_sep_frac": h3a.get("merge_sep_frac"),
        "merge_by": h3a.get("merge_by"),
        "s_start": h3a.get("s_start"),
        "s_min": h3a.get("s_min"),
        "t_at_s_min": h3a.get("t_at_s_min"),
        "t_end": t_end,
        "com_sep_start": h3a.get("com_sep_start"),
        "com_sep_end": h3a.get("com_sep_end"),
        "com_sep_delta": h3a.get("com_sep_delta"),
        "com_sep_source": h3a.get("com_sep_source"),
        "seedmask_source": h3a.get("seedmask_source"),
        "kmeans_inertia_ratio_min": h3a.get("kmeans_inertia_ratio_min"),
        "merge_detector_used": h3a.get("merge_detector_used"),
        "tilt_post_merge_samples": h3a.get("tilt_post_merge_samples"),
        "tilt_post_merge_amp_max": h3a.get("tilt_post_merge_amp_max"),
        "tilt_post_merge_no_nan": h3a.get("tilt_post_merge_no_nan"),
        "tilt_metric_used": h3a.get("tilt_metric_used") or h3b.get("tilt_metric_used"),
        "tilt_slope_bins_min": h3a.get("tilt_slope_bins_min"),
        "tilt_slope_bins_max": h3a.get("tilt_slope_bins_max"),
        "tilt_slope_bins_mean": h3a.get("tilt_slope_bins_mean"),
        "gamma_best": h3b.get("gamma_best"),
        "gamma_fit": h3b.get("gamma_best"),
        "r2_best": h3b.get("r2_best"),
        "r2_fit": h3b.get("r2_best"),
        "gamma_fit_best": h3b.get("gamma_fit_best") or h3b.get("gamma_best"),
        "r2_fit_best": h3b.get("r2_fit_best") or h3b.get("r2_best"),
        "residual_std_best": h3b.get("residual_std_best"),
        "fit_points": h3b.get("fit_points"),
        "fit_found": h3b.get("fit_found"),
        "amp_ratio_fit_best": h3b.get("amp_ratio_fit_best"),
        "fit_window_time0": h3b.get("fit_window_time0") or h3b.get("fit_start_time"),
        "fit_window_time1": h3b.get("fit_window_time1") or h3b.get("fit_end_time"),
        "fit_window_start_time": h3b.get("fit_window_start_time") or h3b.get("fit_start_time"),
        "fit_window_end_time": h3b.get("fit_window_end_time") or h3b.get("fit_end_time"),
        "fit_window_start_idx": h3b.get("fit_window_start_idx"),
        "fit_window_end_idx": h3b.get("fit_window_end_idx"),
        "fit_window_len": h3b.get("fit_window_len"),
        "fit_window_strategy": None,
        "fit_window_len_list": None,
        "fit_window_min_points": None,
        "fit_window_retry_count": None,
        "fit_window_retry_reason": None,
        "tilt_amp_series_len": h3b.get("tilt_amp_series_len") or h3b.get("series_samples"),
        "tilt_amp_initial": h3b.get("tilt_amp_initial"),
        "tilt_amp_final": h3b.get("tilt_amp_final"),
        "tilt_amp_ratio": h3b.get("tilt_amp_ratio"),
        "initial_asymmetry_metric": initial_asymmetry_metric,
        "initial_asymmetry_source": initial_asymmetry_source,
        "tilt_cross_ratio_used": h3b.get("tilt_cross_ratio_used"),
        "tilt_cross_exists": h3b.get("tilt_cross_exists"),
        "tilt_cross_time": h3b.get("tilt_cross_time"),
        "tilt_cross_index": h3b.get("tilt_cross_index"),
        "tilt_cross_amp0": h3b.get("tilt_cross_amp0"),
        "tilt_cross_thresh": h3b.get("tilt_cross_thresh"),
        "tilt_seed_enabled": tilt_seed_enabled,
        "dynamic_drift_enabled": dynamic_drift_enabled,
        "drive_timing_value": None,
        "drive_applied_steps": None,
        "drive_num_applied": None,
        "dv_abs_applied_each": None,
        "dv_abs_linf_each": None,
        "repeat_applied_steps": None,
        "num_cells_modified_mean": None,
        "rho_clip_fraction": None,
        "m1_rho_cos_eps": None,
        "m1_rho_cos_nsteps": None,
        "effective_eta_scale": None,
        "requested_eta_scale": None,
        "plasma_resistivity_scale": None,
        "m1_series_kind": None,
        "m1_ratio_series_len": None,
        "m1_ratio_series_path": None,
        "m1_source_path": None,
        "m1_sha1": None,
        "m1_filesize_bytes": None,
        "m1_mtime": None,
        "m1_first3": None,
        "m1_last3": None,
        "m1_ratio_initial": None,
        "m1_ratio_last": None,
        "m1_ratio_mean": None,
        "m1_ratio_min": None,
        "m1_ratio_max": None,
        "m1_ratio_delta": None,
        "m1_ratio_mean_abs_slope": None,
        "m1v_series_len": None,
        "m1v_ratio_initial": None,
        "m1v_ratio_last": None,
        "m1v_ratio_mean": None,
        "m1v_ratio_min": None,
        "m1v_ratio_max": None,
        "m1v_ratio_delta": None,
        "m1v_ratio_mean_abs_slope": None,
        "m1v_source_path": None,
        "m1v_sha1": None,
        "m1v_floor_tail_median": None,
        "m1v_floor_tail_mean": None,
        "m1v_thr": None,
        "m1v_idx_cross": None,
        "m1v_prefix_len": None,
        "gamma_m1v_fit_prefix": None,
        "r2_m1v_fit_prefix": None,
        "m1v_floor": None,
        "m1v_floor_source": None,
        "m1v_peak": None,
        "m1v_peak_signal": None,
        "m1v_peak_step": None,
        "m1v_peak_time": None,
        "m1v_last": None,
        "m1v_idx_tau_e": None,
        "m1v_tau_e_steps": None,
        "m1v_tau_e_time": None,
        "m1v_idx_t_half": None,
        "m1v_t_half_steps": None,
        "m1v_t_half_time": None,
        "m1v_decay_rate_fit": None,
        "m1v_decay_rate_r2": None,
        "num_windows_above_floor": None,
        "fit_window_start_idx_cap": None,
        "m0_mom_amp_min": None,
        "m0_mom_amp_mean": None,
        "m0_rho_abs_min": None,
        "m0_rho_abs_mean": None,
        "m1_mom_ratio_series_len": None,
        "m1_mom_ratio_initial": None,
        "m1_mom_ratio_last": None,
        "m1_mom_ratio_mean": None,
        "m1_mom_ratio_min": None,
        "m1_mom_ratio_max": None,
        "m1_mom_ratio_delta": None,
        "m1_mom_ratio_mean_abs_slope": None,
        "m1_mom_norm_kind": None,
        "gamma_m1_fit_best": None,
        "r2_m1_fit_best": None,
        "gamma_m1v_fit_best": None,
        "r2_m1v_fit_best": None,
        "gamma_m1mom_fit_best": None,
        "r2_m1mom_fit_best": None,
        "fit_series_source": None,
        "tilt_amp_series_len_tilt": h3b.get("tilt_amp_series_len") or h3b.get("series_samples"),
        "tilt_amp_series_len_source": None,
        "h3a_metrics_path": str(h3a_path),
        "h3b_metrics_path": str(h3b_path),
    }

    args_cfg: dict = {}
    args_fallback: dict = {}
    seed_fallback: dict = {}
    try:
        output_root = h3b_path.parent.parent
        args_fallback, seed_fallback = load_output_inputs(output_root)
    except Exception:
        args_fallback, seed_fallback = {}, {}
    if warpx_meta:
        b_apply = warpx_meta.get("bfield_apply") or {}
        if b_apply:
            combined["bfield_apply_applied"] = b_apply.get("applied")
            combined["bfield_apply_wrapper"] = b_apply.get("wrapper")
        resistivity = warpx_meta.get("resistivity") or {}
        if resistivity:
            combined["effective_eta_scale"] = resistivity.get("effective_eta_scale")
            combined["requested_eta_scale"] = resistivity.get("requested_eta_scale")
            combined["plasma_resistivity_scale"] = resistivity.get("plasma_resistivity_scale")
        drive_timing_value = None
        args_cfg = warpx_meta.get("args") or {}
        cmd_args_cfg = load_args_from_command(warpx_meta.get("command"))
        if cmd_args_cfg:
            merged_args = dict(args_cfg)
            merged_args.update(cmd_args_cfg)
            args_cfg = merged_args
        if args_cfg:
            if "ext_drive_start_step" in args_cfg:
                drive_timing_value = args_cfg.get("ext_drive_start_step")
            dynamic_drift_enabled = bool(args_cfg.get("opmd_double_seed_drift_dynamic", False))
            combined["dynamic_drift_enabled"] = dynamic_drift_enabled
            drive_amp = args_cfg.get("driveAmp_scale")
            if drive_amp is None:
                drive_amp = args_cfg.get("opmd_b_scale")
            combined["driveAmp_effective"] = drive_amp
        b_apply = warpx_meta.get("bfield_apply") or {}
        if drive_timing_value is None and b_apply:
            drive_timing_value = b_apply.get("start_step") or b_apply.get("requested_start_step")
        combined["drive_timing_value"] = drive_timing_value
        m1_drive = warpx_meta.get("m1_drive") or {}
        if m1_drive:
            combined["drive_applied_steps"] = m1_drive.get("drive_applied_steps")
            combined["drive_num_applied"] = m1_drive.get("drive_num_applied")
            combined["dv_abs_applied_each"] = m1_drive.get("dv_abs_applied_each")
            combined["dv_abs_linf_each"] = m1_drive.get("dv_abs_linf_each")
        m1_rho_drive = warpx_meta.get("m1_rho_cos_drive") or {}
        if m1_rho_drive:
            combined["repeat_applied_steps"] = m1_rho_drive.get("repeat_applied_steps")
            combined["num_cells_modified_mean"] = m1_rho_drive.get("num_particles_modified_mean")
            combined["rho_clip_fraction"] = m1_rho_drive.get("rho_clip_fraction_mean")
            combined["m1_rho_cos_eps"] = m1_rho_drive.get("eps")
            combined["m1_rho_cos_nsteps"] = m1_rho_drive.get("nsteps")
        opmd_double_seed = warpx_meta.get("opmd_double_seed") or {}
        tilt_seed = opmd_double_seed.get("tilt_seed") or {}
        if tilt_seed:
            combined["tilt_seed_mode"] = tilt_seed.get("mode")
            combined["tilt_seed_y_offset_amp"] = tilt_seed.get("y_offset_amp")
            combined["tilt_seed_y_offset_profile"] = tilt_seed.get("y_offset_profile")
            combined["tilt_seed_y_offset_params"] = tilt_seed.get("y_offset_params")
            combined["tilt_seed_vkick_frac"] = tilt_seed.get("vkick_frac")
            combined["tilt_seed"] = tilt_seed
            applied = tilt_seed.get("applied")
            if applied is None:
                mode = tilt_seed.get("mode")
                if mode is not None:
                    tilt_seed_enabled = str(mode).strip().lower() not in ("none", "off", "")
            else:
                tilt_seed_enabled = bool(applied)
            combined["tilt_seed_enabled"] = tilt_seed_enabled
        elif args_cfg:
            mode = args_cfg.get("tilt_seed_mode")
            if mode is not None:
                tilt_seed_enabled = str(mode).strip().lower() not in ("none", "off", "")
                combined["tilt_seed_enabled"] = tilt_seed_enabled

    if dynamic_drift_enabled is None:
        dynamic_flag = None
        for cfg in (args_cfg, args_fallback, seed_fallback):
            if cfg and "opmd_double_seed_drift_dynamic" in cfg:
                dynamic_flag = cfg.get("opmd_double_seed_drift_dynamic")
                break
        if dynamic_flag is not None:
            dynamic_drift_enabled = bool(dynamic_flag)
            combined["dynamic_drift_enabled"] = dynamic_drift_enabled

    if tilt_seed_enabled is None:
        mode = None
        for cfg in (args_cfg, args_fallback, seed_fallback):
            if cfg and "tilt_seed_mode" in cfg:
                mode = cfg.get("tilt_seed_mode")
                break
        if mode is not None:
            tilt_seed_enabled = str(mode).strip().lower() not in ("none", "off", "")
            combined["tilt_seed_enabled"] = tilt_seed_enabled

    if combined.get("ran_to_completion") is None and warpx_meta:
        max_steps = None
        if args_cfg:
            max_steps = args_cfg.get("max_steps")
        last_step = warpx_meta.get("heartbeat_last_step", heartbeat.get("last_step"))
        if max_steps is not None and last_step is not None:
            try:
                combined["ran_to_completion"] = int(last_step) >= int(max_steps) - 1
            except (TypeError, ValueError):
                pass

    forced_len = 16
    if case_id is not None and "hires" not in str(case_id):
        forced_len = 24
    r2_threshold = 0.9
    floor_tail_n = 8
    floor_factor = 5.0
    fit_window_start_idx_cap = 8
    min_fit_points_prefix = 6

    series_path = h3b.get("series_path")
    series_file = None
    if series_path:
        candidate = Path(series_path)
        if candidate.exists():
            series_file = candidate
    if series_file is None:
        candidate = h3b_path.parent / "tilt_post_merge_series.csv"
        if candidate.exists():
            series_file = candidate

    tilt_times = np.array([], dtype=float)
    tilt_amps = np.array([], dtype=float)
    if series_file is not None:
        tilt_times, tilt_amps = load_tilt_series(series_file)
        if tilt_times.size == 0:
            fallback = series_file.with_name("tilt_fit_series.csv")
            if fallback.exists():
                tilt_times, tilt_amps = load_tilt_series(fallback)

    if tilt_times.size:
        combined["tilt_amp_series_len_tilt"] = int(tilt_amps.size)
        tilt_start_idx = compute_start_idx(tilt_times, merge_time, guard_dt, forced_len)
        tilt_best = best_window_fit(tilt_times, tilt_amps, forced_len, tilt_start_idx, r2_threshold)
        combined["tilt_r2_fit_len24_best"] = tilt_best.get("r2_fit_len24_best")
        combined["tilt_gamma_fit_len24_best"] = tilt_best.get("gamma_fit_len24_best")
        combined["tilt_fit_window_start_idx"] = tilt_best.get("fit_window_start_idx")
        combined["tilt_fit_window_end_idx"] = tilt_best.get("fit_window_end_idx")
        combined["tilt_fit_window_len"] = tilt_best.get("fit_window_len")
        combined["tilt_fit_window_found"] = tilt_best.get("fit_window_found")
        combined["tilt_num_candidate_windows_len24"] = tilt_best.get("num_candidate_windows_len24")
        combined["tilt_residual_std_best"] = tilt_best.get("residual_std_best")

    m1_kind = None
    m1_extras: dict[str, list[float]] = {}
    m1_path = find_particle_vel_stats_file(warpx_meta_path, warpx_meta)
    if m1_path is not None:
        m1_kind = "m1v"
        m1_times, m1_ratio, m1_audit, m1_extras = load_particle_vel_stats_series(m1_path)
    else:
        m1_path = find_m1mom_file(warpx_meta_path)
        if m1_path is not None:
            m1_kind = "mom"
            m1_times, m1_ratio, m1_audit, m1_extras = load_m1mom_series(m1_path)
        else:
            m1_path = find_m1rho_file(warpx_meta_path)
            if m1_path is not None:
                m1_kind = "rho"
                m1_times, m1_ratio, m1_audit = load_m1rho_series(m1_path)
            else:
                m1_times, m1_ratio, m1_audit = np.array([], dtype=float), np.array([], dtype=float), {}

    m1_floor_median = None
    m1_floor_mean = None
    if m1_path is not None:
        if m1_kind == "mom":
            combined["m1_series_kind"] = "m1mom"
        elif m1_kind == "m1v":
            combined["m1_series_kind"] = "m1v"
        else:
            combined["m1_series_kind"] = "m1rho"
        combined["m1_ratio_series_path"] = str(m1_path)
        combined["m1_source_path"] = m1_audit.get("m1_source_path")
        combined["m1_sha1"] = m1_audit.get("m1_sha1")
        combined["m1_filesize_bytes"] = m1_audit.get("m1_filesize_bytes")
        combined["m1_mtime"] = m1_audit.get("m1_mtime")
        combined["m1_ratio_series_len"] = int(m1_ratio.size)
        if m1_kind == "m1v":
            combined["m1v_series_len"] = int(m1_ratio.size)
            combined["m1v_source_path"] = m1_audit.get("m1_source_path")
            combined["m1v_sha1"] = m1_audit.get("m1_sha1")
        if m1_kind == "mom":
            combined["m1_mom_ratio_series_len"] = int(m1_ratio.size)
            ratio_field = (m1_audit.get("m1_ratio_field") or "").lower()
            if "ratio_b" in ratio_field:
                combined["m1_mom_norm_kind"] = "B"
            elif "ratio_a" in ratio_field or "m1_mom_ratio" in ratio_field:
                combined["m1_mom_norm_kind"] = "A"
        if m1_ratio.size:
            m1_floor_median, m1_floor_mean = compute_tail_floor(m1_ratio, floor_tail_n)
            if m1_kind == "m1v":
                combined["m1v_floor_tail_median"] = m1_floor_median
                combined["m1v_floor_tail_mean"] = m1_floor_mean
            combined["m1_first3"] = [float(v) for v in m1_ratio[:3]]
            combined["m1_last3"] = [float(v) for v in m1_ratio[-3:]]
            combined["m1_ratio_initial"] = float(m1_ratio[0])
            combined["m1_ratio_last"] = float(m1_ratio[-1])
            combined["m1_ratio_mean"] = float(np.mean(m1_ratio))
            combined["m1_ratio_min"] = float(np.min(m1_ratio))
            combined["m1_ratio_max"] = float(np.max(m1_ratio))
            combined["m1_ratio_delta"] = float(m1_ratio[0] - m1_ratio[-1])
            if m1_ratio.size > 1 and m1_times.size == m1_ratio.size:
                dt = np.diff(m1_times)
                dy = np.diff(m1_ratio)
                valid = dt != 0.0
                if np.any(valid):
                    mean_abs_slope = float(np.mean(np.abs(dy[valid] / dt[valid])))
                    combined["m1_ratio_mean_abs_slope"] = mean_abs_slope
                    if m1_kind == "m1v":
                        combined["m1v_ratio_mean_abs_slope"] = mean_abs_slope
            if m1_kind == "m1v":
                combined["m1v_ratio_initial"] = combined["m1_ratio_initial"]
                combined["m1v_ratio_last"] = combined["m1_ratio_last"]
                combined["m1v_ratio_mean"] = combined["m1_ratio_mean"]
                combined["m1v_ratio_min"] = combined["m1_ratio_min"]
                combined["m1v_ratio_max"] = combined["m1_ratio_max"]
                combined["m1v_ratio_delta"] = combined["m1_ratio_delta"]
            if m1_kind == "mom":
                combined["m1_mom_ratio_series_len"] = int(m1_ratio.size)
                combined["m1_mom_ratio_initial"] = combined["m1_ratio_initial"]
                combined["m1_mom_ratio_last"] = combined["m1_ratio_last"]
                combined["m1_mom_ratio_mean"] = combined["m1_ratio_mean"]
                combined["m1_mom_ratio_min"] = combined["m1_ratio_min"]
                combined["m1_mom_ratio_max"] = combined["m1_ratio_max"]
                combined["m1_mom_ratio_delta"] = combined["m1_ratio_delta"]
                combined["m1_mom_ratio_mean_abs_slope"] = combined["m1_ratio_mean_abs_slope"]
                for key, out_min, out_mean in (
                    ("m0_mom_amp", "m0_mom_amp_min", "m0_mom_amp_mean"),
                    ("m0_rho_abs", "m0_rho_abs_min", "m0_rho_abs_mean"),
                ):
                    values = m1_extras.get(key) or []
                    if values:
                        arr = np.array(values, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            combined[out_min] = float(np.min(arr))
                            combined[out_mean] = float(np.mean(arr))
            if m1_kind == "m1v":
                if m1_floor_median is not None and combined.get("m1v_floor") is None:
                    combined["m1v_floor"] = m1_floor_median
                if combined.get("m1v_peak") is None:
                    peak_idx = int(np.nanargmax(m1_ratio))
                    combined["m1v_peak"] = float(m1_ratio[peak_idx])
                    steps = m1_extras.get("step") or []
                    if len(steps) == m1_ratio.size and steps[peak_idx] is not None:
                        combined["m1v_peak_step"] = int(steps[peak_idx])
                    else:
                        combined["m1v_peak_step"] = int(peak_idx)
                if combined.get("m1v_last") is None:
                    combined["m1v_last"] = float(m1_ratio[-1])

        if m1_kind == "m1v" and m1_ratio.size and m1_times.size == m1_ratio.size:
            if m1_floor_median is not None and np.isfinite(m1_floor_median) and m1_floor_median > 0.0:
                thr = float(floor_factor) * float(m1_floor_median)
                combined["m1v_thr"] = thr
                finite = np.isfinite(m1_ratio)
                above = finite & (m1_ratio >= thr)
                if above.size:
                    if np.all(above):
                        idx_cross = int(above.size)
                    else:
                        idx_cross = int(np.where(~above)[0][0])
                    prefix_len = idx_cross
                    combined["m1v_idx_cross"] = idx_cross
                    combined["m1v_prefix_len"] = prefix_len
                    if prefix_len >= min_fit_points_prefix:
                        a_slice = m1_ratio[:prefix_len]
                        t_slice = m1_times[:prefix_len]
                        if np.any(~np.isfinite(a_slice)) or np.any(a_slice <= 0.0):
                            pass
                        else:
                            fit = linear_fit(t_slice, np.log(a_slice))
                            if fit is not None:
                                slope, _intercept, r2, _resid_std = fit
                                combined["gamma_m1v_fit_prefix"] = slope
                                combined["r2_m1v_fit_prefix"] = r2

        if (
            m1_kind == "m1v"
            and case_id is not None
            and "hires" in str(case_id)
            and m1_ratio.size
            and m1_times.size == m1_ratio.size
        ):
            floor_source = case_id
            baseline_floor = None
            if outputs_root is not None and "eps010-hires" in str(case_id):
                case_str = str(case_id)
                prefix = case_str.split("eps010-hires", 1)[0]
                baseline_case = f"{prefix}eps000-hires"
                baseline_path = outputs_root / baseline_case / "raw" / "run" / "particle_vel_stats.csv"
                if baseline_path.exists():
                    _t_base, r_base, _audit, _extras = load_particle_vel_stats_series(baseline_path)
                    baseline_floor, _baseline_mean = compute_tail_floor(r_base, 16)
                    floor_source = baseline_case
            if baseline_floor is None:
                baseline_floor, _baseline_mean = compute_tail_floor(m1_ratio, 16)
            combined["m1v_floor"] = baseline_floor
            combined["m1v_floor_source"] = floor_source
            if baseline_floor is not None and np.isfinite(baseline_floor):
                signal = m1_ratio - float(baseline_floor)
                signal = np.where(np.isfinite(signal), signal, np.nan)
                signal = np.maximum(signal, 0.0)
                peak_signal = float(np.nanmax(signal)) if signal.size else 0.0
                if np.isfinite(peak_signal) and peak_signal > 0.0:
                    peak_idx = int(np.nanargmax(signal))
                    combined["m1v_peak_signal"] = peak_signal
                    combined["m1v_peak"] = float(m1_ratio[peak_idx])
                    combined["m1v_last"] = float(m1_ratio[-1])
                    steps = m1_extras.get("step") or []
                    peak_step = None
                    if len(steps) == m1_ratio.size and steps[peak_idx] is not None:
                        peak_step = int(steps[peak_idx])
                    combined["m1v_peak_step"] = peak_step if peak_step is not None else int(peak_idx)
                    combined["m1v_peak_time"] = float(m1_times[peak_idx])

                    def first_below(target: float) -> int | None:
                        if peak_idx >= signal.size:
                            return None
                        tail = signal[peak_idx:]
                        idxs = np.where(tail <= target)[0]
                        if idxs.size == 0:
                            return None
                        return int(peak_idx + idxs[0])

                    idx_e = first_below(peak_signal / np.e)
                    if idx_e is not None:
                        combined["m1v_idx_tau_e"] = idx_e
                        if len(steps) == m1_ratio.size and steps[idx_e] is not None and peak_step is not None:
                            combined["m1v_tau_e_steps"] = int(steps[idx_e] - peak_step)
                        else:
                            combined["m1v_tau_e_steps"] = int(idx_e - peak_idx)
                        combined["m1v_tau_e_time"] = float(m1_times[idx_e] - m1_times[peak_idx])

                    idx_half = first_below(peak_signal / 2.0)
                    if idx_half is not None:
                        combined["m1v_idx_t_half"] = idx_half
                        if len(steps) == m1_ratio.size and steps[idx_half] is not None and peak_step is not None:
                            combined["m1v_t_half_steps"] = int(steps[idx_half] - peak_step)
                        else:
                            combined["m1v_t_half_steps"] = int(idx_half - peak_idx)
                        combined["m1v_t_half_time"] = float(m1_times[idx_half] - m1_times[peak_idx])

                    valid_idx = np.where(signal > 0.0)[0]
                    valid_idx = valid_idx[valid_idx >= peak_idx]
                    if valid_idx.size >= min_fit_points_prefix:
                        fit_idx = valid_idx[: min_fit_points_prefix]
                        fit = linear_fit(m1_times[fit_idx], np.log(signal[fit_idx]))
                        if fit is not None:
                            slope, _intercept, r2, _resid_std = fit
                            combined["m1v_decay_rate_fit"] = slope
                            combined["m1v_decay_rate_r2"] = r2

        if m1_kind == "m1v":
            combined["m1v_floor_factor_used"] = float(floor_factor)
            floor_val = combined.get("m1v_floor")
            if floor_val is not None and np.isfinite(floor_val) and floor_val > 0.0:
                peak_val = combined.get("m1v_peak")
                last_val = combined.get("m1v_last")
                if peak_val is not None:
                    combined["m1v_peak_over_floor"] = float(peak_val) / float(floor_val)
                if last_val is not None:
                    combined["m1v_last_over_floor"] = float(last_val) / float(floor_val)

        m1_fit_start_step_min = None
        m1_fit_start_idx_min = None
        if m1_kind == "m1v" and m1_ratio.size and warpx_meta:
            m1_rho_drive = warpx_meta.get("m1_rho_cos_drive") or {}
            repeat_steps = m1_rho_drive.get("repeat_applied_steps")
            if repeat_steps:
                try:
                    last_step = max(int(step) for step in repeat_steps if step is not None)
                    m1_fit_start_step_min = last_step + 1
                except ValueError:
                    m1_fit_start_step_min = None
            if m1_fit_start_step_min is None:
                nsteps = m1_rho_drive.get("nsteps")
                if nsteps is None:
                    nsteps = args_cfg.get("m1_rho_cos_nsteps")
                if nsteps is not None:
                    m1_fit_start_step_min = int(nsteps)
            if m1_fit_start_step_min is None:
                m1_drive = warpx_meta.get("m1_drive") or {}
                drive_steps = m1_drive.get("drive_applied_steps")
                if drive_steps:
                    try:
                        last_step = max(int(step) for step in drive_steps if step is not None)
                        m1_fit_start_step_min = last_step + 1
                    except ValueError:
                        m1_fit_start_step_min = None
                elif bool(args_cfg.get("m1_drive_repeat", False)):
                    nsteps = args_cfg.get("m1_drive_nsteps")
                    if nsteps is not None:
                        m1_fit_start_step_min = int(nsteps)
        if m1_fit_start_step_min is not None:
            combined["m1v_fit_start_step_min"] = int(m1_fit_start_step_min)
            steps = m1_extras.get("step") or []
            if len(steps) == m1_ratio.size:
                for idx, step in enumerate(steps):
                    if step is None:
                        continue
                    if int(step) >= m1_fit_start_step_min:
                        m1_fit_start_idx_min = int(idx)
                        break
            if m1_fit_start_idx_min is not None:
                combined["m1v_fit_start_idx_min"] = int(m1_fit_start_idx_min)

        if (
            m1_kind == "m1v"
            and m1_ratio.size
            and m1_times.size == m1_ratio.size
            and combined.get("m1v_tau_e_steps") is None
        ):
            baseline_floor = combined.get("m1v_floor")
            if baseline_floor is not None and np.isfinite(baseline_floor):
                signal = m1_ratio - float(baseline_floor)
                signal = np.where(np.isfinite(signal), signal, np.nan)
                signal = np.maximum(signal, 0.0)
                peak_signal = float(np.nanmax(signal)) if signal.size else 0.0
                if np.isfinite(peak_signal) and peak_signal > 0.0:
                    peak_idx = int(np.nanargmax(signal))
                    combined.setdefault("m1v_peak", float(m1_ratio[peak_idx]))
                    combined.setdefault("m1v_peak_time", float(m1_times[peak_idx]))
                    steps = m1_extras.get("step") or []
                    peak_step = None
                    if len(steps) == m1_ratio.size and steps[peak_idx] is not None:
                        peak_step = int(steps[peak_idx])
                    if combined.get("m1v_peak_step") is None:
                        combined["m1v_peak_step"] = peak_step if peak_step is not None else int(peak_idx)

                    def first_below(target: float) -> int | None:
                        if peak_idx >= signal.size:
                            return None
                        tail = signal[peak_idx:]
                        idxs = np.where(tail <= target)[0]
                        if idxs.size == 0:
                            return None
                        return int(peak_idx + idxs[0])

                    idx_e = first_below(peak_signal / np.e)
                    if idx_e is not None:
                        combined["m1v_idx_tau_e"] = idx_e
                        if len(steps) == m1_ratio.size and steps[idx_e] is not None and peak_step is not None:
                            combined["m1v_tau_e_steps"] = int(steps[idx_e] - peak_step)
                        else:
                            combined["m1v_tau_e_steps"] = int(idx_e - peak_idx)
                        combined["m1v_tau_e_time"] = float(m1_times[idx_e] - m1_times[peak_idx])

                    idx_half = first_below(peak_signal / 2.0)
                    if idx_half is not None:
                        combined["m1v_idx_t_half"] = idx_half
                        if len(steps) == m1_ratio.size and steps[idx_half] is not None and peak_step is not None:
                            combined["m1v_t_half_steps"] = int(steps[idx_half] - peak_step)
                        else:
                            combined["m1v_t_half_steps"] = int(idx_half - peak_idx)
                        combined["m1v_t_half_time"] = float(m1_times[idx_half] - m1_times[peak_idx])

    circuit_cfg = {}
    if warpx_meta:
        circuit_cfg = warpx_meta.get("circuit_mvp") or {}
    if args_cfg:
        args_circuit = args_cfg.get("circuit_mvp") or {}
        if args_circuit:
            merged = dict(circuit_cfg)
            merged.update(args_circuit)
            circuit_cfg = merged
    coil_meta = warpx_meta.get("coil_diag") if warpx_meta else None
    coil_path = None
    if coil_meta and coil_meta.get("path"):
        coil_path = Path(coil_meta.get("path"))
    if coil_path is None and warpx_meta_path is not None:
        candidate = warpx_meta_path.parent / "diag" / "reducedfiles" / "COIL.txt"
        if candidate.exists():
            coil_path = candidate
    if coil_path is not None and coil_path.exists():
        times_c, steps_c, phi_c, area_c, bn_c, coil_audit = load_coil_series(coil_path)
        combined.update(coil_audit)
        combined["coil_series_len"] = int(phi_c.size)
        if phi_c.size:
            combined["coil_phi_peak"] = float(np.max(np.abs(phi_c)))
        if phi_c.size:
            combined["coil_first3_phi"] = [float(v) for v in phi_c[:3]]
            combined["coil_last3_phi"] = [float(v) for v in phi_c[-3:]]
            combined["phi_min"] = float(np.min(phi_c))
            combined["phi_max"] = float(np.max(phi_c))
            combined["phi_delta"] = float(np.max(phi_c) - np.min(phi_c))
            dphi_dt_c = compute_derivative(times_c, phi_c)
            if dphi_dt_c.size:
                combined["dphi_dt_min"] = float(np.min(dphi_dt_c))
                combined["dphi_dt_max"] = float(np.max(dphi_dt_c))
        if area_c.size:
            combined["coil_area_mean"] = float(np.mean(area_c))
        if bn_c.size:
            combined["coil_bn_avg_mean"] = float(np.mean(bn_c))
        if coil_meta:
            combined["coil_axis"] = coil_meta.get("axis")
            combined["coil_center"] = coil_meta.get("center")
            combined["coil_rmax"] = coil_meta.get("rmax")
            combined["coil_plane_pos"] = coil_meta.get("plane_pos")
        if circuit_cfg.get("enabled", False) and phi_c.size and times_c.size == phi_c.size:
            window = circuit_cfg.get("window") or {}
            start_step = window.get("start_step", steps_c[0] if steps_c.size else 0)
            end_step = window.get("end_step", steps_c[-1] if steps_c.size else 0)
            try:
                start_step = int(start_step)
            except (TypeError, ValueError):
                start_step = int(steps_c[0]) if steps_c.size else 0
            try:
                end_step = int(end_step)
            except (TypeError, ValueError):
                end_step = int(steps_c[-1]) if steps_c.size else 0
            mask = (steps_c >= start_step) & (steps_c <= end_step)
            times_w = times_c[mask]
            phi_w = phi_c[mask]
            steps_w = steps_c[mask]
            combined["circuit_window_start_step"] = int(start_step)
            combined["circuit_window_end_step"] = int(end_step)
            if times_w.size >= 2:
                L_H = float(circuit_cfg.get("L_H", 0.0))
                R_ohm = float(circuit_cfg.get("R_ohm", 0.0))
                R_load_ohm = float(circuit_cfg.get("R_load_ohm", 0.0))
                R_switch_ohm = float(circuit_cfg.get("R_switch_ohm", 0.0))
                C_F = float(circuit_cfg.get("C_F", 0.0))
                N_turns = float(circuit_cfg.get("N_turns", 1))
                I0_A = float(circuit_cfg.get("I0_A", 0.0))
                Q0_C = float(circuit_cfg.get("Q0_C", 0.0))
                ode_substeps = int(circuit_cfg.get("ode_substeps", 1))
                ode_method = circuit_cfg.get("ode_method", "rk4")
                ode_interp = circuit_cfg.get("ode_interp", "linear")
                combined["circuit_L_H"] = L_H
                combined["circuit_R_ohm"] = R_ohm
                combined["circuit_R_load_ohm"] = R_load_ohm
                combined["circuit_R_switch_ohm"] = R_switch_ohm
                combined["circuit_C_F"] = C_F
                combined["circuit_N_turns"] = N_turns
                combined["R_total_ohm"] = float(R_ohm + R_load_ohm + R_switch_ohm)
                combined["R_components_ohm"] = {
                    "coil": float(R_ohm),
                    "load": float(R_load_ohm),
                    "switch": float(R_switch_ohm),
                }
                results = simulate_circuit_mvp(
                    times_w,
                    phi_w,
                    N_turns,
                    L_H,
                    R_ohm,
                    R_load_ohm,
                    R_switch_ohm,
                    C_F,
                    I0_A,
                    Q0_C,
                    ode_substeps=ode_substeps,
                    ode_method=ode_method,
                    ode_interp=ode_interp,
                )
                vind_series = results.pop("vind_series", None)
                i_series = results.pop("i_series", None)
                combined.update(results)
                if isinstance(vind_series, np.ndarray) and vind_series.size:
                    combined["vind_first3"] = [float(v) for v in vind_series[:3]]
                    combined["vind_last3"] = [float(v) for v in vind_series[-3:]]
                if isinstance(i_series, np.ndarray) and i_series.size:
                    combined["i_first3"] = [float(v) for v in i_series[:3]]
                    combined["i_last3"] = [float(v) for v in i_series[-3:]]
                if steps_w.size:
                    combined["circuit_window_steps"] = [int(steps_w[0]), int(steps_w[-1])]

    energy_name = None
    if args_cfg and args_cfg.get("energy_diag_name"):
        energy_name = str(args_cfg.get("energy_diag_name")).strip()
    if not energy_name:
        energy_name = "ENERGY0D"
    energy_path = None
    if warpx_meta_path is not None:
        candidate = warpx_meta_path.parent / "diags" / "reducedfiles" / f"{energy_name}.txt"
        if candidate.exists():
            energy_path = candidate
    if energy_path is not None and energy_path.exists():
        times_e, steps_e, wtot_e, we_e, wb_e, energy_audit = load_energy_series(energy_path)
        combined.update(energy_audit)
        combined["energy_obs_kind"] = "field_energy"
        combined["energy_series_len"] = int(wb_e.size)
        if wb_e.size:
            combined["energy_first3_Wb"] = [float(v) for v in wb_e[:3]]
            combined["energy_last3_Wb"] = [float(v) for v in wb_e[-3:]]
            combined["Wb_min"] = float(np.min(wb_e))
            combined["Wb_max"] = float(np.max(wb_e))
            combined["Wb_delta"] = float(np.max(wb_e) - np.min(wb_e))
            combined["Wb_net"] = float(wb_e[-1] - wb_e[0])
            dWb_dt = compute_derivative(times_e, wb_e)
            if dWb_dt.size:
                combined["dWb_dt_min"] = float(np.min(dWb_dt))
                combined["dWb_dt_max"] = float(np.max(dWb_dt))
        if we_e.size:
            combined["We_min"] = float(np.min(we_e))
            combined["We_max"] = float(np.max(we_e))
            combined["We_delta"] = float(np.max(we_e) - np.min(we_e))
        if wtot_e.size:
            combined["Wtot_min"] = float(np.min(wtot_e))
            combined["Wtot_max"] = float(np.max(wtot_e))
            combined["Wtot_delta"] = float(np.max(wtot_e) - np.min(wtot_e))

    # Drift meta proxy (optional)
    drift_meta_path = None
    if warpx_meta_path is not None:
        drift_candidate = warpx_meta_path.parent / "drift_meta.json"
        if drift_candidate.exists():
            drift_meta_path = drift_candidate
    if drift_meta_path is not None:
        drift_meta = _load_json_if_exists(drift_meta_path) or {}
        drift_mag = drift_meta.get("drift_mag")
        if drift_mag is not None:
            combined["drift_rel_diff_max"] = float(drift_mag)
            combined["drift_rel_diff_max_source"] = "drift_meta.drift_mag"

    # Fit window strategy + retry config
    fit_window_len_list = [24, 16, 12]
    min_fit_points = 12

    fit_times = m1_times if m1_times.size else tilt_times
    min_t_for_points = None
    if fit_times.size >= min_fit_points:
        min_t_for_points = float(fit_times[-min_fit_points])

    merge_time_ok = bool(merge_time_exists or merge_time_proxy is not None)
    merge_time_fallback_reason = None
    merge_time_for_fit = merge_time
    merge_time_candidate = merge_time if merge_time_exists else merge_time_proxy
    if merge_time_ok and merge_time_candidate is not None and min_t_for_points is not None:
        start_time_candidate = float(merge_time_candidate) + float(guard_dt or 0.0)
        if start_time_candidate > min_t_for_points:
            merge_time_ok = False
            merge_time_fallback_reason = "merge_time_too_late"
            merge_time_for_fit = None
        else:
            merge_time_for_fit = merge_time_candidate
    elif not merge_time_ok:
        merge_time_fallback_reason = "merge_time_missing"

    combined["merge_time_ok"] = bool(merge_time_ok)
    combined["merge_time_fallback_reason"] = merge_time_fallback_reason

    if merge_time_ok and merge_time_exists and merge_time_candidate is not None:
        combined["fit_window_strategy"] = "post_merge"
    elif merge_time_ok and merge_time_candidate is not None:
        combined["fit_window_strategy"] = "proxy"
    else:
        combined["fit_window_strategy"] = "full"

    if m1_times.size:
        m1_fit_ratio = m1_ratio
        if m1_kind == "m1v":
            m1_fit_ratio = np.maximum(m1_ratio, 1.0e-30)
        floor_median = None
        floor_factor_use = None
        start_idx_cap = None
        if m1_kind == "m1v":
            start_idx_cap = fit_window_start_idx_cap
        start_idx_min = m1_fit_start_idx_min
        if combined.get("fit_window_strategy") == "full":
            start_idx_min = 0

        m1_best, m1_retry = best_window_fit_retry(
            m1_times,
            m1_fit_ratio,
            fit_window_len_list,
            merge_time_for_fit,
            guard_dt,
            r2_threshold,
            min_fit_points,
            floor_median=floor_median,
            floor_factor=floor_factor_use,
            start_idx_cap=start_idx_cap,
            start_idx_min=start_idx_min,
        )

        if not m1_best.get("fit_window_found") and combined.get("fit_window_strategy") in ("post_merge", "proxy"):
            m1_best_fallback, m1_retry_fallback = best_window_fit_retry(
                m1_times,
                m1_fit_ratio,
                fit_window_len_list,
                None,
                guard_dt,
                r2_threshold,
                min_fit_points,
                floor_median=floor_median,
                floor_factor=floor_factor_use,
                start_idx_cap=start_idx_cap,
                start_idx_min=0,
            )
            if m1_best_fallback.get("fit_window_found"):
                m1_best = m1_best_fallback
                m1_retry = m1_retry_fallback
                m1_retry["fit_window_retry_reason"] = "POSTMERGE_FAILED_FALLBACK_FULL"
                combined["fit_window_strategy"] = "fallback_full"
        combined.update(m1_best)
        combined.update(m1_retry)
        combined["r2_m1_fit_best"] = m1_best.get("r2_fit_len24_best")
        combined["gamma_m1_fit_best"] = m1_best.get("gamma_fit_len24_best")
        if m1_kind == "m1v":
            combined["r2_m1v_fit_best"] = combined["r2_m1_fit_best"]
            combined["gamma_m1v_fit_best"] = combined["gamma_m1_fit_best"]
            combined["fit_series_source"] = "m1_vperp_ratio"
        elif m1_kind == "mom":
            combined["r2_m1mom_fit_best"] = combined["r2_m1_fit_best"]
            combined["gamma_m1mom_fit_best"] = combined["gamma_m1_fit_best"]
            combined["fit_series_source"] = "m1_mom_ratio"
        else:
            combined["fit_series_source"] = "m1_ratio"
        combined["r2_fit_best"] = combined["r2_m1_fit_best"]
        combined["gamma_fit_best"] = combined["gamma_m1_fit_best"]
        combined["r2_fit"] = combined["r2_m1_fit_best"]
        combined["gamma_fit"] = combined["gamma_m1_fit_best"]
        combined["tilt_amp_series_len"] = int(m1_ratio.size)
        combined["tilt_amp_series_len_source"] = combined["fit_series_source"]
        if combined.get("tilt_post_merge_amp_max") is None and m1_times.size == m1_ratio.size:
            fit_start = combined.get("fit_window_start_idx")
            fit_end = combined.get("fit_window_end_idx")
            if fit_start is not None and fit_end is not None and fit_end >= fit_start:
                window = m1_ratio[int(fit_start): int(fit_end) + 1]
            else:
                window = m1_ratio
            if window.size:
                combined["tilt_post_merge_amp_max"] = float(np.max(np.abs(window)))
        best_len = m1_best.get("fit_window_len")
        if best_len is not None:
            combined["fit_points_best"] = int(best_len)
            combined["fit_points"] = int(best_len)
            if m1_kind == "m1v":
                combined["fit_points_source"] = "best_window_len24_m1v"
            elif m1_kind == "mom":
                combined["fit_points_source"] = "best_window_len24_m1mom"
            else:
                combined["fit_points_source"] = "best_window_len24_m1"
        else:
            if combined.get("fit_points") in (None, 0):
                combined["fit_points_source"] = "missing"
            else:
                combined["fit_points_source"] = "h3b"

        if m1_kind == "m1v" and m1_ratio.size and m1_times.size == m1_ratio.size:
            fixed_len = 24
            fixed_start = 1
            fixed_fit = fit_window_log(m1_times, m1_fit_ratio, fixed_start, fixed_len)
            if fixed_fit is not None:
                combined["gamma_m1v_fit_fixed24"] = fixed_fit["gamma"]
                combined["r2_m1v_fit_fixed24"] = fixed_fit["r2"]
                combined["fit_fixed24_idx"] = [fixed_fit["start_idx"], fixed_fit["end_idx"]]
                combined["fit_quality_fixed24"] = fit_quality_label(fixed_fit["r2"])
            best_fit = scan_best_window_log(m1_times, m1_fit_ratio, fixed_start, fixed_len)
            if best_fit is not None:
                combined["gamma_m1v_fit_best24"] = best_fit["gamma"]
                combined["r2_m1v_fit_best24"] = best_fit["r2"]
                combined["fit_best24_idx"] = [best_fit["start_idx"], best_fit["end_idx"]]
                combined["fit_quality_best24"] = fit_quality_label(best_fit["r2"])
    elif tilt_times.size:
        combined["m1_ratio_series_len"] = 0
        combined["fit_series_source"] = "tilt_amp"
        combined["tilt_amp_series_len"] = int(tilt_amps.size)
        combined["tilt_amp_series_len_source"] = "tilt_amp"
        tilt_start_idx_min = 0 if combined.get("fit_window_strategy") == "full" else None
        best, tilt_retry = best_window_fit_retry(
            tilt_times,
            tilt_amps,
            fit_window_len_list,
            merge_time_for_fit,
            guard_dt,
            r2_threshold,
            min_fit_points,
            start_idx_min=tilt_start_idx_min,
        )
        if not best.get("fit_window_found") and combined.get("fit_window_strategy") in ("post_merge", "proxy"):
            best_fallback, tilt_retry_fallback = best_window_fit_retry(
                tilt_times,
                tilt_amps,
                fit_window_len_list,
                None,
                guard_dt,
                r2_threshold,
                min_fit_points,
                start_idx_min=0,
            )
            if best_fallback.get("fit_window_found"):
                best = best_fallback
                tilt_retry = tilt_retry_fallback
                tilt_retry["fit_window_retry_reason"] = "POSTMERGE_FAILED_FALLBACK_FULL"
                combined["fit_window_strategy"] = "fallback_full"
        combined.update(best)
        combined.update(tilt_retry)

        h3b_r2 = h3b.get("r2_fit_best") or h3b.get("r2_best")
        h3b_gamma = h3b.get("gamma_fit_best") or h3b.get("gamma_best")
        h3b_fit_points = h3b.get("fit_points")
        h3b_ok = (
            h3b_r2 is not None
            and np.isfinite(h3b_r2)
            and h3b_fit_points is not None
            and int(h3b_fit_points) >= int(min_fit_points)
        )

        best_r2 = best.get("r2_fit_len24_best")
        use_h3b = False
        if h3b_ok:
            if best_r2 is None or not np.isfinite(best_r2) or float(h3b_r2) >= float(best_r2):
                use_h3b = True

        if use_h3b:
            combined["r2_fit_best"] = float(h3b_r2)
            combined["gamma_fit_best"] = h3b_gamma
            combined["r2_fit"] = combined["r2_fit_best"]
            combined["gamma_fit"] = combined["gamma_fit_best"]
            if h3b_fit_points is not None:
                combined["fit_points_best"] = int(h3b_fit_points)
                combined["fit_points"] = int(h3b_fit_points)
            combined["fit_points_source"] = "h3b"
            if h3b.get("fit_window_start_idx") is not None:
                combined["fit_window_start_idx"] = h3b.get("fit_window_start_idx")
            if h3b.get("fit_window_end_idx") is not None:
                combined["fit_window_end_idx"] = h3b.get("fit_window_end_idx")
            if h3b.get("fit_window_len") is not None:
                combined["fit_window_len"] = h3b.get("fit_window_len")
            if h3b.get("fit_window_start_time") is not None:
                combined["fit_window_start_time"] = h3b.get("fit_window_start_time")
            if h3b.get("fit_window_end_time") is not None:
                combined["fit_window_end_time"] = h3b.get("fit_window_end_time")
            if h3b.get("fit_found") is not None:
                combined["fit_window_found"] = bool(h3b.get("fit_found"))
            if h3b.get("residual_std_best") is not None:
                combined["residual_std_best"] = h3b.get("residual_std_best")
        else:
            combined["r2_fit_best"] = best.get("r2_fit_len24_best")
            combined["gamma_fit_best"] = best.get("gamma_fit_len24_best")
            combined["r2_fit"] = combined["r2_fit_best"]
            combined["gamma_fit"] = combined["gamma_fit_best"]
            best_len = best.get("fit_window_len")
            if best_len is not None:
                combined["fit_points_best"] = int(best_len)
                combined["fit_points"] = int(best_len)
                combined["fit_points_source"] = "best_window_len24"
            else:
                if combined.get("fit_points") in (None, 0):
                    combined["fit_points_source"] = "missing"
                else:
                    combined["fit_points_source"] = "h3b"
        if combined.get("tilt_post_merge_amp_max") is None and tilt_times.size == tilt_amps.size:
            fit_start = combined.get("fit_window_start_idx")
            fit_end = combined.get("fit_window_end_idx")
            if fit_start is not None and fit_end is not None and fit_end >= fit_start:
                window = tilt_amps[int(fit_start): int(fit_end) + 1]
            else:
                window = tilt_amps
            if window.size:
                combined["tilt_post_merge_amp_max"] = float(np.max(np.abs(window)))

    # Fit window time audit
    fit_times = m1_times if m1_times.size else tilt_times
    fit_start = combined.get("fit_window_start_idx")
    fit_end = combined.get("fit_window_end_idx")
    if fit_times.size and fit_start is not None and fit_end is not None:
        if 0 <= int(fit_start) < fit_times.size and 0 <= int(fit_end) < fit_times.size:
            combined["tilt_amp_window_t0"] = float(fit_times[int(fit_start)])
            combined["tilt_amp_window_t1"] = float(fit_times[int(fit_end)])
    combined.setdefault("tilt_amp_window_t0", None)
    combined.setdefault("tilt_amp_window_t1", None)

    if combined.get("fail_reason") is None:
        if combined.get("merge_time_conflict"):
            combined["fail_reason"] = "MERGE_CONFLICT"
        elif not combined.get("merge_time_exists") and combined.get("merge_time_proxy") is None:
            combined["fail_reason"] = "MERGE_NOT_FOUND"
        elif combined.get("fit_window_found") is False:
            retry_reason = combined.get("fit_window_retry_reason")
            if retry_reason in ("INSUFFICIENT_POINTS", "WINDOW_OUT_OF_RANGE"):
                combined["fail_reason"] = "FIT_WINDOW_EMPTY"
            else:
                combined["fail_reason"] = "FIT_WINDOW_RETRY_EXHAUSTED"
    if combined.get("no_nan_in_metrics") is False and combined.get("fail_reason") is None:
        combined["fail_reason"] = "NAN_METRICS"

    combined["h3a"] = h3a
    combined["h3b"] = h3b

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
