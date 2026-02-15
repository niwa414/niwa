#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_header_indices(line: str) -> dict:
    cols = {}
    tokens = line.strip().lstrip("#").split()
    for idx, tok in enumerate(tokens):
        if tok.startswith("["):
            end = tok.find("]")
            name = tok[end + 1 :] if end >= 0 else tok
        else:
            name = tok
        cols[name] = idx
    return cols


def load_particle_number(path: Path) -> tuple[list[float], list[int], list[float], list[float], dict]:
    times: list[float] = []
    steps: list[int] = []
    n_macro: list[float] = []
    n_weight: list[float] = []
    header_idx = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header_idx = parse_header_indices(line)
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
            except ValueError:
                continue
            idx_macro = header_idx.get("total_macroparticles()", 2)
            idx_weight = header_idx.get("total_weight()")
            if idx_weight is None:
                # fallback: total_weight is after all species macroparticles
                idx_weight = max(idx_macro + 1, len(parts) - 1)
            try:
                macro = float(parts[idx_macro])
                weight = float(parts[idx_weight])
            except (ValueError, IndexError):
                continue
            steps.append(step)
            times.append(time)
            n_macro.append(macro)
            n_weight.append(weight)
    audit = {
        "particle_number_path": str(path.resolve()),
        "particle_number_sha1": sha1_file(path),
    }
    return times, steps, n_macro, n_weight, audit


def load_particle_energy(path: Path) -> tuple[list[float], list[int], list[float], dict]:
    times: list[float] = []
    steps: list[int] = []
    w_total: list[float] = []
    header_idx = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header_idx = parse_header_indices(line)
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
                idx_total = header_idx.get("total(J)", 2)
                w = float(parts[idx_total])
            except (ValueError, IndexError):
                continue
            steps.append(step)
            times.append(time)
            w_total.append(w)
    audit = {
        "particle_energy_path": str(path.resolve()),
        "particle_energy_sha1": sha1_file(path),
    }
    return times, steps, w_total, audit


def load_rho_max(path: Path) -> tuple[list[float], list[int], list[float], list[float], dict]:
    times: list[float] = []
    steps: list[int] = []
    rho_max: list[float] = []
    rho_min: list[float] = []
    header_idx = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header_idx = parse_header_indices(line)
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
            except ValueError:
                continue
            max_cols = [idx for name, idx in header_idx.items() if name.startswith("max_rho_lev")]
            min_cols = [idx for name, idx in header_idx.items() if name.startswith("min_rho_lev")]
            if not max_cols:
                max_cols = [2]
            if not min_cols:
                min_cols = [3] if len(parts) > 3 else [2]
            try:
                max_val = max(float(parts[i]) for i in max_cols if i < len(parts))
                min_val = min(float(parts[i]) for i in min_cols if i < len(parts))
            except Exception:
                continue
            steps.append(step)
            times.append(time)
            rho_max.append(max_val)
            rho_min.append(min_val)
    audit = {
        "rho_max_path": str(path.resolve()),
        "rho_max_sha1": sha1_file(path),
    }
    return times, steps, rho_max, rho_min, audit


def load_field_energy(path: Path) -> tuple[list[float], list[int], list[float], list[float], list[float], dict]:
    times: list[float] = []
    steps: list[int] = []
    w_tot: list[float] = []
    w_e: list[float] = []
    w_b: list[float] = []
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
            w_tot.append(total_sum)
            w_e.append(e_sum)
            w_b.append(b_sum)
    audit = {
        "field_energy_path": str(path.resolve()),
        "field_energy_sha1": sha1_file(path),
    }
    return times, steps, w_tot, w_e, w_b, audit


def first_last_triplets(
    steps: list[int], times: list[float], values: list[float], count: int = 3
) -> tuple[list[list[float]], list[list[float]]]:
    if not values:
        return [], []
    front = [
        [float(steps[i]), float(times[i]), float(values[i])]
        for i in range(min(count, len(values)))
    ]
    back = [
        [float(steps[i]), float(times[i]), float(values[i])]
        for i in range(max(len(values) - count, 0), len(values))
    ]
    return front, back


def series_has_nan(values: list[float]) -> bool:
    for val in values:
        if not math.isfinite(val):
            return True
    return False


def rounded_unique_count(values: list[float], ndigits: int = 6) -> int:
    if not values:
        return 0
    uniq = {round(val, ndigits) for val in values}
    return len(uniq)


def series_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((val - mean) ** 2 for val in values) / len(values)
    return math.sqrt(var)


def compute_derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    if n == 0 or times.size != n:
        return np.array([], dtype=float)
    deriv = np.zeros(n, dtype=float)
    if n == 1:
        deriv[0] = 0.0
        return deriv
    for i in range(n):
        if i == 0:
            dt = times[i + 1] - times[i]
            deriv[i] = (values[i + 1] - values[i]) / dt if dt != 0.0 else 0.0
        elif i == n - 1:
            dt = times[i] - times[i - 1]
            deriv[i] = (values[i] - values[i - 1]) / dt if dt != 0.0 else 0.0
        else:
            dt = times[i + 1] - times[i - 1]
            deriv[i] = (values[i + 1] - values[i - 1]) / dt if dt != 0.0 else 0.0
    return deriv


def load_coil_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    times: list[float] = []
    steps: list[int] = []
    phi: list[float] = []
    area: list[float] = []
    bn_avg: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 3:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
                phi_val = float(parts[2])
            except ValueError:
                continue
            area_val = None
            bn_val = None
            if len(parts) > 3:
                try:
                    area_val = float(parts[3])
                except ValueError:
                    area_val = None
            if len(parts) > 4:
                try:
                    bn_val = float(parts[4])
                except ValueError:
                    bn_val = None
            steps.append(step)
            times.append(time)
            phi.append(phi_val)
            area.append(area_val if area_val is not None else 0.0)
            bn_avg.append(bn_val if bn_val is not None else 0.0)
    audit = {
        "coil_source_path": str(path.resolve()),
        "coil_sha1": sha1_file(path),
        "coil_filesize_bytes": path.stat().st_size,
    }
    return (
        np.array(times, dtype=float),
        np.array(steps, dtype=int),
        np.array(phi, dtype=float),
        np.array(area, dtype=float),
        np.array(bn_avg, dtype=float),
        audit,
    )


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

        def deriv(I_val, Q_val, dphi_use):
            dI = (-turns * dphi_use - R_total * I_val - Q_val / C_F) / L_H
            dQ = I_val
            return dI, dQ

        for sub in range(ode_substeps):
            dt_sub = dt / ode_substeps
            if ode_interp == "linear":
                frac = (sub + 0.5) / ode_substeps
                dphi_use = dphi_k + frac * (dphi_k1 - dphi_k)
            else:
                dphi_use = dphi_const

            if ode_method == "euler":
                dI, dQ = deriv(I, Q, dphi_use)
                I_next = I + dI * dt_sub
                Q_next = Q + dQ * dt_sub
            else:
                k1_I, k1_Q = deriv(I, Q, dphi_use)
                k2_I, k2_Q = deriv(I + 0.5 * dt_sub * k1_I, Q + 0.5 * dt_sub * k1_Q, dphi_use)
                k3_I, k3_Q = deriv(I + 0.5 * dt_sub * k2_I, Q + 0.5 * dt_sub * k2_Q, dphi_use)
                k4_I, k4_Q = deriv(I + dt_sub * k3_I, Q + dt_sub * k3_Q, dphi_use)
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
    return {
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formation/compression 0D reduced diags")
    parser.add_argument("--particle-number", required=True)
    parser.add_argument("--rho-max", required=True)
    parser.add_argument("--particle-energy", required=True)
    parser.add_argument("--field-energy", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    metrics = {
        "formation_obs_kind": "formation0d",
    }

    pn_path = Path(args.particle_number)
    rm_path = Path(args.rho_max)
    pe_path = Path(args.particle_energy)
    fe_path = Path(args.field_energy)
    run_dir = fe_path.parents[2] if len(fe_path.parents) >= 3 else None
    repo_root = Path(__file__).resolve().parents[1]
    warpx_meta = None
    if run_dir is not None:
        for candidate in run_dir.glob("warpx_run_*.json"):
            try:
                warpx_meta = json.loads(candidate.read_text())
            except Exception:
                warpx_meta = None
            break

    times_pn, steps_pn, n_macro, n_weight, pn_audit = load_particle_number(pn_path)
    times_rm, steps_rm, rho_max, rho_min, rm_audit = load_rho_max(rm_path)
    times_pe, steps_pe, w_k, pe_audit = load_particle_energy(pe_path)
    times_fe, steps_fe, w_tot, w_e, w_b, fe_audit = load_field_energy(fe_path)

    metrics.update(pn_audit)
    metrics.update(rm_audit)
    metrics.update(pe_audit)
    metrics.update(fe_audit)

    metrics["particle_number_series_len"] = len(n_weight)
    metrics["rho_max_series_len"] = len(rho_max)
    metrics["particle_energy_series_len"] = len(w_k)
    field_energy_len = len(w_b) if w_b else len(w_tot)
    metrics["field_energy_series_len"] = field_energy_len
    metrics["formation_series_len"] = min(
        len(n_weight), len(rho_max), len(w_k), field_energy_len
    ) if all([n_weight, rho_max, w_k]) and field_energy_len > 0 else 0

    pn_first3, pn_last3 = first_last_triplets(steps_pn, times_pn, n_weight)
    metrics["particle_number_first3"] = pn_first3
    metrics["particle_number_last3"] = pn_last3
    metrics["particle_number_has_nan"] = series_has_nan(n_weight)

    if rho_max:
        if rho_min:
            rm_abs = [
                max(abs(rmin), abs(rmax))
                for rmax, rmin in zip(rho_max, rho_min)
            ]
        else:
            rm_abs = [abs(rmax) for rmax in rho_max]
    else:
        rm_abs = []
    rm_first3, rm_last3 = first_last_triplets(steps_rm, times_rm, rm_abs if rm_abs else rho_max)
    metrics["rho_max_first3"] = rm_first3
    metrics["rho_max_last3"] = rm_last3
    metrics["rho_max_has_nan"] = series_has_nan(rm_abs if rm_abs else rho_max)
    if rm_abs:
        metrics["rho_abs_first10"] = [float(v) for v in rm_abs[:10]]
        metrics["rho_abs_last10"] = [float(v) for v in rm_abs[-10:]]
        metrics["rho_abs_unique_count"] = rounded_unique_count(rm_abs, ndigits=6)
        metrics["rho_abs_std"] = series_std(rm_abs)

    pe_first3, pe_last3 = first_last_triplets(steps_pe, times_pe, w_k)
    metrics["particle_energy_first3"] = pe_first3
    metrics["particle_energy_last3"] = pe_last3
    metrics["particle_energy_has_nan"] = series_has_nan(w_k)

    fe_series = w_b if w_b else w_tot
    fe_first3, fe_last3 = first_last_triplets(steps_fe, times_fe, fe_series)
    metrics["field_energy_first3"] = fe_first3
    metrics["field_energy_last3"] = fe_last3
    metrics["field_energy_has_nan"] = series_has_nan(fe_series)

    if n_weight:
        metrics["N_total_initial"] = float(n_weight[0])
        metrics["N_total_last"] = float(n_weight[-1])
        metrics["N_total_delta"] = float(n_weight[-1] - n_weight[0])
        metrics["N_macro_last"] = float(n_macro[-1]) if n_macro else None

    if rho_max:
        metrics["rho_max_initial_signed"] = float(rho_max[0])
        metrics["rho_max_last_signed"] = float(rho_max[-1])
        metrics["rho_max_peak_signed"] = float(max(rho_max))
        metrics["rho_min_peak_signed"] = float(min(rho_min)) if rho_min else None
        metrics["rho_min_last_signed"] = float(rho_min[-1]) if rho_min else None
        if rm_abs:
            metrics["rho_max_initial"] = float(rm_abs[0])
            metrics["rho_max_last"] = float(rm_abs[-1])
            metrics["rho_max_peak"] = float(max(rm_abs))
            metrics["rho_max_peak_semantics"] = "abs_max"
            metrics["rho_abs_max_peak"] = float(max(rm_abs))
            metrics["rho_abs_max_last"] = float(rm_abs[-1])
            if rm_abs[0] != 0.0:
                metrics["rho_compress_ratio"] = float(max(rm_abs) / rm_abs[0])
        else:
            metrics["rho_max_initial"] = float(rho_max[0])
            metrics["rho_max_last"] = float(rho_max[-1])
            metrics["rho_max_peak"] = float(max(rho_max))
            metrics["rho_max_peak_semantics"] = "signed_max"
            if rho_max[0] != 0.0:
                metrics["rho_compress_ratio"] = float(max(rho_max) / rho_max[0])

    if w_k:
        metrics["Wk_initial"] = float(w_k[0])
        metrics["Wk_last"] = float(w_k[-1])
        metrics["Wk_peak"] = float(max(w_k))
        metrics["Wk_delta"] = float(max(w_k) - min(w_k))

    if w_tot:
        metrics["W_field_initial"] = float(w_tot[0])
        metrics["W_field_last"] = float(w_tot[-1])
        metrics["W_field_peak"] = float(max(w_tot))
        metrics["W_field_delta"] = float(max(w_tot) - min(w_tot))

    if w_b:
        metrics["Wb_initial"] = float(w_b[0])
        metrics["Wb_last"] = float(w_b[-1])
        metrics["Wb_peak"] = float(max(w_b))
        metrics["Wb_delta"] = float(max(w_b) - min(w_b))
        metrics["Wb_net"] = float(w_b[-1] - w_b[0])
        metrics["Wb_is_total_field"] = False
    elif w_tot:
        metrics["Wb_initial"] = float(w_tot[0])
        metrics["Wb_last"] = float(w_tot[-1])
        metrics["Wb_peak"] = float(max(w_tot))
        metrics["Wb_delta"] = float(max(w_tot) - min(w_tot))
        metrics["Wb_net"] = float(w_tot[-1] - w_tot[0])
        metrics["Wb_is_total_field"] = True

    if w_e:
        metrics["We_initial"] = float(w_e[0])
        metrics["We_last"] = float(w_e[-1])
        metrics["We_peak"] = float(max(w_e))
        metrics["We_delta"] = float(max(w_e) - min(w_e))

    if w_k and w_b and w_b[-1] != 0.0:
        metrics["beta_proxy_last"] = float(w_k[-1] / w_b[-1])
        metrics["beta_proxy_peak"] = float(
            max(wk / wb for wk, wb in zip(w_k, w_b) if wb != 0.0)
        )
    elif w_k and w_tot and w_tot[-1] != 0.0:
        metrics["beta_proxy_last"] = float(w_k[-1] / w_tot[-1])
        metrics["beta_proxy_peak"] = float(
            max(wk / wt for wk, wt in zip(w_k, w_tot) if wt != 0.0)
        )

    metrics["formation_has_nan"] = bool(
        metrics.get("particle_number_has_nan")
        or metrics.get("rho_max_has_nan")
        or metrics.get("particle_energy_has_nan")
        or metrics.get("field_energy_has_nan")
    )

    formation_kpi = {}
    formation_kpi2 = {}
    formation_kpi_phase = {}

    def peak_with_index(values, steps, times):
        if not values:
            return None, None, None
        peak_val = max(values)
        try:
            idx = values.index(peak_val)
        except ValueError:
            return None, None, None
        step = steps[idx] if idx < len(steps) else None
        time = times[idx] if idx < len(times) else None
        return float(peak_val), int(step) if step is not None else None, float(time) if time is not None else None

    rho_series = rm_abs if rm_abs else []
    if rho_series:
        peak_val, peak_step, peak_time = peak_with_index(rho_series, steps_rm, times_rm)
        formation_kpi["rho_max_peak"] = peak_val
        formation_kpi["step_rho_max_peak"] = peak_step
        formation_kpi["t_rho_max_peak"] = peak_time

        rho0 = rho_series[0]
        rhop = max(rho_series)
        rho_delta = rhop - rho0
        denom = max(1.0, abs(rho0))
        rho_delta_rel = rho_delta / denom
        formation_kpi["rho_delta"] = float(rho_delta)
        formation_kpi["rho_delta_rel"] = float(rho_delta_rel)

        tol_rel = 1.0e-4
        if rho_delta_rel < tol_rel:
            formation_kpi["compression_detected"] = False
            formation_kpi["compression_duration_steps"] = None
            formation_kpi["compression_duration_time"] = None
        else:
            formation_kpi["compression_detected"] = True
            thr_a = rho0 + 0.1 * (rhop - rho0)
            thr_b = rho0 + 0.9 * (rhop - rho0)
            step_a = None
            step_b = None
            time_a = None
            time_b = None
            for idx, rho_val in enumerate(rho_series):
                if step_a is None and rho_val >= thr_a:
                    step_a = steps_rm[idx] if idx < len(steps_rm) else idx
                    time_a = times_rm[idx] if idx < len(times_rm) else None
                if step_b is None and rho_val >= thr_b:
                    step_b = steps_rm[idx] if idx < len(steps_rm) else idx
                    time_b = times_rm[idx] if idx < len(times_rm) else None
                if step_a is not None and step_b is not None:
                    break
            formation_kpi["compression_threshold_low"] = float(thr_a)
            formation_kpi["compression_threshold_high"] = float(thr_b)
            formation_kpi["step_rho_threshold_low"] = int(step_a) if step_a is not None else None
            formation_kpi["step_rho_threshold_high"] = int(step_b) if step_b is not None else None
            formation_kpi["t_rho_threshold_low"] = float(time_a) if time_a is not None else None
            formation_kpi["t_rho_threshold_high"] = float(time_b) if time_b is not None else None
            if step_a is not None and step_b is not None:
                formation_kpi["compression_duration_steps"] = int(step_b) - int(step_a)
                if time_a is not None and time_b is not None:
                    formation_kpi["compression_duration_time"] = float(time_b - time_a)
            else:
                formation_kpi["compression_duration_steps"] = None
                formation_kpi["compression_duration_time"] = None

    if w_k:
        peak_val, peak_step, peak_time = peak_with_index(w_k, steps_pe, times_pe)
        formation_kpi["Wk_peak"] = peak_val
        formation_kpi["step_Wk_peak"] = peak_step
        formation_kpi["t_Wk_peak"] = peak_time

    wb_series = w_b if w_b else w_tot
    if wb_series:
        peak_val, peak_step, peak_time = peak_with_index(wb_series, steps_fe, times_fe)
        formation_kpi["Wb_peak"] = peak_val
        formation_kpi["step_Wb_peak"] = peak_step
        formation_kpi["t_Wb_peak"] = peak_time

    if w_k and wb_series:
        ratios = []
        ratio_indices = []
        for idx, (wk, wb) in enumerate(zip(w_k, wb_series)):
            if wb != 0.0:
                ratios.append(wk / wb)
                ratio_indices.append(idx)
        if ratios:
            max_ratio = max(ratios)
            max_idx = ratio_indices[ratios.index(max_ratio)]
            step = steps_pe[max_idx] if max_idx < len(steps_pe) else None
            time = times_pe[max_idx] if max_idx < len(times_pe) else None
            formation_kpi["beta_proxy_peak"] = float(max_ratio)
            formation_kpi["step_beta_proxy_peak"] = int(step) if step is not None else None
            formation_kpi["t_beta_proxy_peak"] = float(time) if time is not None else None

    if formation_kpi:
        phase_min_step = 8
        phase_duration = formation_kpi.get("compression_duration_steps")
        phase_step = formation_kpi.get("step_rho_max_peak")
        phase_valid = (
            phase_duration is not None
            and phase_duration > 0
            and phase_step is not None
            and int(phase_step) >= phase_min_step
        )
        formation_kpi["phase_min_step"] = int(phase_min_step)
        formation_kpi["phase_valid"] = bool(phase_valid)

    if formation_kpi:
        metrics["formation_kpi"] = formation_kpi

    # Phase-limited KPIs: restrict rho peak/delta to envelope window (phase_end).
    phase_end = None
    if warpx_meta:
        args_cfg = warpx_meta.get("args") or {}
        phase_end = args_cfg.get("drive_envelope_off_step")
        if phase_end is None:
            drive_env = warpx_meta.get("drive_envelope") or {}
            phase_end = drive_env.get("off_step")
    if phase_end is not None:
        try:
            phase_end = int(phase_end)
        except (TypeError, ValueError):
            phase_end = None

    if rho_series and phase_end is not None:
        phase_indices = [i for i, s in enumerate(steps_rm) if s <= phase_end]
        if phase_indices:
            last_idx = phase_indices[-1]
            phase_steps = steps_rm[: last_idx + 1]
            phase_times = times_rm[: last_idx + 1]
            phase_rho = rho_series[: last_idx + 1]
            phase_peak_val, phase_peak_step, phase_peak_time = peak_with_index(
                phase_rho, phase_steps, phase_times
            )
            formation_kpi_phase["phase_end_step"] = int(phase_end)
            formation_kpi_phase["rho_max_peak"] = phase_peak_val
            formation_kpi_phase["step_rho_max_peak"] = phase_peak_step
            formation_kpi_phase["t_rho_max_peak"] = phase_peak_time
            rho0_phase = phase_rho[0]
            rhop_phase = max(phase_rho)
            rho_delta_phase = rhop_phase - rho0_phase
            denom_phase = max(1.0, abs(rho0_phase))
            rho_delta_rel_phase = rho_delta_phase / denom_phase
            formation_kpi_phase["rho_delta_rel_phase"] = float(rho_delta_rel_phase)
            if rho_delta_rel_phase < 1.0e-4:
                formation_kpi_phase["compression_detected_phase"] = False
                formation_kpi_phase["compression_duration_steps_phase"] = None
                formation_kpi_phase["compression_duration_time_phase"] = None
            else:
                formation_kpi_phase["compression_detected_phase"] = True
                thr_a = rho0_phase + 0.1 * (rhop_phase - rho0_phase)
                thr_b = rho0_phase + 0.9 * (rhop_phase - rho0_phase)
                step_a = None
                step_b = None
                time_a = None
                time_b = None
                for idx, rho_val in enumerate(phase_rho):
                    if step_a is None and rho_val >= thr_a:
                        step_a = phase_steps[idx] if idx < len(phase_steps) else idx
                        time_a = phase_times[idx] if idx < len(phase_times) else None
                    if step_b is None and rho_val >= thr_b:
                        step_b = phase_steps[idx] if idx < len(phase_steps) else idx
                        time_b = phase_times[idx] if idx < len(phase_times) else None
                    if step_a is not None and step_b is not None:
                        break
                formation_kpi_phase["compression_threshold_low_phase"] = float(thr_a)
                formation_kpi_phase["compression_threshold_high_phase"] = float(thr_b)
                formation_kpi_phase["step_rho_threshold_low_phase"] = int(step_a) if step_a is not None else None
                formation_kpi_phase["step_rho_threshold_high_phase"] = int(step_b) if step_b is not None else None
                if step_a is not None and step_b is not None:
                    formation_kpi_phase["compression_duration_steps_phase"] = int(step_b) - int(step_a)
                    if time_a is not None and time_b is not None:
                        formation_kpi_phase["compression_duration_time_phase"] = float(time_b - time_a)
                else:
                    formation_kpi_phase["compression_duration_steps_phase"] = None
                    formation_kpi_phase["compression_duration_time_phase"] = None

    if formation_kpi_phase:
        metrics["formation_kpi_phase"] = formation_kpi_phase

    # Late-window KPIs to avoid early transient locking.
    max_step = None
    if steps_rm:
        max_step = max(steps_rm)
    elif steps_pe:
        max_step = max(steps_pe)
    elif steps_fe:
        max_step = max(steps_fe)
    if max_step is not None:
        late_start_step = max(64, int(0.25 * int(max_step)))
        formation_kpi2["late_start_step"] = int(late_start_step)

        def find_start_idx(steps, start_step):
            for idx, step in enumerate(steps):
                if step >= start_step:
                    return idx
            return None

        def peak_with_index_range(values, steps, times, start_idx):
            if start_idx is None or start_idx >= len(values):
                return None, None, None
            sub = values[start_idx:]
            if not sub:
                return None, None, None
            peak_val = max(sub)
            rel_idx = sub.index(peak_val)
            idx = start_idx + rel_idx
            step = steps[idx] if idx < len(steps) else None
            time = times[idx] if idx < len(times) else None
            return float(peak_val), int(step) if step is not None else None, float(time) if time is not None else None

        eps_denom = 1.0e-12

        # Late rho KPIs
        if rho_series:
            start_idx = find_start_idx(steps_rm, late_start_step)
            if start_idx is not None and start_idx < len(rho_series):
                metrics["rho_abs_std_late"] = series_std(rho_series[start_idx:])
            rho_peak_late, step_rho_peak_late, t_rho_peak_late = peak_with_index_range(
                rho_series, steps_rm, times_rm, start_idx
            )
            formation_kpi2["rho_max_peak_late"] = rho_peak_late
            formation_kpi2["step_rho_max_peak_late"] = step_rho_peak_late
            formation_kpi2["t_rho_max_peak_late"] = t_rho_peak_late
            if start_idx is not None and start_idx < len(rho_series):
                rho0_late = rho_series[start_idx]
                if rho_peak_late is not None:
                    rho_delta_late = rho_peak_late - rho0_late
                    rho_delta_rel_late = rho_delta_late / (abs(rho0_late) + eps_denom)
                    formation_kpi2["rho_delta_rel_late"] = float(rho_delta_rel_late)
                    formation_kpi2["compression_detected_late"] = bool(
                        rho_delta_rel_late >= 0.01
                    )
                    # Compression duration in late window (10% -> 90% of late rise)
                    if rho_delta_rel_late >= 1.0e-4:
                        thr_a = rho0_late + 0.1 * rho_delta_late
                        thr_b = rho0_late + 0.9 * rho_delta_late
                        step_a = None
                        step_b = None
                        time_a = None
                        time_b = None
                        for idx in range(start_idx, len(rho_series)):
                            rho_val = rho_series[idx]
                            if step_a is None and rho_val >= thr_a:
                                step_a = steps_rm[idx] if idx < len(steps_rm) else idx
                                time_a = times_rm[idx] if idx < len(times_rm) else None
                            if step_b is None and rho_val >= thr_b:
                                step_b = steps_rm[idx] if idx < len(steps_rm) else idx
                                time_b = times_rm[idx] if idx < len(times_rm) else None
                            if step_a is not None and step_b is not None:
                                break
                        if step_a is not None and step_b is not None:
                            formation_kpi2["compression_duration_steps_late"] = int(step_b) - int(step_a)
                            if time_a is not None and time_b is not None:
                                formation_kpi2["compression_duration_time_late"] = float(time_b - time_a)
                        else:
                            formation_kpi2["compression_duration_steps_late"] = None
                            formation_kpi2["compression_duration_time_late"] = None
                    else:
                        formation_kpi2["compression_duration_steps_late"] = None
                        formation_kpi2["compression_duration_time_late"] = None

        # Late energy KPIs
        wb_series = w_b if w_b else w_tot
        if wb_series:
            start_idx_fe = find_start_idx(steps_fe, late_start_step)
            wb_peak_late, step_wb_peak_late, t_wb_peak_late = peak_with_index_range(
                wb_series, steps_fe, times_fe, start_idx_fe
            )
            formation_kpi2["Wb_peak_late"] = wb_peak_late
            formation_kpi2["step_Wb_peak_late"] = step_wb_peak_late
            formation_kpi2["t_Wb_peak_late"] = t_wb_peak_late

        # B-energy proxy (prefer magnetic energy series if available)
        b_energy_series = w_b if w_b else w_tot
        if b_energy_series:
            formation_kpi2["B_energy_is_total_field"] = bool(not w_b)
            start_idx_be = find_start_idx(steps_fe, late_start_step)
            if start_idx_be is not None and start_idx_be < len(b_energy_series):
                b_energy_start = b_energy_series[start_idx_be]
            else:
                b_energy_start = None
            b_energy_peak, step_b_energy_peak, t_b_energy_peak = peak_with_index_range(
                b_energy_series, steps_fe, times_fe, start_idx_be
            )
            formation_kpi2["B_energy_peak"] = b_energy_peak
            formation_kpi2["step_B_energy_peak"] = step_b_energy_peak
            formation_kpi2["t_B_energy_peak"] = t_b_energy_peak
            if step_b_energy_peak is not None and max_step is not None:
                formation_kpi2["step_B_energy_peak_not_tail"] = bool(
                    int(step_b_energy_peak) < (int(max_step) - 16)
                )
            if b_energy_start is not None and b_energy_peak is not None:
                formation_kpi2["B_energy_rise_ratio"] = float(
                    b_energy_peak / (b_energy_start + eps_denom)
                )
        if w_k:
            start_idx_pe = find_start_idx(steps_pe, late_start_step)
            wk_peak_late, step_wk_peak_late, t_wk_peak_late = peak_with_index_range(
                w_k, steps_pe, times_pe, start_idx_pe
            )
            formation_kpi2["Wk_peak_late"] = wk_peak_late
            formation_kpi2["step_Wk_peak_late"] = step_wk_peak_late
            formation_kpi2["t_Wk_peak_late"] = t_wk_peak_late

        # Late beta proxy
        if w_k and wb_series:
            min_len = min(len(w_k), len(wb_series), len(steps_pe))
            ratios = []
            ratio_indices = []
            for idx in range(min_len):
                if steps_pe[idx] < late_start_step:
                    continue
                wb_val = wb_series[idx]
                if wb_val != 0.0:
                    ratios.append(w_k[idx] / wb_val)
                    ratio_indices.append(idx)
            if ratios:
                max_ratio = max(ratios)
                max_idx = ratio_indices[ratios.index(max_ratio)]
                step = steps_pe[max_idx] if max_idx < len(steps_pe) else None
                time = times_pe[max_idx] if max_idx < len(times_pe) else None
                formation_kpi2["beta_proxy_peak_late"] = float(max_ratio)
                formation_kpi2["step_beta_proxy_peak_late"] = int(step) if step is not None else None
                formation_kpi2["t_beta_proxy_peak_late"] = float(time) if time is not None else None

    if formation_kpi2:
        metrics["formation_kpi2"] = formation_kpi2
        if formation_kpi2.get("step_B_energy_peak") is not None:
            metrics["step_B_energy_peak"] = formation_kpi2.get("step_B_energy_peak")
            metrics["step_B_energy_peak_not_tail"] = formation_kpi2.get(
                "step_B_energy_peak_not_tail"
            )

    # Endpoint/plateau formation chain metrics (steady/endpoint proxy)
    formation_chain = {}
    Wb_series = w_b if w_b else w_tot
    if Wb_series:
        Wb_end = float(Wb_series[-1])
        formation_chain["Wb_end"] = Wb_end
        K = 16
        if len(Wb_series) >= K:
            tail = Wb_series[-K:]
        else:
            tail = Wb_series
            K = len(tail)
        formation_chain["Wb_mean_lastK"] = float(sum(tail) / K) if K > 0 else None
        formation_chain["Wb_lastK_count"] = int(K)
        # Slope over last K points
        if K >= 2 and steps_fe and times_fe:
            t_tail = times_fe[-K:]
            y_tail = tail
            t_mean = sum(t_tail) / K
            y_mean = sum(y_tail) / K
            denom = sum((t - t_mean) ** 2 for t in t_tail)
            if denom > 0.0:
                slope = sum((t - t_mean) * (y - y_mean) for t, y in zip(t_tail, y_tail)) / denom
                formation_chain["Wb_slope_lastK"] = float(slope)
    if w_k:
        Wk_end = float(w_k[-1])
        formation_chain["Wk_end"] = Wk_end
        Kk = formation_chain.get("Wb_lastK_count") or 16
        if len(w_k) >= Kk:
            tail_k = w_k[-Kk:]
        else:
            tail_k = w_k
            Kk = len(tail_k)
        formation_chain["Wk_mean_lastK"] = float(sum(tail_k) / Kk) if Kk > 0 else None
        formation_chain["Wk_lastK_count"] = int(Kk)
    if formation_chain.get("Wb_end") is not None and formation_chain.get("Wk_end") is not None:
        formation_chain["beta_end"] = float(
            formation_chain["Wk_end"] / (formation_chain["Wb_end"] + 1.0e-30)
        )
    if formation_chain:
        metrics["formation_chain"] = formation_chain

    # Eta profile debug snapshot (from run metadata if present).
    if warpx_meta is not None:
        eta_debug = {}
        resistivity = warpx_meta.get("resistivity") or {}
        if resistivity:
            eta_debug["resistivity"] = resistivity
        eta_profile_debug = warpx_meta.get("eta_profile_debug")
        if eta_profile_debug:
            eta_debug["eta_profile_debug"] = eta_profile_debug
        if eta_debug:
            metrics["eta_profile_debug"] = eta_debug

    # Circuit chain (coil flux + RLC) if available
    metrics["circuit_import_ok"] = True
    if run_dir is not None:
        coil_path = run_dir / "diag" / "reducedfiles" / "COIL.txt"
        if not coil_path.exists():
            coil_path = run_dir / "diags" / "reducedfiles" / "COIL.txt"
        circuit_chain = {}
        if coil_path.exists():
            times_c, steps_c, phi_c, area_c, bn_c, coil_audit = load_coil_series(coil_path)
            circuit_chain.update(coil_audit)
            circuit_chain["coil_series_len"] = int(phi_c.size)

            # Load circuit config: prefer case warpx_args.json override
            circuit_cfg = {}
            if run_dir is not None:
                try:
                    case_id = run_dir.parents[1].name
                    args_path = repo_root / "cases" / case_id / "inputs" / "warpx_args.json"
                    if args_path.exists():
                        args_cfg = json.loads(args_path.read_text())
                        if isinstance(args_cfg.get("circuit_mvp"), dict):
                            circuit_cfg.update(args_cfg.get("circuit_mvp"))
                except Exception:
                    pass
            # Fallback to warpx_run meta
            meta_path = None
            if run_dir is not None:
                for candidate in run_dir.glob("warpx_run_*.json"):
                    meta_path = candidate
                    break
            if meta_path is not None and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if isinstance(meta.get("circuit_mvp"), dict):
                        cfg = dict(meta.get("circuit_mvp"))
                        cfg.update(circuit_cfg)
                        circuit_cfg = cfg
                except Exception:
                    pass

            if circuit_cfg.get("enabled", False) and phi_c.size and times_c.size == phi_c.size:
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
                results = simulate_circuit_mvp(
                    times_c,
                    phi_c,
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
                for key in (
                    "vind_peak_V",
                    "e_load_J",
                    "e_in_J",
                    "eta_delivered",
                    "energy_residual_rel",
                    "R_total_ohm",
                ):
                    if key in results:
                        circuit_chain[key] = results[key]
                circuit_chain["circuit_enabled"] = True
                circuit_chain["ode_substeps"] = int(ode_substeps)
                circuit_chain["ode_method"] = str(ode_method)
                circuit_chain["ode_interp"] = str(ode_interp)
                circuit_chain["R_load_ohm"] = float(R_load_ohm)
                circuit_chain["R_switch_ohm"] = float(R_switch_ohm)
        if circuit_chain:
            metrics["circuit_chain"] = circuit_chain

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
