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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_variants(path: Path) -> tuple[list[str], dict]:
    data = load_json(path)
    ordered = []
    for key in ("hall_on", "hall_off", "baseline"):
        case_id = data.get(key)
        if case_id:
            ordered.append(case_id)
    for item in data.get("variants", []):
        if isinstance(item, str):
            case_id = item
        else:
            case_id = item.get("id")
        if case_id and case_id not in ordered:
            ordered.append(case_id)
    return ordered, data


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
    x_faces, y_faces, _, data = athena_read.vtk(str(path))
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


def reconnection_series(run_dir: Path) -> tuple[list[float], list[float]]:
    vtk_files = sorted(run_dir.glob("*.vtk"))
    times = []
    rates = []
    for vtk_file in vtk_files:
        try:
            x, y, B, V = load_vtk_fields(vtk_file)
            rate = reconnection_rate(x, y, B, V)
            t = vtk_time(vtk_file)
            times.append(float(t))
            rates.append(float(rate))
        except Exception:
            continue
    order = np.argsort(times)
    return [times[i] for i in order], [rates[i] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A1 Hall on/off comparison.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    variant_ids, cfg = load_variants(Path(args.variants))
    if not variant_ids:
        raise SystemExit("No variants found in variants.json.")

    hall_on_id = cfg.get("hall_on") or (variant_ids[0] if variant_ids else None)
    hall_off_id = cfg.get("hall_off") or (
        variant_ids[1] if len(variant_ids) > 1 else None
    )

    abs_diff_min = float(cfg.get("abs_diff_min", 1.0e-3))
    rel_diff_min = float(cfg.get("rel_diff_min", 0.05))
    eps = float(cfg.get("eps", 1.0e-12))
    eta_tol = float(cfg.get("eta_tol", 1.0e-12))

    variants = {}
    for case_id in variant_ids:
        passfail_path = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
        metrics_path = Path("outputs") / case_id / "analysis" / "metrics.json"
        passfail = load_json(passfail_path)
        metrics = passfail.get("metrics") or load_json(metrics_path)
        result = passfail.get("result") or passfail.get("status")
        variants[case_id] = {
            "case_id": case_id,
            "result": result,
            "metrics": metrics,
            "reconnection_rate_peak": metrics.get("reconnection_rate_peak"),
            "reconnection_rate_peak_time": metrics.get("reconnection_rate_peak_time"),
        }

    hall_on = variants.get(hall_on_id, {})
    hall_off = variants.get(hall_off_id, {})
    hall_on_pass = hall_on.get("result") == "PASS"
    hall_off_pass = hall_off.get("result") == "PASS"

    hall_on_eta = (hall_on.get("metrics") or {}).get("eta_hall_value")
    hall_off_eta = (hall_off.get("metrics") or {}).get("eta_hall_value")
    hall_on_enabled = (hall_on.get("metrics") or {}).get("hall_enabled")
    hall_off_enabled = (hall_off.get("metrics") or {}).get("hall_enabled")

    hall_on_eta_ok = (
        hall_on_eta is not None
        and float(hall_on_eta) > eta_tol
        and hall_on_enabled is True
    )
    hall_off_eta_ok = (
        hall_off_eta is not None
        and abs(float(hall_off_eta)) <= eta_tol
        and hall_off_enabled is False
    )
    hall_config_ok = hall_on_eta_ok and hall_off_eta_ok
    hall_config_reason = "ok" if hall_config_ok else "hall_disablement_not_effective"

    peak_on = hall_on.get("reconnection_rate_peak")
    peak_off = hall_off.get("reconnection_rate_peak")
    peak_time_on = hall_on.get("reconnection_rate_peak_time")
    peak_time_off = hall_off.get("reconnection_rate_peak_time")
    delta_peak_value = None
    ratio_peak_value = None
    ratio_peak_value_reason = None
    delta_peak_time = None
    abs_diff = None
    rel_diff = None
    diff_ok = False
    if peak_on is not None and peak_off is not None:
        delta_peak_value = float(peak_on) - float(peak_off)
        if abs(float(peak_off)) >= eps:
            ratio_peak_value = float(peak_on) / float(peak_off)
        else:
            ratio_peak_value_reason = "peak_off_too_small"
        abs_diff = abs(float(peak_on) - float(peak_off))
        rel_diff = abs_diff / max(abs(float(peak_off)), eps)
        diff_ok = abs_diff >= abs_diff_min or rel_diff >= rel_diff_min
    if peak_time_on is not None and peak_time_off is not None:
        delta_peak_time = float(peak_time_on) - float(peak_time_off)

    compare_pass = hall_on_pass and hall_off_pass and diff_ok and hall_config_ok

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for case_id in variant_ids:
        run_dir = Path("outputs") / case_id / "raw" / "run"
        times, rates = reconnection_series(run_dir)
        if times:
            label = case_id
            if case_id == hall_on_id:
                label = f"{case_id} (Hall on)"
            elif case_id == hall_off_id:
                label = f"{case_id} (Hall off)"
            plt.plot(times, rates, marker="o", label=label)
    plt.xlabel("Time")
    plt.ylabel("Reconnection rate")
    plt.title("Hall-GEM Reconnection Rate (on/off)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reconnection_rate_overlay.png")
    plt.close()

    summary = {
        "variant_ids": variant_ids,
        "hall_on_id": hall_on_id,
        "hall_off_id": hall_off_id,
        "abs_diff_min": abs_diff_min,
        "rel_diff_min": rel_diff_min,
        "eps": eps,
        "eta_tol": eta_tol,
        "hall_on_pass": hall_on_pass,
        "hall_off_pass": hall_off_pass,
        "hall_on_eta_hall_value": hall_on_eta,
        "hall_off_eta_hall_value": hall_off_eta,
        "hall_on_hall_enabled": hall_on_enabled,
        "hall_off_hall_enabled": hall_off_enabled,
        "hall_config_ok": hall_config_ok,
        "hall_config_reason": hall_config_reason,
        "reconnection_rate_peak_on": peak_on,
        "reconnection_rate_peak_off": peak_off,
        "reconnection_rate_peak_time_on": peak_time_on,
        "reconnection_rate_peak_time_off": peak_time_off,
        "delta_peak_value": delta_peak_value,
        "ratio_peak_value": ratio_peak_value,
        "ratio_peak_value_reason": ratio_peak_value_reason,
        "ratio_eps": eps,
        "delta_peak_time": delta_peak_time,
        "reconnection_rate_peak_abs_diff": abs_diff,
        "reconnection_rate_peak_rel_diff": rel_diff,
        "compare_pass": compare_pass,
        "variants": variants,
    }

    metrics_out = {
        "compare_pass": compare_pass,
        "hall_on_pass": hall_on_pass,
        "hall_off_pass": hall_off_pass,
        "hall_config_ok": hall_config_ok,
        "hall_config_reason": hall_config_reason,
        "eta_tol": eta_tol,
        "hall_on_eta_hall_value": hall_on_eta,
        "hall_off_eta_hall_value": hall_off_eta,
        "hall_on_hall_enabled": hall_on_enabled,
        "hall_off_hall_enabled": hall_off_enabled,
        "reconnection_rate_peak_on": peak_on,
        "reconnection_rate_peak_off": peak_off,
        "reconnection_rate_peak_time_on": peak_time_on,
        "reconnection_rate_peak_time_off": peak_time_off,
        "delta_peak_value": delta_peak_value,
        "ratio_peak_value": ratio_peak_value,
        "ratio_peak_value_reason": ratio_peak_value_reason,
        "ratio_eps": eps,
        "delta_peak_time": delta_peak_time,
        "reconnection_rate_peak_abs_diff": abs_diff,
        "reconnection_rate_peak_rel_diff": rel_diff,
        "abs_diff_min": abs_diff_min,
        "rel_diff_min": rel_diff_min,
        "eps": eps,
        "hall_on_id": hall_on_id,
        "hall_off_id": hall_off_id,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


add_athena_vis_path()
import athena_read  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main()
