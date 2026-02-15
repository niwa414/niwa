#!/usr/bin/env python3
import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

B2_GATE = {
    "vdrop_fit_max": -0.20,
    "r2_n_min": 0.98,
    "r2_B_min": 0.95,
    "r2_T_min": 0.95,
    "alpha_n": (-1.20, -1.00),
    "alpha_B": (-0.90, -0.65),
    "alpha_T": (-3.40, -2.70),
    "rel_diff_alpha_n": 0.05,
    "rel_diff_alpha_B": 0.15,
    "rel_diff_alpha_T": 0.12,
    "progress_min": 0.10,
    "progress_max": 0.80,
}


def run(cmd, cwd=None, log_path=None):
    if log_path is None:
        subprocess.check_call(cmd, cwd=cwd)
        return
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def parse_hst(path: Path):
    header = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("# [1]"):
                header = line
                break
    if header is None:
        return {}
    labels = re.findall(r"\[\d+\]=([^\[]+)", header)
    labels = [lab.strip() for lab in labels]
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    data = {name: [] for name in labels}
    for row in rows:
        for name, val in zip(labels, row):
            data[name].append(val)
    return data


def read_json_rows(path: Path):
    return json.loads(path.read_text())


def analyze_kirtley(run_dir: Path, analyze_py: Path, mask_mode: str, mask_threshold: str,
                    waveform: Path, wf_scale: float, wf_bias: float, wall_depth: float,
                    fit_args=None):
    out_prefix = run_dir / f"analysis_{mask_mode}_{mask_threshold.replace('.', 'p')}"
    cmd = [
        "python", str(analyze_py),
        "--mode", "athena",
        "--run-dir", str(run_dir),
        "--vtk-pattern", "*.vtk",
        "--mask-mode", mask_mode,
        "--output-prefix", str(out_prefix),
        "--bext-waveform", str(waveform),
        "--bext-waveform-kind", "fraction",
        "--bext-waveform-scale", f"{wf_scale}",
        "--bext-waveform-bias", f"{wf_bias}",
        "--bext-progress-source", "waveform",
        "--bext-sign-source", "waveform",
        "--wall-depth-max", f"{wall_depth}",
    ]
    if mask_mode == "mass_core":
        cmd += ["--mass-fraction", mask_threshold]
    else:
        cmd += ["--mask-threshold", mask_threshold]
    if fit_args:
        cmd += fit_args
    run(cmd)
    json_path = Path(str(out_prefix) + ".json")
    fit_path = Path(str(out_prefix) + ".fit.json")
    return json_path, fit_path


def monotonic_non_decreasing(values, tol=1.0e-12):
    last = None
    for v in values:
        if not math.isfinite(v):
            return False
        if last is not None and v < last - tol:
            return False
        last = v
    return True


def compute_window_stats(rows, prog_min, prog_max):
    win = [r for r in rows if not r.get("mask_empty")
           and prog_min <= r.get("Bext_progress", float("nan")) <= prog_max]
    if len(win) < 2:
        return None
    vols = [r["volume_m3"] for r in win]
    rho = [r.get("rho_avg_core", float("nan")) for r in win]
    vdrop = (vols[-1] - vols[0]) / vols[0] if vols[0] else float("nan")
    rhodrop = (rho[-1] - rho[0]) / rho[0] if rho[0] else float("nan")
    inc = sum(1 for i in range(1, len(vols)) if vols[i] > vols[i - 1])
    dec = sum(1 for i in range(1, len(vols)) if vols[i] < vols[i - 1])
    running_min = vols[0]
    max_rebound = 0.0
    for v in vols:
        if v < running_min:
            running_min = v
        max_rebound = max(max_rebound, v - running_min)
    total_drop = vols[0] - vols[-1]
    rebound_frac = max_rebound / total_drop if total_drop > 0 else float("nan")
    return {
        "n_points": len(vols),
        "vdrop": vdrop,
        "rho_change": rhodrop,
        "inc": inc,
        "dec": dec,
        "rebound_frac": rebound_frac,
    }


def rel_diff(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a == 0.0:
        return float("nan")
    return abs(a - b) / abs(a)


def eval_stage_0a(json_rows):
    # require finite volume/B avg across rows
    for r in json_rows:
        for key in ("volume_m3", "B_avg_T", "n_avg_m3"):
            val = r.get(key, float("nan"))
            if not math.isfinite(val):
                return False, {"bad_key": key}
    return True, {}


def eval_stage_0b(json_rows):
    progress = [r.get("Bext_frac_waveform", float("nan")) for r in json_rows]
    ok = monotonic_non_decreasing(progress)
    return ok, {"progress_start": progress[0], "progress_end": progress[-1]}


def eval_stage_0c(hst):
    if not hst:
        return False, {"error": "no_hst"}
    vol = hst.get("vol_mv", [])
    mass = hst.get("mass_mv", [])
    etot = hst.get("Etot_mv", [])
    if len(vol) < 2 or len(mass) < 2:
        return False, {"error": "insufficient_hst"}
    vol_drop = (vol[-1] - vol[0]) / vol[0] if vol[0] else float("nan")
    mass_change = (mass[-1] - mass[0]) / mass[0] if mass[0] else float("nan")
    ok = vol_drop < 0.0 and abs(mass_change) < 0.05 and all(math.isfinite(v) for v in etot)
    return ok, {"vol_drop": vol_drop, "mass_change": mass_change}


def eval_stage_1(rows_03, rows_02):
    stats_03 = compute_window_stats(rows_03, 0.10, 0.60)
    stats_02 = compute_window_stats(rows_02, 0.10, 0.60)
    if stats_03 is None or stats_02 is None:
        return False, {"error": "insufficient_window"}
    ok = (stats_03["vdrop"] < 0.0 and stats_02["vdrop"] < 0.0
          and stats_03["rho_change"] > 0.0 and stats_02["rho_change"] > 0.0)
    return ok, {"stats_0p3": stats_03, "stats_0p2": stats_02}


def eval_stage_2(fit_03, fit_02, rows_03, rows_02):
    # full-window sanity (non-blocking)
    full_03 = compute_window_stats(rows_03, B2_GATE["progress_min"], B2_GATE["progress_max"])
    full_02 = compute_window_stats(rows_02, B2_GATE["progress_min"], B2_GATE["progress_max"])

    checks = {
        "full_window_0p3": full_03,
        "full_window_0p2": full_02,
    }

    if not fit_03 or not fit_02:
        return False, {"error": "missing_fit", **checks}

    # primary thresholds
    ok = True
    if fit_03.get("n_points", 0) < 25:
        ok = False
    if fit_03.get("fit_segment_selected") is False:
        ok = False
    if fit_03.get("vol_log_span", 0.0) >= 0.0:
        ok = False
    if "fit_segment_selected" in fit_03 and fit_03["fit_segment_selected"] is False:
        ok = False

    vdrop_fit = fit_03.get("vol_log_span")
    if vdrop_fit is None:
        ok = False

    # Vdrop_fit requirement (use vol_log_span sign + bound by alpha window via vdrop in fit rows)
    if fit_03.get("vdrop_fit") is not None:
        if fit_03["vdrop_fit"] > B2_GATE["vdrop_fit_max"]:
            ok = False

    for key, min_val in ("r2_n", B2_GATE["r2_n_min"]), ("r2_B", B2_GATE["r2_B_min"]), ("r2_T", B2_GATE["r2_T_min"]):
        if fit_03.get(key) is None or not math.isfinite(fit_03.get(key)) or fit_03[key] < min_val:
            ok = False

    for key, bounds in ("alpha_n", B2_GATE["alpha_n"]), ("alpha_B", B2_GATE["alpha_B"]), ("alpha_T", B2_GATE["alpha_T"]):
        val = fit_03.get(key)
        if val is None or not math.isfinite(val) or not (bounds[0] <= val <= bounds[1]):
            ok = False

    # consistency checks
    for key, tol in ("alpha_n", B2_GATE["rel_diff_alpha_n"]), ("alpha_B", B2_GATE["rel_diff_alpha_B"]), ("alpha_T", B2_GATE["rel_diff_alpha_T"]):
        d = rel_diff(fit_03.get(key), fit_02.get(key))
        if not math.isfinite(d) or d > tol:
            ok = False
        checks[f"rel_diff_{key}"] = d

    checks["fit_0p3"] = fit_03
    checks["fit_0p2"] = fit_02
    return ok, checks


def load_fit(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["3d0a", "3d0b", "3d0c", "3d1", "3d2"])
    ap.add_argument("--athena-bin", default="athena-24.0/bin/athena")
    ap.add_argument("--input", required=True)
    ap.add_argument("--run-root", default="outputs/mhd")
    ap.add_argument("--set", action="append", default=[], help="Override parameter, e.g. problem/apply_bext_emf=true")
    ap.add_argument("--analyze-py", default="tools/analyze_kirtley_scaling.py")
    ap.add_argument("--waveform", default="scenes/belova_b2_mirror_ramp_smooth.csv")
    ap.add_argument("--waveform-scale", type=float, default=0.05)
    ap.add_argument("--waveform-bias", type=float, default=0.05)
    ap.add_argument("--wall-depth-max", type=float, default=0.0625)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_root) / f"{args.stage}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.athena_bin, "-i", args.input, "-d", str(run_dir)]
    cmd.extend(args.set)
    log_path = run_dir / "run.log"
    run(cmd, log_path=log_path)

    result = {
        "stage": args.stage,
        "run_dir": str(run_dir),
        "pass": False,
        "checks": {},
    }

    vtk_files = sorted(run_dir.glob("*.vtk"))
    result["checks"]["vtk_count"] = len(vtk_files)

    analyze_py = Path(args.analyze_py)
    waveform = Path(args.waveform)

    if args.stage in ("3d0a", "3d0b"):
        json_path, _ = analyze_kirtley(
            run_dir,
            analyze_py,
            "rho",
            "0.0",
            waveform,
            args.waveform_scale,
            args.waveform_bias,
            args.wall_depth_max,
        )
        rows = read_json_rows(json_path)
        if args.stage == "3d0a":
            ok, info = eval_stage_0a(rows)
        else:
            ok, info = eval_stage_0b(rows)
        result["pass"] = ok
        result["checks"].update(info)

    elif args.stage == "3d0c":
        hst_files = list(run_dir.glob("*.hst"))
        hst = parse_hst(hst_files[0]) if hst_files else {}
        ok, info = eval_stage_0c(hst)
        result["pass"] = ok
        result["checks"].update(info)

    elif args.stage in ("3d1", "3d2"):
        fit_args = None
        if args.stage == "3d2":
            fit_args = [
                "--fit-progress-min", f"{B2_GATE['progress_min']}",
                "--fit-progress-max", f"{B2_GATE['progress_max']}",
                "--fit-segment", "monotonic",
                "--fit-smooth-window", "3",
                "--fit-allow-increase-frac", "0.10",
                "--fit-rebound-frac", "0.05",
                "--fit-min-points", "25",
            ]
        json_03, fit_03 = analyze_kirtley(
            run_dir,
            analyze_py,
            "mass_core",
            "0.30",
            waveform,
            args.waveform_scale,
            args.waveform_bias,
            args.wall_depth_max,
            fit_args=fit_args,
        )
        json_02, fit_02 = analyze_kirtley(
            run_dir,
            analyze_py,
            "mass_core",
            "0.20",
            waveform,
            args.waveform_scale,
            args.waveform_bias,
            args.wall_depth_max,
            fit_args=fit_args,
        )
        rows_03 = read_json_rows(json_03)
        rows_02 = read_json_rows(json_02)
        if args.stage == "3d1":
            ok, info = eval_stage_1(rows_03, rows_02)
        else:
            fit_j_03 = load_fit(fit_03)
            fit_j_02 = load_fit(fit_02)
            ok, info = eval_stage_2(fit_j_03, fit_j_02, rows_03, rows_02)
        result["pass"] = ok
        result["checks"].update(info)

    out_path = run_dir / "PASSFAIL.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[PASSFAIL] {out_path}")


if __name__ == "__main__":
    main()
