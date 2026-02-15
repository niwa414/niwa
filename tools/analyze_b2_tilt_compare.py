#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_variants(path: Path) -> dict:
    return load_json(path)


def get_float(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def get_bool(metrics: dict, key: str) -> bool | None:
    if key not in metrics:
        return None
    val = metrics.get(key)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return None


def load_case_metrics(case_id: str) -> tuple[dict, str | None]:
    passfail_path = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
    metrics_path = Path("outputs") / case_id / "analysis" / "metrics.json"
    passfail = load_json(passfail_path)
    metrics = passfail.get("metrics") or load_json(metrics_path)
    result = passfail.get("result") or passfail.get("status")
    return metrics, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B2 tilt on/off growth metrics.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    args = parser.parse_args()

    cfg = load_variants(Path(args.variants))
    tilt_on = cfg.get("tilt_on")
    tilt_off = cfg.get("tilt_off")
    if not tilt_on or not tilt_off:
        raise SystemExit("variants.json must define tilt_on and tilt_off case IDs.")

    gamma_ratio_min = float(cfg.get("gamma_ratio_min", 5.0))
    tilt_ratio_on_min = float(cfg.get("tilt_ratio_on_min", 1.2))
    tilt_ratio_off_max = float(cfg.get("tilt_ratio_off_max", 1.05))
    t_cross_ratio_min = float(cfg.get("t_cross_ratio_min", 1.10))
    ratio_eps = float(cfg.get("gamma_ratio_eps", 1e-12))

    metrics_on, result_on = load_case_metrics(tilt_on)
    metrics_off, result_off = load_case_metrics(tilt_off)

    gamma_on = get_float(metrics_on, "gamma_fit", "gamma_best")
    gamma_off = get_float(metrics_off, "gamma_fit", "gamma_best")
    r2_on = get_float(metrics_on, "r2_fit", "r2_best")
    r2_off = get_float(metrics_off, "r2_fit", "r2_best")
    fit_points_on = get_float(metrics_on, "fit_points")
    fit_points_off = get_float(metrics_off, "fit_points")
    tilt_ratio_on = get_float(metrics_on, "tilt_amp_ratio")
    tilt_ratio_off = get_float(metrics_off, "tilt_amp_ratio")
    tilt_len_on = get_float(metrics_on, "tilt_amp_series_len")
    tilt_len_off = get_float(metrics_off, "tilt_amp_series_len")
    t_cross_on_exists = get_bool(metrics_on, "tilt_cross_exists")
    t_cross_off_exists = get_bool(metrics_off, "tilt_cross_exists")
    t_cross_on = get_float(metrics_on, "tilt_cross_time")
    t_cross_off = get_float(metrics_off, "tilt_cross_time")
    t_cross_ratio_used = get_float(metrics_on, "tilt_cross_ratio_used")
    ran_on = get_bool(metrics_on, "ran_to_completion")
    ran_off = get_bool(metrics_off, "ran_to_completion")

    gamma_ratio = None
    if gamma_on is not None:
        denom = abs(gamma_off) if gamma_off is not None else 0.0
        gamma_ratio = abs(gamma_on) / max(denom, ratio_eps)

    t_cross_ratio = None
    t_cross_ratio_mode = "missing_on"
    t_cross_pass = False
    if t_cross_on_exists:
        if not t_cross_off_exists:
            t_cross_ratio = 1.0e12
            t_cross_ratio_mode = "off_missing_pass"
            t_cross_pass = True
        else:
            denom = max(t_cross_on or 0.0, ratio_eps)
            t_cross_ratio = (t_cross_off or 0.0) / denom
            t_cross_ratio_mode = "normal"
            t_cross_pass = t_cross_ratio >= t_cross_ratio_min

    compare_pass = bool(ran_on) and bool(ran_off) and t_cross_pass

    summary = {
        "tilt_on": tilt_on,
        "tilt_off": tilt_off,
        "gamma_on": gamma_on,
        "gamma_off": gamma_off,
        "gamma_ratio": gamma_ratio,
        "gamma_ratio_min": gamma_ratio_min,
        "tilt_ratio_on": tilt_ratio_on,
        "tilt_ratio_off": tilt_ratio_off,
        "tilt_ratio_on_min": tilt_ratio_on_min,
        "tilt_ratio_off_max": tilt_ratio_off_max,
        "t_cross_on_exists": t_cross_on_exists,
        "t_cross_off_exists": t_cross_off_exists,
        "t_cross_on": t_cross_on,
        "t_cross_off": t_cross_off,
        "t_cross_ratio": t_cross_ratio,
        "t_cross_ratio_min": t_cross_ratio_min,
        "t_cross_ratio_mode": t_cross_ratio_mode,
        "t_cross_ratio_used": t_cross_ratio_used,
        "t_cross_pass": t_cross_pass,
        "ratio_eps": ratio_eps,
        "r2_on": r2_on,
        "r2_off": r2_off,
        "fit_points_on": fit_points_on,
        "fit_points_off": fit_points_off,
        "tilt_amp_series_len_on": tilt_len_on,
        "tilt_amp_series_len_off": tilt_len_off,
        "ran_to_completion_on": ran_on,
        "ran_to_completion_off": ran_off,
        "result_on": result_on,
        "result_off": result_off,
        "compare_pass": compare_pass,
    }

    metrics_out = {
        "compare_pass": compare_pass,
        "tilt_on": tilt_on,
        "tilt_off": tilt_off,
        "gamma_on": gamma_on,
        "gamma_off": gamma_off,
        "gamma_ratio": gamma_ratio,
        "gamma_ratio_min": gamma_ratio_min,
        "tilt_ratio_on": tilt_ratio_on,
        "tilt_ratio_off": tilt_ratio_off,
        "tilt_ratio_on_min": tilt_ratio_on_min,
        "tilt_ratio_off_max": tilt_ratio_off_max,
        "t_cross_on_exists": t_cross_on_exists,
        "t_cross_off_exists": t_cross_off_exists,
        "t_cross_on": t_cross_on,
        "t_cross_off": t_cross_off,
        "t_cross_ratio": t_cross_ratio,
        "t_cross_ratio_min": t_cross_ratio_min,
        "t_cross_ratio_mode": t_cross_ratio_mode,
        "t_cross_ratio_used": t_cross_ratio_used,
        "t_cross_pass": t_cross_pass,
        "ratio_eps": ratio_eps,
        "r2_on": r2_on,
        "r2_off": r2_off,
        "fit_points_on": fit_points_on,
        "fit_points_off": fit_points_off,
        "tilt_amp_series_len_on": tilt_len_on,
        "tilt_amp_series_len_off": tilt_len_off,
        "ran_to_completion_on": ran_on,
        "ran_to_completion_off": ran_off,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
