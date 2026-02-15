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


def get_int(metrics: dict, *keys: str) -> int | None:
    for key in keys:
        if key in metrics:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                return int(val)
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
    passfail_metrics = passfail.get("metrics") or {}
    metrics = load_json(metrics_path)
    if passfail_metrics:
        merged = dict(passfail_metrics)
        merged.update(metrics)
        metrics = merged
    result = passfail.get("result") or passfail.get("status")
    return metrics, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare m1 diagnostics for B2 tilt on/off variants.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    args = parser.parse_args()

    cfg = load_variants(Path(args.variants))
    tilt_on = cfg.get("tilt_on")
    tilt_off = cfg.get("tilt_off")
    if not tilt_on or not tilt_off:
        raise SystemExit("variants.json must define tilt_on and tilt_off case IDs.")

    gamma_ratio_min = float(cfg.get("gamma_ratio_min", 2.0))
    r2_min = float(cfg.get("r2_m1_min", 0.9))
    m1_len_min = int(cfg.get("m1_len_min", 24))
    ratio_eps = float(cfg.get("gamma_ratio_eps", 1e-12))

    metrics_on, result_on = load_case_metrics(tilt_on)
    metrics_off, result_off = load_case_metrics(tilt_off)

    gamma_on = get_float(metrics_on, "gamma_m1_fit_best", "gamma_fit_best", "gamma_fit")
    gamma_off = get_float(metrics_off, "gamma_m1_fit_best", "gamma_fit_best", "gamma_fit")
    r2_on = get_float(metrics_on, "r2_m1_fit_best", "r2_fit_best", "r2_fit")
    r2_off = get_float(metrics_off, "r2_m1_fit_best", "r2_fit_best", "r2_fit")
    len_on = get_int(metrics_on, "m1_ratio_series_len", "tilt_amp_series_len")
    len_off = get_int(metrics_off, "m1_ratio_series_len", "tilt_amp_series_len")
    kind_on = metrics_on.get("m1_series_kind")
    kind_off = metrics_off.get("m1_series_kind")
    fit_source_on = metrics_on.get("fit_series_source")
    fit_source_off = metrics_off.get("fit_series_source")
    ran_on = get_bool(metrics_on, "ran_to_completion")
    ran_off = get_bool(metrics_off, "ran_to_completion")

    gamma_ratio = None
    if gamma_on is not None:
        denom = abs(gamma_off) if gamma_off is not None else 0.0
        gamma_ratio = abs(gamma_on) / max(denom, ratio_eps)

    len_on_ok = (len_on is not None) and (len_on >= m1_len_min)
    len_off_ok = (len_off is not None) and (len_off >= m1_len_min)
    r2_on_ok = (r2_on is not None) and (r2_on >= r2_min)
    r2_off_ok = (r2_off is not None) and (r2_off >= r2_min)
    gamma_ratio_ok = (gamma_ratio is not None) and (gamma_ratio >= gamma_ratio_min)

    compare_pass = bool(ran_on) and bool(ran_off) and len_on_ok and len_off_ok and r2_on_ok and r2_off_ok and gamma_ratio_ok

    summary = {
        "tilt_on": tilt_on,
        "tilt_off": tilt_off,
        "gamma_on": gamma_on,
        "gamma_off": gamma_off,
        "gamma_ratio": gamma_ratio,
        "gamma_ratio_min": gamma_ratio_min,
        "r2_on": r2_on,
        "r2_off": r2_off,
        "r2_m1_min": r2_min,
        "m1_len_min": m1_len_min,
        "m1_len_on": len_on,
        "m1_len_off": len_off,
        "m1_series_kind_on": kind_on,
        "m1_series_kind_off": kind_off,
        "fit_series_source_on": fit_source_on,
        "fit_series_source_off": fit_source_off,
        "ran_to_completion_on": ran_on,
        "ran_to_completion_off": ran_off,
        "len_on_ok": len_on_ok,
        "len_off_ok": len_off_ok,
        "r2_on_ok": r2_on_ok,
        "r2_off_ok": r2_off_ok,
        "gamma_ratio_ok": gamma_ratio_ok,
        "ratio_eps": ratio_eps,
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
        "r2_on": r2_on,
        "r2_off": r2_off,
        "r2_m1_min": r2_min,
        "m1_len_min": m1_len_min,
        "m1_len_on": len_on,
        "m1_len_off": len_off,
        "m1_series_kind_on": kind_on,
        "m1_series_kind_off": kind_off,
        "fit_series_source_on": fit_source_on,
        "fit_series_source_off": fit_source_off,
        "ran_to_completion_on": ran_on,
        "ran_to_completion_off": ran_off,
        "len_on_ok": len_on_ok,
        "len_off_ok": len_off_ok,
        "r2_on_ok": r2_on_ok,
        "r2_off_ok": r2_off_ok,
        "gamma_ratio_ok": gamma_ratio_ok,
    }

    Path(args.metrics).write_text(json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
