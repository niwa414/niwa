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


def load_variants(path: Path) -> tuple[str, str, dict]:
    data = load_json(path)
    base_id = data.get("base") or data.get("baseline") or data.get("control")
    perturb_id = (
        data.get("ampplus10") or data.get("perturb") or data.get("treatment")
    )
    if not base_id or not perturb_id:
        raise SystemExit("variants.json must define base and perturb case IDs.")
    return base_id, perturb_id, data


def load_case(case_id: str) -> tuple[dict, str | None]:
    passfail_path = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
    metrics_path = Path("outputs") / case_id / "analysis" / "metrics.json"
    passfail = load_json(passfail_path)
    metrics = passfail.get("metrics") or load_json(metrics_path)
    status = passfail.get("result") or passfail.get("status")
    return metrics, status


def load_warpx_meta(case_id: str) -> dict:
    meta_path = (
        Path("outputs") / case_id / "raw" / "run" / f"warpx_run_{case_id}.json"
    )
    return load_json(meta_path)


def last_lcr_value(meta: dict, key: str) -> float | None:
    history = meta.get("lcr_history") or []
    for row in reversed(history):
        if key in row and row.get(key) is not None:
            try:
                return float(row.get(key))
            except Exception:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze A3 circuit response (base vs perturb)."
    )
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    args = parser.parse_args()

    base_id, perturb_id, cfg = load_variants(Path(args.variants))
    metric_key = str(cfg.get("metric", "feedback_signal_range"))
    ratio_min = float(cfg.get("ratio_min", 1.02))
    expected_source = cfg.get("feedback_signal_source")
    driver_ratio_min = float(cfg.get("driver_ratio_min", 1.1))

    base_metrics, base_status = load_case(base_id)
    perturb_metrics, perturb_status = load_case(perturb_id)

    base_val = base_metrics.get(metric_key)
    perturb_val = perturb_metrics.get(metric_key)
    base_source = base_metrics.get("feedback_signal_source")
    perturb_source = perturb_metrics.get("feedback_signal_source")
    base_radius_range = base_metrics.get("radius_rms_range")
    perturb_radius_range = perturb_metrics.get("radius_rms_range")

    ratio = None
    ratio_reason = None
    if base_val is None or perturb_val is None:
        ratio_reason = "missing_metric"
    else:
        try:
            base_val = float(base_val)
            perturb_val = float(perturb_val)
            if base_val > 0.0:
                ratio = perturb_val / base_val
            else:
                ratio_reason = "base_non_positive"
        except Exception:
            ratio_reason = "invalid_metric"

    ratio_ok = ratio is not None and ratio >= ratio_min
    base_ran = base_metrics.get("ran_to_completion") is True
    perturb_ran = perturb_metrics.get("ran_to_completion") is True
    source_ok = True
    if expected_source is not None:
        source_ok = base_source == expected_source and perturb_source == expected_source
    base_meta = load_warpx_meta(base_id)
    perturb_meta = load_warpx_meta(perturb_id)
    base_driver = last_lcr_value(base_meta, "B_est")
    perturb_driver = last_lcr_value(perturb_meta, "B_est")
    driver_response_ratio = None
    if base_driver is not None and perturb_driver is not None and base_driver > 0.0:
        driver_response_ratio = perturb_driver / base_driver
    driver_response_ok = (
        driver_response_ratio is not None and driver_response_ratio >= driver_ratio_min
    )
    compare_pass = bool(base_ran and perturb_ran and driver_response_ok and source_ok)
    variants_pass = (str(base_status).upper() == "PASS") and (
        str(perturb_status).upper() == "PASS"
    )
    radius_rms_range_ratio = None
    if base_radius_range is not None and perturb_radius_range is not None:
        try:
            base_radius_range = float(base_radius_range)
            perturb_radius_range = float(perturb_radius_range)
            if base_radius_range > 0.0:
                radius_rms_range_ratio = perturb_radius_range / base_radius_range
        except Exception:
            radius_rms_range_ratio = None
    r_proxy_b_rms_range_ratio = None
    if expected_source == "r_proxy_b_rms":
        r_proxy_b_rms_range_ratio = ratio

    known_gaps = []
    known_gap_metrics = {}
    if (
        radius_rms_range_ratio is None or radius_rms_range_ratio < ratio_min
    ) and (r_proxy_b_rms_range_ratio is None or r_proxy_b_rms_range_ratio < ratio_min):
        known_gaps.append("a3_radius_unresponsive_in_window")
        known_gap_metrics = {
            "radius_rms_range_ratio": radius_rms_range_ratio,
            "r_proxy_b_rms_range_ratio": r_proxy_b_rms_range_ratio,
            "ratio_min": ratio_min,
        }

    metrics_out = {
        "base_case_id": base_id,
        "perturb_case_id": perturb_id,
        "metric": metric_key,
        "ratio_min": ratio_min,
        "expected_feedback_signal_source": expected_source,
        "base_feedback_signal_source": base_source,
        "perturb_feedback_signal_source": perturb_source,
        "base_metric": base_val,
        "perturb_metric": perturb_val,
        "metric_ratio": ratio,
        "metric_delta": (perturb_val - base_val)
        if (isinstance(base_val, float) and isinstance(perturb_val, float))
        else None,
        "ratio_ok": ratio_ok,
        "ratio_reason": ratio_reason,
        "feedback_signal_source_ok": source_ok,
        "base_ran_to_completion": base_ran,
        "perturb_ran_to_completion": perturb_ran,
        "compare_pass": compare_pass,
        "variants_pass": variants_pass,
        "driver_response_signal": "B_est_last",
        "base_driver_response": base_driver,
        "perturb_driver_response": perturb_driver,
        "driver_response_ratio": driver_response_ratio,
        "driver_ratio_min": driver_ratio_min,
        "driver_response_ok": driver_response_ok,
        "radius_rms_range_base": base_radius_range,
        "radius_rms_range_perturb": perturb_radius_range,
        "radius_rms_range_ratio": radius_rms_range_ratio,
        "r_proxy_b_rms_range_ratio": r_proxy_b_rms_range_ratio,
        "known_gaps": known_gaps,
        "known_gap_metrics": known_gap_metrics,
        "base_status": base_status,
        "perturb_status": perturb_status,
    }

    summary = {
        "base": {
            "case_id": base_id,
            "status": base_status,
            "metrics": base_metrics,
        },
        "perturb": {
            "case_id": perturb_id,
            "status": perturb_status,
            "metrics": perturb_metrics,
        },
        "compare": metrics_out,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
