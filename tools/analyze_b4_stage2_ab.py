#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def to_float(value) -> float | None:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze B4 Stage2 A/B strategy closure metrics.")
    ap.add_argument("--strategy-off-metrics", required=True)
    ap.add_argument("--strategy-on-metrics", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--compression-metric", default="tilt_amp_ratio")
    ap.add_argument("--target-compression", type=float, default=1.5)
    args = ap.parse_args()

    off = load_json(Path(args.strategy_off_metrics))
    on = load_json(Path(args.strategy_on_metrics))
    metric = args.compression_metric

    off_ratio = to_float(off.get(metric))
    on_ratio = to_float(on.get(metric))
    off_no_nan = bool(off.get("no_nan_in_metrics", False))
    on_no_nan = bool(on.get("no_nan_in_metrics", False))
    off_done = bool(off.get("ran_to_completion", False))
    on_done = bool(on.get("ran_to_completion", False))

    control_reached_target_compression = bool(
        off_ratio is not None and off_ratio >= args.target_compression
    )
    treatment_reached_target_compression = bool(
        on_ratio is not None and on_ratio >= args.target_compression
    )

    field_off = to_float(off.get("field_energy_rel_drift"))
    field_on = to_float(on.get("field_energy_rel_drift"))
    mag_off = to_float(off.get("mag_energy_rel_drift"))
    mag_on = to_float(on.get("mag_energy_rel_drift"))
    b_off = to_float(off.get("b_rms_final"))
    b_on = to_float(on.get("b_rms_final"))

    delta_field = None
    delta_mag = None
    delta_b = None
    if field_off is not None and field_on is not None:
        delta_field = abs(field_on - field_off)
    if mag_off is not None and mag_on is not None:
        delta_mag = abs(mag_on - mag_off)
    if b_off is not None and b_on is not None:
        delta_b = abs(b_on - b_off)

    conservation_delta_quantified = bool(
        delta_field is not None and delta_mag is not None and delta_b is not None
    )
    no_nan_in_metrics = bool(off_no_nan and on_no_nan)
    both_ran_to_completion = bool(off_done and on_done)

    metrics = {
        "compression_metric_used": metric,
        "target_compression_ratio": args.target_compression,
        "strategy_off_ratio": off_ratio,
        "strategy_on_ratio": on_ratio,
        "control_reached_target_compression": control_reached_target_compression,
        "treatment_reached_target_compression": treatment_reached_target_compression,
        "both_ran_to_completion": both_ran_to_completion,
        "no_nan_in_metrics": no_nan_in_metrics,
        "delta_field_energy_rel_drift": delta_field,
        "delta_mag_energy_rel_drift": delta_mag,
        "delta_b_rms_final": delta_b,
        "conservation_delta_quantified": conservation_delta_quantified,
        "strategy_off_metrics_path": str(Path(args.strategy_off_metrics).resolve()),
        "strategy_on_metrics_path": str(Path(args.strategy_on_metrics).resolve()),
    }

    summary = {
        "strategy_off": off,
        "strategy_on": on,
        "metrics": metrics,
    }

    metrics_path = Path(args.metrics).resolve()
    summary_path = Path(args.summary).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
