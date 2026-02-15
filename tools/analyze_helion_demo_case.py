#!/usr/bin/env python3
"""Normalize a source case into Helion-style demo KPIs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_nested(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    if not is_finite_number(value):
        return None
    return float(value)


def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = get_nested(data, key)
        if value is not None:
            return value
    return None


def pick_compression_ratio(
    source_metrics: dict[str, Any], formation_metrics: dict[str, Any]
) -> tuple[float | None, str | None]:
    candidates: list[tuple[float, str]] = []
    for key in (
        "rho_compress_ratio",
        "formation_kpi.rho_delta_rel",
        "formation_kpi_phase.rho_delta_rel_phase",
    ):
        value = first_present(formation_metrics, [key])
        v = as_float(value)
        if v is None:
            continue
        if "rho_delta_rel" in key:
            v = 1.0 + v
        candidates.append((v, f"formation:{key}"))
    for key in ("rho_compress_ratio",):
        value = first_present(source_metrics, [key])
        v = as_float(value)
        if v is None:
            continue
        candidates.append((v, f"source:{key}"))
    if not candidates:
        return None, None
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Helion-style demo metrics for one case.")
    parser.add_argument("--source-metrics", required=True, help="Path to source analysis/metrics.json")
    parser.add_argument(
        "--source-formation",
        required=True,
        help="Path to source analysis/metrics_formation.json",
    )
    parser.add_argument("--output-metrics", required=True, help="Output metrics.json path")
    parser.add_argument("--source-case-id", required=True, help="Original source case id")
    parser.add_argument("--knob-name", default="shift", help="Single trade-study knob name")
    parser.add_argument("--knob-value", type=float, required=True, help="Knob value for this case")
    parser.add_argument("--label", required=True, help="Case label: baseline/knob_minus/knob_plus")
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=1.0e-6,
        help="Absolute energy residual threshold for energy_accounting_ok",
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=1.001,
        help="Compression ratio pass threshold used by PASSFAIL gate",
    )
    parser.add_argument(
        "--tilt-threshold",
        type=float,
        default=0.05,
        help="Tilt amplitude pass threshold used by PASSFAIL gate",
    )
    args = parser.parse_args()

    source_metrics_path = Path(args.source_metrics)
    source_formation_path = Path(args.source_formation)
    output_metrics_path = Path(args.output_metrics)

    source_metrics = load_json(source_metrics_path)
    source_formation = load_json(source_formation_path)

    ran_to_completion = bool(source_metrics.get("ran_to_completion"))

    source_no_nan_flag = source_metrics.get("no_nan_in_metrics")
    formation_has_nan = source_formation.get("formation_has_nan")

    merge_time_s = as_float(
        first_present(
            source_metrics,
            [
                "merge_time",
                "merge_time_seedmask",
            ],
        )
    )
    merge_time_exists = source_metrics.get("merge_time_exists")
    if merge_time_exists is None:
        merge_time_exists = merge_time_s is not None
    merge_time_exists = bool(merge_time_exists)

    compression_ratio, compression_ratio_source = pick_compression_ratio(
        source_metrics, source_formation
    )

    tilt_amp_max = as_float(
        first_present(
            source_metrics,
            [
                "tilt_post_merge_amp_max",
                "tilt_amp_max",
                "tilt_amp_ratio",
            ],
        )
    )
    tilt_growth_rate = as_float(
        first_present(
            source_metrics,
            [
                "gamma_fit_best",
                "gamma_m1v_fit_best24",
                "gamma_m1_fit_best",
                "gamma_best",
                "growth_gamma_fit",
            ],
        )
    )

    energy_residual_rel = as_float(
        first_present(
            source_metrics,
            [
                "energy_residual_rel",
                "circuit_chain.energy_residual_rel",
            ],
        )
    )
    energy_accounting_ok = (
        energy_residual_rel is not None and abs(energy_residual_rel) <= args.energy_threshold
    )

    # Use engineering-critical fields to judge NaN safety for gate decisions.
    critical_values = [
        merge_time_s,
        compression_ratio,
        tilt_amp_max,
        tilt_growth_rate,
        energy_residual_rel,
    ]
    no_nan_critical_metrics = all((val is None) or math.isfinite(val) for val in critical_values) and (
        not bool(formation_has_nan)
    )
    if source_no_nan_flag is True:
        no_nan_in_metrics = True
    else:
        no_nan_in_metrics = bool(no_nan_critical_metrics)

    metrics = {
        "source_case_id": args.source_case_id,
        "demo_case_label": args.label,
        "knob_name": args.knob_name,
        "knob_value": float(args.knob_value),
        "ran_to_completion": ran_to_completion,
        "no_nan_in_metrics": no_nan_in_metrics,
        "no_nan_in_source_metrics_flag": source_no_nan_flag,
        "no_nan_in_critical_metrics": bool(no_nan_critical_metrics),
        "merge_time_exists": merge_time_exists,
        "merge_time_s": merge_time_s,
        "compression_ratio": compression_ratio,
        "compression_ratio_source": compression_ratio_source,
        "tilt_amp_max": tilt_amp_max,
        "tilt_growth_rate": tilt_growth_rate,
        "energy_residual_rel": energy_residual_rel,
        "energy_accounting_ok": bool(energy_accounting_ok),
        "recapture_efficiency": as_float(
            first_present(
                source_metrics,
                [
                    "eta_recaptured",
                    "circuit_chain.eta_recaptured",
                ],
            )
        ),
        "load_force_proxy_peak_N": as_float(
            first_present(source_metrics, ["force_proxy_peak_N", "mainline_load_force_peak_N"])
        ),
        "dphi_dt_peak_V": as_float(first_present(source_metrics, ["dphi_dt_peak", "dphi_dt_max"])),
        "source_metrics_path": str(source_metrics_path),
        "source_formation_path": str(source_formation_path),
        "pass_thresholds": {
            "compression_ratio_min": float(args.compression_threshold),
            "tilt_amp_max_max": float(args.tilt_threshold),
            "energy_residual_abs_max": float(args.energy_threshold),
        },
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
