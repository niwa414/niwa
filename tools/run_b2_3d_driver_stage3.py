#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bool_of(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Build B2 stage3 integration summary.")
    parser.add_argument("--output-root", required=True, help="Case raw run directory.")
    parser.add_argument(
        "--stage1-metrics",
        default="outputs/m19-b2-3d-driver-stage1-gate60/analysis/metrics.json",
    )
    parser.add_argument(
        "--stage2-metrics",
        default="outputs/m20-b2-3d-driver-stage2-gate80-dense/analysis/metrics.json",
    )
    parser.add_argument(
        "--tilt-cases",
        nargs="+",
        default=[
            "m17-b2-tilt-seedON-driftON-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedON-driftOFF-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedOFF-driftON-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedOFF-driftOFF-rhocosE002-N008-mainline",
        ],
        help="Case ids used for B2 tilt 2x2 matrix integration checks.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_path = (root / args.stage1_metrics).resolve()
    stage2_path = (root / args.stage2_metrics).resolve()
    stage1 = load_json(stage1_path)
    stage2 = load_json(stage2_path)

    stage1_pass = bool_of(stage1.get("stage1_suite_pass"), False) or bool_of(stage1.get("stage_3d1_pass"), False)
    stage2_transition_pass = bool_of(stage2.get("stage2_transition_pass"), False) or bool_of(stage2.get("stage_transition_pass"), False)
    stage2_strict_pass = bool_of(stage2.get("stage_strict_pass"), False)

    tilt_records: list[dict] = []
    tilt_pass_flags = []
    gamma_values = []
    for case_id in args.tilt_cases:
        passfail_path = root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
        data = load_json(passfail_path)
        result = str(data.get("result", "")).upper()
        passed = result == "PASS"
        tilt_pass_flags.append(passed)
        metrics = data.get("metrics") or {}
        gamma = metrics.get("gamma_m1v_fit_best24")
        gamma_values.append(gamma)
        tilt_records.append(
            {
                "case_id": case_id,
                "passfail_path": str(passfail_path),
                "result": result or "UNKNOWN",
                "gamma_m1v_fit_best24": gamma,
            }
        )

    tilt_matrix_all_pass = all(tilt_pass_flags) if tilt_pass_flags else False
    tilt_gamma_all_negative = all(
        isinstance(g, (int, float)) and g < 0.0 for g in gamma_values
    ) if gamma_values else False

    seed_on = [rec["gamma_m1v_fit_best24"] for rec in tilt_records if "seedON" in rec["case_id"]]
    seed_off = [rec["gamma_m1v_fit_best24"] for rec in tilt_records if "seedOFF" in rec["case_id"]]
    drift_on = [rec["gamma_m1v_fit_best24"] for rec in tilt_records if "driftON" in rec["case_id"]]
    drift_off = [rec["gamma_m1v_fit_best24"] for rec in tilt_records if "driftOFF" in rec["case_id"]]

    def avg(values: list[object]) -> float | None:
        good = [float(v) for v in values if isinstance(v, (int, float))]
        if not good:
            return None
        return sum(good) / float(len(good))

    gamma_seed_on_avg = avg(seed_on)
    gamma_seed_off_avg = avg(seed_off)
    gamma_drift_on_avg = avg(drift_on)
    gamma_drift_off_avg = avg(drift_off)

    stage3_ready = (
        stage1_pass
        and stage2_transition_pass
        and tilt_matrix_all_pass
        and tilt_gamma_all_negative
    )

    summary = {
        "suite": "b2_3d_driver_stage3_integration",
        "stage1_metrics_path": str(stage1_path),
        "stage2_metrics_path": str(stage2_path),
        "stage1_pass": stage1_pass,
        "stage2_transition_pass": stage2_transition_pass,
        "stage2_strict_pass": stage2_strict_pass,
        "tilt_matrix_all_pass": tilt_matrix_all_pass,
        "tilt_gamma_all_negative": tilt_gamma_all_negative,
        "gamma_seed_on_avg": gamma_seed_on_avg,
        "gamma_seed_off_avg": gamma_seed_off_avg,
        "gamma_drift_on_avg": gamma_drift_on_avg,
        "gamma_drift_off_avg": gamma_drift_off_avg,
        "tilt_cases": tilt_records,
        "stage3_ready": stage3_ready,
        "known_gap_strict_stage2": (not stage2_strict_pass),
    }

    (output_root / "stage3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
