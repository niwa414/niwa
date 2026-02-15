#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_passfail(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    data = read_json(path)
    return {
        "case_id": case_id,
        "path": str(path),
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}),
    }


def as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def pick_gamma(metrics: dict) -> float | None:
    for key in ("gamma_fit", "gamma_best", "gamma_m1v_fit_best24", "gamma_m1_fit_best"):
        val = as_float(metrics.get(key))
        if val is not None:
            return val
    return None


def pick_r2(metrics: dict) -> float | None:
    for key in ("r2_fit", "r2_best", "r2_m1v_fit_best24", "r2_fit_best"):
        val = as_float(metrics.get(key))
        if val is not None:
            return val
    return None


def write_seed_drift_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "status", "tilt_seed_enabled", "dynamic_drift_enabled", "gamma_fit", "r2_fit"])
        for row in rows:
            writer.writerow(
                [
                    row.get("case_id"),
                    row.get("status"),
                    row.get("tilt_seed_enabled"),
                    row.get("dynamic_drift_enabled"),
                    row.get("gamma_fit"),
                    row.get("r2_fit"),
                ]
            )


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# B2 Dual-Regime Gate")
    lines.append("")
    lines.append(f"- dual_regime_pass: `{metrics.get('dual_regime_pass')}`")
    lines.append(f"- growth_gamma_fit: `{metrics.get('growth_gamma_fit')}`")
    lines.append(f"- damping_gamma_max: `{metrics.get('damping_gamma_max')}`")
    lines.append(f"- seed_drift_all_negative: `{metrics.get('seed_drift_all_negative')}`")
    lines.append(f"- seed_drift_span_rel: `{metrics.get('seed_drift_span_rel')}`")
    lines.append(f"- seed_effect_rel: `{metrics.get('seed_effect_rel')}`")
    lines.append(f"- drift_effect_rel: `{metrics.get('drift_effect_rel')}`")
    lines.append("")
    lines.append("This gate confirms both growth and damping regimes with a 2x2 seed/drift sensitivity table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 dual-regime gate (growth + damping + seed/drift sensitivity).")
    parser.add_argument("--growth-case", default="m17-b2-tilt-growth-baseline")
    parser.add_argument("--damping-case", default="m17-b2-tilt-m1-compare")
    parser.add_argument(
        "--seed-drift-cases",
        nargs="+",
        default=[
            "m17-b2-tilt-seedON-driftON-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedON-driftOFF-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedOFF-driftON-rhocosE002-N008-mainline",
            "m17-b2-tilt-seedOFF-driftOFF-rhocosE002-N008-mainline",
        ],
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    growth = load_passfail(repo_root, args.growth_case)
    damping = load_passfail(repo_root, args.damping_case)
    seed_cases = [load_passfail(repo_root, case_id) for case_id in args.seed_drift_cases]

    growth_gamma = pick_gamma(growth["metrics"])
    growth_r2 = pick_r2(growth["metrics"])

    damping_gamma_on = as_float(damping["metrics"].get("gamma_on"))
    damping_gamma_off = as_float(damping["metrics"].get("gamma_off"))
    damping_gamma_fit = pick_gamma(damping["metrics"])
    damping_gammas = [g for g in (damping_gamma_on, damping_gamma_off, damping_gamma_fit) if g is not None]
    damping_gamma_max = max(damping_gammas) if damping_gammas else None

    seed_rows = []
    for row in seed_cases:
        m = row["metrics"]
        seed_rows.append(
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "tilt_seed_enabled": m.get("tilt_seed_enabled"),
                "dynamic_drift_enabled": m.get("dynamic_drift_enabled"),
                "gamma_fit": pick_gamma(m),
                "r2_fit": pick_r2(m),
            }
        )

    seed_pass_count = sum(1 for row in seed_rows if row["status"] == "PASS")
    seed_suite_size = len(seed_rows)
    seed_suite_complete = seed_pass_count == seed_suite_size and seed_suite_size >= 4

    seed_gammas = [row["gamma_fit"] for row in seed_rows if row["gamma_fit"] is not None]
    seed_r2 = [row["r2_fit"] for row in seed_rows if row["r2_fit"] is not None]

    seed_drift_all_negative = bool(seed_gammas) and all(g < 0.0 for g in seed_gammas)
    seed_gamma_mean = sum(seed_gammas) / len(seed_gammas) if seed_gammas else None
    seed_gamma_min = min(seed_gammas) if seed_gammas else None
    seed_gamma_max = max(seed_gammas) if seed_gammas else None
    seed_drift_span_rel = None
    if seed_gamma_mean is not None and abs(seed_gamma_mean) > 0.0 and seed_gamma_min is not None and seed_gamma_max is not None:
        seed_drift_span_rel = abs(seed_gamma_max - seed_gamma_min) / abs(seed_gamma_mean)

    seed_on = [row["gamma_fit"] for row in seed_rows if row["tilt_seed_enabled"] is True and row["gamma_fit"] is not None]
    seed_off = [row["gamma_fit"] for row in seed_rows if row["tilt_seed_enabled"] is False and row["gamma_fit"] is not None]
    drift_on = [row["gamma_fit"] for row in seed_rows if row["dynamic_drift_enabled"] is True and row["gamma_fit"] is not None]
    drift_off = [row["gamma_fit"] for row in seed_rows if row["dynamic_drift_enabled"] is False and row["gamma_fit"] is not None]

    seed_effect_rel = None
    if seed_on and seed_off:
        mean_on = sum(seed_on) / len(seed_on)
        mean_off = sum(seed_off) / len(seed_off)
        if mean_off != 0.0:
            seed_effect_rel = abs(mean_on - mean_off) / abs(mean_off)

    drift_effect_rel = None
    if drift_on and drift_off:
        mean_on = sum(drift_on) / len(drift_on)
        mean_off = sum(drift_off) / len(drift_off)
        if mean_off != 0.0:
            drift_effect_rel = abs(mean_on - mean_off) / abs(mean_off)

    growth_case_pass = growth["status"] == "PASS"
    damping_case_pass = damping["status"] == "PASS"
    growth_gamma_positive = growth_gamma is not None and growth_gamma > 0.0
    damping_gamma_negative = damping_gamma_max is not None and damping_gamma_max < 0.0
    seed_r2_all_good = bool(seed_r2) and all(r >= 0.9 for r in seed_r2)
    sensitivity_table_present = (
        seed_suite_complete
        and len(seed_on) > 0
        and len(seed_off) > 0
        and len(drift_on) > 0
        and len(drift_off) > 0
    )

    dual_regime_pass = (
        growth_case_pass
        and damping_case_pass
        and growth_gamma_positive
        and damping_gamma_negative
        and seed_drift_all_negative
        and seed_r2_all_good
        and sensitivity_table_present
    )

    metrics = {
        "growth_case_id": args.growth_case,
        "damping_case_id": args.damping_case,
        "seed_drift_case_ids": list(args.seed_drift_cases),
        "growth_case_pass": growth_case_pass,
        "damping_case_pass": damping_case_pass,
        "seed_drift_pass_count": seed_pass_count,
        "seed_drift_suite_size": seed_suite_size,
        "seed_drift_suite_complete": seed_suite_complete,
        "growth_gamma_fit": growth_gamma,
        "growth_r2_fit": growth_r2,
        "growth_gamma_positive": bool(growth_gamma_positive),
        "damping_gamma_on": damping_gamma_on,
        "damping_gamma_off": damping_gamma_off,
        "damping_gamma_max": damping_gamma_max,
        "damping_gamma_negative": bool(damping_gamma_negative),
        "seed_drift_all_negative": bool(seed_drift_all_negative),
        "seed_drift_r2_all_good": bool(seed_r2_all_good),
        "seed_gamma_mean": seed_gamma_mean,
        "seed_gamma_min": seed_gamma_min,
        "seed_gamma_max": seed_gamma_max,
        "seed_drift_span_rel": seed_drift_span_rel,
        "seed_effect_rel": seed_effect_rel,
        "drift_effect_rel": drift_effect_rel,
        "sensitivity_table_present": bool(sensitivity_table_present),
        "dual_regime_pass": bool(dual_regime_pass),
    }

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    out_table = Path(args.table)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()
    if not out_table.is_absolute():
        out_table = (repo_root / out_table).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_summary, metrics)
    write_seed_drift_table(out_table, seed_rows)


if __name__ == "__main__":
    main()
