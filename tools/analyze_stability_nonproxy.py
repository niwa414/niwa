#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


HEADER_TOKEN_RE = re.compile(r"#\[(\d+)\]([^\s]+)")


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
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {},
    }


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_reduced_dir(repo_root: Path, case_id: str) -> Path:
    base = repo_root / "outputs" / case_id / "raw" / "run"
    candidates = [
        base / "diags" / "reducedfiles",
        base / "diag" / "reducedfiles",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_reduced_series(path: Path) -> tuple[dict[int, str], list[list[float]]]:
    header_map: dict[int, str] = {}
    rows: list[list[float]] = []
    if not path.exists():
        return header_map, rows
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#["):
                    for idx_s, name in HEADER_TOKEN_RE.findall(line):
                        try:
                            header_map[int(idx_s)] = name
                        except Exception:
                            continue
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    rows.append([float(token) for token in parts])
                except Exception:
                    continue
    except Exception:
        return {}, []
    return header_map, rows


def pick_column(header_map: dict[int, str], preferred_names: list[str], fallback_idx: int | None = None) -> int | None:
    norm_map = {idx: normalize_name(name) for idx, name in header_map.items()}
    preferred_norm = [normalize_name(name) for name in preferred_names]
    for candidate in preferred_norm:
        for idx, token in norm_map.items():
            if token == candidate:
                return idx
    if fallback_idx is not None:
        return fallback_idx
    return None


def extract_times_and_values(rows: list[list[float]], time_col: int, value_col: int) -> tuple[list[float], list[float]]:
    times = []
    values = []
    for row in rows:
        if len(row) <= max(time_col, value_col):
            continue
        t = row[time_col]
        v = row[value_col]
        if not (math.isfinite(t) and math.isfinite(v)):
            continue
        if t < 0.0:
            continue
        times.append(float(t))
        values.append(float(v))
    return times, values


def endpoint_gamma(times: list[float], values: list[float], eps: float = 1.0e-30) -> float | None:
    if len(times) < 2 or len(values) < 2:
        return None
    dt = float(times[-1] - times[0])
    if dt <= 0.0:
        return None
    v0 = max(abs(float(values[0])), eps)
    v1 = max(abs(float(values[-1])), eps)
    return float(math.log(v1 / v0) / dt)


def span_rel(values: list[float], eps: float = 1.0e-30) -> float | None:
    if len(values) < 2:
        return None
    denom = max(abs(float(values[0])), eps)
    return float(abs(float(values[-1]) - float(values[0])) / denom)


def monotonic_fraction(values: list[float], increasing: bool = True) -> float | None:
    if len(values) < 2:
        return None
    good = 0
    total = 0
    for idx in range(1, len(values)):
        dv = values[idx] - values[idx - 1]
        total += 1
        if increasing:
            if dv >= 0.0:
                good += 1
        else:
            if dv <= 0.0:
                good += 1
    if total <= 0:
        return None
    return float(good / total)


def write_modal_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "usable",
                "rho_samples",
                "mom_samples",
                "gamma_rho_endpoint",
                "gamma_mom_endpoint",
                "shearing_dominance",
                "rho_span_rel",
                "mom_span_rel",
                "rho_monotonic_decreasing_frac",
                "mom_monotonic_increasing_frac",
                "case_modal_pass",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("case_id"),
                    row.get("usable"),
                    row.get("rho_samples"),
                    row.get("mom_samples"),
                    row.get("gamma_rho_endpoint"),
                    row.get("gamma_mom_endpoint"),
                    row.get("shearing_dominance"),
                    row.get("rho_span_rel"),
                    row.get("mom_span_rel"),
                    row.get("rho_monotonic_decreasing_frac"),
                    row.get("mom_monotonic_increasing_frac"),
                    row.get("case_modal_pass"),
                ]
            )


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# Stability Non-Proxy Summary")
    lines.append("")
    lines.append(f"- dual_regime_pass: `{metrics.get('dual_regime_pass')}`")
    lines.append(f"- modal_cases_usable: `{metrics.get('modal_cases_usable')}`")
    lines.append(f"- shearing_gamma_min: `{metrics.get('shearing_gamma_min')}`")
    lines.append(f"- interchange_abs_gamma_max: `{metrics.get('interchange_abs_gamma_max')}`")
    lines.append(f"- shearing_dominance_min: `{metrics.get('shearing_dominance_min')}`")
    lines.append(f"- stability_nonproxy_pass: `{metrics.get('stability_nonproxy_pass')}`")
    lines.append("")
    lines.append("This gate uses direct reduced diagnostics (M1RHO/M1MOM) to separate interchange and shearing behavior.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-proxy stability gate from direct M1RHO/M1MOM reduced diagnostics."
    )
    parser.add_argument("--dual-case", default="m26-b2-tilt-dual-regime-gate")
    parser.add_argument(
        "--modal-cases",
        nargs="+",
        default=[
            "m17-b2-tilt-m1inject-velkick-eps000",
            "m17-b2-tilt-m1inject-velkick-eps002",
        ],
    )
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-mom-gamma", type=float, default=1.0e8)
    parser.add_argument("--max-rho-abs-gamma", type=float, default=1.0e7)
    parser.add_argument("--min-dominance", type=float, default=100.0)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dual = load_passfail(repo_root, args.dual_case)
    dual_regime_pass = dual["status"] == "PASS" and dual["metrics"].get("dual_regime_pass") is True

    case_rows = []
    for case_id in args.modal_cases:
        reduced_dir = find_reduced_dir(repo_root, case_id)
        rho_path = reduced_dir / "M1RHO.txt"
        mom_path = reduced_dir / "M1MOM.txt"
        rho_header, rho_rows = parse_reduced_series(rho_path)
        mom_header, mom_rows = parse_reduced_series(mom_path)

        rho_col = pick_column(rho_header, ["m1_rho_ratio()", "m1_ratio_raw()"], fallback_idx=5)
        mom_col = pick_column(
            mom_header,
            ["m1_mom_ratio_B()", "m1_mom_ratio_A()", "m1_mom_amp(A/m^2*vol)"],
            fallback_idx=14,
        )
        time_col = 1

        rho_times, rho_vals = extract_times_and_values(rho_rows, time_col, rho_col if rho_col is not None else 5)
        mom_times, mom_vals = extract_times_and_values(mom_rows, time_col, mom_col if mom_col is not None else 14)

        gamma_rho = endpoint_gamma(rho_times, rho_vals)
        gamma_mom = endpoint_gamma(mom_times, mom_vals)

        rho_span = span_rel(rho_vals)
        mom_span = span_rel(mom_vals)
        rho_mon_dec = monotonic_fraction(rho_vals, increasing=False)
        mom_mon_inc = monotonic_fraction(mom_vals, increasing=True)

        dominance = None
        if gamma_mom is not None and gamma_rho is not None:
            dominance = float(gamma_mom / max(abs(gamma_rho), 1.0))

        usable = (
            rho_path.exists()
            and mom_path.exists()
            and len(rho_vals) >= int(args.min_samples)
            and len(mom_vals) >= int(args.min_samples)
        )
        shearing_positive = gamma_mom is not None and gamma_mom >= float(args.min_mom_gamma)
        interchange_bounded = gamma_rho is not None and abs(gamma_rho) <= float(args.max_rho_abs_gamma)
        dominance_ok = dominance is not None and dominance >= float(args.min_dominance)

        case_modal_pass = bool(usable and shearing_positive and interchange_bounded and dominance_ok)
        case_rows.append(
            {
                "case_id": case_id,
                "reduced_dir": str(reduced_dir),
                "rho_path": str(rho_path),
                "mom_path": str(mom_path),
                "usable": bool(usable),
                "rho_samples": int(len(rho_vals)),
                "mom_samples": int(len(mom_vals)),
                "gamma_rho_endpoint": gamma_rho,
                "gamma_mom_endpoint": gamma_mom,
                "shearing_dominance": dominance,
                "rho_span_rel": rho_span,
                "mom_span_rel": mom_span,
                "rho_monotonic_decreasing_frac": rho_mon_dec,
                "mom_monotonic_increasing_frac": mom_mon_inc,
                "case_modal_pass": case_modal_pass,
            }
        )

    modal_cases_usable = bool(case_rows) and all(row["usable"] for row in case_rows)
    shearing_gamma_vals = [row["gamma_mom_endpoint"] for row in case_rows if row["gamma_mom_endpoint"] is not None]
    interchange_gamma_vals = [row["gamma_rho_endpoint"] for row in case_rows if row["gamma_rho_endpoint"] is not None]
    dominance_vals = [row["shearing_dominance"] for row in case_rows if row["shearing_dominance"] is not None]

    shearing_gamma_min = min(shearing_gamma_vals) if shearing_gamma_vals else None
    interchange_abs_gamma_max = max(abs(val) for val in interchange_gamma_vals) if interchange_gamma_vals else None
    shearing_dominance_min = min(dominance_vals) if dominance_vals else None

    modal_case_pass_all = bool(case_rows) and all(row["case_modal_pass"] for row in case_rows)
    stability_nonproxy_pass = bool(
        dual_regime_pass
        and modal_cases_usable
        and modal_case_pass_all
        and shearing_dominance_min is not None
        and shearing_dominance_min >= float(args.min_dominance)
    )

    metrics = {
        "dual_case_id": args.dual_case,
        "dual_regime_pass": bool(dual_regime_pass),
        "modal_case_ids": list(args.modal_cases),
        "modal_cases_usable": bool(modal_cases_usable),
        "modal_case_pass_all": bool(modal_case_pass_all),
        "shearing_gamma_min": shearing_gamma_min,
        "interchange_abs_gamma_max": interchange_abs_gamma_max,
        "shearing_dominance_min": shearing_dominance_min,
        "stability_nonproxy_pass": bool(stability_nonproxy_pass),
        "modal_rows": case_rows,
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
    write_modal_table(out_table, case_rows)


if __name__ == "__main__":
    main()
