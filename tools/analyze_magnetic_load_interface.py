#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_non_none(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def find_coil_path(repo_root: Path, source_case: str, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path

    base = repo_root / "outputs" / source_case / "raw" / "run"
    candidates = [
        base / "diag" / "reducedfiles" / "COIL.txt",
        base / "diags" / "reducedfiles" / "COIL.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_coil_series(path: Path) -> dict[str, np.ndarray]:
    steps = []
    times = []
    phis = []
    areas = []
    bn_avgs = []

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                step = int(float(row.get("step", "nan")))
                time_s = float(row.get("time", "nan"))
                phi_wb = float(row.get("phi", "nan"))
                area_m2 = float(row.get("area", "nan"))
                bn_avg_t = float(row.get("bn_avg", "nan"))
            except Exception:
                continue
            if any(math.isnan(x) for x in (time_s, phi_wb, area_m2, bn_avg_t)):
                continue
            steps.append(step)
            times.append(time_s)
            phis.append(phi_wb)
            areas.append(area_m2)
            bn_avgs.append(bn_avg_t)

    return {
        "step": np.asarray(steps, dtype=float),
        "time": np.asarray(times, dtype=float),
        "phi": np.asarray(phis, dtype=float),
        "area": np.asarray(areas, dtype=float),
        "bn_avg": np.asarray(bn_avgs, dtype=float),
    }


def write_series_csv(
    out_path: Path,
    step: np.ndarray,
    time_s: np.ndarray,
    phi_wb: np.ndarray,
    bn_avg_t: np.ndarray,
    area_m2: np.ndarray,
    dphi_dt_v: np.ndarray,
    p_mag_pa: np.ndarray,
    force_proxy_n: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_s",
                "phi_wb",
                "bn_avg_t",
                "area_m2",
                "dphi_dt_v",
                "p_mag_pa",
                "force_proxy_n",
            ]
        )
        for idx in range(len(time_s)):
            writer.writerow(
                [
                    int(step[idx]),
                    f"{time_s[idx]:.12e}",
                    f"{phi_wb[idx]:.12e}",
                    f"{bn_avg_t[idx]:.12e}",
                    f"{area_m2[idx]:.12e}",
                    f"{dphi_dt_v[idx]:.12e}",
                    f"{p_mag_pa[idx]:.12e}",
                    f"{force_proxy_n[idx]:.12e}",
                ]
            )


def write_summary_md(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# Magnetic Load Interface Summary")
    lines.append("")
    lines.append(f"- source_case: `{metrics.get('source_case_id')}`")
    lines.append(f"- coil_series_len: `{metrics.get('coil_series_len')}`")
    lines.append(f"- bn_avg_peak_T: `{metrics.get('bn_avg_peak_T')}`")
    lines.append(f"- p_mag_peak_Pa: `{metrics.get('p_mag_peak_Pa')}`")
    lines.append(f"- force_proxy_peak_N: `{metrics.get('force_proxy_peak_N')}`")
    lines.append(f"- impulse_proxy_Ns: `{metrics.get('impulse_proxy_Ns')}`")
    lines.append(f"- dphi_dt_peak_V: `{metrics.get('dphi_dt_peak_V')}`")
    lines.append(f"- eta_recaptured: `{metrics.get('eta_recaptured')}`")
    lines.append(f"- eta_delivered: `{metrics.get('eta_delivered')}`")
    lines.append(f"- interface_ready: `{metrics.get('interface_ready')}`")
    lines.append("")
    lines.append("This report exports engineering-facing load proxies from coil diagnostics.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build magnetic-load engineering interface metrics from COIL diagnostics."
    )
    parser.add_argument("--source-case", required=True, help="Case ID under outputs/<case_id>")
    parser.add_argument("--coil-path", default=None, help="Override COIL.txt path")
    parser.add_argument("--source-metrics", default=None, help="Override source metrics.json path")
    parser.add_argument("--mu0", type=float, default=1.2566370614359173e-6)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--series-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument(
        "--append-metrics",
        default=None,
        help="Optional metrics JSON file to append load metrics into.",
    )
    parser.add_argument(
        "--append-prefix",
        default="load_",
        help="Prefix for appended keys when --append-metrics is used.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "outputs" / args.source_case
    source_metrics_path = (
        Path(args.source_metrics)
        if args.source_metrics
        else source_root / "analysis" / "metrics.json"
    )
    if not source_metrics_path.is_absolute():
        source_metrics_path = (repo_root / source_metrics_path).resolve()

    source_metrics = read_json(source_metrics_path)
    coil_path = find_coil_path(repo_root, args.source_case, args.coil_path)

    metrics = {
        "source_case_id": args.source_case,
        "source_case_exists": source_root.exists(),
        "input_case_exists": source_root.exists(),
        "source_metrics_path": str(source_metrics_path),
        "source_metrics_exists": source_metrics_path.exists(),
        "coil_source_path": str(coil_path),
        "coil_path_exists": coil_path.exists(),
        "mu0": float(args.mu0),
    }

    if not coil_path.exists():
        metrics.update(
            {
                "error": "coil_file_missing",
                "coil_series_len": 0,
                "energy_chain_present": False,
                "interface_ready": False,
            }
        )
        out_metrics = Path(args.metrics_out)
        if not out_metrics.is_absolute():
            out_metrics = (repo_root / out_metrics).resolve()
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        summary_out = Path(args.summary_out)
        if not summary_out.is_absolute():
            summary_out = (repo_root / summary_out).resolve()
        write_summary_md(summary_out, metrics)
        if args.append_metrics:
            append_path = Path(args.append_metrics)
            if not append_path.is_absolute():
                append_path = (repo_root / append_path).resolve()
            append_data = read_json(append_path)
            append_data[f"{args.append_prefix}interface_ready"] = False
            append_data[f"{args.append_prefix}error"] = metrics.get("error")
            append_data[f"{args.append_prefix}source_case_id"] = args.source_case
            append_path.parent.mkdir(parents=True, exist_ok=True)
            append_path.write_text(
                json.dumps(append_data, indent=2, sort_keys=True), encoding="utf-8"
            )
        return

    series = parse_coil_series(coil_path)
    step = series["step"]
    time_s = series["time"]
    phi_wb = series["phi"]
    area_m2 = series["area"]
    bn_avg_t = series["bn_avg"]

    if len(time_s) < 2:
        metrics.update(
            {
                "error": "coil_series_too_short",
                "coil_series_len": int(len(time_s)),
                "energy_chain_present": False,
                "interface_ready": False,
            }
        )
        out_metrics = Path(args.metrics_out)
        if not out_metrics.is_absolute():
            out_metrics = (repo_root / out_metrics).resolve()
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        summary_out = Path(args.summary_out)
        if not summary_out.is_absolute():
            summary_out = (repo_root / summary_out).resolve()
        write_summary_md(summary_out, metrics)
        if args.append_metrics:
            append_path = Path(args.append_metrics)
            if not append_path.is_absolute():
                append_path = (repo_root / append_path).resolve()
            append_data = read_json(append_path)
            append_data[f"{args.append_prefix}interface_ready"] = False
            append_data[f"{args.append_prefix}error"] = metrics.get("error")
            append_data[f"{args.append_prefix}source_case_id"] = args.source_case
            append_path.parent.mkdir(parents=True, exist_ok=True)
            append_path.write_text(
                json.dumps(append_data, indent=2, sort_keys=True), encoding="utf-8"
            )
        return

    if np.any(np.diff(time_s) < 0.0):
        idx = np.argsort(time_s)
        step = step[idx]
        time_s = time_s[idx]
        phi_wb = phi_wb[idx]
        area_m2 = area_m2[idx]
        bn_avg_t = bn_avg_t[idx]

    dphi_dt_v = np.gradient(phi_wb, time_s)
    p_mag_pa = (bn_avg_t ** 2) / (2.0 * args.mu0)
    force_proxy_n = p_mag_pa * area_m2
    dforce_dt = np.gradient(force_proxy_n, time_s)

    impulse_proxy_ns = float(np.trapz(force_proxy_n, time_s))
    flux_delta_wb = float(np.max(phi_wb) - np.min(phi_wb))

    e_in_j = maybe_float(
        first_non_none(source_metrics, ["e_in_J", "energy_in_J", "e_in"])
    )
    e_load_j = maybe_float(
        first_non_none(source_metrics, ["e_load_J", "energy_load_J", "e_load"])
    )
    e_recaptured_j = maybe_float(
        first_non_none(
            source_metrics,
            ["e_recaptured_J", "e_stored_end_J", "energy_recaptured_J", "e_recaptured"],
        )
    )
    eta_recaptured = maybe_float(source_metrics.get("eta_recaptured"))
    eta_delivered = maybe_float(source_metrics.get("eta_delivered"))

    if eta_recaptured is None and e_in_j is not None and e_in_j > 0 and e_recaptured_j is not None:
        eta_recaptured = float(e_recaptured_j / e_in_j)
    if eta_delivered is None and e_in_j is not None and e_in_j > 0 and e_load_j is not None:
        eta_delivered = float(e_load_j / e_in_j)

    energy_chain_present = (
        e_in_j is not None
        and e_load_j is not None
        and e_recaptured_j is not None
        and eta_recaptured is not None
    )

    metrics.update(
        {
            "coil_series_len": int(len(time_s)),
            "coil_time_start_s": float(time_s[0]),
            "coil_time_end_s": float(time_s[-1]),
            "coil_step_start": int(step[0]),
            "coil_step_end": int(step[-1]),
            "phi_min_Wb": float(np.min(phi_wb)),
            "phi_max_Wb": float(np.max(phi_wb)),
            "phi_delta_Wb": flux_delta_wb,
            "bn_avg_mean_T": float(np.mean(bn_avg_t)),
            "bn_avg_peak_T": float(np.max(np.abs(bn_avg_t))),
            "coil_area_mean_m2": float(np.mean(area_m2)),
            "coil_area_peak_m2": float(np.max(area_m2)),
            "dphi_dt_min_V": float(np.min(dphi_dt_v)),
            "dphi_dt_max_V": float(np.max(dphi_dt_v)),
            "dphi_dt_peak_V": float(np.max(np.abs(dphi_dt_v))),
            "p_mag_mean_Pa": float(np.mean(p_mag_pa)),
            "p_mag_peak_Pa": float(np.max(p_mag_pa)),
            "force_proxy_mean_N": float(np.mean(force_proxy_n)),
            "force_proxy_peak_N": float(np.max(force_proxy_n)),
            "force_proxy_min_N": float(np.min(force_proxy_n)),
            "force_proxy_dynamic_range": float(
                np.max(force_proxy_n) / max(np.min(force_proxy_n), 1e-300)
            ),
            "dforce_dt_peak_N_per_s": float(np.max(np.abs(dforce_dt))),
            "impulse_proxy_Ns": impulse_proxy_ns,
            "e_in_J": e_in_j,
            "e_load_J": e_load_j,
            "e_recaptured_J": e_recaptured_j,
            "eta_recaptured": eta_recaptured,
            "eta_delivered": eta_delivered,
            "energy_chain_present": bool(energy_chain_present),
        }
    )

    interface_ready = (
        metrics["coil_series_len"] >= 32
        and metrics["dphi_dt_peak_V"] > 0.0
        and metrics["force_proxy_peak_N"] > 0.0
        and bool(energy_chain_present)
    )
    metrics["interface_ready"] = bool(interface_ready)

    out_metrics = Path(args.metrics_out)
    out_series = Path(args.series_out)
    out_summary = Path(args.summary_out)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_series.is_absolute():
        out_series = (repo_root / out_series).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    write_series_csv(
        out_series,
        step=step,
        time_s=time_s,
        phi_wb=phi_wb,
        bn_avg_t=bn_avg_t,
        area_m2=area_m2,
        dphi_dt_v=dphi_dt_v,
        p_mag_pa=p_mag_pa,
        force_proxy_n=force_proxy_n,
    )
    write_summary_md(out_summary, metrics)

    if args.append_metrics:
        append_path = Path(args.append_metrics)
        if not append_path.is_absolute():
            append_path = (repo_root / append_path).resolve()
        append_data = read_json(append_path)
        for key, value in metrics.items():
            if key in {
                "source_case_id",
                "source_case_exists",
                "source_metrics_path",
                "source_metrics_exists",
                "coil_source_path",
                "coil_path_exists",
                "mu0",
                "input_case_exists",
            }:
                continue
            append_data[f"{args.append_prefix}{key}"] = value
        append_data[f"{args.append_prefix}interface_ready"] = bool(interface_ready)
        append_data[f"{args.append_prefix}source_case_id"] = args.source_case
        append_data[f"{args.append_prefix}artifact_metrics_path"] = str(out_metrics)
        append_data[f"{args.append_prefix}artifact_series_path"] = str(out_series)
        append_data[f"{args.append_prefix}artifact_summary_path"] = str(out_summary)
        append_path.parent.mkdir(parents=True, exist_ok=True)
        append_path.write_text(
            json.dumps(append_data, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
