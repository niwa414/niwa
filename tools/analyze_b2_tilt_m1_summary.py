#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_case_metrics(case_id: str) -> tuple[dict, dict, str | None]:
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
    return metrics, passfail, result


def _parse_m1_header(header: str) -> list[str]:
    tokens = header.strip().split()
    names = []
    for tok in tokens:
        if "]" in tok:
            names.append(tok.split("]", 1)[1])
        else:
            names.append(tok)
    return names


def _find_m1_indices(names: list[str], ratio_prefixes: tuple[str, ...]) -> tuple[int | None, int | None]:
    time_idx = None
    ratio_idx = None
    for i, name in enumerate(names):
        if time_idx is None and name.startswith("time"):
            time_idx = i
    if ratio_idx is None:
        for prefix in ratio_prefixes:
            for i, name in enumerate(names):
                if name.startswith(prefix):
                    ratio_idx = i
                    break
            if ratio_idx is not None:
                break
    return time_idx, ratio_idx


def load_m1_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    ratios: list[float] = []
    header = None
    time_idx = None
    ratio_idx = None
    ratio_prefixes = (
        "m1_vperp_ratio",
        "m1_mom_ratio_B",
        "m1_mom_ratio_A",
        "m1_mom_ratio",
        "m1_rho_ratio",
        "m1_ratio_raw",
        "m1_ratio",
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith("#"):
                if header is None:
                    header = line
                    names = _parse_m1_header(header)
                    time_idx, ratio_idx = _find_m1_indices(names, ratio_prefixes)
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if time_idx is None:
                time_idx = 1 if len(parts) > 1 else None
            if ratio_idx is None:
                ratio_idx = 5 if len(parts) > 5 else None
            if time_idx is None or ratio_idx is None:
                continue
            if len(parts) <= max(time_idx, ratio_idx):
                continue
            try:
                times.append(float(parts[time_idx]))
                ratios.append(float(parts[ratio_idx]))
            except ValueError:
                continue
    return np.array(times, dtype=float), np.array(ratios, dtype=float)


def load_m1_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    ratios: list[float] = []
    ratio_keys = ("m1_vperp_ratio", "m1_ratio", "m1_ratio_raw")
    time_keys = ("time", "t", "time_s")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t_raw = None
            for key in time_keys:
                if row.get(key) is not None:
                    t_raw = row.get(key)
                    break
            r_raw = None
            for key in ratio_keys:
                if row.get(key) is not None:
                    r_raw = row.get(key)
                    break
            if t_raw is None or r_raw is None:
                continue
            try:
                times.append(float(t_raw))
                ratios.append(float(r_raw))
            except ValueError:
                continue
    return np.array(times, dtype=float), np.array(ratios, dtype=float)


def load_m1_series(metrics: dict) -> tuple[np.ndarray, np.ndarray] | None:
    path_str = metrics.get("m1_ratio_series_path")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return load_m1_csv(path)
    return load_m1_txt(path)


def compute_series_stats(times: np.ndarray, ratios: np.ndarray) -> dict:
    stats = {}
    if ratios.size:
        stats["m1_ratio_series_len"] = int(ratios.size)
        stats["m1_ratio_initial"] = float(ratios[0])
        stats["m1_ratio_last"] = float(ratios[-1])
        stats["m1_ratio_mean"] = float(np.mean(ratios))
        stats["m1_ratio_min"] = float(np.min(ratios))
        stats["m1_ratio_max"] = float(np.max(ratios))
        stats["m1_ratio_delta"] = float(ratios[0] - ratios[-1])
        if times.size == ratios.size and times.size > 1:
            dt = np.diff(times)
            dy = np.diff(ratios)
            valid = dt != 0.0
            if np.any(valid):
                stats["m1_ratio_mean_abs_slope"] = float(np.mean(np.abs(dy[valid] / dt[valid])))
    return stats


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(val):
        return "n/a"
    abs_val = abs(val)
    if abs_val != 0 and (abs_val >= 1.0e4 or abs_val < 1.0e-3):
        return f"{val:.3e}"
    return f"{val:.6g}"


def format_csv(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {col: format_csv(row.get(col)) for col in columns}
            writer.writerow(out)


def write_md(path: Path, rows: list[dict], columns: list[str], title: str, notes: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [f"# {title}", "", header, sep]
    for row in rows:
        formatted = [format_float(row.get(col)) if isinstance(row.get(col), (int, float)) else row.get(col) for col in columns]
        formatted = [val if val is not None else "n/a" for val in formatted]
        lines.append("| " + " | ".join(str(val) for val in formatted) + " |")
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_series(series_map: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    any_data = False
    y_min = None
    y_max = None
    for label, (times, ratios) in series_map.items():
        if times.size == 0 or ratios.size == 0:
            continue
        any_data = True
        time_ns = times * 1.0e9
        ax.plot(time_ns, ratios, label=label, linewidth=1.6)
        cur_min = float(np.nanmin(ratios))
        cur_max = float(np.nanmax(ratios))
        y_min = cur_min if y_min is None else min(y_min, cur_min)
        y_max = cur_max if y_max is None else max(y_max, cur_max)
    if any_data:
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("m1 ratio")
        if y_min is not None and y_max is not None and y_min > 0.0:
            if y_max / y_min > 20.0:
                ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "No m1 ratio series available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize m1 diagnostics for B2 tilt on/off cases.")
    parser.add_argument("--variants", help="Path to variants.json (expects tilt_on/tilt_off).")
    parser.add_argument("--tilt-on", dest="tilt_on", help="Tilt-on case ID.")
    parser.add_argument("--tilt-off", dest="tilt_off", help="Tilt-off case ID.")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON.")
    parser.add_argument("--summary-csv", required=True, help="Output summary CSV.")
    parser.add_argument("--summary-md", required=True, help="Output summary Markdown.")
    parser.add_argument("--plot", required=True, help="Output plot path.")
    parser.add_argument("--title", default="B2.3 Tilt M1 Diagnostics Summary", help="Title for summary/plot.")
    args = parser.parse_args()

    tilt_on = args.tilt_on
    tilt_off = args.tilt_off
    if args.variants:
        cfg = load_json(Path(args.variants))
        tilt_on = tilt_on or cfg.get("tilt_on")
        tilt_off = tilt_off or cfg.get("tilt_off")
    if not tilt_on or not tilt_off:
        raise SystemExit("Must provide --tilt-on and --tilt-off (or --variants with tilt_on/tilt_off).")

    rows = []
    series_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    combined: dict[str, dict] = {}

    for case_id in (tilt_on, tilt_off):
        metrics, passfail, status = load_case_metrics(case_id)
        series = load_m1_series(metrics)
        series_stats = compute_series_stats(*series) if series is not None else {}
        combined_stats = dict(metrics)
        combined_stats.update(series_stats)
        combined[case_id] = combined_stats
        if series is not None:
            series_map[case_id] = series
        rows.append(
            {
                "case_id": case_id,
                "status": status or "n/a",
                "m1_series_kind": combined_stats.get("m1_series_kind") or "n/a",
                "m1_ratio_series_len": combined_stats.get("m1_ratio_series_len"),
                "m1_ratio_initial": combined_stats.get("m1_ratio_initial"),
                "m1_ratio_last": combined_stats.get("m1_ratio_last"),
                "m1_ratio_delta": combined_stats.get("m1_ratio_delta"),
                "m1_ratio_mean": combined_stats.get("m1_ratio_mean"),
                "gamma_m1_fit_best": combined_stats.get("gamma_m1_fit_best"),
                "r2_m1_fit_best": combined_stats.get("r2_m1_fit_best"),
                "gamma_fit_best": combined_stats.get("gamma_fit_best"),
                "r2_fit_best": combined_stats.get("r2_fit_best"),
                "ran_to_completion": combined_stats.get("ran_to_completion"),
                "done": (passfail.get("metrics") or {}).get("done"),
                "wall_time_s": (passfail.get("metrics") or {}).get("wall_time_s"),
                "archive_size_gb": (passfail.get("metrics") or {}).get("archive_size_gb"),
            }
        )

    gamma_on = combined.get(tilt_on, {}).get("gamma_m1_fit_best")
    gamma_off = combined.get(tilt_off, {}).get("gamma_m1_fit_best")
    ratio_eps = 1.0e-12
    gamma_ratio = None
    if gamma_on is not None:
        try:
            gamma_on_val = float(gamma_on)
            gamma_off_val = float(gamma_off) if gamma_off is not None else 0.0
            gamma_ratio = abs(gamma_on_val) / max(abs(gamma_off_val), ratio_eps)
        except (TypeError, ValueError):
            gamma_ratio = None

    summary = {
        "tilt_on": tilt_on,
        "tilt_off": tilt_off,
        "gamma_m1_on": gamma_on,
        "gamma_m1_off": gamma_off,
        "gamma_m1_ratio": gamma_ratio,
        "ratio_eps": ratio_eps,
        "cases": {tilt_on: combined.get(tilt_on, {}), tilt_off: combined.get(tilt_off, {})},
    }

    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    columns = [
        "case_id",
        "status",
        "m1_series_kind",
        "m1_ratio_series_len",
        "m1_ratio_initial",
        "m1_ratio_last",
        "m1_ratio_delta",
        "m1_ratio_mean",
        "gamma_m1_fit_best",
        "r2_m1_fit_best",
        "gamma_fit_best",
        "r2_fit_best",
        "ran_to_completion",
        "done",
        "wall_time_s",
        "archive_size_gb",
    ]
    write_csv(Path(args.summary_csv), rows, columns)

    notes = [
        "m1_ratio_* are from the m1 ratio series (M1RHO/M1MOM/particle_vel_stats) when available.",
        "gamma_m1_fit_best/r2_m1_fit_best are from m1 series fits in metrics.json.",
        "done, wall_time_s, archive_size_gb are recorded in PASSFAIL when available; missing values are n/a.",
    ]
    if gamma_ratio is not None:
        notes.append(f"gamma_m1_ratio (|on|/|off|) = {format_float(gamma_ratio)}")
    write_md(Path(args.summary_md), rows, columns, args.title, notes)

    plot_series(series_map, Path(args.plot), args.title)


if __name__ == "__main__":
    main()
