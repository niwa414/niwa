#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_hst(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    header = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("# [1]"):
                header = line
                break
    if header is None:
        return {}
    labels = [s.strip() for s in re.findall(r"\[\d+\]=([^\[]+)", header)]
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except Exception:
                continue
    out: dict[str, list[float]] = {k: [] for k in labels}
    for row in rows:
        for k, v in zip(labels, row):
            out[k].append(v)
    return out


def leak_curve(hst: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(hst.get("time") or [], dtype=float)
    m_in = np.asarray(hst.get("mass_cyl_in") or [], dtype=float)
    m_out = np.asarray(hst.get("mass_cyl_out") or [], dtype=float)
    if t.size == 0 or m_in.size == 0 or m_out.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(t.size, m_in.size, m_out.size)
    t = t[:n]
    m_in = m_in[:n]
    m_out = m_out[:n]
    m_in0 = float(m_in[0])
    if m_in0 == 0.0:
        return t, np.zeros_like(t)
    leak = (m_out - float(m_out[0])) / m_in0
    return t, leak


def safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if b == 0.0:
        return None
    return a / b


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare A4 stage2 candidate against baseline.")
    ap.add_argument("--candidate-metrics", required=True)
    ap.add_argument("--baseline-metrics", required=True)
    ap.add_argument("--candidate-hst", required=True)
    ap.add_argument("--baseline-hst", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()

    cand_metrics = load_json(Path(args.candidate_metrics))
    base_metrics = load_json(Path(args.baseline_metrics))
    cand_hst = parse_hst(Path(args.candidate_hst))
    base_hst = parse_hst(Path(args.baseline_hst))

    t_c, leak_c = leak_curve(cand_hst)
    t_b, leak_b = leak_curve(base_hst)

    plot_path = Path(args.plot).resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if t_b.size:
        plt.plot(t_b, leak_b, label="baseline leak_frac", lw=1.8, color="#8c564b")
    if t_c.size:
        plt.plot(t_c, leak_c, label="candidate leak_frac", lw=2.0, color="#1f77b4")
    plt.xlabel("time")
    plt.ylabel("(mass_out - mass_out0)/mass_in0")
    plt.title("A4 Leak Curve Compare")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    cand_res = cand_metrics.get("mass_budget_residual_geom_rel")
    cand_leak = cand_metrics.get("leak_mass_frac_max_geom")
    base_res = base_metrics.get("mass_budget_residual_geom_rel")
    base_leak = base_metrics.get("leak_mass_frac_max_geom")

    residual_improve_factor = None
    leak_improve_factor = None
    if isinstance(base_res, (int, float)) and isinstance(cand_res, (int, float)) and cand_res > 0.0:
        residual_improve_factor = float(base_res) / float(cand_res)
    if isinstance(base_leak, (int, float)) and isinstance(cand_leak, (int, float)) and cand_leak > 0.0:
        leak_improve_factor = float(base_leak) / float(cand_leak)

    metrics = dict(cand_metrics)
    metrics.update(
        {
            "baseline_mass_budget_residual_geom_rel": base_res,
            "baseline_leak_mass_frac_max_geom": base_leak,
            "residual_improve_factor": residual_improve_factor,
            "leak_improve_factor": leak_improve_factor,
            "control_curve_plot_exists": plot_path.exists(),
            "candidate_curve_len": int(t_c.size),
            "baseline_curve_len": int(t_b.size),
            "candidate_hst_path": str(Path(args.candidate_hst).resolve()),
            "baseline_hst_path": str(Path(args.baseline_hst).resolve()),
            "candidate_metrics_path": str(Path(args.candidate_metrics).resolve()),
            "baseline_metrics_path": str(Path(args.baseline_metrics).resolve()),
        }
    )

    summary = {
        "candidate": cand_metrics,
        "baseline": base_metrics,
        "candidate_curve": {
            "time_start": float(t_c[0]) if t_c.size else None,
            "time_end": float(t_c[-1]) if t_c.size else None,
            "leak_max": float(np.max(leak_c)) if leak_c.size else None,
        },
        "baseline_curve": {
            "time_start": float(t_b[0]) if t_b.size else None,
            "time_end": float(t_b[-1]) if t_b.size else None,
            "leak_max": float(np.max(leak_b)) if leak_b.size else None,
        },
        "residual_improve_factor": residual_improve_factor,
        "leak_improve_factor": leak_improve_factor,
        "plot_path": str(plot_path),
    }

    metrics_path = Path(args.metrics).resolve()
    summary_path = Path(args.summary).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
