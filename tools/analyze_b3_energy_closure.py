#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_records(meta: dict) -> list[dict]:
    ee = meta.get("electron_energy")
    if not isinstance(ee, dict):
        return []
    recs = ee.get("records")
    if not isinstance(recs, list):
        return []
    out = []
    for r in recs:
        if isinstance(r, dict):
            out.append(r)
    return out


def finite(x: float) -> bool:
    return math.isfinite(x)


def to_float(value) -> float | None:
    try:
        x = float(value)
        if finite(x):
            return x
        return None
    except Exception:
        return None


def balance_error(records: list[dict], eps: float = 1.0e-30) -> float | None:
    errs: list[float] = []
    for r in records:
        te = to_float(r.get("te_eV"))
        te_model = to_float(r.get("te_model_eV"))
        if te is None or te_model is None:
            continue
        errs.append(abs(te - te_model) / max(abs(te_model), eps))
    if not errs:
        return None
    return float(max(errs))


def scaling_stats(records: list[dict]) -> dict:
    rr: list[float] = []
    te: list[float] = []
    for r in records:
        x = to_float(r.get("r_rms"))
        y = to_float(r.get("te_eV"))
        if x is None or y is None:
            continue
        rr.append(x)
        te.append(y)
    if len(rr) < 3:
        return {
            "points": len(rr),
            "te_min": None,
            "te_max": None,
            "te_range": None,
            "corr_te_vs_r_rms": None,
            "slope_te_vs_r_rms": None,
        }
    x = np.asarray(rr, dtype=float)
    y = np.asarray(te, dtype=float)
    te_min = float(np.min(y))
    te_max = float(np.max(y))
    te_range = float(te_max - te_min)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0.0 and np.std(y) > 0.0 else None
    slope = None
    if np.var(x) > 0.0:
        xm = float(np.mean(x))
        ym = float(np.mean(y))
        slope = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    return {
        "points": int(len(rr)),
        "te_min": te_min,
        "te_max": te_max,
        "te_range": te_range,
        "corr_te_vs_r_rms": corr,
        "slope_te_vs_r_rms": slope,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze B3 electron-energy closure metrics.")
    ap.add_argument("--on-meta", required=True)
    ap.add_argument("--off-meta", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--min-records", type=int, default=40)
    ap.add_argument("--max-balance-err", type=float, default=1.0e-10)
    ap.add_argument("--min-te-range", type=float, default=0.5)
    ap.add_argument("--max-corr", type=float, default=-0.8)
    args = ap.parse_args()

    meta_on = load_json(Path(args.on_meta))
    meta_off = load_json(Path(args.off_meta))
    records_on = extract_records(meta_on)
    records_off = extract_records(meta_off)

    balance_on = balance_error(records_on)
    balance_off = balance_error(records_off)
    balances = [v for v in (balance_on, balance_off) if isinstance(v, float)]
    balance_max = float(max(balances)) if balances else None

    stats_on = scaling_stats(records_on)
    corr_on = stats_on.get("corr_te_vs_r_rms")
    te_range_on = stats_on.get("te_range")
    slope_on = stats_on.get("slope_te_vs_r_rms")

    electron_energy_records = int(len(records_on))
    electron_energy_records_off = int(len(records_off))

    te_scaling_observed = bool(
        electron_energy_records >= args.min_records
        and isinstance(te_range_on, float)
        and te_range_on >= args.min_te_range
        and isinstance(corr_on, float)
        and corr_on <= args.max_corr
        and isinstance(slope_on, float)
        and slope_on < 0.0
    )
    balance_ok = bool(isinstance(balance_max, float) and balance_max <= args.max_balance_err)

    no_nan_in_metrics = True
    for value in (balance_on, balance_off, balance_max, corr_on, te_range_on, slope_on):
        if value is None:
            continue
        if not finite(float(value)):
            no_nan_in_metrics = False
            break

    metrics = {
        "electron_energy_records": electron_energy_records,
        "electron_energy_records_off": electron_energy_records_off,
        "electron_energy_balance_rel_err_on": balance_on,
        "electron_energy_balance_rel_err_off": balance_off,
        "electron_energy_balance_rel_err": balance_max,
        "electron_energy_balance_ok": balance_ok,
        "te_scaling_points": stats_on.get("points"),
        "te_scaling_te_range": te_range_on,
        "te_scaling_corr_te_vs_r_rms": corr_on,
        "te_scaling_slope_te_vs_r_rms": slope_on,
        "te_scaling_observed": te_scaling_observed,
        "no_nan_in_metrics": no_nan_in_metrics,
        "on_meta_path": str(Path(args.on_meta).resolve()),
        "off_meta_path": str(Path(args.off_meta).resolve()),
    }

    summary = {
        "thresholds": {
            "min_records": args.min_records,
            "max_balance_err": args.max_balance_err,
            "min_te_range": args.min_te_range,
            "max_corr": args.max_corr,
        },
        "stats_on": stats_on,
        "balance_on": balance_on,
        "balance_off": balance_off,
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
