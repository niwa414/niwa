#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WarpX KPIs for P1.1 coupling.")
    parser.add_argument("--u2hist", type=Path, required=True, help="metrics_u2hist.json path.")
    parser.add_argument("--coupling-meta", type=Path, required=True, help="coupling_meta.json path.")
    parser.add_argument("--warpx-run", type=Path, required=False, help="warpx_run_*.json path.")
    parser.add_argument("--warpx-meta-dir", type=Path, required=False, help="Directory to search for warpx_run_*.json.")
    parser.add_argument("--baseline-drift", type=float, default=None)
    parser.add_argument("--baseline-shift", type=float, default=None)
    parser.add_argument("--metrics-out", type=Path, required=True)
    args = parser.parse_args()

    u2 = load_json(args.u2hist)
    meta = load_json(args.coupling_meta)
    warpx_run = {}
    if args.warpx_run and args.warpx_run.exists():
        warpx_run = load_json(args.warpx_run)
    elif args.warpx_meta_dir and args.warpx_meta_dir.exists():
        candidates = sorted(args.warpx_meta_dir.glob("warpx_run_*.json"), key=lambda p: p.stat().st_mtime)
        if candidates:
            warpx_run = load_json(candidates[-1])

    p99 = u2.get("u2_p99_end") or u2.get("u2_p99_at_stepEnd")
    p999 = u2.get("u2_p999_end") or u2.get("u2_p999_at_stepEnd")

    anisotropy = None
    if p99 is not None and p999 is not None:
        try:
            p99f = float(p99)
            p999f = float(p999)
            if p99f != 0.0:
                anisotropy = p999f / p99f
        except Exception:
            anisotropy = None

    metrics = {
        "warpx_ran_to_completion": bool(warpx_run.get("ran_to_completion", True)),
        "warpx_kpi_energy_tail": p99,
        "warpx_kpi_anisotropy": anisotropy,
        "warpx_kpi_anisotropy_mode": "u2_p999_over_p99",
        "warpx_kpi_roi_bounds": meta.get("roi_bounds"),
        "warpx_kpi_source_u2hist": str(args.u2hist),
        "warpx_kpi_u2hist_sha1": sha1_file(args.u2hist),
        "baseline_drift": args.baseline_drift,
        "baseline_shift": args.baseline_shift,
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
