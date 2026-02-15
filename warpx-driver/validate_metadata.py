#!/usr/bin/env python3
"""
Validate and optionally enrich WarpX run metadata JSON files.

Checks required top-level fields and args, and can fill in diag_mode_metrics
from the latest diag* directory using yt (thetaMode). Designed for lightweight
sanity after smoke runs:

  python validate_metadata.py --metadata outputs/warpx/warpx_run_*.json --diag-root outputs/warpx/diag --max-mode 2 --fill-diag
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP = [
    "command",
    "mode",
    "args",
    "dropped_particles_total",
    "monitor",
]

REQUIRED_ARGS = [
    "mode",
    "max_steps",
    "dt",
    "nr",
    "nz",
    "n_azimuthal_modes",
    "diag_period",
]


def list_diags(diag_root: Path):
    return sorted([p for p in diag_root.iterdir() if p.is_dir() and p.name.startswith("diag")])


def collect_diag_mode_metrics(diag_root: Path, max_mode: int, fields=("rho", "Bz")):
    try:
        import yt
    except Exception as exc:  # pragma: no cover - optional dep
        return {"error": f"yt unavailable: {exc}"}

    diags = list_diags(diag_root)
    if not diags:
        return {"error": f"no diag* under {diag_root}"}

    diag = diags[-1]
    try:
        ds = yt.load(str(diag))
        ad = ds.all_data()
        metrics: dict[str, float] = {}
        for field in fields:
            arr = ad["boxlib", field].to_ndarray()
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim < 2:
                continue
            nmodes = min(arr.shape[0], max_mode + 1)
            for m in range(nmodes):
                mode_arr = arr[m]
                amp = abs(mode_arr)
                metrics[f"{field}_m{m}_rms"] = float((amp * amp).mean() ** 0.5)
                metrics[f"{field}_m{m}_max"] = float(amp.max())
        return {
            "diag": diag.name,
            "time_s": float(ds.current_time.to_value()),
            "metrics": metrics,
        }
    except Exception as exc:  # pragma: no cover - protects runtime use
        return {"error": f"diag parse failed for {diag}: {exc}", "diag": diag.name}


def validate_one(path: Path, args) -> int:
    with path.open("r", encoding="utf-8") as fh:
        try:
            data: dict[str, Any] = json.load(fh)
        except Exception as exc:
            print(f"[FAIL] {path}: invalid JSON ({exc})")
            return 1

    errors: list[str] = []
    for key in REQUIRED_TOP:
        if key not in data:
            errors.append(f"missing top-level '{key}'")
    if "args" in data and isinstance(data["args"], dict):
        for key in REQUIRED_ARGS:
            if key not in data["args"]:
                errors.append(f"args missing '{key}'")
    else:
        errors.append("missing or invalid 'args' dict")

    diag_metrics = data.get("diag_mode_metrics")
    if args.fill_diag and (diag_metrics is None or "error" in diag_metrics):
        diag_root = Path(args.diag_root) if args.diag_root else path.parent
        inputs = data.get("inputs")
        if isinstance(inputs, dict):
            diag_dir = inputs.get("diag_dir")
            if diag_dir:
                diag_root = Path(diag_dir)
        max_mode = data["args"].get("n_azimuthal_modes", args.max_mode)
        diag_metrics = collect_diag_mode_metrics(diag_root, max_mode)
        data["diag_mode_metrics"] = diag_metrics
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            print(f"[FIXED] {path}: refreshed diag_mode_metrics from {diag_root}")
        except Exception as exc:  # pragma: no cover - runtime guard
            errors.append(f"failed to write updated metadata: {exc}")

    if errors:
        print(f"[FAIL] {path}: " + "; ".join(errors))
        return 1
    print(f"[OK]   {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Validate WarpX run metadata JSONs.")
    ap.add_argument(
        "--metadata",
        required=True,
        help="Glob for metadata files (e.g., outputs/warpx/warpx_run_*.json).",
    )
    ap.add_argument(
        "--diag-root",
        default=None,
        help="diag directory containing diag*/ for diag metrics (default: metadata directory).",
    )
    ap.add_argument("--max-mode", type=int, default=2, help="Highest theta mode to include when filling diag metrics.")
    ap.add_argument("--fill-diag", action="store_true", help="If set, recompute diag_mode_metrics with yt when missing/errored.")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.metadata))
    if not paths:
        print(f"No metadata matched pattern: {args.metadata}")
        sys.exit(1)

    rc = 0
    for path in paths:
        rc |= validate_one(path, args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
