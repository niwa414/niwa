#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_diff(on_val: float | None, off_val: float | None) -> float | None:
    if on_val is None or off_val is None:
        return None
    if off_val == 0.0:
        return None
    try:
        return abs(on_val / off_val - 1.0)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize circuit drift observability")
    parser.add_argument("--on-metrics", required=True)
    parser.add_argument("--off-metrics", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()

    on_path = Path(args.on_metrics)
    off_path = Path(args.off_metrics)
    on = load_json(on_path)
    off = load_json(off_path)

    phi_on = on.get("phi_delta")
    phi_off = off.get("phi_delta")
    wb_on = on.get("Wb_delta")
    wb_off = off.get("Wb_delta")

    rel_phi = rel_diff(phi_on, phi_off)
    rel_wb = rel_diff(wb_on, wb_off)

    diffs = [v for v in (rel_phi, rel_wb) if v is not None]
    drift_rel_diff_max = max(diffs) if diffs else None

    channels = []
    if rel_phi is not None:
        channels.append("coil_flux")
    if rel_wb is not None:
        channels.append("field_energy")

    observable = None
    if drift_rel_diff_max is not None:
        observable = bool(drift_rel_diff_max >= float(args.threshold))

    out = {
        "on_metrics_path": str(on_path),
        "off_metrics_path": str(off_path),
        "on_metrics_sha1": sha1_file(on_path) if on_path.exists() else None,
        "off_metrics_sha1": sha1_file(off_path) if off_path.exists() else None,
        "threshold": float(args.threshold),
        "phi_delta_on": phi_on,
        "phi_delta_off": phi_off,
        "Wb_delta_on": wb_on,
        "Wb_delta_off": wb_off,
        "drift_rel_diff_phi_delta": rel_phi,
        "drift_rel_diff_Wb_delta": rel_wb,
        "drift_rel_diff_max": drift_rel_diff_max,
        "drift_observable_in_b2": observable,
        "drift_observable_channels": channels,
    }

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
