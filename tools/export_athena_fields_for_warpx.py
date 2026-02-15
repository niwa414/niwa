#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_roi(values: list[str] | None) -> dict | None:
    if not values:
        return None
    if len(values) != 6:
        raise SystemExit("ROI requires 6 values: xmin xmax ymin ymax zmin zmax")
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, values)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Athena++ fields for WarpX.")
    parser.add_argument("--config", type=Path, help="JSON config with input paths/ROI.")
    parser.add_argument("--opmd-fluid-src", type=Path, help="Source openPMD fluid file.")
    parser.add_argument("--opmd-b-src", type=Path, help="Source openPMD B field file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output coupling directory.")
    parser.add_argument(
        "--roi",
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Optional ROI bounds (six floats).",
    )
    parser.add_argument("--t-field", type=float, default=None, help="Field time stamp.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fluid_src = args.opmd_fluid_src or Path(cfg.get("athena_fluid_path") or "")
    b_src = args.opmd_b_src or Path(cfg.get("athena_b_path") or "")
    if not fluid_src:
        raise SystemExit("Missing opmd fluid source (--opmd-fluid-src or athena_fluid_path).")
    if not b_src:
        raise SystemExit("Missing opmd B source (--opmd-b-src or athena_b_path).")
    if not fluid_src.exists():
        raise SystemExit(f"opmd fluid source not found: {fluid_src}")
    if not b_src.exists():
        raise SystemExit(f"opmd B source not found: {b_src}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fluid_out = out_dir / "fluid_init.h5"
    b_out = out_dir / "B_ext.h5"
    shutil.copy2(fluid_src, fluid_out)
    shutil.copy2(b_src, b_out)

    roi_bounds = parse_roi(args.roi) or cfg.get("roi_bounds")
    meta = {
        "athena_fluid_src": str(fluid_src),
        "athena_b_src": str(b_src),
        "fluid_out": str(fluid_out),
        "b_out": str(b_out),
        "fluid_sha1": sha1_file(fluid_out),
        "b_sha1": sha1_file(b_out),
        "roi_bounds": roi_bounds,
        "t_field": args.t_field if args.t_field is not None else cfg.get("t_field"),
        "unit_note": cfg.get("unit_note", "internal"),
    }

    with (out_dir / "coupling_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    print(f"[p1p1] staged fluid: {fluid_out}")
    print(f"[p1p1] staged B: {b_out}")


if __name__ == "__main__":
    main()
