#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest(pattern: str) -> Path:
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {pattern}")
    return files[-1]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vtk_time(path: Path) -> float | None:
    try:
        with path.open("rb") as handle:
            for _ in range(3):
                line = handle.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore")
                if "time=" in text:
                    return float(text.split("time=")[1].split()[0])
    except Exception:
        return None
    return None


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare B5 handoff files (Athena VTK -> openPMD).")
    parser.add_argument("--config", required=True, help="Path to handoff_config.json")
    parser.add_argument("--output-dir", required=True, help="Directory for exported openPMD files.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    vtk_path = config.get("vtk_path")
    vtk_pattern = config.get("vtk_pattern")
    if vtk_path:
        vtk = Path(vtk_path)
    elif vtk_pattern:
        vtk = find_latest(str(repo_root / vtk_pattern))
    else:
        raise SystemExit("Config must provide vtk_path or vtk_pattern.")

    if not vtk.is_absolute():
        vtk = (repo_root / vtk).resolve()
    if not vtk.exists():
        raise SystemExit(f"VTK not found: {vtk}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fluid_out = out_dir / "fluid_init.h5"
    b_out = out_dir / "B_ext.h5"

    nr = int(config.get("nr", 64))
    nz = int(config.get("nz", 128))
    r_min = float(config.get("r_min", 0.0))
    r_max = float(config.get("r_max", 4.0))
    z_min = float(config.get("z_min", -8.0))
    z_max = float(config.get("z_max", 8.0))
    axis_mode = config.get("axis_mode", "x_z_y_r")
    fold_r = bool(config.get("fold_r", True))
    resample = bool(config.get("resample", False))

    rho_scale = float(config.get("rho_scale", 1.0))
    vel_scale = float(config.get("vel_scale", 1.0))
    press_scale = float(config.get("press_scale", 1.0))
    b_scale = float(config.get("B_scale", 1.0))
    amu = float(config.get("amu", 1.0))
    te_ratio = float(config.get("te_ratio", 1.0))
    te_const = config.get("te_const", None)
    athena_vis_path = config.get("athena_vis_path")

    common_args = [
        "--nr",
        str(nr),
        "--nz",
        str(nz),
        "--r-min",
        str(r_min),
        "--r-max",
        str(r_max),
        "--z-min",
        str(z_min),
        "--z-max",
        str(z_max),
        "--axis-mode",
        axis_mode,
    ]
    if fold_r:
        common_args.append("--fold-r")
    else:
        common_args.append("--no-fold-r")
    if resample:
        common_args.append("--resample")
    else:
        common_args.append("--no-resample")
    if athena_vis_path:
        common_args += ["--athena-vis-path", athena_vis_path]

    fluid_cmd = [
        sys.executable,
        "warpx-driver/export_fluid_to_opmd.py",
        "--input-vtk",
        str(vtk),
        "--output-fluid",
        str(fluid_out),
        "--amu",
        str(amu),
        "--Te-ratio",
        str(te_ratio),
        "--rho-scale",
        str(rho_scale),
        "--vel-scale",
        str(vel_scale),
        "--press-scale",
        str(press_scale),
    ] + common_args
    if te_const is not None:
        fluid_cmd += ["--Te-const", str(te_const)]

    b_cmd = [
        sys.executable,
        "warpx-driver/export_b_opmd_from_vtk.py",
        "--mode",
        "from-vtk",
        "--input-vtk",
        str(vtk),
        "--output-bfile",
        str(b_out),
        "--B-scale",
        str(b_scale),
    ] + common_args

    run_cmd(fluid_cmd, cwd=repo_root)
    run_cmd(b_cmd, cwd=repo_root)

    meta = {
        "vtk_path": str(vtk),
        "vtk_sha256": hash_file(vtk),
        "vtk_time": parse_vtk_time(vtk),
        "fluid_path": str(fluid_out),
        "b_path": str(b_out),
        "axis_mode": axis_mode,
        "fold_r": fold_r,
        "resample": resample,
        "nr": nr,
        "nz": nz,
        "r_min": r_min,
        "r_max": r_max,
        "z_min": z_min,
        "z_max": z_max,
        "rho_scale": rho_scale,
        "vel_scale": vel_scale,
        "press_scale": press_scale,
        "B_scale": b_scale,
        "amu": amu,
        "te_ratio": te_ratio,
        "te_const": te_const,
        "athena_vis_path": athena_vis_path,
        "generated": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
    }

    meta_path = out_dir / "handoff_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"[handoff] {meta_path}")


if __name__ == "__main__":
    main()
