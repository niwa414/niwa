#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_tlim(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "tlim" not in line:
            continue
        parts = line.split("=")
        if len(parts) < 2:
            continue
        try:
            return float(parts[1].strip())
        except Exception:
            continue
    return None


def find_vtk_files(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))


def select_vtk(
    vtk_files: list[Path],
    target_time: float | None,
    target_index: int | None = None,
) -> tuple[Path, dict]:
    if not vtk_files:
        raise SystemExit("No VTK files matched the provided pattern.")
    if target_time is None:
        selected_index = len(vtk_files) - 1
        return vtk_files[selected_index], {
            "selection_mode": "latest",
            "selected_index": selected_index,
        }

    times = []
    for path in vtk_files:
        t = parse_vtk_time(path)
        times.append(t if t is not None else np.nan)
    if not np.isnan(times).all():
        time_arr = np.asarray(times, dtype=float)
        idx = int(np.nanargmin(np.abs(time_arr - target_time)))
        return vtk_files[idx], {
            "selection_mode": "closest_time",
            "target_time": target_time,
            "selected_time": float(time_arr[idx]),
            "selected_index": idx,
        }

    if target_index is None:
        target_index = len(vtk_files) - 1
    idx = min(len(vtk_files) - 1, max(0, int(target_index)))
    return vtk_files[idx], {
        "selection_mode": "index",
        "target_time": target_time,
        "selected_index": idx,
    }


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def infer_case_id(metrics_path: str | None) -> str | None:
    if not metrics_path:
        return None
    parts = Path(metrics_path).parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def read_openpmd_2d(path: Path, mesh: str, components: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        base = h5f[f"/data/0/meshes/{mesh}"]
        data = {}
        for comp in components:
            if comp not in base:
                raise SystemExit(f"Missing component '{comp}' in {path}")
            arr = np.asarray(base[comp], dtype=float)
            if arr.ndim == 3:
                arr = arr[0]
            data[comp] = arr
        spacing = base[components[0]].attrs.get("gridSpacing", base.attrs.get("gridSpacing", None))
        offset = base[components[0]].attrs.get("gridGlobalOffset", base.attrs.get("gridGlobalOffset", None))
        if spacing is None or offset is None:
            raise SystemExit(f"gridSpacing/gridGlobalOffset missing in {path}")
    spacing = np.asarray(spacing, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if spacing.size != 2 or offset.size != 2:
        raise SystemExit(f"Expected 2D spacing/offset in {path}, got {spacing}, {offset}")
    return data, spacing, offset


def axisym_centers(spacing: np.ndarray, offset: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nr, nz = shape
    r = offset[0] + np.arange(nr, dtype=float) * spacing[0]
    z = offset[1] + np.arange(nz, dtype=float) * spacing[1]
    return r, z


def bilinear_sample(
    arr: np.ndarray,
    r_query: np.ndarray,
    z_query: np.ndarray,
    r0: float,
    z0: float,
    dr: float,
    dz: float,
) -> np.ndarray:
    if dr <= 0.0 or dz <= 0.0:
        raise SystemExit("Invalid spacing for bilinear interpolation.")
    fr = (r_query - r0) / dr
    fz = (z_query - z0) / dz
    i0 = np.floor(fr).astype(int)
    k0 = np.floor(fz).astype(int)
    i0 = np.clip(i0, 0, arr.shape[0] - 2)
    k0 = np.clip(k0, 0, arr.shape[1] - 2)
    i1 = i0 + 1
    k1 = k0 + 1
    wr = fr - i0
    wz = fz - k0
    v00 = arr[i0, k0]
    v10 = arr[i1, k0]
    v01 = arr[i0, k1]
    v11 = arr[i1, k1]
    return (1.0 - wr) * (1.0 - wz) * v00 + wr * (1.0 - wz) * v10 + (1.0 - wr) * wz * v01 + wr * wz * v11


def write_openpmd_cartesian(
    out_path: Path,
    mesh_name: str,
    components: dict[str, np.ndarray],
    spacing: np.ndarray,
    offset: np.ndarray,
    unit_dim: dict,
) -> dict:
    try:
        import openpmd_api as io
    except Exception as exc:
        raise SystemExit(f"openpmd_api required to write openPMD: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = io.Series(str(out_path), io.Access.create)
    iteration = series.iterations[0]
    mesh = iteration.meshes[mesh_name]
    mesh.set_geometry(io.Geometry.cartesian)
    mesh.set_attribute("dataOrder", "C")
    mesh.set_axis_labels(["x", "y", "z"])
    mesh.set_grid_spacing(spacing.tolist())
    mesh.set_grid_global_offset(offset.tolist())
    mesh.unit_dimension = unit_dim

    for comp, arr in components.items():
        rc = mesh[comp]
        rc.reset_dataset(io.Dataset(arr.dtype, arr.shape))
        rc.set_unit_SI(1.0)
        rc.store_chunk(np.ascontiguousarray(arr))
        rc.set_attribute("gridSpacing", np.array(spacing, dtype=np.float64))
        rc.set_attribute("gridGlobalOffset", np.array(offset, dtype=np.float64))

    series.flush()
    series.close()
    return {"path": str(out_path), "shape": list(next(iter(components.values())).shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare H4 handoff (Athena VTK -> 3D openPMD).")
    parser.add_argument("--config", required=True, help="Path to handoff_config.json")
    parser.add_argument("--output-dir", required=True, help="Directory for exported openPMD files.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    vtk_path = config.get("vtk_path")
    vtk_pattern = config.get("vtk_pattern")
    metrics_path = config.get("metrics_path")
    athinput_path = config.get("athinput_path")
    formation_key = config.get("formation_time_frac_key", "formation_time_frac")

    target_time = None
    target_index = None
    selection_meta = {}
    formation_frac = None
    tlim = None
    if metrics_path and athinput_path:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        formation_frac = float(metrics.get(formation_key, 0.0))
        tlim = parse_tlim(Path(athinput_path))
        if tlim is not None:
            target_time = formation_frac * tlim
        target_index = formation_frac

    if vtk_path:
        vtk = Path(vtk_path)
        selection_meta["selection_mode"] = "explicit"
    elif vtk_pattern:
        vtk_files = find_vtk_files(str(repo_root / vtk_pattern))
        if target_index is not None:
            target_index = int(round(target_index * max(len(vtk_files) - 1, 1)))
        vtk, selection_meta = select_vtk(vtk_files, target_time, target_index=target_index)
    else:
        raise SystemExit("Config must provide vtk_path or vtk_pattern.")

    if not vtk.is_absolute():
        vtk = (repo_root / vtk).resolve()
    if not vtk.exists():
        raise SystemExit(f"VTK not found: {vtk}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fluid_2d = out_dir / "fluid_init_2d.h5"
    b_2d = out_dir / "B_ext_2d.h5"
    fluid_3d = out_dir / "fluid_init.h5"
    b_3d = out_dir / "B_ext.h5"

    nr = int(config.get("nr", 64))
    nz = int(config.get("nz", 128))
    r_min = float(config.get("r_min", 0.0))
    r_max = float(config.get("r_max", 1.0))
    z_min = float(config.get("z_min", -1.0))
    z_max = float(config.get("z_max", 1.0))
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
        str(fluid_2d),
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
        str(b_2d),
        "--B-scale",
        str(b_scale),
    ] + common_args

    run_cmd(fluid_cmd, cwd=repo_root)
    run_cmd(b_cmd, cwd=repo_root)

    fluid_data, spacing_2d, offset_2d = read_openpmd_2d(
        fluid_2d, "fluid", ["rho", "vr", "vz", "vphi", "Ti", "Te"]
    )
    b_data, _, _ = read_openpmd_2d(b_2d, "B", ["r", "t", "z"])

    n3 = None
    extrude_length = None
    mapping_mode = str(config.get("mapping_mode", "extrude_x3")).lower()
    if mapping_mode not in {"extrude_x3", "rotate_axisym"}:
        raise SystemExit(f"Unsupported mapping_mode '{mapping_mode}'")

    rotation_checks = {}
    rotation_meta = {}
    cart_grid_meta = None
    vector_map = {}

    if mapping_mode == "extrude_x3":
        n3 = int(config.get("n3_extrude", 16))
        extrude_length = float(config.get("extrude_length", r_max - r_min))
        dy = extrude_length / n3
        y_min = -0.5 * extrude_length
        spacing_3d = np.array([spacing_2d[0], dy, spacing_2d[1]], dtype=float)
        offset_3d = np.array([offset_2d[0], y_min, offset_2d[1]], dtype=float)

        def extrude(arr_2d):
            return np.repeat(arr_2d[:, None, :], n3, axis=1)

        rho3 = extrude(fluid_data["rho"])
        vx3 = extrude(fluid_data["vr"])
        vy3 = extrude(fluid_data["vphi"])
        vz3 = extrude(fluid_data["vz"])
        Ti3 = extrude(fluid_data["Ti"])
        Te3 = extrude(fluid_data["Te"])

        Bx3 = extrude(b_data["r"])
        By3 = extrude(b_data["t"])
        Bz3 = extrude(b_data["z"])
        vector_map = {
            "vx": "vr",
            "vy": "vphi",
            "vz": "vz",
            "Bx": "Br",
            "By": "Bt",
            "Bz": "Bz",
        }
    else:
        cart_grid = config.get("cart_grid", {})
        nx = int(cart_grid.get("nx", config.get("nx_cart", nr)))
        ny = int(cart_grid.get("ny", config.get("ny_cart", nr)))
        nz_cart = int(cart_grid.get("nz", config.get("nz_cart", nz)))
        x_min = float(cart_grid.get("x_min", -r_max))
        x_max = float(cart_grid.get("x_max", r_max))
        y_min = float(cart_grid.get("y_min", -r_max))
        y_max = float(cart_grid.get("y_max", r_max))
        z_min_cart = float(cart_grid.get("z_min", z_min))
        z_max_cart = float(cart_grid.get("z_max", z_max))
        if nx <= 1 or ny <= 1 or nz_cart <= 1:
            raise SystemExit("cart_grid nx/ny/nz must be > 1 for rotation mapping.")
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        dz = (z_max_cart - z_min_cart) / nz_cart
        spacing_3d = np.array([dx, dy, dz], dtype=float)
        offset_3d = np.array([x_min, y_min, z_min_cart], dtype=float)

        r_centers, z_centers = axisym_centers(spacing_2d, offset_2d, fluid_data["rho"].shape)
        r_min_data = float(r_centers[0])
        r_max_data = float(r_centers[-1])
        z_min_data = float(z_centers[0])
        z_max_data = float(z_centers[-1])
        axis_r_min = float(config.get("axis_r_min", spacing_2d[0] * 0.5))
        r_clip_policy = str(config.get("r_clip_policy", "clip")).lower()
        if r_clip_policy not in {"clip", "zero"}:
            raise SystemExit(f"Unsupported r_clip_policy '{r_clip_policy}'")

        x = x_min + np.arange(nx, dtype=float) * dx
        y = y_min + np.arange(ny, dtype=float) * dy
        z = z_min_cart + np.arange(nz_cart, dtype=float) * dz
        xg = x[:, None, None]
        yg = y[None, :, None]
        zg = z[None, None, :]

        r = np.sqrt(xg * xg + yg * yg)
        phi = np.arctan2(yg, xg)
        r_outside = (r < r_min_data) | (r > r_max_data)
        r_clip_fraction = float(np.count_nonzero(r_outside) / r_outside.size)
        r_sample = np.clip(r, r_min_data, r_max_data)
        z_sample = np.clip(zg, z_min_data, z_max_data)

        rho3 = bilinear_sample(fluid_data["rho"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        vr3 = bilinear_sample(fluid_data["vr"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        vphi3 = bilinear_sample(fluid_data["vphi"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        vz3 = bilinear_sample(fluid_data["vz"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        Ti3 = bilinear_sample(fluid_data["Ti"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        Te3 = bilinear_sample(fluid_data["Te"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])

        Br3 = bilinear_sample(b_data["r"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        Bphi3 = bilinear_sample(b_data["t"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])
        Bz3 = bilinear_sample(b_data["z"], r_sample, z_sample, r_min_data, z_min_data, spacing_2d[0], spacing_2d[1])

        if r_clip_policy == "zero":
            rho3 = np.where(r_outside, 0.0, rho3)
            vr3 = np.where(r_outside, 0.0, vr3)
            vphi3 = np.where(r_outside, 0.0, vphi3)
            vz3 = np.where(r_outside, 0.0, vz3)
            Ti3 = np.where(r_outside, 0.0, Ti3)
            Te3 = np.where(r_outside, 0.0, Te3)
            Br3 = np.where(r_outside, 0.0, Br3)
            Bphi3 = np.where(r_outside, 0.0, Bphi3)
            Bz3 = np.where(r_outside, 0.0, Bz3)

        axis_mask = r <= axis_r_min
        if np.any(axis_mask):
            vphi3 = np.where(axis_mask, 0.0, vphi3)
            Bphi3 = np.where(axis_mask, 0.0, Bphi3)
            phi = np.where(axis_mask, 0.0, phi)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        vx3 = vr3 * cos_phi - vphi3 * sin_phi
        vy3 = vr3 * sin_phi + vphi3 * cos_phi
        Bx3 = Br3 * cos_phi - Bphi3 * sin_phi
        By3 = Br3 * sin_phi + Bphi3 * cos_phi

        mass_2d = float(
            np.sum(fluid_data["rho"] * (2.0 * np.pi * r_centers[:, None]))
            * spacing_2d[0]
            * spacing_2d[1]
        )
        mass_3d_full = float(np.sum(rho3) * dx * dy * dz)
        cyl_mask = r <= r_max_data
        mass_3d_cyl = float(np.sum(rho3 * cyl_mask) * dx * dy * dz)
        denom = max(mass_2d, 1.0e-30)
        rotation_checks = {
            "mass_2d": mass_2d,
            "mass_3d_full": mass_3d_full,
            "mass_3d_cyl": mass_3d_cyl,
            "rotation_mass_integral_rel_diff": float(abs(mass_3d_cyl - mass_2d) / denom),
            "rotation_mass_integral_full_rel_diff": float(abs(mass_3d_full - mass_2d) / denom),
        }
        rotation_meta = {
            "interp_method": "bilinear",
            "phi_definition": "atan2(y,x)",
            "r_definition": "sqrt(x^2+y^2)",
            "axis_singularity_policy": "r<axis_r_min: phi=0; Bphi=0; vphi=0",
            "axis_r_min": axis_r_min,
            "r_clip_policy": r_clip_policy,
            "r_clip_fraction": r_clip_fraction,
            "source_axisym_coords": {
                "r_min": r_min_data,
                "r_max": r_max_data,
                "z_min": z_min_data,
                "z_max": z_max_data,
                "dr": float(spacing_2d[0]),
                "dz": float(spacing_2d[1]),
            },
        }
        cart_grid_meta = {
            "nx": nx,
            "ny": ny,
            "nz": nz_cart,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min_cart,
            "z_max": z_max_cart,
            "spacing_3d": spacing_3d.tolist(),
            "offset_3d": offset_3d.tolist(),
        }
        vector_map = {
            "vx": "vr*cos(phi) - vphi*sin(phi)",
            "vy": "vr*sin(phi) + vphi*cos(phi)",
            "vz": "vz",
            "Bx": "Br*cos(phi) - Bphi*sin(phi)",
            "By": "Br*sin(phi) + Bphi*cos(phi)",
            "Bz": "Bz",
        }

    b_mag = np.sqrt(Bx3**2 + By3**2 + Bz3**2)
    source_b_stats = {
        "b_rms": float(np.sqrt(np.mean(b_mag**2))),
        "b_max": float(np.max(b_mag)),
        "b_min": float(np.min(b_mag)),
    }

    import openpmd_api as io
    fluid_summary = write_openpmd_cartesian(
        fluid_3d,
        "fluid",
        {"rho": rho3, "vx": vx3, "vy": vy3, "vz": vz3, "Ti": Ti3, "Te": Te3},
        spacing_3d,
        offset_3d,
        {
            io.Unit_Dimension.L: 0,
            io.Unit_Dimension.T: 0,
            io.Unit_Dimension.M: 0,
        },
    )
    b_summary = write_openpmd_cartesian(
        b_3d,
        "B",
        {"x": Bx3, "y": By3, "z": Bz3},
        spacing_3d,
        offset_3d,
        {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.L: 0,
            io.Unit_Dimension.T: -2,
            io.Unit_Dimension.I: -1,
        },
    )

    selection_mode = selection_meta.get("selection_mode")
    selection_strategy = selection_mode
    selection_reason = None
    fallback_used = False
    fallback_reason = None
    if selection_mode == "explicit":
        selection_reason = "vtk_path provided in config."
    elif selection_mode == "closest_time":
        selection_reason = (
            "target_time = formation_time_frac * tlim; chose VTK with minimal |vtk_time-target_time|."
        )
    elif selection_mode == "index":
        selection_reason = "vtk_time unavailable; selected by index nearest target_time."
        fallback_used = True
        fallback_reason = "vtk_time missing; index-based selection."
        selection_strategy = "closest_index"
    elif selection_mode == "latest":
        selection_reason = "target_time unavailable; selected latest VTK."
        fallback_used = True
        fallback_reason = "target_time unavailable; latest file selection."
    else:
        selection_reason = "selection_mode unspecified."

    chosen_index = selection_meta.get("selected_index")
    index_base = 0 if chosen_index is not None else None
    source_case_id = config.get("source_case_id") or infer_case_id(metrics_path)

    meta = {
        "vtk_path": str(vtk),
        "vtk_sha256": hash_file(vtk),
        "vtk_time": parse_vtk_time(vtk),
        "selection": selection_meta,
        "selection_strategy": selection_strategy,
        "selection_reason": selection_reason,
        "chosen_index": chosen_index,
        "index_base": index_base,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "vtk_pattern": vtk_pattern,
        "source_case_id": source_case_id,
        "formation_time_frac": formation_frac,
        "tlim": tlim,
        "fluid_2d_path": str(fluid_2d),
        "b_2d_path": str(b_2d),
        "fluid_path": str(fluid_3d),
        "b_path": str(b_3d),
        "axis_mode": axis_mode,
        "fold_r": fold_r,
        "resample": resample,
        "nr": nr,
        "nz": nz,
        "n3_extrude": n3 if mapping_mode == "extrude_x3" else None,
        "extrude_length": extrude_length if mapping_mode == "extrude_x3" else None,
        "spacing_2d": spacing_2d.tolist(),
        "offset_2d": offset_2d.tolist(),
        "spacing_3d": spacing_3d.tolist(),
        "offset_3d": offset_3d.tolist(),
        "axis_order_3d": "x,y,z",
        "units_note": "B/fluid units follow the Athena++ output convention used in the source case.",
        "mapping_mode": mapping_mode,
        "vector_map": vector_map,
        "source_b_stats": source_b_stats,
        "rho_scale": rho_scale,
        "vel_scale": vel_scale,
        "press_scale": press_scale,
        "B_scale": b_scale,
        "amu": amu,
        "te_ratio": te_ratio,
        "te_const": te_const,
        "athena_vis_path": athena_vis_path,
        "fluid_summary": fluid_summary,
        "b_summary": b_summary,
        "generated": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
    }
    if mapping_mode == "rotate_axisym":
        meta["cart_grid"] = cart_grid_meta
        meta["integral_checks"] = rotation_checks
        meta["rotation_mass_integral_rel_diff"] = rotation_checks.get(
            "rotation_mass_integral_rel_diff"
        )
        meta["rotation_mass_integral_full_rel_diff"] = rotation_checks.get(
            "rotation_mass_integral_full_rel_diff"
        )
        meta.update(rotation_meta)

    meta_path = out_dir / "handoff_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"[handoff] {meta_path}")


if __name__ == "__main__":
    main()
