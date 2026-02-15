#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hf3d_anim_common import (  # noqa: E402
    FrameWriters,
    align_series,
    flat_to_xyz,
    list_plotfiles,
    load_plotfile,
    normalize01,
    sample_points,
    select_even,
    weighted_centroid,
)


def read_load_series(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True)
    except Exception:
        return {}
    if arr.size == 0:
        return {}
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    out = {"time_s": np.asarray(arr["time_s"], dtype=np.float64)}
    for key in ("phi_wb", "dphi_dt_v", "force_proxy_n"):
        if key in arr.dtype.names:
            out[key] = np.asarray(arr[key], dtype=np.float64)
    return out


def write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# HF3D Merge+Recapture Summary",
        "",
        f"- plotfiles_total: `{metrics.get('plotfiles_total')}`",
        f"- plotfiles_used: `{metrics.get('plotfiles_used')}`",
        f"- frames_rendered: `{metrics.get('frames_rendered')}`",
        f"- merge_sep_drop_ratio: `{metrics.get('merge_sep_drop_ratio')}`",
        f"- compression_gain_ratio: `{metrics.get('compression_gain_ratio')}`",
        f"- recapture_dphi_peak_v: `{metrics.get('recapture_dphi_peak_v')}`",
        f"- recapture_force_peak_n: `{metrics.get('recapture_force_peak_n')}`",
        f"- mp4_written: `{metrics.get('mp4_written')}`",
        f"- gif_written: `{metrics.get('gif_written')}`",
        f"- render_success: `{metrics.get('render_success')}`",
        "",
        "This animation links 3D merge/compression topology proxies with expansion/recapture electrical response.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render high-fidelity 3D merge/compression + recapture animation.")
    parser.add_argument(
        "--diag-dir",
        default="outputs/m9-h4-1-mhd-seeded-hybrid-3d-rotation/raw/run/diag",
        help="WarpX plotfile directory.",
    )
    parser.add_argument(
        "--load-series",
        default="outputs/m26-d2-magnetic-load-interface/analysis/magnetic_load_series.csv",
        help="CSV with magnetic load/recapture time series.",
    )
    parser.add_argument("--r-load-ohm", type=float, default=2000.0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--point-budget", type=int, default=12000)
    parser.add_argument("--density-quantile", type=float, default=0.988)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--mp4-out", required=True)
    parser.add_argument("--gif-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    diag_dir = Path(args.diag_dir)
    if not diag_dir.is_absolute():
        diag_dir = (repo_root / diag_dir).resolve()
    load_path = Path(args.load_series)
    if not load_path.is_absolute():
        load_path = (repo_root / load_path).resolve()
    mp4_out = Path(args.mp4_out)
    if not mp4_out.is_absolute():
        mp4_out = (repo_root / mp4_out).resolve()
    gif_out = Path(args.gif_out)
    if not gif_out.is_absolute():
        gif_out = (repo_root / gif_out).resolve()
    metrics_out = Path(args.metrics_out)
    if not metrics_out.is_absolute():
        metrics_out = (repo_root / metrics_out).resolve()
    summary_out = Path(args.summary_out)
    if not summary_out.is_absolute():
        summary_out = (repo_root / summary_out).resolve()

    all_plotfiles = list_plotfiles(diag_dir)
    plotfiles = select_even(all_plotfiles, args.max_frames)
    if not plotfiles:
        metrics = {
            "plotfiles_total": int(len(all_plotfiles)),
            "plotfiles_used": 0,
            "frames_rendered": 0,
            "render_success": False,
            "reason": "no_plotfiles",
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_summary(summary_out, metrics)
        return

    fields = ("rho", "Bx", "By", "Bz", "Ex", "Ey", "Ez")
    frame_state = []
    x_flat = y_flat = z_flat = None
    shape = None
    time_s = []
    sep_series = []
    comp_series = []
    tilt_series = []

    for frame_idx, path in enumerate(plotfiles):
        snap = load_plotfile(path, fields)
        if x_flat is None:
            shape = snap["rho"].shape
            xx, yy, zz = np.meshgrid(snap["x"], snap["y"], snap["z"], indexing="ij")
            x_flat = xx.ravel()
            y_flat = yy.ravel()
            z_flat = zz.ravel()

        rho = snap["rho"].ravel()
        bx = snap["Bx"].ravel()
        by = snap["By"].ravel()
        bz = snap["Bz"].ravel()
        ex = snap["Ex"].ravel()
        ey = snap["Ey"].ravel()
        ez = snap["Ez"].ravel()

        b2 = bx * bx + by * by + bz * bz + 1.0e-30
        cx = ey * bz - ez * by
        cy = ez * bx - ex * bz
        cz = ex * by - ey * bx
        v_exb2 = (cx * cx + cy * cy + cz * cz) / (b2 * b2)

        rho_q = float(np.nanquantile(rho, args.density_quantile))
        mask = rho >= rho_q
        idx = sample_points(mask.reshape(shape), args.point_budget, 1000 + frame_idx)
        px, py, pz = flat_to_xyz(idx, shape, snap["x"], snap["y"], snap["z"])
        cval = np.log10(v_exb2[idx] + 1.0e-30) if idx.size else np.array([], dtype=np.float64)

        w = np.clip(rho, 0.0, None)
        mask_left = x_flat < 0.0
        mask_right = ~mask_left
        cx_l, cy_l, cz_l = weighted_centroid(x_flat[mask_left], y_flat[mask_left], z_flat[mask_left], w[mask_left])
        cx_r, cy_r, cz_r = weighted_centroid(x_flat[mask_right], y_flat[mask_right], z_flat[mask_right], w[mask_right])
        sep = float(math.sqrt((cx_r - cx_l) ** 2 + (cy_r - cy_l) ** 2 + (cz_r - cz_l) ** 2))
        cx_g, cy_g, cz_g = weighted_centroid(x_flat, y_flat, z_flat, w)
        r2 = (x_flat - cx_g) ** 2 + (y_flat - cy_g) ** 2 + (z_flat - cz_g) ** 2
        rms = float(np.sqrt(np.sum(r2 * w) / max(np.sum(w), 1.0e-30)))
        comp = float(1.0 / max(rms, 1.0e-30))

        frame_state.append(
            {
                "time_s": float(snap["time_s"]),
                "px": px,
                "py": py,
                "pz": pz,
                "cval": cval,
                "left": (cx_l, cy_l, cz_l),
                "right": (cx_r, cy_r, cz_r),
                "sep": sep,
                "comp": comp,
                "tilt": float(cy_g),
                "bounds": (snap["left"], snap["right"]),
            }
        )
        time_s.append(float(snap["time_s"]))
        sep_series.append(sep)
        comp_series.append(comp)
        tilt_series.append(float(cy_g))

    time_s_arr = np.asarray(time_s, dtype=np.float64)
    sep_arr = np.asarray(sep_series, dtype=np.float64)
    comp_arr = np.asarray(comp_series, dtype=np.float64)
    tilt_arr = np.asarray(tilt_series, dtype=np.float64)

    load = read_load_series(load_path)
    load_present = bool(load)
    phi_aligned = np.zeros_like(time_s_arr)
    dphi_aligned = np.zeros_like(time_s_arr)
    force_aligned = np.zeros_like(time_s_arr)
    if load_present:
        src_t = np.asarray(load.get("time_s", []), dtype=np.float64)
        phi_aligned = align_series(time_s_arr, src_t, np.asarray(load.get("phi_wb", []), dtype=np.float64))
        dphi_aligned = align_series(time_s_arr, src_t, np.asarray(load.get("dphi_dt_v", []), dtype=np.float64))
        force_aligned = align_series(time_s_arr, src_t, np.asarray(load.get("force_proxy_n", []), dtype=np.float64))
    i_induced = -dphi_aligned / max(float(args.r_load_ohm), 1.0e-30)

    sep_n = normalize01(sep_arr)
    comp_n = normalize01(comp_arr)
    tilt_n = normalize01(np.abs(tilt_arr))
    phi_n = normalize01(phi_aligned)
    dphi_n = normalize01(np.abs(dphi_aligned))
    i_n = normalize01(np.abs(i_induced))
    force_n = normalize01(np.abs(force_aligned))

    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.28)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    writers = FrameWriters(mp4_out, gif_out, args.fps)
    writers.open()
    frames_rendered = 0
    try:
        for idx, st in enumerate(frame_state):
            ax3d.cla()
            ax_top.cla()
            ax_bot.cla()

            cval = st["cval"]
            if cval.size:
                cv = normalize01(cval)
                ax3d.scatter(
                    st["px"],
                    st["py"],
                    st["pz"],
                    c=cv,
                    cmap="turbo",
                    s=4,
                    alpha=0.35,
                    linewidths=0.0,
                )
            lx, ly, lz = st["left"]
            rx, ry, rz = st["right"]
            ax3d.plot([lx, rx], [ly, ry], [lz, rz], color="white", linewidth=2.0, alpha=0.9)
            ax3d.scatter([lx, rx], [ly, ry], [lz, rz], color=["cyan", "magenta"], s=46)
            left_b, right_b = st["bounds"]
            ax3d.set_xlim(float(left_b[0]), float(right_b[0]))
            ax3d.set_ylim(float(left_b[1]), float(right_b[1]))
            ax3d.set_zlim(float(left_b[2]), float(right_b[2]))
            ax3d.view_init(elev=22.0, azim=35.0 + 0.8 * idx)
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title(
                f"3D Merge+Compression Topology  t={st['time_s']*1e9:.3f} ns\n"
                f"sep={st['sep']:.4f}, comp={st['comp']:.4f}, tilt_y={st['tilt']:.4e}",
                fontsize=11,
            )

            tns = time_s_arr * 1.0e9
            ax_top.plot(tns, sep_n, label="merge separation (norm)", color="#1f77b4", linewidth=2.0)
            ax_top.plot(tns, comp_n, label="compression proxy (norm)", color="#ff7f0e", linewidth=2.0)
            ax_top.plot(tns, tilt_n, label="tilt |yc| (norm)", color="#2ca02c", linewidth=1.8)
            ax_top.axvline(tns[idx], color="black", linestyle="--", linewidth=1.2)
            ax_top.set_ylabel("Normalized")
            ax_top.set_title("Merge/Compression/Tilt Timeline", fontsize=11)
            ax_top.grid(alpha=0.25)
            ax_top.legend(loc="upper right", fontsize=8)

            ax_bot.plot(tns, phi_n, label="phi (norm)", color="#9467bd", linewidth=1.8)
            ax_bot.plot(tns, dphi_n, label="|dphi/dt| (norm)", color="#d62728", linewidth=2.0)
            ax_bot.plot(tns, i_n, label="|I_ind| proxy (norm)", color="#8c564b", linewidth=1.8)
            ax_bot.plot(tns, force_n, label="|force| (norm)", color="#17becf", linewidth=1.8)
            ax_bot.axvline(tns[idx], color="black", linestyle="--", linewidth=1.2)
            ax_bot.set_xlabel("time [ns]")
            ax_bot.set_ylabel("Normalized")
            ax_bot.set_title("Expansion/Recapture Electrical Chain", fontsize=11)
            ax_bot.grid(alpha=0.25)
            ax_bot.legend(loc="upper right", fontsize=8)

            fig.suptitle("HF3D-1: Merge+Compression and Expansion/Recapture", fontsize=14)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writers.append(frame)
            frames_rendered += 1
    finally:
        writers.close()
        plt.close(fig)

    sep_drop = float(sep_arr[-1] / max(sep_arr[0], 1.0e-30))
    comp_gain = float(np.max(comp_arr) / max(np.min(comp_arr), 1.0e-30))
    metrics = {
        "plotfiles_total": int(len(all_plotfiles)),
        "plotfiles_used": int(len(plotfiles)),
        "frames_rendered": int(frames_rendered),
        "load_series_present": bool(load_present),
        "recapture_dphi_peak_v": float(np.max(np.abs(dphi_aligned))) if dphi_aligned.size else 0.0,
        "recapture_force_peak_n": float(np.max(np.abs(force_aligned))) if force_aligned.size else 0.0,
        "merge_sep_drop_ratio": float(sep_drop),
        "compression_gain_ratio": float(comp_gain),
        "mp4_written": bool(mp4_out.exists() and mp4_out.stat().st_size > 0),
        "gif_written": bool(gif_out.exists() and gif_out.stat().st_size > 0),
        "mp4_error": writers.mp4_error,
    }
    metrics["render_success"] = bool(
        metrics["frames_rendered"] >= 3
        and metrics["gif_written"]
        and metrics["load_series_present"]
    )

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_out, metrics)


if __name__ == "__main__":
    main()
