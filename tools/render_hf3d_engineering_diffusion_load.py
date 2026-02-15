#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
)

MU0 = 1.25663706127e-6


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    for key in ("dphi_dt_v", "force_proxy_n", "phi_wb"):
        if key in arr.dtype.names:
            out[key] = np.asarray(arr[key], dtype=np.float64)
    return out


def write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# HF3D Engineering Summary",
        "",
        f"- plotfiles_total: `{metrics.get('plotfiles_total')}`",
        f"- plotfiles_used: `{metrics.get('plotfiles_used')}`",
        f"- frames_rendered: `{metrics.get('frames_rendered')}`",
        f"- j_proxy_p99_peak: `{metrics.get('j_proxy_p99_peak')}`",
        f"- p_mag_p99_peak_pa: `{metrics.get('p_mag_p99_peak_pa')}`",
        f"- force_peak_n: `{metrics.get('force_peak_n')}`",
        f"- dphi_dt_peak_v: `{metrics.get('dphi_dt_peak_v')}`",
        f"- diffusion_activation_ok: `{metrics.get('diffusion_activation_ok')}`",
        f"- etaj2_scaling_ok: `{metrics.get('etaj2_scaling_ok')}`",
        f"- mp4_written: `{metrics.get('mp4_written')}`",
        f"- gif_written: `{metrics.get('gif_written')}`",
        f"- render_success: `{metrics.get('render_success')}`",
        "",
        "This animation packages 3D engineering observables for diffusion/load acceptance.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render high-fidelity 3D engineering eddy/diffusion/load animation.")
    parser.add_argument(
        "--diag-dir",
        default="outputs/m9-h4-1-mhd-seeded-hybrid-3d-rotation/raw/run/diag",
        help="WarpX plotfile directory.",
    )
    parser.add_argument(
        "--load-series",
        default="outputs/m26-d2-magnetic-load-interface/analysis/magnetic_load_series.csv",
        help="Engineering load interface CSV.",
    )
    parser.add_argument(
        "--diffusion-metrics",
        default="outputs/m28-s3-em-diffusion-nonproxy-gate/analysis/metrics.json",
        help="Diffusion/etaJ2 gate metrics JSON.",
    )
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--point-budget", type=int, default=12000)
    parser.add_argument("--j-quantile", type=float, default=0.992)
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
    diff_path = Path(args.diffusion_metrics)
    if not diff_path.is_absolute():
        diff_path = (repo_root / diff_path).resolve()
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

    diff_metrics = read_json(diff_path)
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

    fields = ("Bx", "By", "Bz")
    frame_state = []
    time_s = []
    j99_series = []
    p99_series = []
    shape = None

    for i, path in enumerate(plotfiles):
        snap = load_plotfile(path, fields)
        bx = snap["Bx"]
        by = snap["By"]
        bz = snap["Bz"]

        if shape is None:
            shape = bx.shape
            xx, yy, zz = np.meshgrid(snap["x"], snap["y"], snap["z"], indexing="ij")
            phi = np.arctan2(yy.ravel(), xx.ravel() + 1.0e-30)
            _ = phi  # keep explicit for possible future azimuthal diagnostics

        dby_dz = np.gradient(by, snap["dz"], axis=2, edge_order=1)
        dbz_dy = np.gradient(bz, snap["dy"], axis=1, edge_order=1)
        dbz_dx = np.gradient(bz, snap["dx"], axis=0, edge_order=1)
        dbx_dz = np.gradient(bx, snap["dz"], axis=2, edge_order=1)
        dbx_dy = np.gradient(bx, snap["dy"], axis=1, edge_order=1)
        dby_dx = np.gradient(by, snap["dx"], axis=0, edge_order=1)
        jx = dbz_dy - dby_dz
        jy = dbx_dz - dbz_dx
        jz = dby_dx - dbx_dy
        jmag = np.sqrt(jx * jx + jy * jy + jz * jz)

        b2 = bx * bx + by * by + bz * bz
        pmag = b2 / (2.0 * MU0)

        jf = jmag.ravel()
        pf = pmag.ravel()
        jthr = float(np.nanquantile(jf, args.j_quantile))
        idx = sample_points((jf >= jthr).reshape(shape), args.point_budget, 6000 + i)
        px, py, pz = flat_to_xyz(idx, shape, snap["x"], snap["y"], snap["z"])
        cval = np.log10(pf[idx] + 1.0e-30) if idx.size else np.array([], dtype=np.float64)

        j99 = float(np.nanpercentile(jf, 99.0))
        p99 = float(np.nanpercentile(pf, 99.0))
        frame_state.append(
            {
                "time_s": float(snap["time_s"]),
                "px": px,
                "py": py,
                "pz": pz,
                "cval": cval,
                "j99": j99,
                "p99": p99,
                "bounds": (snap["left"], snap["right"]),
            }
        )
        time_s.append(float(snap["time_s"]))
        j99_series.append(j99)
        p99_series.append(p99)

    t = np.asarray(time_s, dtype=np.float64)
    j99_arr = np.asarray(j99_series, dtype=np.float64)
    p99_arr = np.asarray(p99_series, dtype=np.float64)

    load = read_load_series(load_path)
    load_present = bool(load)
    dphi = np.zeros_like(t)
    force = np.zeros_like(t)
    if load_present:
        src_t = np.asarray(load.get("time_s", []), dtype=np.float64)
        dphi = align_series(t, src_t, np.asarray(load.get("dphi_dt_v", []), dtype=np.float64))
        force = align_series(t, src_t, np.asarray(load.get("force_proxy_n", []), dtype=np.float64))

    jn = normalize01(j99_arr)
    pn = normalize01(p99_arr)
    dn = normalize01(np.abs(dphi))
    fn = normalize01(np.abs(force))

    diff_text = (
        f"diff_coeff: {diff_metrics.get('diffusion_coeff_1', 'n/a')} -> {diff_metrics.get('diffusion_coeff_2', 'n/a')}\n"
        f"etaJ2_gain_2_over_1: {diff_metrics.get('etaJ2_gain_2_over_1', 'n/a')}\n"
        f"diffusion_activation_ok: {diff_metrics.get('diffusion_activation_ok', 'n/a')}\n"
        f"etaj2_scaling_ok: {diff_metrics.get('etaj2_scaling_ok', 'n/a')}"
    )

    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.28)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    writers = FrameWriters(mp4_out, gif_out, args.fps)
    writers.open()
    frames_rendered = 0
    try:
        for i, st in enumerate(frame_state):
            ax3d.cla()
            ax_top.cla()
            ax_bot.cla()

            if st["cval"].size:
                cn = normalize01(st["cval"])
                ax3d.scatter(
                    st["px"],
                    st["py"],
                    st["pz"],
                    c=cn,
                    cmap="inferno",
                    s=4,
                    alpha=0.35,
                    linewidths=0.0,
                )
            left, right = st["bounds"]
            ax3d.set_xlim(float(left[0]), float(right[0]))
            ax3d.set_ylim(float(left[1]), float(right[1]))
            ax3d.set_zlim(float(left[2]), float(right[2]))
            ax3d.view_init(elev=23.0, azim=40.0 + 0.8 * i)
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title(
                f"3D Eddy/Diffusion Hotspots  t={st['time_s']*1e9:.3f} ns\n"
                f"J99={st['j99']:.3e}, P99={st['p99']:.3e} Pa",
                fontsize=11,
            )

            tns = t * 1.0e9
            ax_top.plot(tns, jn, color="#d62728", linewidth=2.0, label="J proxy p99 (norm)")
            ax_top.plot(tns, pn, color="#1f77b4", linewidth=2.0, label="Pmag p99 (norm)")
            ax_top.axvline(tns[i], color="black", linestyle="--", linewidth=1.2)
            ax_top.set_ylabel("Normalized")
            ax_top.set_title("3D Engineering Field Indicators", fontsize=11)
            ax_top.grid(alpha=0.25)
            ax_top.legend(loc="upper right", fontsize=8)

            ax_bot.plot(tns, dn, color="#9467bd", linewidth=2.0, label="|dphi/dt| (norm)")
            ax_bot.plot(tns, fn, color="#17becf", linewidth=2.0, label="|force| (norm)")
            ax_bot.axvline(tns[i], color="black", linestyle="--", linewidth=1.2)
            ax_bot.set_xlabel("time [ns]")
            ax_bot.set_ylabel("Normalized")
            ax_bot.set_title("Load Interface Timeline", fontsize=11)
            ax_bot.grid(alpha=0.25)
            ax_bot.legend(loc="upper right", fontsize=8)
            ax_bot.text(
                0.01,
                0.02,
                diff_text,
                transform=ax_bot.transAxes,
                fontsize=8,
                family="monospace",
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )

            fig.suptitle("HF3D-3: Eddy Current / Diffusion / Load Engineering View", fontsize=14)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writers.append(frame)
            frames_rendered += 1
    finally:
        writers.close()
        plt.close(fig)

    metrics = {
        "plotfiles_total": int(len(all_plotfiles)),
        "plotfiles_used": int(len(plotfiles)),
        "frames_rendered": int(frames_rendered),
        "load_series_present": bool(load_present),
        "diffusion_metrics_present": bool(diff_metrics),
        "j_proxy_p99_peak": float(np.max(j99_arr)),
        "p_mag_p99_peak_pa": float(np.max(p99_arr)),
        "force_peak_n": float(np.max(np.abs(force))) if force.size else 0.0,
        "dphi_dt_peak_v": float(np.max(np.abs(dphi))) if dphi.size else 0.0,
        "diffusion_activation_ok": bool(diff_metrics.get("diffusion_activation_ok") is True),
        "etaj2_scaling_ok": bool(diff_metrics.get("etaj2_scaling_ok") is True),
        "etaJ2_gain_2_over_1": diff_metrics.get("etaJ2_gain_2_over_1"),
        "mp4_written": bool(mp4_out.exists() and mp4_out.stat().st_size > 0),
        "gif_written": bool(gif_out.exists() and gif_out.stat().st_size > 0),
        "mp4_error": writers.mp4_error,
    }
    metrics["render_success"] = bool(
        metrics["frames_rendered"] >= 3
        and metrics["gif_written"]
        and metrics["load_series_present"]
        and metrics["diffusion_metrics_present"]
    )

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_out, metrics)


if __name__ == "__main__":
    main()
