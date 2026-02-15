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
    flat_to_xyz,
    list_plotfiles,
    load_plotfile,
    normalize01,
    sample_points,
    select_even,
)


def write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# HF3D Formation Summary",
        "",
        f"- plotfiles_total: `{metrics.get('plotfiles_total')}`",
        f"- plotfiles_used: `{metrics.get('plotfiles_used')}`",
        f"- frames_rendered: `{metrics.get('frames_rendered')}`",
        f"- m1_amp_peak: `{metrics.get('m1_amp_peak')}`",
        f"- flux_loss_frac_peak: `{metrics.get('flux_loss_frac_peak')}`",
        f"- mag_energy_loss_frac_peak: `{metrics.get('mag_energy_loss_frac_peak')}`",
        f"- micro_grad_proxy_peak: `{metrics.get('micro_grad_proxy_peak')}`",
        f"- mp4_written: `{metrics.get('mp4_written')}`",
        f"- gif_written: `{metrics.get('gif_written')}`",
        f"- render_success: `{metrics.get('render_success')}`",
        "",
        "This animation emphasizes why 3D is required: non-axisymmetric mode growth and flux/energy loss proxies.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render high-fidelity 3D formation micro-instability animation.")
    parser.add_argument(
        "--diag-dir",
        default="outputs/m9-h4-1-mhd-seeded-hybrid-3d-rotation/raw/run/diag",
        help="WarpX 3D plotfile directory.",
    )
    parser.add_argument("--max-frames", type=int, default=18)
    parser.add_argument("--formation-frac", type=float, default=0.5, help="Only use early formation-like segment.")
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
    if all_plotfiles:
        use_n = max(2, int(np.ceil(len(all_plotfiles) * float(args.formation_frac))))
        candidate = all_plotfiles[:use_n]
    else:
        candidate = []
    plotfiles = select_even(candidate, args.max_frames)
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

    fields = ("rho", "Bx", "By", "Bz")
    frame_state = []
    time_s = []
    m1_amp = []
    flux_x0 = []
    mag_energy = []
    grad_proxy = []
    shape = None

    for i, path in enumerate(plotfiles):
        snap = load_plotfile(path, fields)
        rho = snap["rho"]
        bx = snap["Bx"]
        by = snap["By"]
        bz = snap["Bz"]
        if shape is None:
            shape = rho.shape
            xx, yy, zz = np.meshgrid(snap["x"], snap["y"], snap["z"], indexing="ij")
            x_flat = xx.ravel()
            y_flat = yy.ravel()
            z_flat = zz.ravel()
            phi_flat = np.arctan2(y_flat, x_flat + 1.0e-30)

        rho_f = rho.ravel()
        rho_q = float(np.nanquantile(rho_f, args.density_quantile))
        idx = sample_points((rho_f >= rho_q).reshape(shape), args.point_budget, 3000 + i)
        px, py, pz = flat_to_xyz(idx, shape, snap["x"], snap["y"], snap["z"])
        c_phase = phi_flat[idx] if idx.size else np.array([], dtype=np.float64)

        w = np.clip(rho_f, 0.0, None)
        re = float(np.sum(w * np.cos(phi_flat)))
        im = float(np.sum(w * np.sin(phi_flat)))
        m1 = float(np.sqrt(re * re + im * im) / max(np.sum(w), 1.0e-30))

        ix0 = int(np.argmin(np.abs(np.asarray(snap["x"], dtype=np.float64))))
        flux = float(np.sum(bx[ix0, :, :]) * snap["dy"] * snap["dz"])
        b2 = bx * bx + by * by + bz * bz
        e_mag = float(0.5 * np.sum(b2) * snap["dx"] * snap["dy"] * snap["dz"])
        gx, gy, gz = np.gradient(rho, snap["dx"], snap["dy"], snap["dz"], edge_order=1)
        grad2 = gx * gx + gy * gy + gz * gz
        gproxy = float(np.mean(grad2) / max(np.mean(rho * rho), 1.0e-30))

        frame_state.append(
            {
                "time_s": float(snap["time_s"]),
                "px": px,
                "py": py,
                "pz": pz,
                "c_phase": c_phase,
                "m1": m1,
                "flux": flux,
                "e_mag": e_mag,
                "gproxy": gproxy,
                "bounds": (snap["left"], snap["right"]),
            }
        )
        time_s.append(float(snap["time_s"]))
        m1_amp.append(m1)
        flux_x0.append(flux)
        mag_energy.append(e_mag)
        grad_proxy.append(gproxy)

    t = np.asarray(time_s, dtype=np.float64)
    m1_arr = np.asarray(m1_amp, dtype=np.float64)
    flux_arr = np.asarray(flux_x0, dtype=np.float64)
    emag_arr = np.asarray(mag_energy, dtype=np.float64)
    gproxy_arr = np.asarray(grad_proxy, dtype=np.float64)

    flux_loss = (flux_arr[0] - flux_arr) / max(abs(flux_arr[0]), 1.0e-30)
    emag_loss = (emag_arr[0] - emag_arr) / max(abs(emag_arr[0]), 1.0e-30)

    m1_n = normalize01(m1_arr)
    gproxy_n = normalize01(gproxy_arr)
    flux_n = normalize01(np.abs(flux_loss))
    emag_n = normalize01(np.abs(emag_loss))

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

            if st["c_phase"].size:
                phase_norm = (st["c_phase"] + np.pi) / (2.0 * np.pi)
                ax3d.scatter(
                    st["px"],
                    st["py"],
                    st["pz"],
                    c=phase_norm,
                    cmap="hsv",
                    s=4,
                    alpha=0.35,
                    linewidths=0.0,
                )
            left, right = st["bounds"]
            ax3d.set_xlim(float(left[0]), float(right[0]))
            ax3d.set_ylim(float(left[1]), float(right[1]))
            ax3d.set_zlim(float(left[2]), float(right[2]))
            ax3d.view_init(elev=22.0, azim=30.0 + 0.8 * i)
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title(
                f"3D Formation Micro-Instability  t={st['time_s']*1e9:.3f} ns\n"
                f"m1={st['m1']:.3e}, flux_loss={((flux_arr[0]-st['flux'])/max(abs(flux_arr[0]),1e-30)):.3e}",
                fontsize=11,
            )

            tns = t * 1.0e9
            ax_top.plot(tns, m1_n, color="#d62728", linewidth=2.0, label="m1 non-axisym (norm)")
            ax_top.plot(tns, gproxy_n, color="#1f77b4", linewidth=2.0, label="micro-grad proxy (norm)")
            ax_top.axvline(tns[i], color="black", linestyle="--", linewidth=1.2)
            ax_top.set_ylabel("Normalized")
            ax_top.set_title("3D Micro-Instability Indicators", fontsize=11)
            ax_top.grid(alpha=0.25)
            ax_top.legend(loc="upper right", fontsize=8)

            ax_bot.plot(tns, flux_n, color="#2ca02c", linewidth=2.0, label="|flux loss| (norm)")
            ax_bot.plot(tns, emag_n, color="#9467bd", linewidth=2.0, label="|mag energy loss| (norm)")
            ax_bot.axvline(tns[i], color="black", linestyle="--", linewidth=1.2)
            ax_bot.set_xlabel("time [ns]")
            ax_bot.set_ylabel("Normalized")
            ax_bot.set_title("Flux/Magnetic-Energy Loss", fontsize=11)
            ax_bot.grid(alpha=0.25)
            ax_bot.legend(loc="upper right", fontsize=8)

            fig.suptitle("HF3D-2: Formation Micro-Instability and Flux Loss", fontsize=14)
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
        "m1_amp_peak": float(np.max(m1_arr)),
        "flux_loss_frac_peak": float(np.max(np.abs(flux_loss))),
        "mag_energy_loss_frac_peak": float(np.max(np.abs(emag_loss))),
        "micro_grad_proxy_peak": float(np.max(gproxy_arr)),
        "mp4_written": bool(mp4_out.exists() and mp4_out.stat().st_size > 0),
        "gif_written": bool(gif_out.exists() and gif_out.stat().st_size > 0),
        "mp4_error": writers.mp4_error,
    }
    metrics["render_success"] = bool(metrics["frames_rendered"] >= 3 and metrics["gif_written"])

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_out, metrics)


if __name__ == "__main__":
    main()
