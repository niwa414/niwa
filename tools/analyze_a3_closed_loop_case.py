#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_corr(x, y):
    if x.size < 2 or y.size < 2:
        return None
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A3 closed-loop LCR gate.")
    parser.add_argument("--metadata", required=True, help="WarpX run metadata JSON.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--lcr-csv", required=True, help="Output CSV for LCR history.")
    parser.add_argument("--feedback-csv", required=True, help="Output CSV for feedback signal series.")
    parser.add_argument(
        "--feedback-signal-source",
        default="radius_rms",
        help="Feedback signal source: radius_rms (default) or r_proxy_b_rms.",
    )
    args = parser.parse_args()

    meta = load_metadata(Path(args.metadata))
    run_args = meta.get("args", {})
    lcr_history = meta.get("lcr_history") or []
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def to_float(val):
        return float(val) if val is not None else float("nan")

    lcr_rows = [row for row in lcr_history if row.get("t") is not None]
    lcr_times = np.array([to_float(row.get("t")) for row in lcr_rows], dtype=float)
    lcr_I = np.array([to_float(row.get("I")) for row in lcr_rows], dtype=float)
    lcr_R = np.array([to_float(row.get("R_plasma")) for row in lcr_rows], dtype=float)
    lcr_feedback = np.array([to_float(row.get("feedback_signal")) for row in lcr_rows], dtype=float)
    driver_amp = np.array([to_float(row.get("B_est")) for row in lcr_rows], dtype=float)
    lcr_feedback_used = [row.get("feedback_used") for row in lcr_history]
    feedback_updates = int(sum(1 for val in lcr_feedback_used if val))
    feedback_used_fraction = float(feedback_updates / len(lcr_history)) if lcr_history else None

    radius_times = []
    radius_vals = []
    signal_times = []
    signal_vals = []
    source = str(args.feedback_signal_source or "radius_rms").lower()
    source_detail = None
    proxy_eps = 1.0e-30
    for rec in records:
        r_val = rec.get("radius_rms")
        if r_val is not None:
            radius_times.append(rec.get("time"))
            radius_vals.append(r_val)
        if source == "radius_rms":
            if r_val is None:
                continue
            signal_times.append(rec.get("time"))
            signal_vals.append(r_val)
        elif source == "r_proxy_b_rms":
            mode_energy = rec.get("mode_energy") or {}
            m0 = mode_energy.get("m0")
            if m0 is None:
                continue
            try:
                m0 = float(m0)
            except Exception:
                continue
            if m0 < 0.0:
                continue
            b_proxy = float(np.sqrt(m0))
            r_proxy = 1.0 / max(b_proxy, proxy_eps)
            signal_times.append(rec.get("time"))
            signal_vals.append(r_proxy)
            source_detail = "r_proxy_b_rms = 1 / max(sqrt(mode_energy.m0), eps)"

    radius_times = np.array(radius_times, dtype=float) if radius_times else np.array([])
    radius_vals = np.array(radius_vals, dtype=float) if radius_vals else np.array([])
    signal_times = np.array(signal_times, dtype=float) if signal_times else np.array([])
    signal_vals = np.array(signal_vals, dtype=float) if signal_vals else np.array([])

    feedback_signal_present = bool(signal_vals.size)
    feedback_signal_std = float(np.std(signal_vals)) if feedback_signal_present else None
    feedback_signal_range = (
        float(np.max(signal_vals) - np.min(signal_vals)) if feedback_signal_present else None
    )
    feedback_signal_mean = float(np.mean(signal_vals)) if feedback_signal_present else None
    feedback_signal_min = float(np.min(signal_vals)) if feedback_signal_present else None
    feedback_signal_max = float(np.max(signal_vals)) if feedback_signal_present else None

    radius_rms_present = bool(radius_vals.size)
    radius_rms_std = float(np.std(radius_vals)) if radius_rms_present else None
    radius_rms_range = (
        float(np.max(radius_vals) - np.min(radius_vals)) if radius_rms_present else None
    )
    radius_rms_min = float(np.min(radius_vals)) if radius_rms_present else None
    radius_rms_max = float(np.max(radius_vals)) if radius_rms_present else None

    feedback_corr = None
    if signal_vals.size and lcr_times.size:
        interp_fb = np.interp(lcr_times, signal_times, signal_vals)
        feedback_corr = compute_corr(interp_fb, lcr_R)

    max_steps = run_args.get("max_steps")
    last_step = records[-1].get("step") if records else None
    ran_to_completion = None
    if max_steps is not None:
        if last_step is not None:
            ran_to_completion = last_step >= (max_steps - 1)
        elif lcr_history:
            ran_to_completion = len(lcr_history) >= max_steps
    if ran_to_completion is None:
        ran_to_completion = bool(lcr_history)

    circuit_update_count = meta.get("lcr_circuit_update_count")
    if circuit_update_count is None:
        circuit_update_count = len(lcr_history)
    driver_writeback_count = meta.get("lcr_driver_writeback_count")
    if driver_writeback_count is None and lcr_history:
        driver_writeback_count = len(lcr_history)
    coupling_mode = str(run_args.get("lcr_coupling_mode", "weak")).lower()
    coupling_stride = run_args.get("lcr_coupling_stride")
    driver_amp_std = float(np.std(driver_amp)) if driver_amp.size else None
    driver_amp_mean = float(np.mean(driver_amp)) if driver_amp.size else None
    driver_amp_range = float(np.max(driver_amp) - np.min(driver_amp)) if driver_amp.size else None
    circuit_update_fraction = (
        float(circuit_update_count) / float(max_steps)
        if max_steps and circuit_update_count is not None
        else None
    )
    driver_writeback_fraction = (
        float(driver_writeback_count) / float(max_steps)
        if max_steps and driver_writeback_count is not None
        else None
    )
    driver_writeback_match = None
    if driver_writeback_count is not None and circuit_update_count is not None:
        driver_writeback_match = abs(driver_writeback_count - circuit_update_count) <= 1

    no_nan_in_metrics = True
    for arr in (lcr_times, lcr_I, lcr_R, lcr_feedback, driver_amp, signal_vals, radius_vals):
        if arr.size and np.isnan(arr).any():
            no_nan_in_metrics = False

    metrics = {
        "ran_to_completion": ran_to_completion,
        "no_nan_in_metrics": no_nan_in_metrics,
        "closed_loop_enabled": bool(run_args.get("use_lcr")) and run_args.get("lcr_feedback") not in (None, "off"),
        "strong_coupling_enabled": coupling_mode == "strong",
        "coupling_stride": coupling_stride,
        "feedback_signal": run_args.get("lcr_feedback"),
        "feedback_signal_source": source,
        "feedback_signal_source_detail": source_detail,
        "feedback_signal_present": feedback_signal_present,
        "feedback_signal_std": feedback_signal_std,
        "feedback_signal_range": feedback_signal_range,
        "feedback_signal_mean": feedback_signal_mean,
        "feedback_signal_min": feedback_signal_min,
        "feedback_signal_max": feedback_signal_max,
        "radius_rms_present": radius_rms_present,
        "radius_rms_std": radius_rms_std,
        "radius_rms_range": radius_rms_range,
        "radius_rms_min": radius_rms_min,
        "radius_rms_max": radius_rms_max,
        "feedback_update_count": feedback_updates,
        "feedback_used_fraction": feedback_used_fraction,
        "feedback_corr": feedback_corr,
        "circuit_update_count": circuit_update_count,
        "circuit_update_fraction": circuit_update_fraction,
        "driver_writeback_count": driver_writeback_count,
        "driver_writeback_fraction": driver_writeback_fraction,
        "driver_writeback_match": driver_writeback_match,
        "driver_amp_std": driver_amp_std,
        "driver_amp_mean": driver_amp_mean,
        "driver_amp_range": driver_amp_range,
        "lcr_steps": len(lcr_history),
        "monitor_records": len(records),
        "metadata_path": str(Path(args.metadata)),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # CSV outputs
    lcr_csv = Path(args.lcr_csv)
    lcr_csv.parent.mkdir(parents=True, exist_ok=True)
    with lcr_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t",
                "I",
                "R_plasma",
                "R_plasma_source",
                "feedback_signal",
                "feedback_used",
            ],
        )
        writer.writeheader()
        for row in lcr_history:
            writer.writerow(
                {
                    "t": row.get("t"),
                    "I": row.get("I"),
                    "R_plasma": row.get("R_plasma"),
                    "R_plasma_source": row.get("R_plasma_source"),
                    "feedback_signal": row.get("feedback_signal"),
                    "feedback_used": row.get("feedback_used"),
                }
            )

    fb_csv = Path(args.feedback_csv)
    fb_csv.parent.mkdir(parents=True, exist_ok=True)
    with fb_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "radius_rms"])
        writer.writeheader()
        for t_val, r_val in zip(radius_times, radius_vals):
            writer.writerow({"time_s": t_val, "radius_rms": r_val})

    # Plots
    if lcr_times.size and lcr_I.size:
        plt.figure(figsize=(6, 4))
        plt.plot(lcr_times, lcr_I, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Current I (A)")
        plt.title("LCR Current vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "lcr_current_vs_time.png")
        plt.close()

    if lcr_times.size and lcr_R.size:
        plt.figure(figsize=(6, 4))
        plt.plot(lcr_times, lcr_R, marker="o", label="R_plasma")
        if radius_times.size and radius_vals.size:
            plt.plot(radius_times, radius_vals, marker="o", label="radius_rms")
        plt.xlabel("Time (s)")
        plt.ylabel("Radius (m)")
        plt.title("Feedback Signal vs R_plasma")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "feedback_vs_rplasma.png")
        plt.close()

    if radius_times.size and radius_vals.size:
        plt.figure(figsize=(6, 4))
        plt.plot(radius_times, radius_vals, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("radius_rms (m)")
        plt.title("radius_rms vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "radius_rms_vs_time.png")
        plt.close()

    if lcr_times.size and driver_amp.size:
        plt.figure(figsize=(6, 4))
        plt.plot(lcr_times, driver_amp, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("B_est (T)")
        plt.title("Driver Amplitude vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "driver_amp_vs_time.png")
        plt.close()

    if radius_times.size and radius_vals.size and lcr_times.size and driver_amp.size:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(
            radius_times, radius_vals, color="tab:blue", marker="o", label="radius_rms"
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("radius_rms (m)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(lcr_times, driver_amp, color="tab:orange", marker="o", label="B_est")
        ax2.set_ylabel("B_est (T)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        plt.title("radius_rms and Driver Amplitude")
        fig.tight_layout()
        plt.savefig(plots_dir / "radius_rms_and_driver_amp_overlay.png")
        plt.close()


if __name__ == "__main__":
    main()
