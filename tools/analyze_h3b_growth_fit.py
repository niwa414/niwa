#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_series(path: Path):
    times = []
    amps = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t = row.get("time_s")
            a = row.get("amp_xy")
            if a is None:
                a = row.get("amp_smooth") or row.get("amp_raw") or row.get("amp")
            if t is None or a is None:
                continue
            try:
                times.append(float(t))
                amps.append(float(a))
            except ValueError:
                continue
    return np.array(times), np.array(amps)


def smooth_series(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if mode == "median":
        half = window // 2
        out = np.empty_like(values)
        for i in range(values.size):
            lo = max(0, i - half)
            hi = min(values.size, i + half + 1)
            out[i] = np.median(values[lo:hi])
        return out
    kernel = np.ones(window)
    num = np.convolve(values, kernel, mode="same")
    den = np.convolve(np.ones_like(values), kernel, mode="same")
    return num / den


def linear_fit(t: np.ndarray, y: np.ndarray):
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    dt = t - t_mean
    dy = y - y_mean
    var = float(np.sum(dt * dt))
    if var <= 0.0:
        return None
    slope = float(np.sum(dt * dy) / var)
    intercept = float(y_mean - slope * t_mean)
    y_pred = slope * t + intercept
    resid = y - y_pred
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    resid_std = float(np.sqrt(ss_res / max(1, y.size)))
    return slope, intercept, r2, resid_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit post-merge tilt growth rate.")
    parser.add_argument("--series", required=True, help="CSV with time_s, amp_xy.")
    parser.add_argument("--source-metrics", default=None, help="H3a metrics JSON (optional).")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--fit-csv", required=True, help="Output CSV with raw/smooth series.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Smoothing window size.")
    parser.add_argument("--smooth-mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--floor-percentile", type=float, default=10.0)
    parser.add_argument("--abs-eps", type=float, default=1e-12)
    parser.add_argument("--min-amp-ratio", type=float, default=1.05)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument(
        "--cross-ratio",
        type=float,
        default=2.0,
        help="Amplitude ratio threshold for tilt cross time.",
    )
    args = parser.parse_args()

    series_path = Path(args.series)
    times, amps = load_series(series_path)
    if times.size == 0:
        fallback = series_path.with_name("tilt_fit_series.csv")
        if fallback.exists():
            times, amps = load_series(fallback)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if times.size == 0:
        raise SystemExit("No samples found in tilt series.")

    series_sort_applied = False
    series_drop_nonfinite = 0
    if times.size:
        finite_mask = np.isfinite(times) & np.isfinite(amps)
        if not np.all(finite_mask):
            series_drop_nonfinite = int(times.size - np.sum(finite_mask))
            times = times[finite_mask]
            amps = amps[finite_mask]
    if times.size > 1:
        order = np.argsort(times)
        if not np.all(order == np.arange(times.size)):
            series_sort_applied = True
            times = times[order]
            amps = amps[order]

    amp_raw = amps
    amp_smooth = smooth_series(amp_raw, args.smooth_window, args.smooth_mode)
    tilt_amp_series_len = int(times.size)
    tilt_amp_initial = float(amp_smooth[0]) if amp_smooth.size else None
    tilt_amp_final = float(amp_smooth[-1]) if amp_smooth.size else None
    tilt_amp_ratio = None
    if tilt_amp_initial is not None and tilt_amp_final is not None:
        denom = max(abs(tilt_amp_initial), float(args.abs_eps))
        tilt_amp_ratio = float(tilt_amp_final / denom)
    tilt_cross_ratio_used = float(args.cross_ratio)
    tilt_cross_exists = False
    tilt_cross_time = None
    tilt_cross_index = None
    tilt_cross_amp0 = tilt_amp_initial
    tilt_cross_thresh = None
    if tilt_amp_initial is not None and np.isfinite(tilt_amp_initial):
        tilt_cross_thresh = float(tilt_amp_initial * tilt_cross_ratio_used)
        for idx, val in enumerate(amp_smooth):
            if val >= tilt_cross_thresh:
                tilt_cross_exists = True
                tilt_cross_index = int(idx)
                tilt_cross_time = float(times[idx])
                break
    floor_val = max(float(np.percentile(amp_smooth, args.floor_percentile)), float(args.abs_eps))
    valid_mask = amp_smooth > floor_val
    t_valid = times[valid_mask]
    a_valid = amp_smooth[valid_mask]

    fit_found = False
    gamma_best = None
    r2_best = None
    residual_std_best = None
    fit_points = 0
    fit_start = None
    fit_end = None
    amp_ratio_fit_best = None
    best_indices = None
    r2_eps = 1.0e-6

    if t_valid.size >= args.min_points:
        lnA = np.log(a_valid)
        n = t_valid.size
        for i in range(0, n - args.min_points + 1):
            for j in range(i + args.min_points - 1, n):
                t_slice = t_valid[i : j + 1]
                y_slice = lnA[i : j + 1]
                fit = linear_fit(t_slice, y_slice)
                if fit is None:
                    continue
                slope, intercept, r2, resid_std = fit
                amp_ratio = float(a_valid[j] / a_valid[i])
                if amp_ratio < args.min_amp_ratio:
                    continue
                choose = False
                window_len = int(j - i + 1)
                if (r2_best is None) or (r2 > r2_best + r2_eps):
                    choose = True
                elif r2_best is not None and abs(r2 - r2_best) <= r2_eps:
                    if window_len > fit_points:
                        choose = True
                    elif window_len == fit_points:
                        if residual_std_best is None or (
                            resid_std is not None and resid_std < residual_std_best
                        ):
                            choose = True
                if choose:
                    r2_best = r2
                    gamma_best = slope
                    residual_std_best = resid_std
                    fit_points = window_len
                    fit_start = float(t_slice[0])
                    fit_end = float(t_slice[-1])
                    amp_ratio_fit_best = amp_ratio
                    best_indices = (i, j, intercept)
        fit_found = gamma_best is not None

    no_nan = True
    for val in (gamma_best, r2_best, amp_ratio_fit_best):
        if val is None or not np.isfinite(val):
            no_nan = False
    if fit_points <= 0:
        no_nan = False

    # Save fit series CSV
    fit_csv = Path(args.fit_csv)
    fit_csv.parent.mkdir(parents=True, exist_ok=True)
    with fit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_s", "amp_raw", "amp_smooth", "ln_amp_smooth", "used_for_fit"],
        )
        writer.writeheader()
        used_idx = set()
        if best_indices is not None:
            i, j, _ = best_indices
            used_idx = set(range(i, j + 1))
        valid_indices = np.where(valid_mask)[0]
        valid_pos = {int(idx): int(pos) for pos, idx in enumerate(valid_indices)}
        for idx, (t, raw, smooth) in enumerate(zip(times, amp_raw, amp_smooth)):
            ln_val = np.log(smooth) if smooth > 0.0 else None
            used = 0
            if idx in valid_pos:
                pos = valid_pos[idx]
                if pos in used_idx:
                    used = 1
            writer.writerow(
                {
                    "time_s": t,
                    "amp_raw": raw,
                    "amp_smooth": smooth,
                    "ln_amp_smooth": ln_val,
                    "used_for_fit": used,
                }
            )

    # Plot: amplitude
    plt.figure(figsize=(6, 4))
    plt.plot(times, amp_raw, label="raw", alpha=0.5)
    plt.plot(times, amp_smooth, label="smooth")
    plt.axhline(floor_val, color="gray", linestyle="--", label="floor")
    if fit_found:
        plt.axvspan(fit_start, fit_end, color="orange", alpha=0.2, label="fit window")
    plt.xlabel("Time (s)")
    tilt_label = "Centroid amplitude (m)"
    tilt_title = "Post-merge Tilt Amplitude"
    if args.source_metrics:
        try:
            _src = json.loads(Path(args.source_metrics).read_text(encoding="utf-8"))
            if _src.get("tilt_metric_used") == "slope":
                tilt_label = "Tilt slope (m/m)"
                tilt_title = "Post-merge Tilt Slope"
        except Exception:
            pass
    plt.ylabel(tilt_label)
    plt.title(tilt_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "tilt_amp_post_merge.png")
    plt.close()

    # Plot: log fit
    plt.figure(figsize=(6, 4))
    if t_valid.size:
        plt.plot(t_valid, np.log(a_valid), marker="o", linestyle="none", label="ln(A_smooth)")
    if fit_found and best_indices is not None:
        i, j, intercept = best_indices
        t_fit = t_valid[i : j + 1]
        y_fit = gamma_best * t_fit + intercept
        plt.plot(t_fit, y_fit, color="red", label=f"fit gamma={gamma_best:.3e}, R2={r2_best:.3f}")
    plt.xlabel("Time (s)")
    if tilt_label.startswith("Tilt slope"):
        plt.ylabel("ln(slope)")
        plt.title("Post-merge Tilt Log Fit (slope)")
    else:
        plt.ylabel("ln(A)")
        plt.title("Post-merge Tilt Log Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "tilt_log_fit.png")
    plt.close()

    # Metrics output
    source_metrics = None
    tilt_metric_used = None
    if args.source_metrics:
        try:
            source_metrics = json.loads(Path(args.source_metrics).read_text(encoding="utf-8"))
            tilt_metric_used = source_metrics.get("tilt_metric_used")
        except Exception:
            source_metrics = None

    fit_window_start_idx = None
    fit_window_end_idx = None
    fit_window_len = None
    if best_indices is not None:
        i, j, _ = best_indices
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size:
            fit_window_start_idx = int(valid_indices[i])
            fit_window_end_idx = int(valid_indices[j])
            fit_window_len = int(j - i + 1)

    metrics = {
        "fit_found": fit_found,
        "fit_points": fit_points,
        "gamma_best": gamma_best,
        "r2_best": r2_best,
        "gamma_fit_best": gamma_best,
        "r2_fit_best": r2_best,
        "residual_std_best": residual_std_best,
        "amp_ratio_fit_best": amp_ratio_fit_best,
        "fit_start_time": fit_start,
        "fit_end_time": fit_end,
        "fit_window_time0": fit_start,
        "fit_window_time1": fit_end,
        "fit_window_start_time": fit_start,
        "fit_window_end_time": fit_end,
        "fit_window_start_idx": fit_window_start_idx,
        "fit_window_end_idx": fit_window_end_idx,
        "fit_window_len": fit_window_len,
        "smooth_window": args.smooth_window,
        "smooth_mode": args.smooth_mode,
        "floor_percentile": args.floor_percentile,
        "abs_eps": args.abs_eps,
        "min_amp_ratio": args.min_amp_ratio,
        "min_points": args.min_points,
        "amp_floor": floor_val,
        "series_samples": int(times.size),
        "series_valid_samples": int(t_valid.size),
        "tilt_amp_series_len": tilt_amp_series_len,
        "tilt_amp_initial": tilt_amp_initial,
        "tilt_amp_final": tilt_amp_final,
        "tilt_amp_ratio": tilt_amp_ratio,
        "tilt_cross_ratio_used": tilt_cross_ratio_used,
        "tilt_cross_exists": tilt_cross_exists,
        "tilt_cross_time": tilt_cross_time,
        "tilt_cross_index": tilt_cross_index,
        "tilt_cross_amp0": tilt_cross_amp0,
        "tilt_cross_thresh": tilt_cross_thresh,
        "series_path": str(series_path),
        "source_metrics_path": args.source_metrics,
        "no_nan_in_metrics": no_nan,
        "tilt_metric_used": tilt_metric_used,
        "series_sort_applied": series_sort_applied,
        "series_drop_nonfinite": series_drop_nonfinite,
    }
    if source_metrics is not None:
        metrics["source_metrics"] = {
            "merge_time": source_metrics.get("merge_time"),
            "merge_time_frac": source_metrics.get("merge_time_frac"),
            "tilt_post_merge_samples": source_metrics.get("tilt_post_merge_samples"),
        }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
