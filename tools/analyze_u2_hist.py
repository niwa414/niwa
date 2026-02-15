#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha1_file(path: Path) -> str | None:
    if not path.exists():
        return None
    sha1 = hashlib.sha1()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha1.update(chunk)
    return sha1.hexdigest()


def parse_histogram(path: Path):
    header = []
    rows = []
    lengths = []
    has_nan = False
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith('#'):
            header.append(line.rstrip())
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            step = float(parts[0])
            time = float(parts[1])
            counts = [float(val) for val in parts[2:]]
        except Exception:
            continue
        if any(math.isnan(val) for val in counts) or math.isnan(step) or math.isnan(time):
            has_nan = True
        rows.append((step, time, counts))
        lengths.append(len(counts))
    return header, rows, lengths, has_nan


def histogram_quantile(centers, weights, q):
    total = float(sum(weights))
    if total <= 0.0:
        return None
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += float(w)
        if acc / total >= q:
            return float(centers[idx])
    return float(centers[-1]) if centers else None


def extract_centers_from_header(header_lines, expected_len):
    centers = []
    for line in header_lines:
        tokens = line.strip().lstrip('#').split()
        for tok in tokens:
            if 'bin' in tok and '=' in tok:
                try:
                    val = tok.split('=', 1)[1]
                    val = val.split('(')[0]
                    centers.append(float(val))
                except Exception:
                    continue
    if expected_len and len(centers) >= expected_len:
        return centers[:expected_len]
    return centers if centers else None


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze U2 histogram (ParticleHistogram).')
    parser.add_argument('--hist', required=True, help='Path to U2Hist.txt')
    parser.add_argument('--metrics-out', required=True, help='Output metrics JSON')
    parser.add_argument('--warpx-args', default=None, help='Optional warpx_args.json for bin config')
    args = parser.parse_args()

    hist_path = Path(args.hist)
    metrics = {
        'u2hist_path': str(hist_path),
        'u2hist_obs_kind': 'histogram',
    }

    if not hist_path.exists():
        metrics['u2hist_error'] = 'hist_path_missing'
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
        return

    metrics['u2hist_sha1'] = sha1_file(hist_path)
    header, rows, lengths, has_nan = parse_histogram(hist_path)
    metrics['u2hist_header'] = header[:3]
    metrics['u2hist_series_len_raw'] = len(rows)
    metrics['u2hist_has_nan'] = bool(has_nan)

    if not rows:
        metrics['u2hist_error'] = 'hist_no_rows'
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
        return

    length_mode = None
    if lengths:
        length_mode = Counter(lengths).most_common(1)[0][0]
    length_mode = length_mode or len(rows[0][2])
    metrics['u2hist_bin_count_mode'] = int(length_mode)
    metrics['u2hist_bin_count_min'] = int(min(lengths)) if lengths else None
    metrics['u2hist_bin_count_max'] = int(max(lengths)) if lengths else None

    filtered = [row for row in rows if len(row[2]) == length_mode]
    metrics['u2hist_rows_skipped'] = int(len(rows) - len(filtered))
    metrics['u2hist_series_len'] = len(filtered)

    if not filtered:
        metrics['u2hist_error'] = 'hist_all_rows_filtered'
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
        return

    # Load bin configuration from warpx_args.json if provided.
    bin_min = None
    bin_max = None
    bin_number_cfg = None
    species = None
    if args.warpx_args:
        try:
            cfg = json.loads(Path(args.warpx_args).read_text())
        except Exception:
            cfg = {}
        if cfg:
            bin_min = cfg.get('u2_hist_bin_min')
            bin_max = cfg.get('u2_hist_bin_max')
            bin_number_cfg = cfg.get('u2_hist_bin_number')
            species = cfg.get('u2_hist_species')
            metrics['u2hist_species'] = species
            metrics['u2hist_bin_min'] = bin_min
            metrics['u2hist_bin_max'] = bin_max
            metrics['u2hist_bin_number_config'] = bin_number_cfg

    bin_number_used = length_mode
    if bin_number_cfg is not None:
        try:
            bin_number_cfg = int(bin_number_cfg)
            if bin_number_cfg > 0:
                bin_number_used = bin_number_cfg
        except Exception:
            bin_number_cfg = None

    if bin_min is None:
        bin_min = 0.0
    if bin_max is None:
        bin_max = float(bin_number_used)

    if bin_number_used != length_mode:
        bin_number_used = length_mode
    metrics['u2hist_bin_number_used'] = int(bin_number_used)

    centers = extract_centers_from_header(header, length_mode)
    if centers is None or len(centers) != length_mode:
        centers = [
            bin_min + (idx + 0.5) * (bin_max - bin_min) / bin_number_used
            for idx in range(bin_number_used)
        ]

    steps = [row[0] for row in filtered]
    times = [row[1] for row in filtered]
    steps_int = [int(round(step)) for step in steps]
    step_to_index = {}
    for idx, step in enumerate(steps_int):
        # last-write-wins for repeated steps
        step_to_index[step] = idx

    p99_series = []
    p999_series = []
    weight_series = []
    for _, _, counts in filtered:
        p99_series.append(histogram_quantile(centers, counts, 0.99))
        p999_series.append(histogram_quantile(centers, counts, 0.999))
        weight_series.append(float(sum(counts)))

    def last_valid(series):
        for val in reversed(series):
            if val is not None:
                return val
        return None

    def peak_step(series):
        best = None
        best_step = None
        for val, step in zip(series, steps):
            if val is None:
                continue
            if best is None or val > best:
                best = val
                best_step = step
        return best, best_step

    metrics['u2_p99_end'] = last_valid(p99_series)
    metrics['u2_p999_end'] = last_valid(p999_series)
    p99_peak, p99_step = peak_step(p99_series)
    p999_peak, p999_step = peak_step(p999_series)
    metrics['u2_p99_peak'] = p99_peak
    metrics['u2_p99_peak_step'] = p99_step
    metrics['u2_p999_peak'] = p999_peak
    metrics['u2_p999_peak_step'] = p999_step

    metrics['u2hist_first3'] = [[float(steps[i]), float(times[i]), float(sum(filtered[i][2]))] for i in range(min(3, len(filtered)))]
    metrics['u2hist_last3'] = [[float(steps[i]), float(times[i]), float(sum(filtered[i][2]))] for i in range(max(len(filtered)-3, 0), len(filtered))]

    # Additional step-specific diagnostics for validating monotonic decay.
    step_end = max(steps_int) if steps_int else None
    metrics['u2_step_end'] = step_end

    def series_at_step(target_step, series):
        if target_step is None:
            return None, None
        if target_step in step_to_index:
            idx = step_to_index[target_step]
            return series[idx], target_step
        if not steps_int:
            return None, None
        nearest = min(steps_int, key=lambda s: abs(s - target_step))
        idx = step_to_index[nearest]
        return series[idx], nearest

    off_step = None
    if args.warpx_args:
        try:
            cfg = json.loads(Path(args.warpx_args).read_text())
        except Exception:
            cfg = {}
        if cfg:
            off_step = cfg.get('drive_envelope_off_step')
    metrics['u2_step_off_config'] = off_step

    def counts_at_step(target_step):
        if target_step is None:
            return None, None
        if target_step in step_to_index:
            idx = step_to_index[target_step]
            return filtered[idx][2], target_step
        if not steps_int:
            return None, None
        nearest = min(steps_int, key=lambda s: abs(s - target_step))
        idx = step_to_index[nearest]
        return filtered[idx][2], nearest

    p99_step0, step0_used = series_at_step(0, p99_series)
    p99_off, step_off_used = series_at_step(off_step, p99_series)
    p99_end, step_end_used = series_at_step(step_end, p99_series)
    metrics['u2_p99_at_step0'] = p99_step0
    metrics['u2_p99_at_stepOff'] = p99_off
    metrics['u2_p99_at_stepEnd'] = p99_end
    metrics['u2_p99_step0_used'] = step0_used
    metrics['u2_p99_stepOff_used'] = step_off_used
    metrics['u2_p99_stepEnd_used'] = step_end_used
    if p99_off is not None and p99_end is not None:
        try:
            metrics['u2_p99_ratio_off_to_end'] = float(p99_off) / float(p99_end) if float(p99_end) != 0.0 else None
        except Exception:
            metrics['u2_p99_ratio_off_to_end'] = None
    else:
        metrics['u2_p99_ratio_off_to_end'] = None

    w_step0, w_step0_used = series_at_step(0, weight_series)
    w_off, w_off_used = series_at_step(off_step, weight_series)
    w_end, w_end_used = series_at_step(step_end, weight_series)
    metrics['u2_hist_total_weight_at_step0'] = w_step0
    metrics['u2_hist_total_weight_at_stepOff'] = w_off
    metrics['u2_hist_total_weight_at_stepEnd'] = w_end
    metrics['u2_hist_weight_step0_used'] = w_step0_used
    metrics['u2_hist_weight_stepOff_used'] = w_off_used
    metrics['u2_hist_weight_stepEnd_used'] = w_end_used

    # Histogram integrity diagnostics.
    metrics['hist_bins'] = int(bin_number_used)
    metrics['hist_bin_min'] = float(bin_min)
    metrics['hist_bin_max'] = float(bin_max)
    metrics['row_len_mode'] = int(length_mode)

    def nonzero_bins(counts):
        if counts is None:
            return None
        return int(sum(1 for val in counts if float(val) != 0.0))

    def topbin_index(counts):
        if not counts:
            return None
        max_val = None
        max_idx = None
        for idx, val in enumerate(counts):
            val_f = float(val)
            if max_val is None or val_f > max_val:
                max_val = val_f
                max_idx = idx
        return int(max_idx) if max_idx is not None else None

    counts_step0, counts_step0_used = counts_at_step(0)
    counts_stepOff, counts_stepOff_used = counts_at_step(off_step)
    counts_stepEnd, counts_stepEnd_used = counts_at_step(step_end)
    metrics['nonzero_bins_step0'] = nonzero_bins(counts_step0)
    metrics['nonzero_bins_stepOff'] = nonzero_bins(counts_stepOff)
    metrics['nonzero_bins_stepEnd'] = nonzero_bins(counts_stepEnd)
    metrics['topbin_index_step0'] = topbin_index(counts_step0)
    metrics['topbin_index_stepEnd'] = topbin_index(counts_stepEnd)
    metrics['topbin_index_end'] = metrics['topbin_index_stepEnd']
    metrics['u2_hist_counts_step0_used'] = counts_step0_used
    metrics['u2_hist_counts_stepOff_used'] = counts_stepOff_used
    metrics['u2_hist_counts_stepEnd_used'] = counts_stepEnd_used

    # Step-off mean/p50 (more sensitive KPIs than p99).
    def weighted_mean(centers_arr, weights_arr):
        if weights_arr is None:
            return None
        total = float(sum(weights_arr))
        if total <= 0.0:
            return None
        return float(sum(c * w for c, w in zip(centers_arr, weights_arr))) / total

    def weighted_quantile(centers_arr, weights_arr, q):
        if weights_arr is None:
            return None
        return histogram_quantile(centers_arr, weights_arr, q)

    metrics['u2_mean_at_stepOff'] = weighted_mean(centers, counts_stepOff)
    metrics['u2_p50_at_stepOff'] = weighted_quantile(centers, counts_stepOff, 0.5)
    metrics['u2_mean_at_stepEnd'] = weighted_mean(centers, counts_stepEnd)
    metrics['u2_p50_at_stepEnd'] = weighted_quantile(centers, counts_stepEnd, 0.5)

    # Decay timing after drive-off: first step after off where p99 drops 10%.
    decay10_step = None
    decay10_threshold = None
    if off_step is not None and p99_off is not None:
        decay10_threshold = 0.9 * float(p99_off)
        for step, val in zip(steps_int, p99_series):
            if step < int(off_step):
                continue
            if val is None:
                continue
            if float(val) <= decay10_threshold:
                decay10_step = int(step)
                break
    metrics['u2_p99_decay10_threshold'] = decay10_threshold
    metrics['u2_p99_decay10_step_after_off'] = decay10_step

    # Snapshot of histogram rows (first/last bins) for manual inspection.
    snap_path = Path(args.metrics_out).with_name('u2hist_snap.txt')
    def format_snapshot(step_label, counts, used_step):
        if counts is None:
            return f"{step_label} step=NA bins=NA"
        first = [f"{float(val):.6g}" for val in counts[:8]]
        last = [f"{float(val):.6g}" for val in counts[-8:]] if len(counts) >= 8 else [f"{float(val):.6g}" for val in counts]
        return f"{step_label} step={used_step} first8={' '.join(first)} last8={' '.join(last)}"

    snap_lines = [
        format_snapshot('step0', counts_step0, counts_step0_used),
        format_snapshot('stepOff', counts_stepOff, counts_stepOff_used),
        format_snapshot('stepEnd', counts_stepEnd, counts_stepEnd_used),
    ]
    snap_path.write_text("\n".join(snap_lines) + "\n", encoding='utf-8')
    metrics['u2hist_snap_path'] = str(snap_path)

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    main()
