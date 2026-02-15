#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path


def read_signal_csv(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                t = float(r["time"])
                s = float(r["signal"])
            except Exception:
                continue
            if math.isfinite(t) and math.isfinite(s):
                rows.append((t, s))
    return rows


def read_waveform_csv(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("t,"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                y = float(parts[1])
            except Exception:
                continue
            if math.isfinite(t) and math.isfinite(y):
                rows.append((t, y))
    return rows


def interp_linear(xs: list[float], ys: list[float], xq: float) -> float:
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    j = bisect_right(xs, xq) - 1
    x0, x1 = xs[j], xs[j + 1]
    y0, y1 = ys[j], ys[j + 1]
    if x1 == x0:
        return y0
    w = (xq - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def normalize(vals: list[float]) -> list[float]:
    vmin = min(vals)
    vmax = max(vals)
    if vmax == vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def mean(vals: list[float]) -> float:
    return sum(vals) / float(len(vals)) if vals else float("nan")


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0.0:
        return my, 0.0
    b = num / den
    a = my - b * mx
    return a, b


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = mean(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def corrcoef(x: list[float], y: list[float]) -> float:
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare simulation signal against benchmark waveform.")
    ap.add_argument("--signal", required=True, help="Simulation signal csv path (time,signal).")
    ap.add_argument("--target-waveform", required=True, help="Target waveform csv path (t,frac).")
    ap.add_argument("--summary", required=True, help="Output summary json path.")
    ap.add_argument("--plot", required=True, help="Output alignment plot path.")
    ap.add_argument("--min-points", type=int, default=40)
    args = ap.parse_args()

    signal_rows = read_signal_csv(Path(args.signal))
    target_rows = read_waveform_csv(Path(args.target_waveform))
    ok = bool(signal_rows and target_rows)
    summary = {
        "signal_path": str(Path(args.signal).resolve()),
        "target_waveform_path": str(Path(args.target_waveform).resolve()),
        "alignment_ok": False,
        "alignment_points": 0,
        "alignment_r2": None,
        "alignment_corrcoef": None,
        "fit_offset": None,
        "fit_scale": None,
        "plot_path": str(Path(args.plot).resolve()),
        "plot_exists": False,
    }
    if not ok:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return

    sx = [t for t, _ in signal_rows]
    sy = [v for _, v in signal_rows]
    tx = [t for t, _ in target_rows]
    ty = [v for _, v in target_rows]
    t0 = max(min(sx), min(tx))
    t1 = min(max(sx), max(tx))
    if t1 <= t0:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return

    n = max(args.min_points, min(400, len(sx), len(tx)))
    grid = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    s_interp = [interp_linear(sx, sy, t) for t in grid]
    t_interp = [interp_linear(tx, ty, t) for t in grid]
    s_norm = normalize(s_interp)
    t_norm = normalize(t_interp)

    a, b = linear_fit(t_norm, s_norm)
    pred = [a + b * x for x in t_norm]
    r2 = r2_score(s_norm, pred)
    cc = corrcoef(s_norm, t_norm)
    summary.update(
        {
            "alignment_ok": True,
            "alignment_points": n,
            "alignment_r2": r2,
            "alignment_corrcoef": cc,
            "fit_offset": a,
            "fit_scale": b,
        }
    )

    import matplotlib.pyplot as plt  # lazy import

    plot_path = Path(args.plot).resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(grid, t_norm, label="benchmark(norm)", lw=2.0)
    ax.plot(grid, s_norm, label="simulation(norm)", lw=2.0)
    ax.plot(grid, pred, label="fit(sim~a+b*bench)", lw=1.5, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized signal")
    ax.set_title(f"Signal Alignment: R2={r2:.3f}, corr={cc:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    summary["plot_exists"] = plot_path.exists()

    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
