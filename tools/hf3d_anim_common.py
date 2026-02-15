#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import imageio.v2 as iio
import numpy as np
import yt


UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
}

yt.funcs.mylog.setLevel(40)


def parse_diag_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def list_plotfiles(diag_dir: Path) -> list[Path]:
    if not diag_dir.exists():
        return []
    items = [p for p in diag_dir.iterdir() if p.is_dir() and p.name.startswith("diag")]
    items.sort(key=parse_diag_index)
    return items


def select_even(paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
    return [paths[i] for i in idx]


def _axis_centers(left: float, right: float, n: int) -> np.ndarray:
    step = (right - left) / float(n)
    return np.linspace(left + 0.5 * step, right - 0.5 * step, n, dtype=np.float64)


def load_plotfile(path: Path, fields: Iterable[str]) -> dict:
    ds = yt.load(str(path), units_override=UNITS_OVERRIDE)
    dims = ds.domain_dimensions.astype(int)
    left = ds.domain_left_edge.to_ndarray().astype(np.float64)
    right = ds.domain_right_edge.to_ndarray().astype(np.float64)
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    out = {
        "path": str(path),
        "time_s": float(ds.current_time.to_value()),
        "dims": (int(dims[0]), int(dims[1]), int(dims[2])),
        "left": left,
        "right": right,
        "dx": float((right[0] - left[0]) / dims[0]),
        "dy": float((right[1] - left[1]) / dims[1]),
        "dz": float((right[2] - left[2]) / dims[2]),
    }
    out["x"] = _axis_centers(left[0], right[0], int(dims[0]))
    out["y"] = _axis_centers(left[1], right[1], int(dims[1]))
    out["z"] = _axis_centers(left[2], right[2], int(dims[2]))
    for name in fields:
        out[name] = np.asarray(grid[("boxlib", name)], dtype=np.float64)
    return out


def sample_points(mask: np.ndarray, budget: int, seed: int) -> np.ndarray:
    flat = np.flatnonzero(mask.ravel())
    if flat.size == 0:
        return np.array([], dtype=np.int64)
    if budget <= 0 or flat.size <= budget:
        return flat
    rng = np.random.default_rng(seed)
    return rng.choice(flat, size=budget, replace=False)


def flat_to_xyz(indices: np.ndarray, shape: tuple[int, int, int], x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    ix, iy, iz = np.unravel_index(indices, shape)
    return x[ix], y[iy], z[iz]


def weighted_centroid(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    wsum = float(np.sum(w))
    if not math.isfinite(wsum) or wsum <= 0.0:
        return 0.0, 0.0, 0.0
    cx = float(np.sum(x * w) / wsum)
    cy = float(np.sum(y * w) / wsum)
    cz = float(np.sum(z * w) / wsum)
    return cx, cy, cz


def align_series(frame_t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    if frame_t.size == 0:
        return np.array([], dtype=np.float64)
    if src_t.size == 0 or src_v.size == 0:
        return np.zeros_like(frame_t, dtype=np.float64)
    order = np.argsort(src_t)
    t = src_t[order]
    v = src_v[order]
    return np.interp(frame_t, t, v)


class FrameWriters:
    def __init__(self, mp4_path: Path, gif_path: Path, fps: int) -> None:
        self.mp4_path = mp4_path
        self.gif_path = gif_path
        self.fps = max(1, int(fps))
        self._mp4 = None
        self._gif = None
        self.mp4_error = ""
        self.mp4_opened = False
        self.gif_opened = False

    def open(self) -> None:
        for path in (self.gif_path, self.mp4_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                pass
        self.gif_path.parent.mkdir(parents=True, exist_ok=True)
        self._gif = iio.get_writer(str(self.gif_path), mode="I", duration=1.0 / self.fps, loop=0)
        self.gif_opened = True
        try:
            self.mp4_path.parent.mkdir(parents=True, exist_ok=True)
            # Prefer FFMPEG explicitly to avoid backend ambiguity for .mp4 targets.
            self._mp4 = iio.get_writer(
                str(self.mp4_path),
                format="FFMPEG",
                mode="I",
                fps=self.fps,
                codec="libx264",
                quality=8,
            )
            self.mp4_opened = True
        except Exception as exc:  # noqa: BLE001
            self.mp4_error = str(exc)
            self._mp4 = None
            self.mp4_opened = False

    def append(self, frame_rgb: np.ndarray) -> None:
        if self._gif is not None:
            self._gif.append_data(frame_rgb)
        if self._mp4 is not None:
            try:
                self._mp4.append_data(frame_rgb)
            except Exception as exc:  # noqa: BLE001
                # MP4 is optional for gate success; continue with GIF/metrics.
                if self.mp4_error:
                    self.mp4_error = f"{self.mp4_error} | append: {exc}"
                else:
                    self.mp4_error = f"append: {exc}"
                try:
                    self._mp4.close()
                except Exception:  # noqa: BLE001
                    pass
                self._mp4 = None
                self.mp4_opened = False

    def close(self) -> None:
        if self._gif is not None:
            self._gif.close()
            self._gif = None
        if self._mp4 is not None:
            self._mp4.close()
            self._mp4 = None


def normalize01(arr: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    if arr.size == 0:
        return arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or abs(vmax - vmin) < eps:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)
