#!/usr/bin/env python3
"""
Lightweight scene loader for YAML/JSON specs under scenes/.
Intended to reduce flag duplication across exporters and the WarpX driver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_scene(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    text = p.read_text(encoding="utf-8")
    # Prefer YAML if available; fall back to JSON; raise clear error otherwise.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        try:
            data = json.loads(text)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse scene file {p}. Install PyYAML or provide JSON. Error: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Scene root must be a mapping, got {type(data)}")
    data["__path__"] = str(p)
    return data


def scene_geometry(scene: Dict[str, Any]) -> Dict[str, float]:
    geom = scene.get("geometry", {}) if isinstance(scene.get("geometry", {}), dict) else {}
    r_max = float(geom.get("r_max", 0.1))
    z_span = float(geom.get("z_span", 0.2))
    return {
        "r_min": 0.0,
        "r_max": r_max,
        "z_min": -0.5 * z_span,
        "z_max": 0.5 * z_span,
    }


def scene_export_mesh(scene: Dict[str, Any]) -> Dict[str, int | float]:
    exp = scene.get("export", {}) if isinstance(scene.get("export", {}), dict) else {}
    geo = scene_geometry(scene)
    return {
        "nr": int(exp.get("nr", scene.get("mesh", {}).get("warpx", {}).get("nr", 64))),
        "nz": int(exp.get("nz", scene.get("mesh", {}).get("warpx", {}).get("nz", 128))),
        "r_min": float(exp.get("r_min", geo["r_min"])),
        "r_max": float(exp.get("r_max", geo["r_max"])),
        "z_min": float(exp.get("z_min", geo["z_min"])),
        "z_max": float(exp.get("z_max", geo["z_max"])),
    }


def scene_warpx_mesh(scene: Dict[str, Any]) -> Dict[str, int | float]:
    wp = scene.get("mesh", {}).get("warpx", {}) if isinstance(scene.get("mesh", {}), dict) else {}
    geo = scene_geometry(scene)
    return {
        "nr": int(wp.get("nr", 64)),
        "nz": int(wp.get("nz", 128)),
        "r_max": float(wp.get("r_max", geo["r_max"])),
        "z_max": float(wp.get("z_max", geo["z_max"])),
        "dt": float(wp.get("dt", 1.0e-11)),
        "diag_period": int(wp.get("diag_period", 10)),
        "n_azimuthal_modes": int(wp.get("n_azimuthal_modes", 1)),
    }


__all__ = ["load_scene", "scene_geometry", "scene_export_mesh", "scene_warpx_mesh"]
