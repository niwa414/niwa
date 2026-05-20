from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .moire_errors import MoReWorkflowError


DEFAULT_MOIRE_PAIR_MARKER_HOST_TYPE = 1
BCC_BASIS = ((0, 0, 0), (1, 1, 1))
BCC_NN1_OFFSETS = (
    (-1, -1, -1),
    (-1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1),
)


@dataclass(frozen=True)
class EventLatticeInfo:
    style: str
    step: float
    box_lo: tuple[float, float, float]
    box_hi: tuple[float, float, float]
    half_steps: tuple[int, int, int]
    cells: tuple[int, int, int]
    basis: tuple[tuple[int, int, int], ...]

def _load_event(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise MoReWorkflowError(f"event.json not found: {path}")
    with path.open() as fp:
        event = json.load(fp)
    required = [
        "pair_type",
        "re_concentration",
        "box_lo",
        "box_hi",
        "initsite",
        "jumpsite",
        "normal_sites",
        "other_pair_sites",
    ]
    missing = [key for key in required if key not in event]
    if missing:
        raise MoReWorkflowError(f"event.json missing keys: {', '.join(missing)}")
    return event


def _unique_sorted(values: list[float], tol: float = 1.0e-8) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
    return out


def _closest_int(value: float, *, tol: float = 1.0e-6, label: str = "value") -> int:
    rounded = int(round(value))
    if abs(value - rounded) > tol:
        raise MoReWorkflowError(f"{label} is not close to an integer: {value}")
    return rounded


def _all_sites_from_event(event: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [event["initsite"], event["jumpsite"], *event["normal_sites"], *event["other_pair_sites"]]


def _infer_event_step(event: Dict[str, Any]) -> float:
    coords_by_axis = [[], [], []]
    for site in _all_sites_from_event(event):
        coords_by_axis[0].append(float(site["x"]))
        coords_by_axis[1].append(float(site["y"]))
        coords_by_axis[2].append(float(site["z"]))

    diffs: list[float] = []
    for axis_values in coords_by_axis:
        unique = _unique_sorted(axis_values)
        for left, right in zip(unique, unique[1:]):
            diff = right - left
            if diff > 1.0e-8:
                diffs.append(diff)
    if not diffs:
        raise MoReWorkflowError("Could not infer the lattice half-step from event.json")
    return min(diffs)


def _coord_to_grid(coord: float, lo: float, step: float, axis_name: str) -> int:
    return _closest_int((coord - lo) / step, label=f"{axis_name}-grid")


def _wrap_grid(grid: tuple[int, int, int], info: EventLatticeInfo) -> tuple[int, int, int]:
    return (
        grid[0] % info.half_steps[0],
        grid[1] % info.half_steps[1],
        grid[2] % info.half_steps[2],
    )


def _site_id_from_grid(grid: tuple[int, int, int], info: EventLatticeInfo) -> int:
    gx, gy, gz = grid
    parity = (gx & 1, gy & 1, gz & 1)
    try:
        basis_index = info.basis.index(parity)
    except ValueError as exc:
        raise MoReWorkflowError(f"grid point {grid} is not a valid {info.style} lattice site") from exc
    cx, cy, cz = gx // 2, gy // 2, gz // 2
    nx, ny, _ = info.cells
    return ((cz * ny + cy) * nx + cx) * len(info.basis) + basis_index + 1


def _site_grid(site: Dict[str, Any], info: EventLatticeInfo) -> tuple[int, int, int]:
    return _wrap_grid(
        (
            _coord_to_grid(float(site["x"]), info.box_lo[0], info.step, "x"),
            _coord_to_grid(float(site["y"]), info.box_lo[1], info.step, "y"),
            _coord_to_grid(float(site["z"]), info.box_lo[2], info.step, "z"),
        ),
        info,
    )


def _infer_event_lattice_info(event: Dict[str, Any]) -> EventLatticeInfo:
    step = _infer_event_step(event)
    box_lo = tuple(float(v) for v in event["box_lo"])
    box_hi = tuple(float(v) for v in event["box_hi"])
    half_steps = tuple(
        _closest_int((hi - lo) / step, label=f"box-{axis}-half-steps")
        for axis, (lo, hi) in zip("xyz", zip(box_lo, box_hi))
    )
    for axis, count in zip("xyz", half_steps):
        if count <= 0 or count % 2 != 0:
            raise MoReWorkflowError(
                f"box size on {axis} does not look like a valid even half-step count: {count}"
            )

    parities = set()
    for site in _all_sites_from_event(event):
        gx = _coord_to_grid(float(site["x"]), box_lo[0], step, "x")
        gy = _coord_to_grid(float(site["y"]), box_lo[1], step, "y")
        gz = _coord_to_grid(float(site["z"]), box_lo[2], step, "z")
        parities.add((gx & 1, gy & 1, gz & 1))

    if not parities.issubset(set(BCC_BASIS)):
        raise MoReWorkflowError(
            f"event.json coordinates do not match the expected bcc basis; parities={sorted(parities)}"
        )

    return EventLatticeInfo(
        style="bcc",
        step=step,
        box_lo=box_lo,
        box_hi=box_hi,
        half_steps=half_steps,
        cells=(half_steps[0] // 2, half_steps[1] // 2, half_steps[2] // 2),
        basis=BCC_BASIS,
    )


def _parse_data_lmp_site_types(data_lmp: Path, info: EventLatticeInfo) -> Dict[tuple[int, int, int], int]:
    if not data_lmp.exists():
        raise MoReWorkflowError(f"data.lmp not found: {data_lmp}")

    lines = data_lmp.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, raw in enumerate(lines) if raw.strip() == "Atoms")
    except StopIteration as exc:
        raise MoReWorkflowError(f"Could not find an 'Atoms' section in {data_lmp}") from exc

    lookup: Dict[tuple[int, int, int], int] = {}
    tol = 1.0e-6
    for raw in lines[start + 1 :]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
            continue
        atom_type = int(parts[1])
        x, y, z = map(float, parts[2:5])
        if not (
            info.box_lo[0] - tol <= x <= info.box_hi[0] + tol
            and info.box_lo[1] - tol <= y <= info.box_hi[1] + tol
            and info.box_lo[2] - tol <= z <= info.box_hi[2] + tol
        ):
            continue
        try:
            grid = _wrap_grid(
                (
                    _coord_to_grid(x, info.box_lo[0], info.step, "x"),
                    _coord_to_grid(y, info.box_lo[1], info.step, "y"),
                    _coord_to_grid(z, info.box_lo[2], info.step, "z"),
                ),
                info,
            )
        except MoReWorkflowError:
            continue
        lookup[grid] = atom_type
    return lookup


def _parse_data_lmp_box(data_lmp: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x_bounds = y_bounds = z_bounds = None
    for raw in data_lmp.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) >= 4 and parts[2:] == ["xlo", "xhi"]:
            x_bounds = (float(parts[0]), float(parts[1]))
        elif len(parts) >= 4 and parts[2:] == ["ylo", "yhi"]:
            y_bounds = (float(parts[0]), float(parts[1]))
        elif len(parts) >= 4 and parts[2:] == ["zlo", "zhi"]:
            z_bounds = (float(parts[0]), float(parts[1]))
    if not x_bounds or not y_bounds or not z_bounds:
        raise MoReWorkflowError(f"Could not parse box bounds from {data_lmp}")
    return (x_bounds[0], y_bounds[0], z_bounds[0]), (x_bounds[1], y_bounds[1], z_bounds[1])


def _infer_dominant_spacing(values: list[float], tol: float = 1.0e-6) -> float:
    unique = _unique_sorted(values, tol=tol)
    diffs = [right - left for left, right in zip(unique, unique[1:]) if right - left > tol]
    if not diffs:
        raise MoReWorkflowError("Could not infer a dominant lattice spacing from data.lmp")

    clusters: list[list[float]] = []
    for diff in diffs:
        placed = False
        for cluster in clusters:
            if abs(diff - cluster[0]) <= 1.0e-3:
                cluster.append(diff)
                placed = True
                break
        if not placed:
            clusters.append([diff])
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))
    dominant = clusters[0]
    return sum(dominant) / len(dominant)


def _infer_lattice_info_from_data_lmp(data_lmp: Path) -> EventLatticeInfo:
    box_lo, box_hi = _parse_data_lmp_box(data_lmp)
    lines = data_lmp.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, raw in enumerate(lines) if raw.strip() == "Atoms")
    except StopIteration as exc:
        raise MoReWorkflowError(f"Could not find an 'Atoms' section in {data_lmp}") from exc

    coords_by_axis = [[], [], []]
    for raw in lines[start + 1 :]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
            continue
        coords_by_axis[0].append(float(parts[2]))
        coords_by_axis[1].append(float(parts[3]))
        coords_by_axis[2].append(float(parts[4]))

    step = min(_infer_dominant_spacing(values) for values in coords_by_axis)
    half_steps = tuple(
        _closest_int((hi - lo) / step, label=f"box-{axis}-half-steps")
        for axis, (lo, hi) in zip("xyz", zip(box_lo, box_hi))
    )
    return EventLatticeInfo(
        style="bcc",
        step=step,
        box_lo=box_lo,
        box_hi=box_hi,
        half_steps=half_steps,
        cells=(half_steps[0] // 2, half_steps[1] // 2, half_steps[2] // 2),
        basis=BCC_BASIS,
    )


def _grid_to_coords(grid: tuple[int, int, int], info: EventLatticeInfo) -> tuple[float, float, float]:
    return (
        info.box_lo[0] + grid[0] * info.step,
        info.box_lo[1] + grid[1] * info.step,
        info.box_lo[2] + grid[2] * info.step,
    )


def _normalize_vector(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5
    if norm <= 0.0:
        raise MoReWorkflowError("Could not normalize the jump direction for the generated event")
    return (dx / norm, dy / norm, dz / norm)


def _choose_seed_vacancy_and_jump(
    occupancy: Dict[tuple[int, int, int], int],
    info: EventLatticeInfo,
    preferred_vacancy_grid: Optional[tuple[int, int, int]] = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    center = (info.half_steps[0] / 2.0, info.half_steps[1] / 2.0, info.half_steps[2] / 2.0)

    def rank(grid: tuple[int, int, int]) -> tuple[float, int, int, int]:
        dx = grid[0] - center[0]
        dy = grid[1] - center[1]
        dz = grid[2] - center[2]
        return (dx * dx + dy * dy + dz * dz, grid[2], grid[1], grid[0])

    if preferred_vacancy_grid is not None:
        candidate_vacancies = [preferred_vacancy_grid]
    else:
        candidate_vacancies = sorted(
            (grid for grid, atom_type in occupancy.items() if atom_type == 1),
            key=rank,
        )
        if not candidate_vacancies:
            candidate_vacancies = sorted(occupancy.keys(), key=rank)

    for vacancy_grid in candidate_vacancies:
        neighbors = []
        for dx, dy, dz in BCC_NN1_OFFSETS:
            neighbor = _wrap_grid((vacancy_grid[0] + dx, vacancy_grid[1] + dy, vacancy_grid[2] + dz), info)
            atom_type = occupancy.get(neighbor)
            if atom_type is not None:
                neighbors.append((atom_type, rank(neighbor), neighbor))
        if not neighbors:
            continue
        neighbors.sort(key=lambda item: (0 if item[0] == 1 else 1, item[1]))
        return vacancy_grid, neighbors[0][2]
    raise MoReWorkflowError("Could not find a valid vacancy/jump pair in data.lmp to seed repo KMC")


def _site_record(site_id: int, atom_type: int, coords: tuple[float, float, float]) -> Dict[str, Any]:
    return {
        "site_id": site_id,
        "atom_type": atom_type,
        "x": round(coords[0], 6),
        "y": round(coords[1], 6),
        "z": round(coords[2], 6),
    }


def _build_seed_event_from_data_lmp(
    *,
    data_lmp: Path,
    pair_type: str = "MoRe",
    pair_displacement: float = 0.3,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    info = _infer_lattice_info_from_data_lmp(data_lmp)
    occupancy = _parse_data_lmp_site_types(data_lmp, info)
    missing_grids = []
    for cz in range(info.cells[2]):
        for cy in range(info.cells[1]):
            for cx in range(info.cells[0]):
                base = (2 * cx, 2 * cy, 2 * cz)
                for parity in info.basis:
                    grid = (base[0] + parity[0], base[1] + parity[1], base[2] + parity[2])
                    if grid not in occupancy:
                        missing_grids.append(grid)

    vacancy_source = "synthetic_seed"
    preferred_vacancy = None
    if len(missing_grids) == 1:
        preferred_vacancy = missing_grids[0]
        vacancy_source = "data_lmp_missing_site"
    vacancy_grid, jump_grid = _choose_seed_vacancy_and_jump(
        occupancy,
        info,
        preferred_vacancy_grid=preferred_vacancy,
    )
    vacancy_coords = _grid_to_coords(vacancy_grid, info)
    jump_coords = _grid_to_coords(jump_grid, info)
    dx = jump_coords[0] - vacancy_coords[0]
    dy = jump_coords[1] - vacancy_coords[1]
    dz = jump_coords[2] - vacancy_coords[2]
    jump_direction = _normalize_vector(dx, dy, dz)

    total_occupied_after_vacancy = len(occupancy) - 1
    re_sites_after_vacancy = sum(1 for grid, atom_type in occupancy.items() if atom_type == 2 and grid != vacancy_grid)
    re_concentration = re_sites_after_vacancy / total_occupied_after_vacancy if total_occupied_after_vacancy else 0.0

    event = {
        "pair_type": pair_type,
        "re_concentration": re_concentration,
        "pair_displacement": pair_displacement,
        "box_lo": [round(v, 6) for v in info.box_lo],
        "box_hi": [round(v, 6) for v in info.box_hi],
        "jump_direction": [round(v, 8) for v in jump_direction],
        "initsite": {
            "site_id": _site_id_from_grid(vacancy_grid, info),
            "x": round(vacancy_coords[0], 6),
            "y": round(vacancy_coords[1], 6),
            "z": round(vacancy_coords[2], 6),
        },
        "jumpsite": _site_record(
            _site_id_from_grid(jump_grid, info),
            occupancy[jump_grid],
            jump_coords,
        ),
        "normal_sites": [],
        "other_pair_sites": [],
        "metadata": {
            "source": "generated_from_data_lmp",
            "data_lmp": str(data_lmp),
            "lattice_style": info.style,
            "lattice_constant": info.step * 2.0,
            "cells": list(info.cells),
            "generated_vacancy_site_id": _site_id_from_grid(vacancy_grid, info),
            "generated_jump_site_id": _site_id_from_grid(jump_grid, info),
            "vacancy_source": vacancy_source,
            "missing_lattice_sites_in_data_lmp": len(missing_grids),
            "note": (
                "This is a mietclaw-generated KMC seed event derived from the local MoRe case. "
                "It is used so the agent can build a repo-compatible initial KMC state without relying on a pre-existing demo event file."
            ),
        },
    }
    return event, {
        "source": "generated_from_data_lmp",
        "data_lmp": str(data_lmp),
        "lattice_style": info.style,
        "lattice_constant": info.step * 2.0,
        "cells": list(info.cells),
        "vacancy_site_id": event["initsite"]["site_id"],
        "jump_site_id": event["jumpsite"]["site_id"],
        "jump_atom_type": event["jumpsite"]["atom_type"],
        "vacancy_source": vacancy_source,
        "missing_lattice_sites_in_data_lmp": len(missing_grids),
        "re_concentration": re_concentration,
    }


def _generate_seed_event_from_data_lmp(
    *,
    data_lmp: Path,
    output_path: Path,
    pair_type: str = "MoRe",
    pair_displacement: float = 0.3,
) -> Dict[str, Any]:
    event, summary = _build_seed_event_from_data_lmp(
        data_lmp=data_lmp,
        pair_type=pair_type,
        pair_displacement=pair_displacement,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "event_json": str(output_path),
        **summary,
    }


def _event_site_signature(site: Dict[str, Any]) -> Dict[str, Any]:
    signature = {
        "site_id": int(site["site_id"]) if site.get("site_id") is not None else None,
        "x": round(float(site["x"]), 6),
        "y": round(float(site["y"]), 6),
        "z": round(float(site["z"]), 6),
    }
    if site.get("atom_type") is not None:
        signature["atom_type"] = int(site["atom_type"])
    return signature


def _format_event_site(site: Dict[str, Any]) -> str:
    atom_type = f", type {site['atom_type']}" if site.get("atom_type") is not None else ""
    return (
        f"site {site.get('site_id', '—')}{atom_type} "
        f"@ ({float(site['x']):.6f}, {float(site['y']):.6f}, {float(site['z']):.6f})"
    )


def _format_event_pair(event: Dict[str, Any]) -> str:
    return f"vacancy {_format_event_site(event['initsite'])}; jump {_format_event_site(event['jumpsite'])}"


def _event_site_matches(expected: Dict[str, Any], candidate: Dict[str, Any], *, tol: float = 1.0e-6) -> bool:
    expected_site_id = expected.get("site_id")
    candidate_site_id = candidate.get("site_id")
    if expected_site_id is not None and candidate_site_id is not None:
        if int(expected_site_id) != int(candidate_site_id):
            return False
    else:
        for axis in ("x", "y", "z"):
            if abs(float(expected[axis]) - float(candidate[axis])) > tol:
                return False
    expected_atom_type = expected.get("atom_type")
    candidate_atom_type = candidate.get("atom_type")
    if expected_atom_type is not None and candidate_atom_type is not None:
        return int(expected_atom_type) == int(candidate_atom_type)
    return True


def _event_pair_mismatch_reasons(expected_event: Dict[str, Any], requested_event: Dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _event_site_matches(expected_event["initsite"], requested_event["initsite"]):
        reasons.append(
            "vacancy site does not match "
            f"(case expects {_format_event_site(expected_event['initsite'])}, "
            f"requested {_format_event_site(requested_event['initsite'])})"
        )
    if not _event_site_matches(expected_event["jumpsite"], requested_event["jumpsite"]):
        reasons.append(
            "jump site does not match "
            f"(case expects {_format_event_site(expected_event['jumpsite'])}, "
            f"requested {_format_event_site(requested_event['jumpsite'])})"
        )
    return reasons


def _derive_static_case_event_binding(source_case_dir: Path) -> Dict[str, Any]:
    data_lmp = source_case_dir / "data.lmp"
    expected_event, seed_summary = _build_seed_event_from_data_lmp(data_lmp=data_lmp)
    return {
        "mode": "static_case_seed_event",
        "source": "data_lmp",
        "data_lmp": str(data_lmp),
        "vacancy_source": seed_summary["vacancy_source"],
        "expected_pair": {
            "vacancy": _event_site_signature(expected_event["initsite"]),
            "jump": _event_site_signature(expected_event["jumpsite"]),
        },
    }


def _validate_requested_event_against_static_case(*, source_case_dir: Path, event_json: Path) -> Dict[str, Any]:
    binding = _derive_static_case_event_binding(source_case_dir)
    requested_event = _load_event(event_json)
    expected_event = {
        "initsite": binding["expected_pair"]["vacancy"],
        "jumpsite": binding["expected_pair"]["jump"],
    }
    mismatches = _event_pair_mismatch_reasons(expected_event, requested_event)
    binding.update(
        {
            "requested_event_json": str(event_json),
            "requested_pair": {
                "vacancy": _event_site_signature(requested_event["initsite"]),
                "jump": _event_site_signature(requested_event["jumpsite"]),
            },
            "matches_requested_event": not mismatches,
        }
    )
    if mismatches:
        raise MoReWorkflowError(
            "This MoRe LAMMPS path currently replays the static NEB case stored in the case directory. "
            "Your event.json does not match that built-in jump pair, so the barrier would otherwise look unchanged. "
            f"Static case pair: {_format_event_pair(expected_event)}. "
            f"Requested pair: {_format_event_pair(requested_event)}. "
            f"Mismatch details: {'; '.join(mismatches)}."
        )
    return binding


def _generate_repo_kmc_state_from_event(
    *,
    event_json: Path,
    output_path: Path,
    data_lmp: Optional[Path] = None,
    pair_marker_host_type: int = DEFAULT_MOIRE_PAIR_MARKER_HOST_TYPE,
) -> Dict[str, Any]:
    event = _load_event(event_json)
    info = _infer_event_lattice_info(event)
    data_lmp_lookup = _parse_data_lmp_site_types(data_lmp, info) if data_lmp else {}

    occupancy: Dict[tuple[int, int, int], int] = {}
    source_labels: Dict[tuple[int, int, int], str] = {}
    source_counts: Dict[str, int] = {"event_json": 0, "data_lmp": 0, "fallback_host_type": 0}
    output_counts: Dict[str, int] = {}

    def insert(site: Dict[str, Any], site_value: int, label: str) -> None:
        grid = _site_grid(site, info)
        expected_site_id = int(site["site_id"])
        derived_site_id = _site_id_from_grid(grid, info)
        if derived_site_id != expected_site_id:
            raise MoReWorkflowError(
                f"site_id mismatch for site {expected_site_id}: derived {derived_site_id} from coordinates"
            )
        occupancy[grid] = site_value
        source_labels[grid] = label
        output_counts[str(site_value)] = output_counts.get(str(site_value), 0) + 1

    insert(event["initsite"], 0, "event_json")
    source_counts["event_json"] += 1
    insert(event["jumpsite"], int(event["jumpsite"]["atom_type"]), "event_json")
    source_counts["event_json"] += 1

    for site in event["normal_sites"]:
        insert(site, int(site["atom_type"]), "event_json")
        source_counts["event_json"] += 1

    pair_sites_from_data_lmp = 0
    defaulted_pair_sites = 0
    for site in event["other_pair_sites"]:
        grid = _site_grid(site, info)
        site_value = data_lmp_lookup.get(grid)
        if site_value is None:
            site_value = pair_marker_host_type
            defaulted_pair_sites += 1
            source_counts["fallback_host_type"] += 1
            label = "fallback_host_type"
        else:
            pair_sites_from_data_lmp += 1
            source_counts["data_lmp"] += 1
            label = "data_lmp"
        insert(site, site_value, label)

    expected_sites = info.cells[0] * info.cells[1] * info.cells[2] * len(info.basis)
    rows: list[str] = []
    missing_sites = 0
    filled_from_data_lmp = 0
    for cz in range(info.cells[2]):
        for cy in range(info.cells[1]):
            for cx in range(info.cells[0]):
                base = (2 * cx, 2 * cy, 2 * cz)
                for parity in info.basis:
                    grid = (base[0] + parity[0], base[1] + parity[1], base[2] + parity[2])
                    site_id = _site_id_from_grid(grid, info)
                    site_value = occupancy.get(grid)
                    if site_value is None and grid in data_lmp_lookup:
                        site_value = data_lmp_lookup[grid]
                        filled_from_data_lmp += 1
                        source_counts["data_lmp"] += 1
                    if site_value is None:
                        missing_sites += 1
                        raise MoReWorkflowError(
                            f"event.json does not fully define the repo KMC state; missing lattice site {site_id}"
                        )
                    rows.append(f"{site_id} {site_value} {site_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        f"# repo KMC read_sites file generated by mietclaw from {event_json.name}",
        f"{len(rows)} sites",
        "id site i2 values",
        "",
        "Values",
        "",
        *rows,
    ]
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")

    return {
        "source": "event_json",
        "event_json": str(event_json),
        "data_lmp": str(data_lmp) if data_lmp else None,
        "state_values_sites": str(output_path),
        "lattice_style": info.style,
        "lattice_constant": info.step * 2.0,
        "cells": list(info.cells),
        "total_sites": len(rows),
        "expected_sites": expected_sites,
        "vacancy_site_id": int(event["initsite"]["site_id"]),
        "jump_site_id": int(event["jumpsite"]["site_id"]),
        "pair_marker_host_type": pair_marker_host_type,
        "pair_sites_from_data_lmp": pair_sites_from_data_lmp,
        "defaulted_pair_sites": defaulted_pair_sites,
        "filled_from_data_lmp": filled_from_data_lmp,
        "missing_sites": missing_sites,
        "converted_pair_markers": len(event["other_pair_sites"]),
        "source_site_counts": source_counts,
        "output_site_counts": output_counts,
        "note": (
            "mietclaw now generates the repo KMC initial state directly from event.json. "
            "Vacancy and species sites come from the event payload; pair-marker sites are mapped back to physical species "
            "using data.lmp when available, otherwise they fall back to the Mo host type."
        ),
    }
