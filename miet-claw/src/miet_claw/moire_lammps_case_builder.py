from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .moire_errors import MoReWorkflowError
from .moire_event_model import (
    _event_site_signature,
    _grid_to_coords,
    _infer_event_lattice_info,
    _infer_lattice_info_from_data_lmp,
    _load_event,
    _normalize_vector,
    _parse_data_lmp_site_types,
    _site_grid,
    _site_id_from_grid,
)
from .moire_visualization import _read_lammps_atom_records


PAIR_TYPE_TO_GAP_TYPES = {
    "MoMo": (1, 1),
    "MoRe": (1, 2),
    "ReMo": (2, 1),
    "ReRe": (2, 2),
}


def _wrap_position(
    position: tuple[float, float, float],
    *,
    box_lo: tuple[float, float, float],
    box_hi: tuple[float, float, float],
) -> tuple[float, float, float]:
    wrapped = []
    for value, lo, hi in zip(position, box_lo, box_hi):
        box = hi - lo
        if box <= 0:
            wrapped.append(value)
            continue
        coord = float(value)
        while coord < lo:
            coord += box
        while coord >= hi:
            coord -= box
        wrapped.append(coord)
    return (wrapped[0], wrapped[1], wrapped[2])


def _shortest_image_direction(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    box_lo: tuple[float, float, float],
    box_hi: tuple[float, float, float],
) -> tuple[float, float, float]:
    delta = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
    for index, (lo, hi) in enumerate(zip(box_lo, box_hi)):
        box = hi - lo
        if box <= 0:
            continue
        if delta[index] > 0.5 * box:
            delta[index] -= box
        elif delta[index] < -0.5 * box:
            delta[index] += box
    norm = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    if norm <= 1.0e-12:
        diagonal = 1.0 / math.sqrt(3.0)
        return (diagonal, diagonal, diagonal)
    return _normalize_vector(delta[0], delta[1], delta[2])


def _resolve_event_jump_direction(event: Dict[str, Any]) -> tuple[float, float, float]:
    direction = event.get("jump_direction")
    if direction is not None and len(direction) == 3:
        dx, dy, dz = (float(direction[0]), float(direction[1]), float(direction[2]))
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm > 1.0e-12:
            return (dx / norm, dy / norm, dz / norm)
    init_center = (
        float(event["initsite"]["x"]),
        float(event["initsite"]["y"]),
        float(event["initsite"]["z"]),
    )
    jump_center = (
        float(event["jumpsite"]["x"]),
        float(event["jumpsite"]["y"]),
        float(event["jumpsite"]["z"]),
    )
    return _shortest_image_direction(
        init_center,
        jump_center,
        box_lo=tuple(float(v) for v in event["box_lo"]),
        box_hi=tuple(float(v) for v in event["box_hi"]),
    )


def _resolve_gap_types(event: Dict[str, Any]) -> tuple[int, int]:
    raw_gap_types = event.get("gap_types")
    if raw_gap_types is not None:
        if len(raw_gap_types) != 2:
            raise MoReWorkflowError("event.json gap_types must contain exactly two atom types")
        return (int(raw_gap_types[0]), int(raw_gap_types[1]))
    pair_type = str(event.get("pair_type", "MoRe"))
    gap_types = PAIR_TYPE_TO_GAP_TYPES.get(pair_type)
    if gap_types is None:
        raise MoReWorkflowError(f"Unsupported pair_type for LAMMPS event modeling: {pair_type}")
    return gap_types


def _make_pair_atoms(
    *,
    center: tuple[float, float, float],
    direction: tuple[float, float, float],
    displacement: float,
    atom_ids: tuple[int, int],
    atom_types: tuple[int, int],
    box_lo: tuple[float, float, float],
    box_hi: tuple[float, float, float],
) -> list[tuple[int, int, tuple[float, float, float]]]:
    dx = direction[0] * displacement
    dy = direction[1] * displacement
    dz = direction[2] * displacement
    atom_a = _wrap_position((center[0] + dx, center[1] + dy, center[2] + dz), box_lo=box_lo, box_hi=box_hi)
    atom_b = _wrap_position((center[0] - dx, center[1] - dy, center[2] - dz), box_lo=box_lo, box_hi=box_hi)
    return [
        (int(atom_ids[0]), int(atom_types[0]), atom_a),
        (int(atom_ids[1]), int(atom_types[1]), atom_b),
    ]


def _parse_lammps_masses(data_lmp: Path) -> Dict[int, float]:
    lines = data_lmp.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, raw in enumerate(lines) if raw.strip() == "Masses")
    except StopIteration:
        return {}
    masses: Dict[int, float] = {}
    for raw in lines[start + 1 :]:
        line = raw.strip()
        if not line:
            if masses:
                break
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            if masses:
                break
            continue
        try:
            masses[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return masses


def _write_lammps_data_file(
    *,
    path: Path,
    atoms: Sequence[tuple[int, int, tuple[float, float, float]]],
    box_lo: tuple[float, float, float],
    box_hi: tuple[float, float, float],
    source_masses: Optional[Dict[int, float]] = None,
) -> str:
    atoms_sorted = sorted(atoms, key=lambda item: item[0])
    if not atoms_sorted:
        raise MoReWorkflowError("Cannot write an empty LAMMPS data file")
    max_type = max(int(atom_type) for _, atom_type, _ in atoms_sorted)
    masses = {int(atom_type): float(source_masses.get(atom_type, 1.0)) for atom_type in range(1, max_type + 1)} if source_masses else {atom_type: 1.0 for atom_type in range(1, max_type + 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("LAMMPS data file generated by mietclaw.\n\n")
        fp.write(f"{len(atoms_sorted)} atoms\n")
        fp.write(f"{max_type} atom types\n\n")
        fp.write(f"{box_lo[0]:.6f} {box_hi[0]:.6f} xlo xhi\n")
        fp.write(f"{box_lo[1]:.6f} {box_hi[1]:.6f} ylo yhi\n")
        fp.write(f"{box_lo[2]:.6f} {box_hi[2]:.6f} zlo zhi\n\n")
        fp.write("Masses\n\n")
        for atom_type in range(1, max_type + 1):
            fp.write(f"{atom_type} {masses[atom_type]:.6f}\n")
        fp.write("\nAtoms\n\n")
        for atom_id, atom_type, position in atoms_sorted:
            fp.write(
                f"{int(atom_id)} {int(atom_type)} "
                f"{float(position[0]):.6f} {float(position[1]):.6f} {float(position[2]):.6f}\n"
            )
    return str(path)


def _write_final_mosia(
    *,
    path: Path,
    atoms: Sequence[tuple[int, int, tuple[float, float, float]]],
) -> str:
    atoms_sorted = sorted(atoms, key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms_sorted))]
    for atom_id, _, position in atoms_sorted:
        lines.append(f"{int(atom_id)} {float(position[0]):.6f} {float(position[1]):.6f} {float(position[2]):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _derive_requested_event_binding(event_json: Path) -> Dict[str, Any]:
    requested_event = _load_event(event_json)
    requested_pair = {
        "vacancy": _event_site_signature(requested_event["initsite"]),
        "jump": _event_site_signature(requested_event["jumpsite"]),
    }
    return {
        "mode": "event_json_requested_model",
        "source": "event_json",
        "event_json": str(event_json),
        "expected_pair": requested_pair,
        "requested_pair": requested_pair,
        "matches_requested_event": True,
    }


def _build_dynamic_lammps_case_from_event(
    *,
    event_json: Path,
    source_data_lmp: Path,
    copied_case_dir: Path,
) -> Dict[str, Any]:
    copied_case_dir.mkdir(parents=True, exist_ok=True)
    event = _load_event(event_json)
    info = _infer_event_lattice_info(event)
    source_info = _infer_lattice_info_from_data_lmp(source_data_lmp)
    if info.cells != source_info.cells:
        raise MoReWorkflowError(
            f"event.json lattice cells {list(info.cells)} do not match case data.lmp cells {list(source_info.cells)}"
        )
    for axis, event_bounds, source_bounds in zip(
        "xyz",
        zip(info.box_lo, info.box_hi),
        zip(source_info.box_lo, source_info.box_hi),
    ):
        if abs(event_bounds[0] - source_bounds[0]) > 1.0e-6 or abs(event_bounds[1] - source_bounds[1]) > 1.0e-6:
            raise MoReWorkflowError(
                f"event.json box on {axis} does not match case data.lmp: {event_bounds} vs {source_bounds}"
            )

    source_lookup = _parse_data_lmp_site_types(source_data_lmp, info)
    source_masses = _parse_lammps_masses(source_data_lmp)
    source_atom_records = _read_lammps_atom_records(source_data_lmp)
    source_max_atom_id = max((int(record["atom_id"]) for record in source_atom_records), default=0)
    pair_displacement = float(event.get("pair_displacement", 0.3))
    gap_types = _resolve_gap_types(event)
    direction = _resolve_event_jump_direction(event)
    box_lo = tuple(float(v) for v in event["box_lo"])
    box_hi = tuple(float(v) for v in event["box_hi"])
    vacancy_site_id = int(event["initsite"]["site_id"])
    jump_site_id = int(event["jumpsite"]["site_id"])
    jump_atom_type = int(event["jumpsite"]["atom_type"])
    init_center = (
        float(event["initsite"]["x"]),
        float(event["initsite"]["y"]),
        float(event["initsite"]["z"]),
    )
    jump_center = (
        float(event["jumpsite"]["x"]),
        float(event["jumpsite"]["y"]),
        float(event["jumpsite"]["z"]),
    )
    other_pair_centers = {int(site["site_id"]) for site in event["other_pair_sites"]}
    lattice_site_ids = [vacancy_site_id, jump_site_id, *other_pair_centers, *[int(site["site_id"]) for site in event["normal_sites"]]]
    max_reserved_atom_id = max([source_max_atom_id, *lattice_site_ids], default=source_max_atom_id)
    current_pair_atom_ids = (max_reserved_atom_id + 1, max_reserved_atom_id + 2)
    next_extra_atom_id = max_reserved_atom_id + 3

    lattice_atoms: Dict[int, tuple[int, tuple[float, float, float]]] = {}
    source_counts = {"event_json": 0, "data_lmp": 0}

    def insert_lattice_site(site: Dict[str, Any], atom_type: int) -> None:
        site_id = int(site["site_id"])
        grid = _site_grid(site, info)
        derived_site_id = _site_id_from_grid(grid, info)
        if derived_site_id != site_id:
            raise MoReWorkflowError(
                f"site_id mismatch for LAMMPS event modeling at site {site_id}: derived {derived_site_id}"
            )
        lattice_atoms[site_id] = (
            int(atom_type),
            (float(site["x"]), float(site["y"]), float(site["z"])),
        )

    insert_lattice_site(event["jumpsite"], jump_atom_type)
    source_counts["event_json"] += 1
    for site in event["normal_sites"]:
        site_id = int(site["site_id"])
        if site_id == vacancy_site_id or site_id in other_pair_centers:
            continue
        insert_lattice_site(site, int(site["atom_type"]))
        source_counts["event_json"] += 1

    filled_from_data_lmp = 0
    for cz in range(info.cells[2]):
        for cy in range(info.cells[1]):
            for cx in range(info.cells[0]):
                base = (2 * cx, 2 * cy, 2 * cz)
                for parity in info.basis:
                    grid = (base[0] + parity[0], base[1] + parity[1], base[2] + parity[2])
                    site_id = _site_id_from_grid(grid, info)
                    if site_id == vacancy_site_id or site_id in other_pair_centers:
                        continue
                    if site_id in lattice_atoms:
                        continue
                    atom_type = source_lookup.get(grid)
                    if atom_type is None:
                        raise MoReWorkflowError(
                            "event.json does not provide a complete LAMMPS lattice reconstruction; "
                            f"missing occupied lattice site {site_id}. "
                            "Provide a full KMC event payload with normal_sites or a compatible data.lmp."
                        )
                    lattice_atoms[site_id] = (int(atom_type), _grid_to_coords(grid, info))
                    filled_from_data_lmp += 1
                    source_counts["data_lmp"] += 1

    initial_atoms: list[tuple[int, int, tuple[float, float, float]]] = [
        (site_id, atom_type, coords)
        for site_id, (atom_type, coords) in lattice_atoms.items()
    ]
    final_atoms: list[tuple[int, int, tuple[float, float, float]]] = [
        (site_id, atom_type, coords)
        for site_id, (atom_type, coords) in lattice_atoms.items()
        if site_id != jump_site_id
    ]

    diagonal = 1.0 / math.sqrt(3.0)
    other_pair_records = []
    default_other_pair_type = str(event.get("other_pair_type", event.get("pair_type", "MoRe")))
    for site in event["other_pair_sites"]:
        pair_type = str(site.get("pair_type", default_other_pair_type))
        pair_atom_types = PAIR_TYPE_TO_GAP_TYPES.get(pair_type)
        if pair_atom_types is None:
            raise MoReWorkflowError(f"Unsupported other_pair_sites pair_type for LAMMPS event modeling: {pair_type}")
        pair_atom_ids = (next_extra_atom_id, next_extra_atom_id + 1)
        next_extra_atom_id += 2
        center = (float(site["x"]), float(site["y"]), float(site["z"]))
        pair_atoms = _make_pair_atoms(
            center=center,
            direction=(diagonal, diagonal, diagonal),
            displacement=pair_displacement,
            atom_ids=pair_atom_ids,
            atom_types=pair_atom_types,
            box_lo=box_lo,
            box_hi=box_hi,
        )
        initial_atoms.extend(pair_atoms)
        final_atoms.extend(pair_atoms)
        other_pair_records.append(
            {
                "site_id": int(site["site_id"]),
                "pair_type": pair_type,
                "atom_ids": [pair_atom_ids[0], pair_atom_ids[1]],
            }
        )

    current_pair_atoms = _make_pair_atoms(
        center=init_center,
        direction=direction,
        displacement=pair_displacement,
        atom_ids=current_pair_atom_ids,
        atom_types=gap_types,
        box_lo=box_lo,
        box_hi=box_hi,
    )
    initial_atoms.extend(current_pair_atoms)
    final_gap = _wrap_position(
        (
            jump_center[0] - direction[0] * pair_displacement,
            jump_center[1] - direction[1] * pair_displacement,
            jump_center[2] - direction[2] * pair_displacement,
        ),
        box_lo=box_lo,
        box_hi=box_hi,
    )
    final_target = _wrap_position(
        (
            jump_center[0] + direction[0] * pair_displacement,
            jump_center[1] + direction[1] * pair_displacement,
            jump_center[2] + direction[2] * pair_displacement,
        ),
        box_lo=box_lo,
        box_hi=box_hi,
    )
    final_atoms.append((current_pair_atom_ids[0], int(gap_types[0]), final_gap))
    final_atoms.append((current_pair_atom_ids[1], int(gap_types[1]), init_center))
    final_atoms.append((jump_site_id, jump_atom_type, final_target))

    generated_data_lmp = copied_case_dir / "data.lmp"
    generated_data_final_lmp = copied_case_dir / "data_final.lmp"
    generated_final_mosia = copied_case_dir / "final.mosia"
    source_data_copy = copied_case_dir / "source_data.lmp"
    shutil.copy2(source_data_lmp, source_data_copy)
    _write_lammps_data_file(
        path=generated_data_lmp,
        atoms=initial_atoms,
        box_lo=box_lo,
        box_hi=box_hi,
        source_masses=source_masses,
    )
    _write_lammps_data_file(
        path=generated_data_final_lmp,
        atoms=final_atoms,
        box_lo=box_lo,
        box_hi=box_hi,
        source_masses=source_masses,
    )
    _write_final_mosia(path=generated_final_mosia, atoms=final_atoms)
    return {
        "copied_assets": {
            "data.lmp": str(generated_data_lmp),
            "data_final.lmp": str(generated_data_final_lmp),
            "final.mosia": str(generated_final_mosia),
            "source_data.lmp": str(source_data_copy),
        },
        "kmc_data_lmp_assist": str(source_data_copy),
        "model": {
            "mode": "generated_from_event_json",
            "event_json": str(event_json),
            "pair_type": event.get("pair_type"),
            "pair_displacement": pair_displacement,
            "vacancy_site_id": vacancy_site_id,
            "jump_site_id": jump_site_id,
            "jump_atom_type": jump_atom_type,
            "jump_direction": [round(v, 8) for v in direction],
            "gap_types": [int(gap_types[0]), int(gap_types[1])],
            "current_pair_atom_ids": [current_pair_atom_ids[0], current_pair_atom_ids[1]],
            "dynamic_atom_id_start": max_reserved_atom_id + 1,
            "other_pair_count": len(other_pair_records),
            "other_pairs": other_pair_records,
            "lattice_site_count": len(lattice_atoms),
            "filled_from_data_lmp": filled_from_data_lmp,
            "source_site_counts": source_counts,
            "generated_data_lmp": str(generated_data_lmp),
            "generated_data_final_lmp": str(generated_data_final_lmp),
            "generated_final_mosia": str(generated_final_mosia),
            "kmc_data_lmp_assist": str(source_data_copy),
        },
    }
