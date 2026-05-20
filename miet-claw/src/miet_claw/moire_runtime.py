import csv
import json
import math
import sys
import shutil
import subprocess
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .moire_errors import MoReWorkflowError
from .moire_event_model import (
    _derive_static_case_event_binding,
    _generate_repo_kmc_state_from_event,
    _generate_seed_event_from_data_lmp,
    _load_event,
    _validate_requested_event_against_static_case,
)
from .moire_lammps_case_builder import (
    _build_dynamic_lammps_case_from_event,
    _derive_requested_event_binding,
)
from .moire_health import build_repo_kmc_runtime_health, parse_repo_kmc_run_output
from .moire_stats import (
    DEFAULT_MOIRE_DIFFUSION_TEMPERATURES,
    DEFAULT_MOIRE_KMC_SEED,
    KB_EV_PER_K,
    apply_retry_seeds as _apply_retry_seeds,
    linear_fit as _linear_fit,
    normalize_kmc_seeds as _normalize_kmc_seeds,
    normalize_temperature_list as _normalize_temperature_list,
    series_stats as _series_stats,
    summarize_seed_runs as _summarize_seed_runs,
    temperature_dir_label as _temperature_dir_label,
)
from .moire_plots import (
    _render_image_sequence_gif,
    _render_seed_comparison_svg,
    _render_temperature_relationship_svg,
)
from .moire_visualization import (
    _build_lammps_visualization_inputs,
    _list_dump_paths,
    _maybe_render_seed_ovito,
    _render_ovito_snapshot,
)

DEFAULT_CONDA_EXEC = Path("conda")
DEFAULT_CONDA_ENV = "miet-stack"
DEFAULT_NEB_INPUT = "in.neb.mosia"
DEFAULT_POST_SCRIPT = "neb_right.sh"
DEFAULT_MPI_PROCS = 5
DEFAULT_MOIRE_KMC_BINARY = Path(
    "/path/to/misa-kmc"
)
DEFAULT_MOIRE_KMC_EAM_FILE = Path(
    "/path/to/MoRe.eam.fs"
)
DEFAULT_MOIRE_KMC_TEMPERATURE = 1100.0
DEFAULT_MOIRE_KMC_STATS_STEP = "1e-10"
DEFAULT_MOIRE_KMC_RUN_TIME = "1e-10"
DEFAULT_MOIRE_DIFFUSION_STATS_STEP = "1e-7"
DEFAULT_MOIRE_DIFFUSION_RUN_TIME = "1e-6"
DEFAULT_MOIRE_DIFFUSION_SWEEP_SEEDS = (3401, 3402, 3403)
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _emit_progress(callback: Optional[ProgressCallback], stage: str, **payload: Any) -> None:
    if callback is None:
        return
    callback(stage, payload)


def _run(command: list[str], cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
    }


def _parse_barrier_from_neb(neb_txt_path: Path) -> float:
    if not neb_txt_path.exists():
        raise MoReWorkflowError(f"neb.txt not found: {neb_txt_path}")

    values = []
    for raw in neb_txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
    if not values:
        raise MoReWorkflowError(f"Could not parse a barrier from {neb_txt_path}")
    return max(values)


def _copy_case_assets(source_case_dir: Path, copied_case_dir: Path, *, reuse_existing_neb: bool) -> Dict[str, str]:
    copied_case_dir.mkdir(parents=True, exist_ok=True)
    assets = ["data.lmp", "final.mosia", "MoRe.eam.fs"]
    if reuse_existing_neb and (source_case_dir / "neb.txt").exists():
        assets.append("neb.txt")

    copied: Dict[str, str] = {}
    for name in assets:
        source = source_case_dir / name
        if not source.exists():
            raise MoReWorkflowError(f"Required MoRe case asset not found: {source}")
        target = copied_case_dir / name
        shutil.copy2(source, target)
        copied[name] = str(target)
    return copied


def _render_lammps_ovito_views(
    *,
    copied_case_dir: Path,
    enabled: bool,
    ovito_python: Optional[str],
) -> Dict[str, Any]:
    inputs = _build_lammps_visualization_inputs(copied_case_dir=copied_case_dir)
    visualization = {
        "requested": enabled,
        "ovito_python": ovito_python,
        "initial_structure": inputs["initial_structure"],
        "final_structure": inputs["final_structure"],
        "initial_particle_count": inputs["initial_particle_count"],
        "final_particle_count": inputs["final_particle_count"],
        "source_data_lmp": inputs["source_data_lmp"],
        "source_final_mosia": inputs["source_final_mosia"],
    }
    if not enabled:
        visualization.update({"status": "disabled", "warnings": []})
        return visualization
    if ovito_python is None:
        visualization.update({"status": "skipped", "warnings": ["ovito-python-not-found"]})
        return visualization

    initial_png = copied_case_dir / "ovito_initial.png"
    final_png = copied_case_dir / "ovito_final.png"
    initial_render = _render_ovito_snapshot(Path(inputs["initial_structure"]), initial_png, ovito_python)
    final_render = _render_ovito_snapshot(Path(inputs["final_structure"]), final_png, ovito_python)
    warnings = []
    if initial_render["returncode"] != 0 or not initial_png.exists():
        warnings.append(initial_render["output"].strip() or "initial-ovito-render-failed")
    if final_render["returncode"] != 0 or not final_png.exists():
        warnings.append(final_render["output"].strip() or "final-ovito-render-failed")
    visualization.update(
        {
            "status": "completed" if not warnings else ("partial" if initial_png.exists() or final_png.exists() else "failed"),
            "warnings": warnings,
            "initial_snapshot": str(initial_png) if initial_png.exists() else None,
            "final_snapshot": str(final_png) if final_png.exists() else None,
        }
    )
    return visualization


def _render_generated_neb_input(*, data_file: str, final_file: str, eam_file: str) -> str:
    return f"""# generated by mietclaw
# local MoRe NEB input regenerated in the workdir so the agent can show the exact LAMMPS script it used
units           metal
dimension       3
atom_style      atomic
atom_modify     map array
boundary        p p p
atom_modify     sort 0 0.0
neighbor        3 bin
neigh_modify    delay 5

read_data       {data_file}
write_dump      all custom neb.initial.dump id type x y z

pair_style      eam/fs
pair_coeff      * * {eam_file} Mo Re

min_style       cg
minimize        1.0e-10 1.0e-10 100000 100000
reset_timestep  0

fix             1 all neb 1.0
thermo          100
timestep        0.001
min_style       quickmin
neb             1e-12 1e-12 10000 10000 200 final {final_file}
run             0
write_dump      all custom neb.final.dump id type x y z
"""


def _render_generated_barrier_script(*, mpi_procs: int) -> str:
    last_partition = max(1, mpi_procs - 1)
    return f"""#!/bin/bash
set -euo pipefail

echo '#reaction_coordinate de' > neb.txt
echo '0 0' >> neb.txt

E0=$(grep -A 3 'next-to-last' log.lammps.0 | tail -3 | head -1 | awk '{{printf "%012.6f\\n", $3}}')

for i in $(seq 1 {last_partition})
do
  E=$(grep -A 3 'next-to-last' log.lammps.$i | tail -3 | head -1 | awk '{{printf "%012.6f\\n", $3}}')
  de=$(awk -v e=\"$E\" -v e0=\"$E0\" 'BEGIN {{printf \"%0.6f\\n\", e - e0}}')
  rc=$(awk -v idx=\"$i\" -v n=\"{last_partition}\" 'BEGIN {{printf \"%0.3f\\n\", idx / n}}')
  echo \"$rc $de\" >> neb.txt
done
"""


def _render_generated_repo_kmc_input(
    *,
    state_values_sites: Path,
    eam_file: Path,
    barrier_eV: float,
    seed: int,
    lattice_constant: float,
    cells: tuple[int, int, int],
    temperature: float,
    stats_step: str,
    run_time: str,
) -> str:
    xlo, ylo, zlo = 0, 0, 0
    xhi, yhi, zhi = cells
    return f"""# generated by mietclaw
# local repo misa-kmc input generated from a LAMMPS NEB barrier
seed                 {seed}
app_style            vacancy

dimension            3
lattice              bcc {lattice_constant:.12g}
region               box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}
create_box           box
create_sites         box value site 1
read_sites           {state_values_sites}

nei                  0.9
barrier              Mo {barrier_eV:.6f}
barrier              Re {barrier_eV:.6f}
nbody                eam/fs {eam_file}
solve_style          tree
temperature          {temperature}
diag_style           energy stats yes
stats                {stats_step}
dump                 1 text {stats_step} *.dump id site i2 x y z
run                  {run_time}
"""



def _parse_vacancy_dump_snapshot(dump_path: Path) -> Optional[Dict[str, Any]]:
    lines = dump_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 10 or not lines[0].startswith("ITEM: TIMESTEP"):
        return None
    timestep_parts = lines[1].split()
    if not timestep_parts:
        return None
    if len(timestep_parts) == 1:
        dump_index = int(float(timestep_parts[0]))
        simulation_time = float(timestep_parts[0])
    else:
        dump_index = int(float(timestep_parts[0]))
        simulation_time = float(timestep_parts[1])

    try:
        box_idx = next(index for index, raw in enumerate(lines) if raw.startswith("ITEM: BOX BOUNDS"))
        atoms_idx = next(index for index, raw in enumerate(lines) if raw.startswith("ITEM: ATOMS"))
    except StopIteration:
        return None
    if atoms_idx + 1 >= len(lines):
        return None

    xlo, xhi = (float(value) for value in lines[box_idx + 1].split()[:2])
    ylo, yhi = (float(value) for value in lines[box_idx + 2].split()[:2])
    zlo, zhi = (float(value) for value in lines[box_idx + 3].split()[:2])
    headers = lines[atoms_idx].split()[2:]
    try:
        type_index = headers.index("type")
        site_index = headers.index("i2")
        x_index = headers.index("x")
        y_index = headers.index("y")
        z_index = headers.index("z")
    except ValueError:
        return None

    for raw in lines[atoms_idx + 1 :]:
        parts = raw.split()
        if len(parts) < len(headers):
            continue
        try:
            particle_type = int(float(parts[type_index]))
        except ValueError:
            continue
        if particle_type != 0:
            continue
        return {
            "dump_file": str(dump_path),
            "dump_index": dump_index,
            "simulation_time": simulation_time,
            "vacancy_site_id": int(float(parts[site_index])),
            "x": float(parts[x_index]),
            "y": float(parts[y_index]),
            "z": float(parts[z_index]),
            "box_lengths": {
                "x": xhi - xlo,
                "y": yhi - ylo,
                "z": zhi - zlo,
            },
        }
    return None


def _unwrap_periodic_delta(delta: float, box_length: float) -> float:
    if box_length <= 0:
        return delta
    half = box_length / 2.0
    while delta > half:
        delta -= box_length
    while delta < -half:
        delta += box_length
    return delta


def _analyze_vacancy_diffusion_from_dumps(
    *,
    run_dir: Path,
    initial_vacancy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    dump_paths = _list_dump_paths(run_dir)
    if not dump_paths:
        return {
            "status": "skipped",
            "reason": "no-dump-file-found",
            "dump_count": 0,
        }

    snapshots = [item for item in (_parse_vacancy_dump_snapshot(path) for path in dump_paths) if item]
    if not snapshots:
        return {
            "status": "skipped",
            "reason": "no-vacancy-snapshots-found",
            "dump_count": len(dump_paths),
        }

    csv_path = run_dir / "vacancy_diffusion.csv"
    summary_path = run_dir / "vacancy_diffusion_summary.json"

    initial_site_id = None
    previous_position: Optional[tuple[float, float, float]] = None
    if initial_vacancy:
        if initial_vacancy.get("site_id") is not None:
            initial_site_id = int(initial_vacancy["site_id"])
        if all(key in initial_vacancy for key in ("x", "y", "z")):
            previous_position = (
                float(initial_vacancy["x"]),
                float(initial_vacancy["y"]),
                float(initial_vacancy["z"]),
            )
    if previous_position is None:
        first = snapshots[0]
        previous_position = (float(first["x"]), float(first["y"]), float(first["z"]))
        initial_site_id = int(first["vacancy_site_id"])

    previous_site_id = initial_site_id
    displacement = [0.0, 0.0, 0.0]
    observed_site_changes = 0
    rows: list[Dict[str, Any]] = [
        {
            "dump_index": 0,
            "simulation_time": 0.0,
            "vacancy_site_id": initial_site_id,
            "x": previous_position[0],
            "y": previous_position[1],
            "z": previous_position[2],
            "dx_unwrapped": 0.0,
            "dy_unwrapped": 0.0,
            "dz_unwrapped": 0.0,
            "msd": 0.0,
            "diffusion_coefficient": 0.0,
        }
    ]
    for snapshot in snapshots:
        box_lengths = snapshot["box_lengths"]
        dx = _unwrap_periodic_delta(float(snapshot["x"]) - previous_position[0], float(box_lengths["x"]))
        dy = _unwrap_periodic_delta(float(snapshot["y"]) - previous_position[1], float(box_lengths["y"]))
        dz = _unwrap_periodic_delta(float(snapshot["z"]) - previous_position[2], float(box_lengths["z"]))
        displacement[0] += dx
        displacement[1] += dy
        displacement[2] += dz
        if previous_site_id is not None and int(snapshot["vacancy_site_id"]) != int(previous_site_id):
            observed_site_changes += 1
        msd = displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2
        cur_time = float(snapshot["simulation_time"])
        rows.append(
            {
                "dump_index": int(snapshot["dump_index"]),
                "simulation_time": cur_time,
                "vacancy_site_id": int(snapshot["vacancy_site_id"]),
                "x": float(snapshot["x"]),
                "y": float(snapshot["y"]),
                "z": float(snapshot["z"]),
                "dx_unwrapped": displacement[0],
                "dy_unwrapped": displacement[1],
                "dz_unwrapped": displacement[2],
                "msd": msd,
                "diffusion_coefficient": (msd / (6.0 * cur_time)) if cur_time > 0 else 0.0,
            }
        )
        previous_position = (float(snapshot["x"]), float(snapshot["y"]), float(snapshot["z"]))
        previous_site_id = int(snapshot["vacancy_site_id"])

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dump_index",
                "simulation_time",
                "vacancy_site_id",
                "x",
                "y",
                "z",
                "dx_unwrapped",
                "dy_unwrapped",
                "dz_unwrapped",
                "msd",
                "diffusion_coefficient",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    final_row = rows[-1]
    summary = {
        "status": "completed",
        "csv_path": str(csv_path),
        "dump_count": len(dump_paths),
        "snapshot_count": len(rows),
        "initial_vacancy_site_id": initial_site_id,
        "final_vacancy_site_id": final_row["vacancy_site_id"],
        "observed_site_changes": observed_site_changes,
        "final_time": final_row["simulation_time"],
        "final_msd": final_row["msd"],
        "final_diffusion_coefficient": final_row["diffusion_coefficient"],
        "final_unwrapped_displacement": {
            "x": final_row["dx_unwrapped"],
            "y": final_row["dy_unwrapped"],
            "z": final_row["dz_unwrapped"],
        },
        "first_dump": str(dump_paths[0]),
        "last_dump": str(dump_paths[-1]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _resolve_module_python(
    module_name: str,
    override: Optional[str] = None,
    extra_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    conda_env_python = DEFAULT_CONDA_EXEC.parent.parent / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
    candidates = [override]
    if extra_candidates:
        candidates.extend(str(item) for item in extra_candidates)
    candidates.extend([str(conda_env_python), sys.executable, "python3"])
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            probe = subprocess.run(
                [candidate, "-c", f"import importlib; importlib.import_module('{module_name}')"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    return None


def _resolve_ovito_python(override: Optional[str] = None) -> Optional[str]:
    return _resolve_module_python("ovito", override=override, extra_candidates=["ovitos"])


def _resolve_pillow_python(override: Optional[str] = None) -> Optional[str]:
    return _resolve_module_python("PIL", override=override)



def _run_moire_repo_kmc_once(
    *,
    seed: int,
    barrier_eV: float,
    run_dir: Path,
    state_values_sites: Path,
    misa_kmc_binary: Path,
    eam_file: Path,
    lattice_constant: float,
    cells: tuple[int, int, int],
    temperature: float,
    stats_step: str,
    run_time: str,
    render_ovito: bool,
    ovito_python: Optional[str],
    initial_vacancy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "generated_kmc.repo.in"
    input_path.write_text(
        _render_generated_repo_kmc_input(
            state_values_sites=state_values_sites,
            eam_file=eam_file,
            barrier_eV=barrier_eV,
            seed=seed,
            lattice_constant=lattice_constant,
            cells=cells,
            temperature=temperature,
            stats_step=stats_step,
            run_time=run_time,
        ),
        encoding="utf-8",
    )

    command = [str(misa_kmc_binary), "-in", input_path.name]
    proc = subprocess.run(
        command,
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    run_out_path = run_dir / "run.out"
    run_out_path.write_text(proc.stdout, encoding="utf-8")
    parsed = parse_repo_kmc_run_output(proc.stdout)
    derived_jump_frequency = None
    if parsed.get("accepted_events") is not None and parsed.get("final_time"):
        try:
            final_time = float(parsed["final_time"])
            if final_time > 0:
                derived_jump_frequency = float(parsed["accepted_events"]) / final_time
        except (TypeError, ValueError, ZeroDivisionError):
            derived_jump_frequency = None

    status, runtime_health = build_repo_kmc_runtime_health(
        returncode=proc.returncode,
        parsed=parsed,
        run_text=proc.stdout,
    )
    visualization = _maybe_render_seed_ovito(
        run_dir=run_dir,
        seed=seed,
        enabled=render_ovito,
        python_cmd=ovito_python,
    )
    diffusion_analysis = _analyze_vacancy_diffusion_from_dumps(run_dir=run_dir, initial_vacancy=initial_vacancy)
    summary = {
        "seed": seed,
        "status": status,
        "workdir": str(run_dir),
        "files": {
            "input_kmc": str(input_path),
            "run_out": str(run_out_path),
            "ovito_snapshot": visualization.get("output_png"),
            "vacancy_diffusion_csv": diffusion_analysis.get("csv_path"),
        },
        "command": command,
        "returncode": proc.returncode,
        "runtime_health": runtime_health,
        "parsed_run": parsed,
        "derived_metrics": {
            "jump_frequency_hz": derived_jump_frequency,
        },
        "diffusion_analysis": diffusion_analysis,
        "visualization": visualization,
    }
    seed_summary_path = run_dir / "seed_summary.json"
    seed_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(seed_summary_path)
    return summary


def _build_seed_ensemble_outputs(
    *,
    seed_runs: Sequence[Dict[str, Any]],
    workdir_path: Path,
    preferred_python: Optional[str],
) -> Dict[str, Any]:
    if len(seed_runs) < 2:
        return {
            "comparison_chart_svg": None,
            "animated_gif": None,
            "gif_status": "disabled",
            "gif_warnings": [],
        }

    chart_path = workdir_path / "kmc_seed_comparison.svg"
    chart_svg = _render_seed_comparison_svg(seed_runs, chart_path)

    image_paths = [
        Path(item["visualization"]["output_png"])
        for item in seed_runs
        if (item.get("visualization") or {}).get("status") == "completed" and (item.get("visualization") or {}).get("output_png")
    ]
    if len(image_paths) < 2:
        return {
            "comparison_chart_svg": chart_svg,
            "animated_gif": None,
            "gif_status": "skipped",
            "gif_warnings": ["not-enough-seed-snapshots-for-gif"],
        }

    pillow_python = _resolve_pillow_python(preferred_python)
    if pillow_python is None:
        return {
            "comparison_chart_svg": chart_svg,
            "animated_gif": None,
            "gif_status": "skipped",
            "gif_warnings": ["pillow-python-not-found"],
        }

    gif_path = workdir_path / "kmc_seed_animation.gif"
    gif = _render_image_sequence_gif(image_paths, gif_path, python_cmd=pillow_python)
    if gif["returncode"] != 0 or not gif_path.exists():
        return {
            "comparison_chart_svg": chart_svg,
            "animated_gif": None,
            "gif_status": "failed",
            "gif_warnings": [gif["output"].strip() or "gif-render-failed"],
        }
    return {
        "comparison_chart_svg": chart_svg,
        "animated_gif": str(gif_path),
        "gif_status": "completed",
        "gif_warnings": [],
    }


def run_moire_repo_kmc(
    *,
    barrier_eV: float,
    workdir: str,
    event_json: Optional[str] = None,
    data_lmp: Optional[str] = None,
    misa_kmc_binary: Path = DEFAULT_MOIRE_KMC_BINARY,
    eam_file: Path = DEFAULT_MOIRE_KMC_EAM_FILE,
    temperature: float = DEFAULT_MOIRE_KMC_TEMPERATURE,
    stats_step: str = DEFAULT_MOIRE_KMC_STATS_STEP,
    run_time: str = DEFAULT_MOIRE_KMC_RUN_TIME,
    kmc_seed: Optional[int] = None,
    kmc_seeds: Optional[Sequence[int]] = None,
    retry_attempts: int = 0,
    render_ovito: bool = False,
    ovito_python: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    data_lmp_path = Path(data_lmp).expanduser().resolve() if data_lmp else None
    misa_kmc_binary = Path(misa_kmc_binary).expanduser().resolve()
    eam_file = Path(eam_file).expanduser().resolve()
    seed_list, auto_retry = _apply_retry_seeds(
        _normalize_kmc_seeds(kmc_seed=kmc_seed, kmc_seeds=kmc_seeds),
        retry_attempts,
    )

    if not misa_kmc_binary.exists():
        raise MoReWorkflowError(f"repo misa-kmc binary not found: {misa_kmc_binary}")
    if not eam_file.exists():
        raise MoReWorkflowError(f"MoRe eam/fs file not found: {eam_file}")
    if event_json is None and data_lmp_path is None:
        raise MoReWorkflowError("repo KMC stage needs either event.json or data.lmp so it can generate the initial state")

    generated_event = None
    if event_json is not None:
        event_json_path = Path(event_json).expanduser().resolve()
    else:
        event_json_path = workdir_path / "generated_seed_event.json"
        generated_event = _generate_seed_event_from_data_lmp(
            data_lmp=data_lmp_path,
            output_path=event_json_path,
        )
    event_payload = _load_event(event_json_path)
    initial_vacancy = event_payload.get("initsite") or {}

    _emit_progress(progress_callback, "moire.kmc.start", workdir=str(workdir_path), barrier_eV=barrier_eV)
    state_summary = _generate_repo_kmc_state_from_event(
        event_json=event_json_path,
        output_path=workdir_path / "state.repo.values.sites",
        data_lmp=data_lmp_path,
    )
    state_values_sites = Path(state_summary["state_values_sites"])
    resolved_ovito_python = _resolve_ovito_python(ovito_python) if render_ovito else None
    seed_runs = []
    for seed in seed_list:
        run_dir = workdir_path if len(seed_list) == 1 else (workdir_path / f"seed_{seed}")
        _emit_progress(progress_callback, "moire.kmc.seed.start", seed=seed, workdir=str(run_dir))
        seed_summary = _run_moire_repo_kmc_once(
            seed=seed,
            barrier_eV=barrier_eV,
            run_dir=run_dir,
            state_values_sites=state_values_sites,
            misa_kmc_binary=misa_kmc_binary,
            eam_file=eam_file,
            lattice_constant=float(state_summary["lattice_constant"]),
            cells=tuple(int(v) for v in state_summary["cells"]),
            temperature=temperature,
            stats_step=stats_step,
            run_time=run_time,
            render_ovito=render_ovito,
            ovito_python=resolved_ovito_python,
            initial_vacancy=initial_vacancy,
        )
        seed_runs.append(seed_summary)
        _emit_progress(
            progress_callback,
            f"moire.kmc.seed.{seed_summary['status']}",
            seed=seed,
            workdir=str(run_dir),
        )

    representative_run = next((item for item in seed_runs if item.get("status") == "completed"), seed_runs[0])
    ensemble = _summarize_seed_runs(seed_runs)
    if len(seed_list) == 1:
        status = representative_run["status"]
        runtime_health = representative_run["runtime_health"]
    elif ensemble["completed_count"] == len(seed_runs):
        status = "completed"
        runtime_health = {
            "status": "ok",
            "warnings": [],
            "checks": {
                "seed_count": len(seed_runs),
                "completed_runs": ensemble["completed_count"],
                "failed_runs": ensemble["failed_count"],
            },
        }
    elif ensemble["completed_count"] == 0:
        status = "failed"
        runtime_warnings = [
            f"seed {item['seed']}: {'; '.join(item.get('runtime_health', {}).get('warnings') or ['run failed'])}"
            for item in seed_runs
        ]
        runtime_health = {
            "status": "failed",
            "warnings": runtime_warnings,
            "checks": {
                "seed_count": len(seed_runs),
                "completed_runs": ensemble["completed_count"],
                "failed_runs": ensemble["failed_count"],
            },
        }
    else:
        status = "warning"
        runtime_warnings = []
        for item in seed_runs:
            if item.get("status") != "completed":
                runtime_warnings.extend(
                    f"seed {item['seed']}: {warning}"
                    for warning in (item.get("runtime_health", {}).get("warnings") or ["run failed"])
                )
        runtime_health = {
            "status": "warning",
            "warnings": runtime_warnings,
            "checks": {
                "seed_count": len(seed_runs),
                "completed_runs": ensemble["completed_count"],
                "failed_runs": ensemble["failed_count"],
            },
        }

    visualization_items = [item.get("visualization") for item in seed_runs if item.get("visualization")]
    visualization_warnings = [
        f"seed {item['seed']}: {item.get('reason')}"
        for item in visualization_items
        if item.get("status") not in {"completed", "disabled"}
    ]
    if render_ovito:
        completed_visualizations = [item for item in visualization_items if item.get("status") == "completed"]
        failed_visualizations = [item for item in visualization_items if item.get("status") == "failed"]
        skipped_visualizations = [item for item in visualization_items if item.get("status") == "skipped"]
        if completed_visualizations and not failed_visualizations and not skipped_visualizations:
            visualization_status = "completed"
        elif completed_visualizations:
            visualization_status = "partial"
        elif failed_visualizations:
            visualization_status = "failed"
        else:
            visualization_status = "skipped"
        visualization = {
            "requested": True,
            "status": visualization_status,
            "ovito_python": resolved_ovito_python,
            "completed_count": len(completed_visualizations),
            "failed_count": len(failed_visualizations),
            "skipped_count": len(skipped_visualizations),
            "warnings": visualization_warnings,
            "per_seed": visualization_items,
        }
    else:
        visualization = {
            "requested": False,
            "status": "disabled",
            "ovito_python": None,
            "completed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "warnings": [],
            "per_seed": [],
        }

    ensemble_outputs = _build_seed_ensemble_outputs(
        seed_runs=seed_runs,
        workdir_path=workdir_path,
        preferred_python=resolved_ovito_python,
    )
    visualization["warnings"] = list(visualization.get("warnings") or []) + list(ensemble_outputs["gif_warnings"] or [])
    visualization["comparison_chart_svg"] = ensemble_outputs["comparison_chart_svg"]
    visualization["animated_gif"] = ensemble_outputs["animated_gif"]
    visualization["gif_status"] = ensemble_outputs["gif_status"]
    visualization["gif_warnings"] = ensemble_outputs["gif_warnings"]

    summary = {
        "status": status,
        "event_json": str(event_json_path),
        "generated_event": generated_event,
        "barrier_eV": barrier_eV,
        "temperature_k": temperature,
        "stats_step": stats_step,
        "run_time": run_time,
        "barrier_assignment": {
            "Mo": barrier_eV,
            "Re": barrier_eV,
            "note": (
                "The repo misa-kmc binary used here accepts species-level barrier lines. "
                "mietclaw therefore writes the single MoRe LAMMPS barrier into both `barrier Mo` and `barrier Re`."
            ),
        },
        "state_generation": state_summary,
        "state_transform": state_summary,
        "files": {
            "state_values_sites": state_summary["state_values_sites"],
            "input_kmc": representative_run["files"].get("input_kmc"),
            "run_out": representative_run["files"].get("run_out"),
            "ovito_snapshot": representative_run["files"].get("ovito_snapshot"),
            "vacancy_diffusion_csv": representative_run["files"].get("vacancy_diffusion_csv"),
            "comparison_chart_svg": ensemble_outputs["comparison_chart_svg"],
            "animated_gif": ensemble_outputs["animated_gif"],
        },
        "lattice_constant": state_summary["lattice_constant"],
        "cells": state_summary["cells"],
        "misa_kmc_binary": str(misa_kmc_binary),
        "eam_file": str(eam_file),
        "command": representative_run["command"],
        "returncode": representative_run["returncode"],
        "seed": representative_run["seed"],
        "seeds": seed_list,
        "representative_seed": representative_run["seed"],
        "auto_retry": auto_retry,
        "runtime_health": runtime_health,
        "parsed_run": representative_run["parsed_run"],
        "derived_metrics": representative_run.get("derived_metrics") or {},
        "diffusion_analysis": representative_run.get("diffusion_analysis") or {},
        "diffusion_ensemble": ((ensemble.get("metrics") or {}).get("final_diffusion_coefficient") if ensemble else None),
        "seed_runs": seed_runs,
        "ensemble": ensemble if len(seed_list) > 1 else None,
        "visualization": visualization,
    }
    summary_path = workdir_path / "kmc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    if status == "failed":
        _emit_progress(progress_callback, "moire.kmc.failed", workdir=str(workdir_path), status=status)
        if len(seed_list) == 1:
            raise MoReWorkflowError(f"repo misa-kmc failed. See {representative_run['files']['run_out']}")
        raise MoReWorkflowError(f"repo misa-kmc failed for all seeds. See {summary_path}")
    _emit_progress(
        progress_callback,
        "moire.kmc.complete",
        workdir=str(workdir_path),
        status=status,
        barrier_eV=barrier_eV,
        seed_count=len(seed_list),
    )
    return summary


def run_moire_lammps_case(
    *,
    case_dir: str,
    workdir: str,
    event_json: Optional[str] = None,
    reuse_existing_neb: bool = False,
    conda_exec: Path = DEFAULT_CONDA_EXEC,
    conda_env: str = DEFAULT_CONDA_ENV,
    neb_input: str = DEFAULT_NEB_INPUT,
    post_script: str = DEFAULT_POST_SCRIPT,
    mpi_procs: int = DEFAULT_MPI_PROCS,
    render_ovito: bool = False,
    ovito_python: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    source_case_dir = Path(case_dir).expanduser().resolve()
    workdir_path = Path(workdir).expanduser().resolve()
    event_json_path = Path(event_json).expanduser().resolve() if event_json else None

    if not source_case_dir.exists():
        raise MoReWorkflowError(f"MoRe case directory not found: {source_case_dir}")
    if not conda_exec.exists() and shutil.which(str(conda_exec)) is None:
        raise MoReWorkflowError(f"conda executable not found: {conda_exec}")
    if event_json_path is not None and not event_json_path.exists():
        raise MoReWorkflowError(f"event.json not found: {event_json_path}")
    if reuse_existing_neb and event_json_path is not None:
        raise MoReWorkflowError(
            "reuse_existing_neb cannot be used together with event.json because the LAMMPS model now needs to be rebuilt from the requested event."
        )

    workdir_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, "moire.run.start", workdir=str(workdir_path), case_dir=str(source_case_dir))
    copied_case_dir = workdir_path / "lammps_case"
    if copied_case_dir.exists():
        shutil.rmtree(copied_case_dir)
    copied_case_dir.mkdir(parents=True, exist_ok=True)
    eam_source = source_case_dir / "MoRe.eam.fs"
    if not eam_source.exists():
        raise MoReWorkflowError(f"Required MoRe case asset not found: {eam_source}")
    eam_target = copied_case_dir / "MoRe.eam.fs"
    shutil.copy2(eam_source, eam_target)
    copied_assets: Dict[str, str]
    kmc_data_lmp_assist: str
    model_summary: Dict[str, Any]
    if event_json_path is not None:
        dynamic_assets = _build_dynamic_lammps_case_from_event(
            event_json=event_json_path,
            source_data_lmp=source_case_dir / "data.lmp",
            copied_case_dir=copied_case_dir,
        )
        copied_assets = {
            **dynamic_assets["copied_assets"],
            "MoRe.eam.fs": str(eam_target),
        }
        kmc_data_lmp_assist = dynamic_assets["kmc_data_lmp_assist"]
        model_summary = dynamic_assets["model"]
    else:
        copied_assets = _copy_case_assets(source_case_dir, copied_case_dir, reuse_existing_neb=reuse_existing_neb)
        copied_assets["MoRe.eam.fs"] = str(eam_target)
        kmc_data_lmp_assist = copied_assets["data.lmp"]
        model_summary = {
            "mode": "static_case_assets",
            "source_case_dir": str(source_case_dir),
            "generated_data_lmp": copied_assets["data.lmp"],
            "generated_final_mosia": copied_assets["final.mosia"],
            "kmc_data_lmp_assist": kmc_data_lmp_assist,
        }
    generated_input_path = copied_case_dir / "generated_in.neb.mietclaw"
    generated_input_path.write_text(
        _render_generated_neb_input(
            data_file=Path(copied_assets["data.lmp"]).name,
            final_file=Path(copied_assets["final.mosia"]).name,
            eam_file=Path(copied_assets["MoRe.eam.fs"]).name,
        ),
        encoding="utf-8",
    )
    generated_post_path = copied_case_dir / "extract_barrier.mietclaw.sh"
    generated_post_path.write_text(_render_generated_barrier_script(mpi_procs=mpi_procs), encoding="utf-8")
    generated_post_path.chmod(0o755)
    _emit_progress(progress_callback, "moire.case.copied", copied_case_dir=str(copied_case_dir))
    resolved_ovito_python = _resolve_ovito_python(ovito_python) if render_ovito else None

    neb_txt_path = copied_case_dir / "neb.txt"
    lammps_log_path = workdir_path / "lammps_run.out"
    post_log_path = workdir_path / "neb_postprocess.out"

    lammps_result: Dict[str, Any] = {
        "status": "skipped" if reuse_existing_neb else "pending",
        "returncode": None,
        "command": None,
        "log": str(lammps_log_path),
    }
    post_result: Dict[str, Any] = {
        "status": "skipped" if reuse_existing_neb else "pending",
        "returncode": None,
        "command": None,
        "log": str(post_log_path),
    }

    if not reuse_existing_neb:
        _emit_progress(progress_callback, "moire.lammps.start", cwd=str(copied_case_dir), mpi_procs=mpi_procs)
        lammps_command = [
            str(conda_exec),
            "run",
            "-n",
            conda_env,
            "mpirun",
            "--oversubscribe",
            "-np",
            str(mpi_procs),
            "lmp",
            "-partition",
            f"{mpi_procs}x1",
            "-in",
            generated_input_path.name,
        ]
        lammps_result = _run(lammps_command, copied_case_dir)
        lammps_log_path.write_text(lammps_result["stdout"], encoding="utf-8")
        lammps_result["status"] = "executed" if lammps_result["returncode"] == 0 else "failed"
        lammps_result["log"] = str(lammps_log_path)
        if lammps_result["returncode"] != 0:
            _emit_progress(
                progress_callback,
                "moire.lammps.failed",
                log=str(lammps_log_path),
                returncode=lammps_result["returncode"],
            )
            raise MoReWorkflowError(
                f"LAMMPS/NEB run failed with return code {lammps_result['returncode']}. "
                f"See {lammps_log_path}"
            )
        _emit_progress(progress_callback, "moire.lammps.complete", log=str(lammps_log_path))

        _emit_progress(progress_callback, "moire.postprocess.start", cwd=str(copied_case_dir))
        post_command = ["bash", generated_post_path.name]
        post_result = _run(post_command, copied_case_dir)
        post_log_path.write_text(post_result["stdout"], encoding="utf-8")
        post_result["status"] = "executed" if post_result["returncode"] == 0 else "failed"
        post_result["log"] = str(post_log_path)
        if post_result["returncode"] != 0:
            _emit_progress(
                progress_callback,
                "moire.postprocess.failed",
                log=str(post_log_path),
                returncode=post_result["returncode"],
            )
            raise MoReWorkflowError(
                f"NEB post-process failed with return code {post_result['returncode']}. "
                f"See {post_log_path}"
            )
        _emit_progress(progress_callback, "moire.postprocess.complete", log=str(post_log_path))

    if not neb_txt_path.exists():
        raise MoReWorkflowError(f"neb.txt not found after LAMMPS stage: {neb_txt_path}")

    visualization = _render_lammps_ovito_views(
        copied_case_dir=copied_case_dir,
        enabled=render_ovito,
        ovito_python=resolved_ovito_python,
    )
    barrier_eV = _parse_barrier_from_neb(neb_txt_path)
    summary = {
        "status": "completed",
        "source_case_dir": str(source_case_dir),
        "copied_case_dir": str(copied_case_dir),
        "copied_assets": copied_assets,
        "kmc_data_lmp_assist": kmc_data_lmp_assist,
        "model": model_summary,
        "generated_lammps_input": str(generated_input_path),
        "generated_barrier_script": str(generated_post_path),
        "neb_txt": str(neb_txt_path),
        "barrier_eV": barrier_eV,
        "lammps": lammps_result,
        "postprocess": post_result,
        "visualization": visualization,
    }
    summary_path = workdir_path / "lammps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    _emit_progress(progress_callback, "moire.barrier.ready", neb_txt=str(neb_txt_path), barrier_eV=barrier_eV)
    return summary


def run_moire_lammps_to_kmc(
    *,
    event_json: Optional[str] = None,
    case_dir: str,
    workdir: str,
    validate: bool = True,
    reuse_existing_neb: bool = False,
    conda_exec: Path = DEFAULT_CONDA_EXEC,
    conda_env: str = DEFAULT_CONDA_ENV,
    neb_input: str = DEFAULT_NEB_INPUT,
    post_script: str = DEFAULT_POST_SCRIPT,
    mpi_procs: int = DEFAULT_MPI_PROCS,
    kmc_seed: Optional[int] = None,
    kmc_seeds: Optional[Sequence[int]] = None,
    kmc_retry_attempts: int = 0,
    misa_kmc_binary: Path = DEFAULT_MOIRE_KMC_BINARY,
    eam_file: Path = DEFAULT_MOIRE_KMC_EAM_FILE,
    kmc_temperature: float = DEFAULT_MOIRE_KMC_TEMPERATURE,
    kmc_stats_step: str = DEFAULT_MOIRE_KMC_STATS_STEP,
    kmc_run_time: str = DEFAULT_MOIRE_KMC_RUN_TIME,
    render_ovito: bool = False,
    ovito_python: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    workdir_path = Path(workdir).expanduser().resolve()

    event_json_path = Path(event_json).expanduser().resolve() if event_json else None
    if event_json_path is not None and not event_json_path.exists():
        raise MoReWorkflowError(f"event.json not found: {event_json_path}")
    lammps_event_binding = (
        _derive_requested_event_binding(event_json_path)
        if event_json_path is not None
        else _derive_static_case_event_binding(Path(case_dir).expanduser().resolve())
    )

    lammps_summary = run_moire_lammps_case(
        case_dir=case_dir,
        workdir=str(workdir_path),
        event_json=str(event_json_path) if event_json_path is not None else None,
        reuse_existing_neb=reuse_existing_neb,
        conda_exec=conda_exec,
        conda_env=conda_env,
        neb_input=neb_input,
        post_script=post_script,
        mpi_procs=mpi_procs,
        render_ovito=render_ovito,
        ovito_python=ovito_python,
        progress_callback=progress_callback,
    )

    bridge_workdir = workdir_path / "kmc_bridge"
    _emit_progress(
        progress_callback,
        "moire.kmc.start",
        bridge_workdir=str(bridge_workdir),
        barrier_eV=lammps_summary["barrier_eV"],
    )
    kmc_summary = run_moire_repo_kmc(
        event_json=str(event_json_path) if event_json_path is not None else None,
        barrier_eV=lammps_summary["barrier_eV"],
        workdir=str(bridge_workdir),
        data_lmp=lammps_summary.get("kmc_data_lmp_assist") or (lammps_summary.get("copied_assets") or {}).get("data.lmp"),
        misa_kmc_binary=misa_kmc_binary,
        eam_file=eam_file,
        temperature=kmc_temperature,
        stats_step=kmc_stats_step,
        run_time=kmc_run_time,
        kmc_seed=kmc_seed,
        kmc_seeds=kmc_seeds,
        retry_attempts=kmc_retry_attempts,
        render_ovito=render_ovito,
        ovito_python=ovito_python,
        progress_callback=progress_callback,
    )
    _emit_progress(
        progress_callback,
        "moire.kmc.complete",
        bridge_workdir=str(bridge_workdir),
        health=(kmc_summary.get("runtime_health") or {}).get("status"),
        status=kmc_summary.get("status"),
    )

    final_status = "completed"
    warnings = list((kmc_summary.get("runtime_health") or {}).get("warnings") or [])
    warnings.extend((kmc_summary.get("visualization") or {}).get("warnings") or [])
    warnings.extend((lammps_summary.get("visualization") or {}).get("warnings") or [])
    if warnings:
        final_status = "warning"
    summary = {
        "status": final_status,
        "event_json": kmc_summary.get("event_json") or (str(event_json_path) if event_json_path is not None else None),
        "generated_event": kmc_summary.get("generated_event"),
        "source_case_dir": lammps_summary.get("source_case_dir"),
        "copied_case_dir": lammps_summary.get("copied_case_dir"),
        "copied_assets": lammps_summary.get("copied_assets"),
        "kmc_data_lmp_assist": lammps_summary.get("kmc_data_lmp_assist"),
        "generated_lammps_input": lammps_summary.get("generated_lammps_input"),
        "generated_barrier_script": lammps_summary.get("generated_barrier_script"),
        "neb_txt": lammps_summary.get("neb_txt"),
        "lammps": lammps_summary.get("lammps"),
        "postprocess": lammps_summary.get("postprocess"),
        "lammps_event_binding": lammps_event_binding,
        "lammps_model": lammps_summary.get("model"),
        "lammps_visualization": lammps_summary.get("visualization"),
        "barrier_eV": kmc_summary.get("barrier_eV", lammps_summary.get("barrier_eV")),
        "runtime_health": kmc_summary.get("runtime_health"),
        "lammps_stage": lammps_summary,
        "kmc": kmc_summary,
    }
    summary_path = workdir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    _emit_progress(progress_callback, "moire.run.complete", summary_json=str(summary_path), status=final_status)
    return summary


def _slugify_event_label(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return cleaned or "event"


def run_moire_event_compare(
    *,
    case_dir: str,
    event_jsons: Sequence[str],
    workdir: str,
    validate: bool = True,
    reuse_existing_neb: bool = False,
    conda_exec: Path = DEFAULT_CONDA_EXEC,
    conda_env: str = DEFAULT_CONDA_ENV,
    neb_input: str = DEFAULT_NEB_INPUT,
    post_script: str = DEFAULT_POST_SCRIPT,
    mpi_procs: int = DEFAULT_MPI_PROCS,
    kmc_seed: Optional[int] = None,
    kmc_seeds: Optional[Sequence[int]] = None,
    kmc_retry_attempts: int = 0,
    misa_kmc_binary: Path = DEFAULT_MOIRE_KMC_BINARY,
    eam_file: Path = DEFAULT_MOIRE_KMC_EAM_FILE,
    kmc_temperature: float = DEFAULT_MOIRE_KMC_TEMPERATURE,
    kmc_stats_step: str = DEFAULT_MOIRE_KMC_STATS_STEP,
    kmc_run_time: str = DEFAULT_MOIRE_KMC_RUN_TIME,
    run_kmc: bool = True,
    render_ovito: bool = False,
    ovito_python: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    source_case_dir = Path(case_dir).expanduser().resolve()
    if not source_case_dir.exists():
        raise MoReWorkflowError(f"MoRe case directory not found: {source_case_dir}")

    resolved_events: list[Path] = []
    seen_events: set[Path] = set()
    for raw in event_jsons:
        candidate = Path(str(raw)).expanduser().resolve()
        if candidate in seen_events:
            continue
        if not candidate.exists():
            raise MoReWorkflowError(f"event.json not found: {candidate}")
        resolved_events.append(candidate)
        seen_events.add(candidate)
    if len(resolved_events) < 2:
        raise MoReWorkflowError("MoRe compare needs at least two distinct event.json files.")

    workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)

    label_counts: Dict[str, int] = {}
    event_labels: list[str] = []
    for index, event_path in enumerate(resolved_events, start=1):
        base = _slugify_event_label(event_path.stem)
        label_counts[base] = label_counts.get(base, 0) + 1
        suffix = label_counts[base]
        label = base if suffix == 1 else f"{base}_{suffix}"
        if label == "event":
            label = f"event_{index:02d}"
        event_labels.append(label)

    seed_list = _normalize_kmc_seeds(kmc_seed=kmc_seed, kmc_seeds=kmc_seeds) if run_kmc else []
    _emit_progress(
        progress_callback,
        "moire.compare.start",
        case_dir=str(source_case_dir),
        workdir=str(workdir_path),
        event_count=len(resolved_events),
        run_kmc=run_kmc,
    )

    event_runs: list[Dict[str, Any]] = []
    for index, (event_path, label) in enumerate(zip(resolved_events, event_labels), start=1):
        event_workdir = workdir_path / label
        _emit_progress(
            progress_callback,
            "moire.compare.event.start",
            index=index,
            label=label,
            event_json=str(event_path),
            workdir=str(event_workdir),
        )
        try:
            if run_kmc:
                event_summary = run_moire_lammps_to_kmc(
                    event_json=str(event_path),
                    case_dir=str(source_case_dir),
                    workdir=str(event_workdir),
                    validate=validate,
                    reuse_existing_neb=reuse_existing_neb,
                    conda_exec=conda_exec,
                    conda_env=conda_env,
                    neb_input=neb_input,
                    post_script=post_script,
                    mpi_procs=mpi_procs,
                    kmc_seed=kmc_seed,
                    kmc_seeds=kmc_seeds,
                    kmc_retry_attempts=kmc_retry_attempts,
                    misa_kmc_binary=misa_kmc_binary,
                    eam_file=eam_file,
                    kmc_temperature=kmc_temperature,
                    kmc_stats_step=kmc_stats_step,
                    kmc_run_time=kmc_run_time,
                    render_ovito=render_ovito,
                    ovito_python=ovito_python,
                    progress_callback=progress_callback,
                )
                lammps_summary = event_summary.get("lammps_stage") or {}
                kmc_summary = event_summary.get("kmc") or {}
                parsed_run = kmc_summary.get("parsed_run") or {}
                ensemble = kmc_summary.get("ensemble") or {}
                kmc_visualization = kmc_summary.get("visualization") or {}
                runtime_health = kmc_summary.get("runtime_health") or event_summary.get("runtime_health") or {}
                record = {
                    "index": index,
                    "label": label,
                    "event_json": str(event_path),
                    "workdir": str(event_workdir),
                    "status": event_summary.get("status", "completed"),
                    "barrier_eV": event_summary.get("barrier_eV"),
                    "summary_json": event_summary.get("summary_json"),
                    "neb_txt": event_summary.get("neb_txt"),
                    "lammps_summary_json": lammps_summary.get("summary_json"),
                    "lammps_model_mode": (event_summary.get("lammps_model") or {}).get("mode"),
                    "lammps_initial_snapshot": ((event_summary.get("lammps_visualization") or {}).get("initial_snapshot")),
                    "lammps_final_snapshot": ((event_summary.get("lammps_visualization") or {}).get("final_snapshot")),
                    "kmc_enabled": True,
                    "kmc_status": kmc_summary.get("status"),
                    "kmc_summary_json": kmc_summary.get("summary_json"),
                    "kmc_visualization_status": kmc_visualization.get("status"),
                    "kmc_visualization_sample": next(
                        (
                            item.get("output_png")
                            for item in (kmc_visualization.get("per_seed") or [])
                            if item.get("output_png")
                        ),
                        kmc_summary.get("files", {}).get("ovito_snapshot"),
                    ),
                    "kmc_seed_count": ensemble.get("count") or len(kmc_summary.get("seeds") or []) or (1 if kmc_summary else 0),
                    "kmc_completed_seed_count": ensemble.get("completed_count")
                    if ensemble
                    else (1 if kmc_summary.get("status") == "completed" else 0),
                    "kmc_seeds": ensemble.get("seeds") or kmc_summary.get("seeds"),
                    "accepted_events": parsed_run.get("accepted_events"),
                    "rejected_events": parsed_run.get("rejected_events"),
                    "final_time": parsed_run.get("final_time"),
                    "final_energy": parsed_run.get("final_energy"),
                    "loop_time_seconds": parsed_run.get("loop_time_seconds"),
                    "runtime_health": runtime_health,
                }
            else:
                event_summary = run_moire_lammps_case(
                    event_json=str(event_path),
                    case_dir=str(source_case_dir),
                    workdir=str(event_workdir),
                    reuse_existing_neb=reuse_existing_neb,
                    conda_exec=conda_exec,
                    conda_env=conda_env,
                    neb_input=neb_input,
                    post_script=post_script,
                    mpi_procs=mpi_procs,
                    render_ovito=render_ovito,
                    ovito_python=ovito_python,
                    progress_callback=progress_callback,
                )
                lammps_visualization = event_summary.get("visualization") or {}
                record = {
                    "index": index,
                    "label": label,
                    "event_json": str(event_path),
                    "workdir": str(event_workdir),
                    "status": event_summary.get("status", "completed"),
                    "barrier_eV": event_summary.get("barrier_eV"),
                    "summary_json": event_summary.get("summary_json"),
                    "neb_txt": event_summary.get("neb_txt"),
                    "lammps_summary_json": event_summary.get("summary_json"),
                    "lammps_model_mode": (event_summary.get("model") or {}).get("mode"),
                    "lammps_initial_snapshot": lammps_visualization.get("initial_snapshot"),
                    "lammps_final_snapshot": lammps_visualization.get("final_snapshot"),
                    "kmc_enabled": False,
                    "kmc_status": None,
                    "kmc_summary_json": None,
                    "kmc_visualization_status": None,
                    "kmc_visualization_sample": None,
                    "kmc_seed_count": 0,
                    "kmc_completed_seed_count": 0,
                    "kmc_seeds": [],
                    "accepted_events": None,
                    "rejected_events": None,
                    "final_time": None,
                    "final_energy": None,
                    "loop_time_seconds": None,
                    "runtime_health": None,
                }
            _emit_progress(
                progress_callback,
                "moire.compare.event.complete",
                index=index,
                label=label,
                barrier_eV=record.get("barrier_eV"),
                status=record.get("status"),
            )
        except MoReWorkflowError as exc:
            record = {
                "index": index,
                "label": label,
                "event_json": str(event_path),
                "workdir": str(event_workdir),
                "status": "failed",
                "barrier_eV": None,
                "summary_json": str(event_workdir / "summary.json"),
                "neb_txt": None,
                "lammps_summary_json": str(event_workdir / "lammps_summary.json"),
                "lammps_model_mode": None,
                "lammps_initial_snapshot": None,
                "lammps_final_snapshot": None,
                "kmc_enabled": run_kmc,
                "kmc_status": "failed" if run_kmc else None,
                "kmc_summary_json": str(event_workdir / "kmc_bridge" / "kmc_summary.json") if run_kmc else None,
                "kmc_visualization_status": None,
                "kmc_visualization_sample": None,
                "kmc_seed_count": len(seed_list),
                "kmc_completed_seed_count": 0,
                "kmc_seeds": seed_list,
                "accepted_events": None,
                "rejected_events": None,
                "final_time": None,
                "final_energy": None,
                "loop_time_seconds": None,
                "runtime_health": {"status": "failed", "warnings": [str(exc)], "checks": {}},
                "error": str(exc),
            }
            _emit_progress(
                progress_callback,
                "moire.compare.event.failed",
                index=index,
                label=label,
                event_json=str(event_path),
                error=str(exc),
            )
        event_runs.append(record)

    successful_runs = [item for item in event_runs if item.get("barrier_eV") is not None]
    barrier_values = [float(item["barrier_eV"]) for item in successful_runs]
    barrier_stats = _series_stats(barrier_values) if barrier_values else {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    reference_barrier = successful_runs[0]["barrier_eV"] if successful_runs else None
    lowest_barrier = min(barrier_values) if barrier_values else None
    barrier_ranking = []
    for rank, item in enumerate(sorted(successful_runs, key=lambda current: float(current["barrier_eV"])), start=1):
        barrier = float(item["barrier_eV"])
        barrier_ranking.append(
            {
                "rank": rank,
                "label": item["label"],
                "event_json": item["event_json"],
                "barrier_eV": barrier,
                "delta_vs_lowest_eV": (barrier - lowest_barrier) if lowest_barrier is not None else None,
                "delta_vs_first_eV": (barrier - float(reference_barrier)) if reference_barrier is not None else None,
                "summary_json": item.get("summary_json"),
            }
        )

    kmc_metrics: Dict[str, Any] = {}
    if run_kmc:
        for field in ("accepted_events", "rejected_events", "final_time", "final_energy", "loop_time_seconds", "barrier_eV"):
            values = [float(item[field]) for item in successful_runs if item.get(field) is not None]
            if values:
                kmc_metrics[field] = _series_stats(values)

    warning_messages: list[str] = []
    for item in event_runs:
        runtime_health = item.get("runtime_health") or {}
        for warning in runtime_health.get("warnings") or []:
            warning_messages.append(f"{item['label']}: {warning}")

    failed_count = sum(1 for item in event_runs if item.get("barrier_eV") is None)
    warning_count = sum(1 for item in event_runs if str(item.get("status")) == "warning")
    if failed_count == len(event_runs):
        status = "failed"
    elif failed_count or warning_count or warning_messages:
        status = "warning"
    else:
        status = "completed"

    summary = {
        "status": status,
        "mode": "moire_event_compare",
        "case_dir": str(source_case_dir),
        "workdir": str(workdir_path),
        "validate": bool(validate),
        "run_kmc": run_kmc,
        "kmc_seeds": seed_list,
        "ovito_requested": bool(render_ovito),
        "ovito_python": ovito_python if render_ovito else None,
        "event_jsons": [str(item) for item in resolved_events],
        "event_runs": event_runs,
        "event_count": len(event_runs),
        "completed_count": len(successful_runs),
        "failed_count": failed_count,
        "barrier_stats": barrier_stats,
        "barrier_ranking": barrier_ranking,
        "barrier_span_eV": (max(barrier_values) - min(barrier_values)) if len(barrier_values) >= 2 else 0.0 if barrier_values else None,
        "reference_event": {
            "label": successful_runs[0]["label"],
            "event_json": successful_runs[0]["event_json"],
            "barrier_eV": successful_runs[0]["barrier_eV"],
        }
        if successful_runs
        else None,
        "kmc_metrics": kmc_metrics if run_kmc else None,
        "warnings": warning_messages,
    }
    comparison_path = workdir_path / "comparison.json"
    summary_path = workdir_path / "summary.json"
    summary["comparison_json"] = str(comparison_path)
    summary["summary_json"] = str(summary_path)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    comparison_path.write_text(payload, encoding="utf-8")
    summary_path.write_text(payload, encoding="utf-8")
    _emit_progress(
        progress_callback,
        "moire.compare.complete",
        summary_json=str(summary_path),
        comparison_json=str(comparison_path),
        status=status,
        completed_count=len(successful_runs),
        event_count=len(event_runs),
    )
    return summary


def run_moire_diffusion_sweep(
    *,
    event_json: str,
    case_dir: str,
    workdir: str,
    temperatures_k: Optional[Sequence[float]] = None,
    validate: bool = True,
    kmc_seed: Optional[int] = None,
    kmc_seeds: Optional[Sequence[int]] = None,
    kmc_retry_attempts: int = 0,
    misa_kmc_binary: Path = DEFAULT_MOIRE_KMC_BINARY,
    eam_file: Path = DEFAULT_MOIRE_KMC_EAM_FILE,
    render_ovito: bool = False,
    ovito_python: Optional[str] = None,
    run_time: str = DEFAULT_MOIRE_DIFFUSION_RUN_TIME,
    stats_step: str = DEFAULT_MOIRE_DIFFUSION_STATS_STEP,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    source_case_dir = Path(case_dir).expanduser().resolve()
    event_json_path = Path(event_json).expanduser().resolve()
    workdir_path = Path(workdir).expanduser().resolve()
    if not source_case_dir.exists():
        raise MoReWorkflowError(f"MoRe case directory not found: {source_case_dir}")
    if not event_json_path.exists():
        raise MoReWorkflowError(f"event.json not found: {event_json_path}")

    temperatures = _normalize_temperature_list(temperatures_k)
    if kmc_seed is None and not kmc_seeds:
        seed_list = list(DEFAULT_MOIRE_DIFFUSION_SWEEP_SEEDS)
    else:
        seed_list = _normalize_kmc_seeds(kmc_seed=kmc_seed, kmc_seeds=kmc_seeds)

    workdir_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress_callback,
        "moire.diffusion.start",
        case_dir=str(source_case_dir),
        event_json=str(event_json_path),
        workdir=str(workdir_path),
    )

    lammps_summary = run_moire_lammps_case(
        case_dir=str(source_case_dir),
        workdir=str(workdir_path),
        event_json=str(event_json_path),
        render_ovito=render_ovito,
        ovito_python=ovito_python,
        progress_callback=progress_callback,
    )
    barrier_eV = float(lammps_summary["barrier_eV"])

    temperature_runs: list[Dict[str, Any]] = []
    for temperature in temperatures:
        temp_dir = workdir_path / f"T_{_temperature_dir_label(temperature)}K"
        _emit_progress(progress_callback, "moire.diffusion.temperature.start", temperature_k=temperature, workdir=str(temp_dir))
        try:
            kmc_summary = run_moire_repo_kmc(
                barrier_eV=barrier_eV,
                workdir=str(temp_dir),
                event_json=str(event_json_path),
                data_lmp=lammps_summary.get("kmc_data_lmp_assist"),
                temperature=temperature,
                misa_kmc_binary=misa_kmc_binary,
                eam_file=eam_file,
                stats_step=stats_step,
                run_time=run_time,
                kmc_seeds=seed_list,
                retry_attempts=kmc_retry_attempts,
                render_ovito=render_ovito,
                ovito_python=ovito_python,
                progress_callback=progress_callback,
            )
            diffusion_ensemble = kmc_summary.get("diffusion_ensemble") or {}
            jump_stats = ((kmc_summary.get("ensemble") or {}).get("metrics") or {}).get("jump_frequency_hz") or {}
            accepted_stats = ((kmc_summary.get("ensemble") or {}).get("metrics") or {}).get("accepted_events") or {}
            temperature_runs.append(
                {
                    "temperature_k": float(temperature),
                    "label": f"{float(temperature):.0f} K" if abs(float(temperature) - round(float(temperature))) < 1.0e-9 else f"{float(temperature):g} K",
                    "status": kmc_summary.get("status"),
                    "workdir": str(temp_dir),
                    "kmc_summary_json": kmc_summary.get("summary_json"),
                    "representative_seed": kmc_summary.get("representative_seed"),
                    "completed_seed_count": (kmc_summary.get("ensemble") or {}).get("completed_count", 1 if kmc_summary.get("status") == "completed" else 0),
                    "diffusion_coefficient": diffusion_ensemble.get("mean") if diffusion_ensemble else (kmc_summary.get("diffusion_analysis") or {}).get("final_diffusion_coefficient"),
                    "diffusion_std": diffusion_ensemble.get("std") if diffusion_ensemble else 0.0,
                    "jump_frequency_hz": jump_stats.get("mean") if jump_stats else (kmc_summary.get("derived_metrics") or {}).get("jump_frequency_hz"),
                    "accepted_events": accepted_stats.get("mean") if accepted_stats else (kmc_summary.get("parsed_run") or {}).get("accepted_events"),
                    "ovito_sample": (kmc_summary.get("files") or {}).get("ovito_snapshot"),
                    "vacancy_diffusion_csv": (kmc_summary.get("files") or {}).get("vacancy_diffusion_csv"),
                }
            )
        except MoReWorkflowError as exc:
            temperature_runs.append(
                {
                    "temperature_k": float(temperature),
                    "label": f"{float(temperature):g} K",
                    "status": "failed",
                    "workdir": str(temp_dir),
                    "error": str(exc),
                }
            )
        _emit_progress(
            progress_callback,
            f"moire.diffusion.temperature.{temperature_runs[-1]['status']}",
            temperature_k=temperature,
            workdir=str(temp_dir),
        )

    completed_runs = [item for item in temperature_runs if item.get("status") == "completed" and item.get("diffusion_coefficient") is not None]
    failed_runs = [item for item in temperature_runs if item.get("status") != "completed"]
    diffusion_values = [float(item["diffusion_coefficient"]) for item in completed_runs if item.get("diffusion_coefficient") is not None]
    monotonic_increasing = None
    if len(completed_runs) >= 2:
        ordered = sorted(completed_runs, key=lambda item: float(item["temperature_k"]))
        monotonic_increasing = all(
            float(next_item["diffusion_coefficient"]) >= float(current["diffusion_coefficient"])
            for current, next_item in zip(ordered, ordered[1:])
        )

    diffusion_csv_path = workdir_path / "diffusion_vs_temperature.csv"
    with diffusion_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "temperature_k",
                "completed_seed_count",
                "representative_seed",
                "diffusion_coefficient",
                "diffusion_std",
                "jump_frequency_hz",
                "accepted_events",
                "vacancy_diffusion_csv",
                "ovito_sample",
                "status",
            ],
        )
        writer.writeheader()
        for row in temperature_runs:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    diffusion_svg_path = workdir_path / "diffusion_vs_temperature.svg"
    diffusion_svg = _render_temperature_relationship_svg(
        rows=temperature_runs,
        output_path=diffusion_svg_path,
        title="MoRe diffusion coefficient vs temperature",
        subtitle=f"Barrier fixed at {barrier_eV:.6f} eV from one LAMMPS/NEB event",
        x_key="temperature_k",
        y_key="diffusion_coefficient",
        x_label="Temperature (K)",
        y_label="Diffusion coefficient",
        y_error_key="diffusion_std",
    )

    arrhenius_rows = [item for item in completed_runs if item.get("diffusion_coefficient") and float(item["diffusion_coefficient"]) > 0]
    arrhenius_csv_path = workdir_path / "arrhenius.csv"
    with arrhenius_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["temperature_k", "inverse_temperature_1000_per_k", "diffusion_coefficient", "ln_diffusion_coefficient"],
        )
        writer.writeheader()
        for row in arrhenius_rows:
            writer.writerow(
                {
                    "temperature_k": row["temperature_k"],
                    "inverse_temperature_1000_per_k": 1000.0 / float(row["temperature_k"]),
                    "diffusion_coefficient": row["diffusion_coefficient"],
                    "ln_diffusion_coefficient": math.log(float(row["diffusion_coefficient"])),
                }
            )

    fit = _linear_fit(
        [1000.0 / float(row["temperature_k"]) for row in arrhenius_rows],
        [math.log(float(row["diffusion_coefficient"])) for row in arrhenius_rows],
    )
    arrhenius_svg_path = workdir_path / "arrhenius.svg"
    arrhenius_svg = _render_temperature_relationship_svg(
        rows=[
            {
                **row,
                "inverse_temperature_1000_per_k": 1000.0 / float(row["temperature_k"]),
                "ln_diffusion_coefficient": math.log(float(row["diffusion_coefficient"])),
            }
            for row in arrhenius_rows
        ],
        output_path=arrhenius_svg_path,
        title="MoRe Arrhenius plot",
        subtitle="ln(D) vs 1000/T reconstructed from the vacancy trajectory in KMC dumps",
        x_key="inverse_temperature_1000_per_k",
        y_key="ln_diffusion_coefficient",
        x_label="1000 / T (1/K)",
        y_label="ln(diffusion coefficient)",
        fit=fit,
    )

    activation_energy_eV = None
    if fit is not None:
        activation_energy_eV = -fit["slope"] * 1000.0 * KB_EV_PER_K

    if completed_runs and not failed_runs:
        status = "completed"
    elif completed_runs:
        status = "warning"
    else:
        status = "failed"

    summary = {
        "status": status,
        "mode": "moire_diffusion_sweep",
        "case_dir": str(source_case_dir),
        "event_json": str(event_json_path),
        "workdir": str(workdir_path),
        "validate": bool(validate),
        "barrier_eV": barrier_eV,
        "temperatures_k": temperatures,
        "kmc_seeds": seed_list,
        "stats_step": stats_step,
        "run_time": run_time,
        "ovito_requested": bool(render_ovito),
        "ovito_python": ovito_python if render_ovito else None,
        "lammps_summary_json": lammps_summary.get("summary_json"),
        "lammps_visualization": lammps_summary.get("visualization"),
        "temperature_runs": temperature_runs,
        "completed_count": len(completed_runs),
        "failed_count": len(failed_runs),
        "diffusion_stats": _series_stats(diffusion_values) if diffusion_values else {"count": 0, "mean": None, "std": None, "min": None, "max": None},
        "temperature_trend": {
            "monotonic_increasing": monotonic_increasing,
            "points_used": len(completed_runs),
        },
        "arrhenius_fit": {
            "slope": fit.get("slope") if fit else None,
            "intercept": fit.get("intercept") if fit else None,
            "activation_energy_eV": activation_energy_eV,
            "point_count": len(arrhenius_rows),
        },
        "files": {
            "diffusion_vs_temperature_csv": str(diffusion_csv_path),
            "diffusion_vs_temperature_svg": diffusion_svg,
            "arrhenius_csv": str(arrhenius_csv_path),
            "arrhenius_svg": arrhenius_svg,
        },
        "warnings": [item.get("error") for item in failed_runs if item.get("error")],
    }
    summary_path = workdir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    _emit_progress(
        progress_callback,
        "moire.diffusion.complete",
        workdir=str(workdir_path),
        status=status,
        completed_count=len(completed_runs),
        temperature_count=len(temperature_runs),
    )
    return summary
