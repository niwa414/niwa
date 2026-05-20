import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


NUMERIC_RE = r'[-+0-9.eE]+'


def _campaign_mass_lines(campaign: Dict[str, Any]) -> str:
    return '\n'.join(
        f"mass            {index} {campaign['masses'][species]}"
        for index, species in enumerate(campaign['species_order'], start=1)
    )


def render_reference_preflight_input(campaign: Dict[str, Any]) -> str:
    pair_species = ' '.join(campaign['pair_coeff_species'])
    return f"""# mietclaw autonomy-generated reference preflight
units           metal
dimension       3
boundary        p p p
atom_style      atomic

lattice         bcc {campaign['lattice_constant']}
region          box block 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']}
create_box      {len(campaign['species_order'])} box
create_atoms    1 box

{_campaign_mass_lines(campaign)}

pair_style      {campaign['pair_style']}
pair_coeff      * * {campaign['potential_path']} {pair_species}

neighbor        2.0 bin
compute         peratom all pe/atom
thermo          0
run             0

print           \"Total energy: $(etotal) eV\"
print           \"Potential energy per atom: $(pe/atoms) eV\"
"""


def render_neb_species_relax_input(campaign: Dict[str, Any], task: Dict[str, Any], endpoint: str) -> str:
    pair_species = ' '.join(campaign['pair_coeff_species'])
    vacancy_x, vacancy_y, vacancy_z = task['vacancy_coords']
    position_line = ''
    if endpoint == 'final':
        x, y, z = task['target_coords']
        position_line = f"set             atom {task['moving_atom_id']} x {x} y {y} z {z}\\n"

    return f"""# {task['species']} {endpoint} endpoint relaxation drafted by mietclaw
units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map array

lattice         bcc {campaign['lattice_constant']}
region          box block 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']}
create_box      {len(campaign['species_order'])} box
create_atoms    {task['moving_type']} box

{_campaign_mass_lines(campaign)}

pair_style      {campaign['pair_style']}
pair_coeff      * * {campaign['potential_path']} {pair_species}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
region          vacancy sphere {vacancy_x} {vacancy_y} {vacancy_z} 0.1 units box
delete_atoms    region vacancy compress no
group           mobile id {task['moving_atom_id']}
group           frozen subtract all mobile
fix             hold frozen setforce 0.0 0.0 0.0
{position_line}compute         peratom all pe/atom
thermo          0
min_style       {campaign['neb']['min_style']}
min_modify      line quadratic
minimize        0.0 0.001 200 2000
print           \"Total energy: $(etotal) eV\"
write_dump      all custom {endpoint}.relaxed.dump id type x y z
"""


def render_neb_species_final_coords(task: Dict[str, Any]) -> str:
    x, y, z = task['target_coords']
    return f"""# coords.final for {task['species']}
# Only the moved atom is listed because this is a draft endpoint hint.
1
{task['moving_atom_id']} {x} {y} {z} # {task['target_coords_comment']}
"""


def render_neb_species_input(campaign: Dict[str, Any], task: Dict[str, Any]) -> str:
    pair_species = ' '.join(campaign['pair_coeff_species'])
    neb = campaign['neb']
    vacancy_x, vacancy_y, vacancy_z = task['vacancy_coords']
    return f"""# {task['species']} CI-NEB workflow drafted by mietclaw
# Run with: {neb['run_hint']}
units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map array

lattice         bcc {campaign['lattice_constant']}
region          box block 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']} 0 {campaign['reference_region_cells']}
create_box      {len(campaign['species_order'])} box
create_atoms    {task['moving_type']} box

{_campaign_mass_lines(campaign)}

pair_style      {campaign['pair_style']}
pair_coeff      * * {campaign['potential_path']} {pair_species}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
region          vacancy sphere {vacancy_x} {vacancy_y} {vacancy_z} 0.1 units box
delete_atoms    region vacancy compress no
group           mobile id {task['moving_atom_id']}
group           frozen subtract all mobile
fix             hold frozen setforce 0.0 0.0 0.0
min_style       {neb['min_style']}
min_modify      line quadratic
timestep        {neb['timestep']}
thermo          {neb['print_every']}
thermo_style    custom step pe fnorm
fix             nebfix mobile neb {neb['spring_constant']} parallel {neb['parallel_style']}
neb             {neb['energy_tolerance']} {neb['force_tolerance']} {neb['path_iterations']} {neb['climb_iterations']} {neb['print_every']} final coords.final verbosity terse
write_dump      all custom neb.final.dump id type x y z
"""


def render_neb_campaign_readme(campaign: Dict[str, Any]) -> str:
    lines = [
        '# mietclaw generated NEB / CI-NEB campaign',
        '',
        f"- material: {campaign['material_name']}",
        f"- pathway: {campaign['pathway']}",
        f"- NEB images: {campaign['neb']['images']}",
        f"- run hint: `{campaign['neb']['run_hint']}`",
        '',
        '## Species tasks',
        '',
    ]
    for task in campaign['species_tasks']:
        lines.append(f"- {task['species']}: barrier seed {task['barrier_ev']:.4f} eV, files under `{task['slug']}/`")
    lines.extend(['', '## Notes', ''])
    lines.extend([f'- {note}' for note in campaign['notes']])
    return '\n'.join(lines) + '\n'


def command_exists(command: str) -> bool:
    candidate = Path(command)
    if candidate.is_absolute() or '/' in str(command):
        return candidate.exists()
    return shutil.which(str(command)) is not None


def parse_total_energy(stdout: str) -> Optional[float]:
    match = re.search(rf"Total energy:\s*({NUMERIC_RE})\s*eV", stdout)
    if not match:
        return None
    return float(match.group(1))


def parse_neb_terse_output(stdout: str) -> Dict[str, Any]:
    line_pattern = re.compile(
        rf"^\s*(\d+)\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s+({NUMERIC_RE})\s*$",
        flags=re.MULTILINE,
    )
    climb_match = re.search(r"Climbing replica\s*=\s*(\d+)", stdout)

    rows = []
    for match in line_pattern.finditer(stdout):
        rows.append(
            {
                'step': int(match.group(1)),
                'max_replica_force': float(match.group(2)),
                'max_atom_force': float(match.group(3)),
                'grad_v0': float(match.group(4)),
                'grad_v1': float(match.group(5)),
                'grad_vc': float(match.group(6)),
                'barrier_forward_ev': float(match.group(7)),
                'barrier_reverse_ev': float(match.group(8)),
                'reaction_distance': float(match.group(9)),
            }
        )

    final_row = rows[-1] if rows else None
    return {
        'climbing_replica': int(climb_match.group(1)) if climb_match else None,
        'samples': rows,
        'final': final_row,
        'barrier_forward_ev': final_row.get('barrier_forward_ev') if final_row else None,
        'barrier_reverse_ev': final_row.get('barrier_reverse_ev') if final_row else None,
        'reaction_distance': final_row.get('reaction_distance') if final_row else None,
        'converged_step': final_row.get('step') if final_row else None,
        'parsed': final_row is not None,
    }


def run_reference_preflight(reference_dir: Path, campaign: Dict[str, Any]) -> Dict[str, Any]:
    potential_path = Path(campaign['potential_path'])
    if not potential_path.exists():
        return {
            'status': 'skipped',
            'reason': f'potential file not found: {potential_path}',
            'reference_energy_ev': None,
            'stdout_file': None,
            'input_file': None,
        }

    lammps_bin = 'lmp'
    if not command_exists(lammps_bin):
        return {
            'status': 'skipped',
            'reason': f'LAMMPS executable not found: {lammps_bin}',
            'reference_energy_ev': None,
            'stdout_file': None,
            'input_file': None,
        }

    reference_dir.mkdir(parents=True, exist_ok=True)
    input_path = reference_dir / 'in.reference.lmp'
    stdout_path = reference_dir / 'lammps_preflight.out'
    input_path.write_text(render_reference_preflight_input(campaign), encoding='utf-8')
    completed = subprocess.run(
        [lammps_bin, '-in', input_path.name],
        cwd=reference_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    reference_energy = parse_total_energy(completed.stdout)
    return {
        'status': 'executed' if completed.returncode == 0 and reference_energy is not None else 'degraded',
        'returncode': completed.returncode,
        'reference_energy_ev': reference_energy,
        'stdout_file': stdout_path.name,
        'input_file': input_path.name,
    }


def write_species_campaign(workflow_root: Path, campaign: Dict[str, Any]) -> List[Dict[str, Any]]:
    species_files: List[Dict[str, Any]] = []
    for task in campaign['species_tasks']:
        species_dir = workflow_root / task['slug']
        species_dir.mkdir(parents=True, exist_ok=True)
        initial_path = species_dir / 'in.relax.initial.lmp'
        final_path = species_dir / 'in.relax.final.lmp'
        coords_path = species_dir / 'coords.final'
        neb_path = species_dir / 'in.neb.ci.lmp'

        initial_path.write_text(render_neb_species_relax_input(campaign, task, 'initial'), encoding='utf-8')
        final_path.write_text(render_neb_species_relax_input(campaign, task, 'final'), encoding='utf-8')
        coords_path.write_text(render_neb_species_final_coords(task), encoding='utf-8')
        neb_path.write_text(render_neb_species_input(campaign, task), encoding='utf-8')

        species_files.append(
            {
                'species': task['species'],
                'slug': task['slug'],
                'initial_relax': str(initial_path.relative_to(workflow_root.parent)),
                'final_relax': str(final_path.relative_to(workflow_root.parent)),
                'coords_final': str(coords_path.relative_to(workflow_root.parent)),
                'ci_neb_input': str(neb_path.relative_to(workflow_root.parent)),
            }
        )
    return species_files


def run_endpoint_relaxation(species_dir: Path, input_name: str, label: str) -> Dict[str, Any]:
    lammps_bin = 'lmp'
    if not command_exists(lammps_bin):
        return {
            'status': 'skipped',
            'reason': f'LAMMPS executable not found: {lammps_bin}',
            'returncode': None,
            'total_energy_ev': None,
            'stdout_file': None,
            'input_file': input_name,
        }

    stdout_path = species_dir / f'{label}.relax.out'
    completed = subprocess.run(
        [lammps_bin, '-in', input_name],
        cwd=species_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    total_energy = parse_total_energy(completed.stdout)
    return {
        'status': 'executed' if completed.returncode == 0 and total_energy is not None else 'degraded',
        'returncode': completed.returncode,
        'total_energy_ev': total_energy,
        'stdout_file': stdout_path.name,
        'input_file': input_name,
    }


def run_ci_neb(species_dir: Path, campaign: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    lammps_bin = 'lmp'
    if not command_exists(lammps_bin):
        return {
            'status': 'skipped',
            'reason': f'LAMMPS executable not found: {lammps_bin}',
            'returncode': None,
            'stdout_file': None,
            'neb_parse': {'parsed': False},
        }

    mpirun_bin = shutil.which('mpirun') or shutil.which('mpiexec')
    if not mpirun_bin:
        return {
            'status': 'skipped',
            'reason': 'MPI launcher not found: mpirun/mpiexec',
            'returncode': None,
            'stdout_file': None,
            'neb_parse': {'parsed': False},
        }

    images = int(campaign['neb']['images'])
    stdout_path = species_dir / 'neb.screen.out'
    completed = subprocess.run(
        [mpirun_bin, '--oversubscribe', '-np', str(images), lammps_bin, '-partition', f'{images}x1', '-in', 'in.neb.ci.lmp'],
        cwd=species_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    neb_parse = parse_neb_terse_output(completed.stdout)
    status = 'executed' if completed.returncode == 0 and neb_parse.get('parsed') else 'degraded'
    return {
        'status': status,
        'returncode': completed.returncode,
        'stdout_file': stdout_path.name,
        'mpi_launcher': mpirun_bin,
        'command': [mpirun_bin, '--oversubscribe', '-np', str(images), lammps_bin, '-partition', f'{images}x1', '-in', 'in.neb.ci.lmp'],
        'neb_parse': neb_parse,
    }


def execute_species_neb_campaign(workflow_root: Path, campaign: Dict[str, Any]) -> Dict[str, Any]:
    species_runs: Dict[str, Any] = {}
    for task in campaign['species_tasks']:
        species_dir = workflow_root / task['slug']
        initial = run_endpoint_relaxation(species_dir, 'in.relax.initial.lmp', 'initial')
        final = run_endpoint_relaxation(species_dir, 'in.relax.final.lmp', 'final')
        neb = run_ci_neb(species_dir, campaign, task)
        execution_payload = {
            'species': task['species'],
            'seed_barrier_ev': float(task['barrier_ev']),
            'initial_relaxation': initial,
            'final_relaxation': final,
            'neb': neb,
        }
        (species_dir / 'neb.execution.json').write_text(json.dumps(execution_payload, indent=2, ensure_ascii=False), encoding='utf-8')
        species_runs[task['species']] = execution_payload
    return species_runs


def determine_barrier_source_mode(species_runs: Dict[str, Any]) -> str:
    parsed = 0
    total = 0
    for record in species_runs.values():
        total += 1
        if (record.get('neb') or {}).get('neb_parse', {}).get('parsed'):
            parsed += 1
    if total == 0:
        return 'unavailable'
    if parsed == total:
        return 'parsed-neb'
    if parsed == 0:
        return 'seed-fallback'
    return 'mixed'


def build_barrier_payload(
    campaign: Dict[str, Any],
    preflight: Dict[str, Any],
    species_files: List[Dict[str, Any]],
    species_runs: Dict[str, Any],
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    parsed_count = 0
    for task in campaign['species_tasks']:
        record = species_runs.get(task['species']) or {}
        neb_parse = (record.get('neb') or {}).get('neb_parse', {})
        parsed_barrier = neb_parse.get('barrier_forward_ev')
        barrier_source = 'seed-fallback'
        barrier_value = float(task['barrier_ev'])
        if parsed_barrier is not None:
            barrier_value = float(parsed_barrier)
            barrier_source = 'parsed-neb'
            parsed_count += 1
        events.append(
            {
                'event_id': task['event_id'],
                'species': task['species'],
                'barrier_ev': barrier_value,
                'prefactor_hz': float(task['prefactor_hz']),
                'pathway': task['pathway'],
                'barrier_seed_ev': float(task['barrier_ev']),
                'barrier_source': barrier_source,
                'reverse_barrier_ev': neb_parse.get('barrier_reverse_ev'),
                'reaction_distance': neb_parse.get('reaction_distance'),
                'neb_converged_step': neb_parse.get('converged_step'),
            }
        )

    return {
        'source': 'mietclaw-autonomy-neb-workflow',
        'material_system': campaign['material_name'],
        'attempt_frequency_hz': campaign['attempt_frequency_hz'],
        'events': events,
        'metadata': {
            'workflow_kind': 'lammps-ci-neb',
            'reference_energy_ev': preflight.get('reference_energy_ev'),
            'reference_preflight': preflight,
            'potential_path': campaign['potential_path'],
            'neb_images': campaign['neb']['images'],
            'neb_variant': campaign['neb']['variant'],
            'run_hint': campaign['neb']['run_hint'],
            'species_workflows': species_files,
            'species_runs': species_runs,
            'barrier_source_mode': determine_barrier_source_mode(species_runs),
            'parsed_species_count': parsed_count,
            'species_count': len(campaign['species_tasks']),
            'notes': campaign['notes'],
            'generated_by': 'mietclaw autonomy layer',
        },
    }


def run_generated_neb_workflow(campaign: Dict[str, Any]) -> int:
    workdir = Path.cwd()
    workflow_root = workdir / 'neb_campaign'
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / 'neb_campaign.json').write_text(json.dumps(campaign, indent=2, ensure_ascii=False), encoding='utf-8')
    (workflow_root / 'README.generated.md').write_text(render_neb_campaign_readme(campaign), encoding='utf-8')

    species_files = write_species_campaign(workflow_root, campaign)
    preflight = run_reference_preflight(workflow_root / 'reference', campaign)
    species_runs = execute_species_neb_campaign(workflow_root, campaign)
    barrier_output_path = workdir / 'barriers.generated.json'
    payload = build_barrier_payload(campaign, preflight, species_files, species_runs)
    barrier_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Generated {barrier_output_path.name}')
    print(f'Drafted NEB workflow in {workflow_root}')
    if preflight.get('reference_energy_ev') is not None:
        print(f"Reference total energy: {preflight['reference_energy_ev']:.6f} eV")
    else:
        print(f"Reference preflight status: {preflight.get('status')}")

    for event in payload['events']:
        source = event.get('barrier_source', 'unknown')
        print(f"{event['species']} barrier: {event['barrier_ev']:.6f} eV ({source})")

    if payload['metadata']['barrier_source_mode'] != 'parsed-neb':
        print(
            'Barrier extraction degraded: one or more species fell back to seed barriers. '
            'Check neb.execution.json under each species directory for details.',
            file=sys.stderr,
        )
    return 0


def render_md_neb_workflow_script(campaign: Dict[str, Any], project_root: Path) -> str:
    campaign_literal = json.dumps(campaign, indent=2, ensure_ascii=False)
    return f"""# mietclaw-autonomy-neb-workflow
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path({project_root.as_posix()!r})
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from miet_claw.neb_runtime import run_generated_neb_workflow

CAMPAIGN = json.loads({json.dumps(campaign_literal)})


def main():
    return run_generated_neb_workflow(CAMPAIGN)


if __name__ == '__main__':
    raise SystemExit(main())
"""
