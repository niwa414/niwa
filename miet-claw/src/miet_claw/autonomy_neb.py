"""NEB campaign drafting helpers for autonomy workspaces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_BARRIER_LIBRARY = {
    'Fe': 0.65,
    'Cu': 0.56,
    'Ni': 0.55,
}


def _slugify(value: str) -> str:
    return re.sub(r'(^_+|_+$)', '', re.sub(r'[^a-z0-9]+', '_', value.strip().lower())) or 'mietclaw_job'


def extract_neb_images(prompt: str) -> Optional[int]:
    match = re.search(r'(\d+)\s*(?:images|replicas|replica)\b', prompt, flags=re.IGNORECASE)
    if not match:
        return None
    return max(3, int(match.group(1)))


def infer_pathway_label(prompt: str) -> str:
    text = prompt.lower()
    prefix = 'autonomy-generated'
    if 'vacancy' in text or '空位' in text:
        if '1nn' in text or 'nearest neighbor' in text or '最近邻' in text:
            return f'{prefix} 1nn vacancy hop'
        return f'{prefix} vacancy hop'
    if 'interstitial' in text or '间隙' in text:
        return f'{prefix} interstitial hop'
    if 'surface' in text or '表面' in text:
        return f'{prefix} surface diffusion path'
    return f'{prefix} migration path'


def build_neb_campaign(
    spec: Dict[str, Any],
    material_name: str,
    barrier_map: Dict[str, float],
    project_root: Path,
    prompt: str,
) -> Dict[str, Any]:
    kmc_template = spec.get('kmc', {}).get('template', {})
    species_order = list(kmc_template.get('species_order') or barrier_map.keys() or DEFAULT_BARRIER_LIBRARY.keys())
    masses = kmc_template.get('species_masses') or {species: 55.0 for species in species_order}
    lattice_constant = float((kmc_template.get('lattice') or {}).get('constant', 2.86))
    potential_assets = kmc_template.get('potential_assets') or []
    default_potential = project_root / 'crystalkmc-fix-diffusion-coef' / 'TEST' / '1Compatibility' / 'case2' / 'FeCuNi.eam.alloy'
    potential_path = Path(potential_assets[0]) if potential_assets else default_potential
    region_cells = int(float((kmc_template.get('region_block') or [0, 4, 0, 4, 0, 4])[1])) if kmc_template.get('region_block') else 4
    neb_images = extract_neb_images(prompt) or 7
    hop_delta = round(lattice_constant * 0.5, 6)
    vacancy_coords = [hop_delta, hop_delta, hop_delta]

    campaign = {
        'material_name': material_name,
        'species_order': species_order,
        'masses': masses,
        'lattice_constant': lattice_constant,
        'reference_region_cells': max(4, min(region_cells, 8)),
        'potential_path': str(potential_path),
        'pair_style': 'eam/alloy',
        'pair_coeff_species': species_order,
        'attempt_frequency_hz': 1.0e13,
        'barriers_ev': {species: float(barrier_map[species]) for species in species_order if species in barrier_map},
        'pathway': infer_pathway_label(prompt),
        'notes': [
            'This workflow drafts and, when LAMMPS plus MPI are available, executes endpoint-relaxation and CI-NEB input decks for each species-specific hop.',
            'Species-specific barriers are extracted from the final EBF value in the LAMMPS NEB terse output. If execution is unavailable or fails, the chain falls back to prompt hints or template defaults.',
            'The current autonomous MD model uses a pure-species BCC lattice with one vacancy and frozen background atoms to keep local runs stable; mixed local environments are the next upgrade.',
        ],
        'neb': {
            'variant': 'ci-neb',
            'images': neb_images,
            'spring_constant': 1.0,
            'parallel_style': 'ideal',
            'path_iterations': 800,
            'climb_iterations': 400,
            'print_every': 50,
            'energy_tolerance': 0.0,
            'force_tolerance': 0.001,
            'timestep': 0.02,
            'min_style': 'fire',
            'run_hint': f'lmp -partition {neb_images}x1 -in in.neb.ci.lmp',
        },
        'species_tasks': [],
    }

    for index, species in enumerate(species_order, start=1):
        if species not in barrier_map:
            continue
        target_coords = [hop_delta, hop_delta, hop_delta]
        campaign['species_tasks'].append(
            {
                'species': species,
                'slug': _slugify(species),
                'event_id': f'vacancy_jump_{species.lower()}_1nn',
                'pathway': f"{campaign['pathway']} ({species})",
                'barrier_ev': float(barrier_map[species]),
                'prefactor_hz': 1.0e13,
                'moving_atom_id': 1,
                'moving_type': index,
                'target_coords': target_coords,
                'vacancy_coords': vacancy_coords,
                'target_coords_comment': 'draft 1nn endpoint derived from the lattice constant',
            }
        )

    return campaign
