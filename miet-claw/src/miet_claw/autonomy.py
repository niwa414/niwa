import asyncio
import copy
import importlib
import importlib.util
import json
import os
import re
import stat
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .planner import build_plan_payload
from .specs import load_job_spec
from .transforms import render_kmc_input
from .executor import run_job
from . import neb_runtime
from .autonomy_neb import DEFAULT_BARRIER_LIBRARY, build_neb_campaign, extract_neb_images, infer_pathway_label


DEFAULT_NATIVE_TEMPLATE = 'md_to_kmc_chain_native.json'
CLAUDE_AUTH_ENV_VARS = (
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_VERTEX',
    'CLAUDE_CODE_USE_FOUNDRY',
)


class AutonomyError(RuntimeError):
    pass


ProgressCallback = Callable[[str, Dict[str, Any]], None]
CancelCheck = Callable[[str, Path, Dict[str, Any]], Optional[Any]]
CheckpointCallback = Callable[[str, Dict[str, Any]], None]


def emit_progress(callback: Optional[ProgressCallback], stage: str, **payload: Any) -> None:
    if callback is None:
        return
    callback(stage, payload)


def _load_run_recovery(run_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if run_dir is None:
        return None
    state_path = run_dir / 'state.json'
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None
    job_payload = state.get('job') if isinstance(state, dict) else {}
    checkpoints = state.get('checkpoints') if isinstance(state, dict) else []
    return {
        'status': state.get('status'),
        'resume_summary': job_payload.get('resume_summary'),
        'recovery_plan': job_payload.get('recovery_plan'),
        'checkpoint_count': len(checkpoints) if isinstance(checkpoints, list) else 0,
    }


def detect_project_root(start: Optional[Path] = None) -> Path:
    cursor = Path(start or __file__).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / 'src' / 'miet_claw' / 'cli.py').exists() and (candidate / 'examples' / 'jobs').exists():
            return candidate
    raise AutonomyError('Could not detect mietclaw project root')


def slugify(value: str) -> str:
    return re.sub(r'(^_+|_+$)', '', re.sub(r'[^a-z0-9]+', '_', value.strip().lower())) or 'mietclaw_job'


def supports_claude_sdk() -> bool:
    has_sdk = importlib.util.find_spec('claude_agent_sdk') is not None
    has_auth = any(os.environ.get(name) for name in CLAUDE_AUTH_ENV_VARS)
    return has_sdk and has_auth


def resolve_provider(provider: str) -> str:
    normalized = (provider or 'auto').strip().lower()
    if normalized == 'auto':
        return 'claude-sdk' if supports_claude_sdk() else 'local-heuristic'
    if normalized == 'claude':
        if not supports_claude_sdk():
            raise AutonomyError(
                'Claude Agent SDK mode requires `pip install claude-agent-sdk` and one of '
                f'{", ".join(CLAUDE_AUTH_ENV_VARS)}.'
            )
        return 'claude-sdk'
    if normalized in {'local', 'local-heuristic'}:
        return 'local-heuristic'
    raise AutonomyError(f'Unsupported autonomy provider: {provider}')


def load_template_catalog(project_root: Path) -> List[Dict[str, Any]]:
    templates_root = project_root / 'examples' / 'jobs'
    catalog: List[Dict[str, Any]] = []
    for spec_path in sorted(templates_root.glob('*.json')):
        spec = load_job_spec(str(spec_path))
        catalog.append(
            {
                'path': spec_path,
                'file_name': spec_path.name,
                'mode': spec['mode'],
                'job_id': spec['job_id'],
                'material_name': spec.get('material_system', {}).get('name', spec['job_id']),
                'is_native': '_native' in spec_path.name,
                'spec': spec,
            }
        )

    def sort_key(item: Dict[str, Any]) -> tuple:
        score = 0
        if item['file_name'] != DEFAULT_NATIVE_TEMPLATE:
            score += 2
        if not item['is_native']:
            score += 1
        return (score, item['file_name'])

    return sorted(catalog, key=sort_key)


def infer_mode_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    if any(token in text for token in ['kmc only', 'only kmc', '只做kmc', '只做 kmc', '仅做kmc', '仅做 kmc']):
        return 'kmc_only'
    if any(token in text for token in ['md only', 'only md', '只做md', '只做 md', '仅做md', '仅做 md']):
        return 'md_only'
    if any(token in text for token in ['md to kmc', 'md->kmc', 'md → kmc', 'multiscale', 'multi-scale', '多尺度', '能垒', 'barrier']):
        return 'md_to_kmc_chain'
    return 'md_to_kmc_chain'


def choose_template(
    catalog: List[Dict[str, Any]],
    prompt: str,
    mode_hint: Optional[str] = None,
    template_path: Optional[str] = None,
) -> Dict[str, Any]:
    if template_path:
        absolute = Path(template_path).resolve()
        for item in catalog:
            if item['path'].resolve() == absolute:
                return item
        raise AutonomyError(f'Template not found in catalog: {template_path}')

    desired_mode = mode_hint or infer_mode_from_prompt(prompt)
    for item in catalog:
        if item['mode'] == desired_mode:
            return item
    return catalog[0]


def extract_temperature(prompt: str) -> Optional[float]:
    match = re.search(r'(\d+(?:\.\d+)?)\s*[kK]\b', prompt)
    if match:
        return float(match.group(1))
    return None


def extract_owner(prompt: str) -> Optional[str]:
    patterns = [
        r'owner\s*[:=]\s*([A-Za-z0-9_-]+)',
        r'owned\s+by\s+([A-Za-z0-9_-]+)',
        r'负责人\s*[:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)',
        r'归属\s*[:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_material_name(prompt: str) -> str:
    quoted = re.search(r'["“](.+?)["”]', prompt)
    if quoted:
        return quoted.group(1).strip()
    lower = prompt.lower()
    if (
        any(token in prompt or token in lower for token in ['lammps', 'neb', 'ci-neb', '迁移能垒', '扩散能垒', 'barrier'])
        and any(token in prompt or token in lower for token in ['kmc', 'misa-kmc', '继续模拟', 'continue kmc', '工作流'])
    ):
        return 'LAMMPS to KMC migration barrier workflow'
    first_line = prompt.strip().splitlines()[0].strip()
    if len(first_line) > 80:
        first_line = f'{first_line[:77]}...'
    return first_line or 'Autonomy-generated material system'


def extract_barrier_map(prompt: str, species_order: List[str]) -> Dict[str, float]:
    barriers: Dict[str, float] = {}
    for species in species_order:
        patterns = [
            rf'\b{re.escape(species)}\b\s*barrier\s*(?:=|is)?\s*(\d+(?:\.\d+)?)',
            rf'\b{re.escape(species)}\b\s*(?:=|:)\s*(\d+(?:\.\d+)?)\s*(?:eV|ev)?',
            rf'{re.escape(species)}.*?(\d+(?:\.\d+)?)\s*(?:eV|ev)',
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                barriers[species] = float(match.group(1))
                break
    return barriers


def default_barrier_map(spec: Dict[str, Any]) -> Dict[str, float]:
    template = spec.get('kmc', {}).get('template', {})
    precomputed = template.get('precomputed_barriers') or {}
    if precomputed:
        return {str(key): float(value) for key, value in precomputed.items()}

    species_order = template.get('species_order') or list(DEFAULT_BARRIER_LIBRARY.keys())
    fallback = {}
    for species in species_order:
        fallback[species] = float(DEFAULT_BARRIER_LIBRARY.get(species, 0.60))
    return fallback


def build_notes_markdown(prompt: str, report: Dict[str, Any]) -> str:
    lines = [
        '# mietclaw autonomy draft',
        '',
        '## Prompt',
        '',
        prompt.strip(),
        '',
        '## Selected template',
        '',
        f"- file: `{report['selected_template']['file_name']}`",
        f"- mode: `{report['mode']}`",
        f"- provider: `{report['provider_used']}`",
    ]

    assumptions = report.get('assumptions') or []
    warnings = report.get('warnings') or []
    generated = report.get('generated_files') or {}

    if assumptions:
        lines.extend(['', '## Assumptions', ''])
        lines.extend([f'- {item}' for item in assumptions])

    if warnings:
        lines.extend(['', '## Warnings', ''])
        lines.extend([f'- {item}' for item in warnings])

    lines.extend(['', '## Generated files', ''])
    for label, file_path in generated.items():
        if file_path:
            lines.append(f'- {label}: `{file_path}`')

    return '\n'.join(lines) + '\n'


render_reference_preflight_input = neb_runtime.render_reference_preflight_input
render_neb_species_relax_input = neb_runtime.render_neb_species_relax_input
render_neb_species_final_coords = neb_runtime.render_neb_species_final_coords
render_neb_species_input = neb_runtime.render_neb_species_input
render_neb_campaign_readme = neb_runtime.render_neb_campaign_readme


def render_md_neb_workflow_script(campaign: Dict[str, Any]) -> str:
    return neb_runtime.render_md_neb_workflow_script(campaign, detect_project_root())


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if executable:
        current = path.stat().st_mode
        path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def render_shell_script(project_root: Path, command: List[str]) -> str:
    command_literal = ' '.join(json.dumps(part) for part in command)
    return textwrap.dedent(
        f'''\
        #!/bin/zsh
        set -euo pipefail
        cd {json.dumps(str(project_root))}
        export PYTHONPATH={json.dumps(str(project_root / 'src'))}:${{PYTHONPATH:-}}
        exec {command_literal} "$@"
        '''
    )


def materialize_support_files(
    spec: Dict[str, Any],
    workspace_dir: Path,
    project_root: Path,
    barrier_map: Dict[str, float],
    prompt: str,
) -> Dict[str, Optional[str]]:
    generated: Dict[str, Optional[str]] = {}

    if 'md' in spec:
        md_dir = workspace_dir / 'md'
        md_dir.mkdir(parents=True, exist_ok=True)
        campaign = build_neb_campaign(
            spec,
            spec.get('material_system', {}).get('name', spec['job_id']),
            barrier_map,
            project_root,
            prompt,
        )

        neb_preview_root = md_dir / 'neb'
        campaign_path = neb_preview_root / 'neb_campaign.json'
        readme_path = neb_preview_root / 'README.generated.md'
        write_text(campaign_path, json.dumps(campaign, indent=2, ensure_ascii=False))
        write_text(readme_path, neb_runtime.render_neb_campaign_readme(campaign))
        for task in campaign['species_tasks']:
            species_dir = neb_preview_root / task['slug']
            write_text(species_dir / 'in.relax.initial.lmp', neb_runtime.render_neb_species_relax_input(campaign, task, 'initial'))
            write_text(species_dir / 'in.relax.final.lmp', neb_runtime.render_neb_species_relax_input(campaign, task, 'final'))
            write_text(species_dir / 'coords.final', neb_runtime.render_neb_species_final_coords(task))
            write_text(species_dir / 'in.neb.ci.lmp', neb_runtime.render_neb_species_input(campaign, task))

        adapter_path = md_dir / 'generated_md_neb_workflow.py'
        write_text(
            adapter_path,
            neb_runtime.render_md_neb_workflow_script(campaign, project_root),
        )
        generated['md_script'] = str(adapter_path)
        generated['md_neb_preview_root'] = str(neb_preview_root)
        generated['md_neb_campaign'] = str(campaign_path)
        generated['md_neb_readme'] = str(readme_path)
        if campaign['species_tasks']:
            primary_task = campaign['species_tasks'][0]
            generated['md_neb_primary_input'] = str(neb_preview_root / primary_task['slug'] / 'in.neb.ci.lmp')

        original_command = list(spec['md'].get('command') or [])
        if original_command and str(original_command[-1]).endswith('.py'):
            original_command[-1] = str(adapter_path)
            spec['md']['command'] = original_command
        else:
            spec['md']['command'] = ['python3', str(adapter_path)]
        working_dir = (md_dir / 'workspace').resolve()
        working_dir.mkdir(parents=True, exist_ok=True)
        spec['md']['working_dir'] = str(working_dir)
        spec['md']['barriers_source'] = 'barriers.generated.json'
        spec['md'].setdefault('environment', {})

        dry_run_barrier_payload = {
            'source': 'mietclaw-autonomy-dry-run-seed',
            'material_system': spec.get('material_system', {}).get('name', spec['job_id']),
            'attempt_frequency_hz': 1.0e13,
            'events': [
                {
                    'event_id': f'vacancy_jump_{species.lower()}_1nn',
                    'species': species,
                    'barrier_ev': float(barrier_map[species]),
                    'prefactor_hz': 1.0e13,
                    'pathway': 'autonomy dry-run seed',
                }
                for species in barrier_map
            ],
            'metadata': {
                'reference_energy_ev': None,
                'workflow_kind': 'lammps-ci-neb-draft',
                'neb_images': campaign['neb']['images'],
                'barrier_source_mode': 'seed-fallback',
                'parsed_species_count': 0,
                'species_count': len(campaign['species_tasks']),
                'generated_by': 'mietclaw autonomy layer',
                'purpose': 'dry-run validation seed; real MD execution overwrites this file',
            },
        }
        write_text(working_dir / 'barriers.generated.json', json.dumps(dry_run_barrier_payload, indent=2, ensure_ascii=False))

    if 'kmc' in spec:
        kmc_dir = workspace_dir / 'kmc'
        kmc_dir.mkdir(parents=True, exist_ok=True)
        preview_path = kmc_dir / 'generated_kmc.preview.in'
        template = copy.deepcopy(spec['kmc']['template'])
        cluster_path = template.get('cluster_xyz')
        cluster_ref = Path(cluster_path).name if cluster_path else None
        preview_text = render_kmc_input(template, barrier_map, float(spec['kmc']['temperature_k']), cluster_ref=cluster_ref)
        write_text(preview_path, preview_text)
        generated['kmc_preview_input'] = str(preview_path)

    scripts_dir = workspace_dir / 'scripts'
    plan_script = scripts_dir / 'plan.sh'
    dry_run_script = scripts_dir / 'dry_run.sh'
    run_script = scripts_dir / 'run.sh'
    generated['plan_script'] = str(plan_script)
    generated['dry_run_script'] = str(dry_run_script)
    generated['run_script'] = str(run_script)
    return generated


def _apply_fact_overrides(
    spec: Dict[str, Any],
    prompt: str,
    job_id: Optional[str],
    material_name: Optional[str],
) -> Dict[str, Any]:
    updated = copy.deepcopy(spec)
    template = updated.get('kmc', {}).get('template', {})
    species_order = list(template.get('species_order') or DEFAULT_BARRIER_LIBRARY.keys())
    barriers = extract_barrier_map(prompt, species_order)
    temperature = extract_temperature(prompt)
    owner = extract_owner(prompt)
    inferred_material = material_name or extract_material_name(prompt)

    assumptions: List[str] = []
    warnings: List[str] = []

    updated['job_id'] = slugify(job_id or inferred_material or updated['job_id'])
    updated.setdefault('material_system', {})
    updated['material_system']['name'] = inferred_material
    if owner:
        updated['material_system']['owner'] = owner
    elif updated['material_system'].get('owner'):
        assumptions.append('Owner was inherited from the selected template.')

    if 'kmc' in updated and temperature is not None:
        updated['kmc']['temperature_k'] = temperature
    elif 'kmc' in updated:
        assumptions.append(f"Temperature was inherited from the selected template ({updated['kmc']['temperature_k']} K).")

    if updated['mode'] == 'kmc_only':
        if barriers:
            updated['kmc']['template']['precomputed_barriers'] = {key: float(value) for key, value in barriers.items()}
        else:
            defaults = default_barrier_map(updated)
            updated['kmc']['template']['precomputed_barriers'] = defaults
            assumptions.append('Prompt did not specify KMC barriers; using the template/default vacancy diffusion barrier map.')
    elif barriers:
        assumptions.append('Prompt provided barrier hints; they will seed the generated NEB / CI-NEB workflow and will only be used if real NEB extraction degrades.')
    else:
        assumptions.append('Prompt did not specify MD barriers; the agent will attempt real NEB extraction and will fall back to the template/default barrier seeds only if execution degrades.')

    return {
        'spec': updated,
        'facts': {
            'temperature_k': temperature,
            'owner': owner,
            'material_name': inferred_material,
            'barrier_hints_ev': barriers,
        },
        'assumptions': assumptions,
        'warnings': warnings,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        raise AutonomyError('Claude SDK output did not contain a JSON object.')
    return json.loads(candidate)


async def _draft_with_claude_agent_sdk_async(prompt: str, template_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    sdk = importlib.import_module('claude_agent_sdk')
    query = getattr(sdk, 'query')
    ClaudeAgentOptions = getattr(sdk, 'ClaudeAgentOptions')

    instruction = textwrap.dedent(
        f'''\
        You are drafting a mietclaw multiscale materials simulation job.
        Choose the best template from this catalog and return ONLY one JSON object with keys:
        template_file, mode, job_id, material_name, owner, temperature_k, barriers_ev, assumptions, warnings.

        Template catalog:
        {json.dumps(template_summary, indent=2, ensure_ascii=False)}

        Rules:
        - Prefer native templates when possible.
        - If the prompt is vague, keep the template defaults and explain assumptions.
        - barriers_ev should be a species->barrier map only when the prompt gives values or when you must provide a draft guess.
        - Do not return markdown outside the JSON object.

        User task:
        {prompt}
        '''
    )

    transcript: List[str] = []
    async for message in query(
        prompt=instruction,
        options=ClaudeAgentOptions(
            allowed_tools=['Read', 'Glob'],
        ),
    ):
        if hasattr(message, 'result') and getattr(message, 'result', None):
            transcript.append(str(message.result))
        if hasattr(message, 'content'):
            for block in getattr(message, 'content', []) or []:
                text = getattr(block, 'text', None)
                if text:
                    transcript.append(str(text))

    return _extract_json_object('\n'.join(transcript))


def draft_with_claude_agent_sdk(prompt: str, catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    template_summary = [
        {
            'file_name': item['file_name'],
            'mode': item['mode'],
            'material_name': item['material_name'],
            'is_native': item['is_native'],
        }
        for item in catalog
    ]
    return asyncio.run(_draft_with_claude_agent_sdk_async(prompt, template_summary))


def materialize_autonomy_workspace(
    prompt: str,
    project_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    provider: str = 'auto',
    mode_hint: Optional[str] = None,
    template_path: Optional[str] = None,
    job_id: Optional[str] = None,
    material_name: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    emit_progress(progress_callback, 'autonomy.workspace.start', provider=provider)
    root = detect_project_root(Path(project_root) if project_root else None)
    catalog = load_template_catalog(root)
    provider_used = resolve_provider(provider)
    emit_progress(progress_callback, 'autonomy.provider.resolved', provider_used=provider_used)

    chosen_template: Optional[Dict[str, Any]] = None
    claude_payload: Optional[Dict[str, Any]] = None
    if provider_used == 'claude-sdk':
        claude_payload = draft_with_claude_agent_sdk(prompt, catalog)
        template_file = claude_payload.get('template_file')
        chosen_template = choose_template(catalog, prompt, mode_hint=claude_payload.get('mode') or mode_hint, template_path=str(root / 'examples' / 'jobs' / template_file) if template_file else template_path)
        job_id = job_id or claude_payload.get('job_id')
        material_name = material_name or claude_payload.get('material_name')
        mode_hint = claude_payload.get('mode') or mode_hint
    else:
        chosen_template = choose_template(catalog, prompt, mode_hint=mode_hint, template_path=template_path)

    if chosen_template is None:
        raise AutonomyError('Failed to choose a template for the autonomy draft.')
    emit_progress(progress_callback, 'autonomy.template.selected', template=chosen_template['file_name'], mode=chosen_template['mode'])

    draft = _apply_fact_overrides(chosen_template['spec'], prompt, job_id=job_id, material_name=material_name)
    spec = draft['spec']
    assumptions = list(draft['assumptions'])
    warnings = list(draft['warnings'])
    facts = draft['facts']

    if claude_payload:
        assumptions.extend([str(item) for item in claude_payload.get('assumptions') or []])
        warnings.extend([str(item) for item in claude_payload.get('warnings') or []])
        barrier_hints = claude_payload.get('barriers_ev') or {}
        if barrier_hints and spec['mode'] == 'kmc_only':
            spec['kmc']['template']['precomputed_barriers'] = {key: float(value) for key, value in barrier_hints.items()}
        if barrier_hints:
            facts['barrier_hints_ev'] = {key: float(value) for key, value in barrier_hints.items()}
        if claude_payload.get('temperature_k') is not None and 'kmc' in spec:
            spec['kmc']['temperature_k'] = float(claude_payload['temperature_k'])

    barrier_map = dict(default_barrier_map(spec))
    barrier_map.update({key: float(value) for key, value in (facts.get('barrier_hints_ev') or {}).items()})

    workspace_base = Path(workspace_root).resolve() if workspace_root else (root / '.autonomy').resolve()
    workspace_base.mkdir(parents=True, exist_ok=True)
    workspace_dir = workspace_base / f"{spec['job_id']}-{int(asyncio.get_event_loop_policy().get_event_loop().time() * 1000) if False else int(__import__('time').time() * 1000)}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    generated_files = materialize_support_files(spec, workspace_dir, root, barrier_map, prompt)
    emit_progress(progress_callback, 'autonomy.files.materialized', workspace_dir=str(workspace_dir))

    spec_path = workspace_dir / 'job_spec.generated.json'
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding='utf-8')

    load_job_spec(str(spec_path))
    generated_files['job_spec'] = str(spec_path)

    notes_path = workspace_dir / 'autonomy_notes.md'
    prompt_path = workspace_dir / 'prompt.txt'
    report_path = workspace_dir / 'autonomy_report.json'
    prompt_path.write_text(prompt.strip() + '\n', encoding='utf-8')
    generated_files['prompt'] = str(prompt_path)

    plan_script_path = Path(generated_files['plan_script'])
    dry_run_script_path = Path(generated_files['dry_run_script'])
    run_script_path = Path(generated_files['run_script'])
    write_text(plan_script_path, render_shell_script(root, ['python3', '-m', 'miet_claw.cli', 'plan', str(spec_path)]), executable=True)
    write_text(dry_run_script_path, render_shell_script(root, ['python3', '-m', 'miet_claw.cli', 'run', str(spec_path), '--output-dir', str(root / 'runs-autonomy-validation'), '--dry-run']), executable=True)
    write_text(run_script_path, render_shell_script(root, ['python3', '-m', 'miet_claw.cli', 'run', str(spec_path), '--output-dir', str(root / 'runs')]), executable=True)

    report = {
        'provider_requested': provider,
        'provider_used': provider_used,
        'workspace_dir': str(workspace_dir),
        'job_id': spec['job_id'],
        'material_name': spec.get('material_system', {}).get('name'),
        'mode': spec['mode'],
        'selected_template': {
            'file_name': chosen_template['file_name'],
            'path': str(chosen_template['path']),
            'is_native': chosen_template['is_native'],
        },
        'facts': facts,
        'assumptions': assumptions,
        'warnings': warnings,
        'generated_files': generated_files,
        'job_spec': spec,
    }
    notes_path.write_text(build_notes_markdown(prompt, report), encoding='utf-8')
    generated_files['notes'] = str(notes_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    generated_files['report'] = str(report_path)
    emit_progress(progress_callback, 'autonomy.workspace.ready', job_id=spec['job_id'], workspace_dir=str(workspace_dir))
    return report


def run_autonomy_job(
    prompt: str,
    project_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    provider: str = 'auto',
    mode_hint: Optional[str] = None,
    template_path: Optional[str] = None,
    job_id: Optional[str] = None,
    material_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    dry_run_only: bool = False,
    resume_existing: bool = False,
    overwrite_existing: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
) -> Dict[str, Any]:
    emit_progress(progress_callback, 'autonomy.run.start', provider=provider, dry_run_only=dry_run_only)
    report = materialize_autonomy_workspace(
        prompt=prompt,
        project_root=project_root,
        workspace_root=workspace_root,
        provider=provider,
        mode_hint=mode_hint,
        template_path=template_path,
        job_id=job_id,
        material_name=material_name,
        progress_callback=progress_callback,
    )

    root = detect_project_root(Path(project_root) if project_root else None)
    spec_path = report['generated_files']['job_spec']
    plan_payload = build_plan_payload(load_job_spec(spec_path))
    emit_progress(progress_callback, 'autonomy.plan.ready', steps=len(plan_payload))

    validation_root = Path(report['workspace_dir']) / 'validation_runs'
    validation_root.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, 'autonomy.validation.start', output_dir=str(validation_root))
    validation_run_dir = run_job(
        spec_path,
        str(validation_root),
        dry_run=True,
        overwrite_existing=True,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        checkpoint_callback=checkpoint_callback,
    )
    emit_progress(progress_callback, 'autonomy.validation.complete', run_dir=str(validation_run_dir))

    final_run_dir = None
    if not dry_run_only:
        final_output_dir = Path(output_dir).resolve() if output_dir else (root / 'runs').resolve()
        final_output_dir.mkdir(parents=True, exist_ok=True)
        emit_progress(
            progress_callback,
            'autonomy.final_run.start',
            output_dir=str(final_output_dir),
            resume_existing=resume_existing,
            overwrite_existing=overwrite_existing,
        )
        final_run_dir = run_job(
            spec_path,
            str(final_output_dir),
            dry_run=False,
            resume=resume_existing,
            overwrite_existing=overwrite_existing,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            checkpoint_callback=checkpoint_callback,
        )
        emit_progress(progress_callback, 'autonomy.final_run.complete', run_dir=str(final_run_dir))

    execution = {
        'plan': plan_payload,
        'validation_run_dir': str(validation_run_dir),
        'final_run_dir': str(final_run_dir) if final_run_dir else None,
        'dry_run_only': dry_run_only,
        'resume_existing': resume_existing,
        'overwrite_existing': overwrite_existing,
        'validation_recovery': _load_run_recovery(validation_run_dir),
        'final_recovery': _load_run_recovery(final_run_dir),
    }

    report['execution'] = execution
    report_path = Path(report['generated_files']['report'])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    emit_progress(progress_callback, 'autonomy.run.complete', final_run_dir=execution['final_run_dir'])
    return report
