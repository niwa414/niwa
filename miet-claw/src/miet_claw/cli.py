import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .autonomy import materialize_autonomy_workspace, run_autonomy_job
from .bridge import run_kmc_lookup_bridge
from .chat import (
    chat_with_local_model,
    get_local_model_status,
    get_log_excerpt,
    inspect_run,
    list_artifacts,
    list_runs,
    run_chat,
    run_chat_once_payload,
)
from .local_profile import get_local_model_settings, get_runtime_settings
from .mcp_server import main as run_mcp_server
from .moire_runtime import (
    run_moire_diffusion_sweep,
    run_moire_event_compare,
    run_moire_lammps_case,
    run_moire_lammps_to_kmc,
    run_moire_repo_kmc,
)
from .shell_runtime import DOMAIN_TOOLS, collect_runtime_doctor, dump_json, format_runtime_doctor, format_shell_tools
from .executor import run_job
from .planner import build_plan_payload
from .runtime.router_eval import run_router_golden_eval
from .runtime.runtime_eval import run_runtime_health_golden_eval
from .specs import load_job_spec


def cmd_plan(args: argparse.Namespace) -> int:
    spec = load_job_spec(args.job_spec)
    plan = build_plan_payload(spec)
    print(json.dumps({"job_id": spec["job_id"], "mode": spec["mode"], "plan": plan}, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = run_job(
        args.job_spec,
        args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite_existing=args.overwrite_existing,
    )
    print(str(run_dir))
    return 0


def _load_prompt_from_args(args: argparse.Namespace) -> str:
    if getattr(args, 'prompt_file', None):
        return Path(args.prompt_file).read_text(encoding='utf-8')
    if args.prompt:
        return args.prompt
    raise SystemExit('Provide either a prompt argument or --prompt-file.')


def cmd_autonomy_draft(args: argparse.Namespace) -> int:
    payload = materialize_autonomy_workspace(
        prompt=_load_prompt_from_args(args),
        project_root=args.project_root,
        workspace_root=args.workspace_root,
        provider=args.provider,
        mode_hint=args.mode,
        template_path=args.template_path,
        job_id=args.job_id,
        material_name=args.material_name,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_autonomy_run(args: argparse.Namespace) -> int:
    payload = run_autonomy_job(
        prompt=_load_prompt_from_args(args),
        project_root=args.project_root,
        workspace_root=args.workspace_root,
        provider=args.provider,
        mode_hint=args.mode,
        template_path=args.template_path,
        job_id=args.job_id,
        material_name=args.material_name,
        output_dir=args.output_dir,
        dry_run_only=args.dry_run_only,
        resume_existing=args.resume_existing,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    once_prompt = args.once
    if args.json:
        if not once_prompt:
            once_prompt = _load_prompt_from_args(args)
        history_messages = None
        if getattr(args, 'history_file', None):
            history_messages = json.loads(Path(args.history_file).read_text(encoding='utf-8'))
        payload = run_chat_once_payload(
            project_root=args.project_root,
            workspace_root=args.workspace_root,
            output_dir=args.output_dir,
            provider=args.provider,
            mode_hint=args.mode,
            model=args.model,
            prompt=once_prompt,
            history_messages=history_messages,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    return run_chat(
        project_root=args.project_root,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        provider=args.provider,
        mode_hint=args.mode,
        model=args.model,
        once=args.once,
        ui=args.ui,
    )


def cmd_runs(args: argparse.Namespace) -> int:
    print(json.dumps({"runs": list_runs(Path(args.output_dir).resolve(), limit=args.limit)}, indent=2, ensure_ascii=False))
    return 0


def _resolve_run_path(output_dir: str, run_dir: Optional[str]) -> Path:
    output_root = Path(output_dir).resolve()
    if run_dir:
        candidate = Path(run_dir).expanduser()
        if candidate.exists():
            return candidate.resolve()
        nested = output_root / run_dir
        if nested.exists():
            return nested.resolve()
        raise SystemExit(f'Run directory not found: {run_dir}')
    runs = list_runs(output_root, limit=1)
    if not runs:
        raise SystemExit('No runs found.')
    return Path(runs[0]['path']).resolve()


def cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_run(_resolve_run_path(args.output_dir, args.run_dir))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_path(args.output_dir, args.run_dir)
    print(
        json.dumps(
            {"run_dir": str(run_dir), "artifacts": list_artifacts(run_dir, limit=args.limit)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_path(args.output_dir, args.run_dir)
    info = get_log_excerpt(run_dir, target=args.target, max_lines=args.max_lines)
    print(json.dumps({"run_dir": str(run_dir), **info}, indent=2, ensure_ascii=False))
    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"bridge_{int(time.time())}").resolve())
    summary = run_kmc_lookup_bridge(
        event_json=args.event_json,
        neb_txt=args.neb_txt,
        barrier=args.barrier,
        workdir=workdir,
        validate=args.validate,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _parse_int_list_arg(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    parts = [item for item in re.split(r"[\s,，、]+", raw.strip()) if item]
    if not parts:
        return None
    return [int(item) for item in parts]


def _parse_float_list_arg(raw: Optional[str]) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [item for item in re.split(r"[\s,，、]+", raw.strip()) if item]
    if not parts:
        return None
    return [float(item) for item in parts]


def _runtime_settings() -> Dict[str, Any]:
    return get_runtime_settings(Path(__file__).resolve().parents[2])


def _configured_kmc_retries(args: argparse.Namespace, runtime: Dict[str, Any]) -> int:
    value = getattr(args, "kmc_retries", None)
    if value is None:
        return int(runtime.get("kmc_retry_attempts") or 0)
    return max(0, int(value))


def cmd_moire_run(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"moire_run_{int(time.time())}").resolve())
    runtime = _runtime_settings()
    summary = run_moire_lammps_to_kmc(
        event_json=args.event_json,
        case_dir=args.case_dir,
        workdir=workdir,
        validate=args.validate,
        conda_exec=Path(runtime["conda_exec"]),
        conda_env=runtime["conda_env"],
        neb_input=runtime["neb_input"],
        post_script=runtime["post_script"],
        mpi_procs=int(runtime["mpi_procs"]),
        kmc_seed=args.kmc_seed,
        kmc_seeds=_parse_int_list_arg(args.kmc_seeds),
        kmc_retry_attempts=_configured_kmc_retries(args, runtime),
        misa_kmc_binary=Path(runtime["kmc_binary"]),
        eam_file=Path(runtime["eam_file"]),
        kmc_temperature=float(runtime["kmc_temperature"]),
        kmc_stats_step=runtime["kmc_stats_step"],
        kmc_run_time=runtime["kmc_run_time"],
        render_ovito=bool(args.ovito),
        ovito_python=args.ovito_python,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_moire_lammps(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"moire_lammps_{int(time.time())}").resolve())
    runtime = _runtime_settings()
    summary = run_moire_lammps_case(
        event_json=args.event_json,
        case_dir=args.case_dir,
        workdir=workdir,
        conda_exec=Path(runtime["conda_exec"]),
        conda_env=runtime["conda_env"],
        neb_input=runtime["neb_input"],
        post_script=runtime["post_script"],
        mpi_procs=int(runtime["mpi_procs"]),
        render_ovito=bool(args.ovito),
        ovito_python=args.ovito_python,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_moire_kmc(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"moire_kmc_{int(time.time())}").resolve())
    runtime = _runtime_settings()
    summary = run_moire_repo_kmc(
        barrier_eV=args.barrier_eV,
        event_json=args.event_json,
        data_lmp=args.data_lmp,
        workdir=workdir,
        misa_kmc_binary=Path(runtime["kmc_binary"]),
        eam_file=Path(runtime["eam_file"]),
        temperature=float(runtime["kmc_temperature"]),
        stats_step=runtime["kmc_stats_step"],
        run_time=runtime["kmc_run_time"],
        kmc_seed=args.kmc_seed,
        kmc_seeds=_parse_int_list_arg(args.kmc_seeds),
        retry_attempts=_configured_kmc_retries(args, runtime),
        render_ovito=bool(args.ovito),
        ovito_python=args.ovito_python,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_moire_compare(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"moire_compare_{int(time.time())}").resolve())
    runtime = _runtime_settings()
    summary = run_moire_event_compare(
        case_dir=args.case_dir,
        event_jsons=args.event_jsons,
        workdir=workdir,
        validate=args.validate,
        conda_exec=Path(runtime["conda_exec"]),
        conda_env=runtime["conda_env"],
        neb_input=runtime["neb_input"],
        post_script=runtime["post_script"],
        mpi_procs=int(runtime["mpi_procs"]),
        kmc_seed=args.kmc_seed,
        kmc_seeds=_parse_int_list_arg(args.kmc_seeds),
        kmc_retry_attempts=_configured_kmc_retries(args, runtime),
        misa_kmc_binary=Path(runtime["kmc_binary"]),
        eam_file=Path(runtime["eam_file"]),
        kmc_temperature=float(runtime["kmc_temperature"]),
        kmc_stats_step=runtime["kmc_stats_step"],
        kmc_run_time=runtime["kmc_run_time"],
        run_kmc=not bool(args.lammps_only),
        render_ovito=bool(args.ovito),
        ovito_python=args.ovito_python,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_moire_diffusion_sweep(args: argparse.Namespace) -> int:
    workdir = args.workdir or str((Path(args.output_dir).resolve() / f"moire_diffusion_{int(time.time())}").resolve())
    runtime = _runtime_settings()
    temperatures = _parse_float_list_arg(args.temperatures) or list(runtime["diffusion_temperatures"])
    summary = run_moire_diffusion_sweep(
        event_json=args.event_json,
        case_dir=args.case_dir,
        workdir=workdir,
        temperatures_k=temperatures,
        validate=args.validate,
        kmc_seed=args.kmc_seed,
        kmc_seeds=_parse_int_list_arg(args.kmc_seeds),
        kmc_retry_attempts=_configured_kmc_retries(args, runtime),
        misa_kmc_binary=Path(runtime["kmc_binary"]),
        eam_file=Path(runtime["eam_file"]),
        render_ovito=bool(args.ovito),
        ovito_python=args.ovito_python,
        run_time=args.run_time or runtime["diffusion_run_time"],
        stats_step=args.stats_step or runtime["diffusion_stats_step"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    if args.json:
        print(dump_json({"tools": DOMAIN_TOOLS}))
    else:
        print(format_shell_tools())
    return 0


def cmd_router_golden_eval(args: argparse.Namespace) -> int:
    payload = run_router_golden_eval(
        args.golden_file,
        output_dir=args.output_dir,
        current_run_dir=args.current_run_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


def cmd_runtime_golden_eval(args: argparse.Namespace) -> int:
    payload = run_runtime_health_golden_eval(args.golden_file)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    status = get_local_model_status()
    payload = collect_runtime_doctor(Path(args.project_root).resolve(), local_status=status)
    if args.json:
        print(dump_json(payload))
    else:
        print(format_runtime_doctor(payload))
    checks = payload.get("checks", {})
    probes = payload.get("probes", {})
    healthy = bool(status.get("healthy")) and all(
        [
            checks.get("profile_exists"),
            checks.get("kmc_binary_exists"),
            checks.get("conda_exec_exists"),
            probes.get("lmp", {}).get("ok", False),
            probes.get("mpirun", {}).get("ok", False),
        ]
    )
    return 0 if healthy else 1


def cmd_mcp_server(args: argparse.Namespace) -> int:
    return run_mcp_server(
        [
            "--project-root",
            args.project_root,
            "--workspace-root",
            args.workspace_root,
            "--output-dir",
            args.output_dir,
            "--provider",
            args.provider,
        ]
    )


def cmd_local_status(args: argparse.Namespace) -> int:
    settings = get_local_model_settings()
    runtime = get_runtime_settings(Path(__file__).resolve().parents[2])
    status = get_local_model_status()
    payload = {
        "agent_name": settings["agent_name"],
        "profile_path": settings["profile_path"],
        "base_url": settings["base_url"],
        "preferred_model": settings["preferred_model"],
        "healthy": status.get("healthy", False),
        "resolved_default_model": status.get("default_model"),
        "available_models": status.get("models", []),
        "runtime": runtime,
    }
    if status.get("error"):
        payload["error"] = status["error"]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_local_self_check(args: argparse.Namespace) -> int:
    status = get_local_model_status()
    payload = {
        "healthy": status.get("healthy", False),
        "profile_path": status.get("profile_path"),
        "base_url": status.get("base_url"),
        "preferred_model": status.get("preferred_model"),
        "resolved_default_model": status.get("default_model"),
    }
    if not status.get("healthy"):
        payload["error"] = status.get("error") or "Local model is unavailable"
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    reply = chat_with_local_model(
        [
            {"role": "system", "content": "You are a local model connectivity probe. Reply in one short line only."},
            {"role": "user", "content": args.prompt},
        ],
        model=args.model,
        purpose="chat",
    )
    payload["probe_model"] = reply.get("model")
    payload["probe_reply"] = reply.get("content")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Miet Claw orchestration MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the execution plan for a job spec")
    plan_parser.add_argument("job_spec")
    plan_parser.set_defaults(func=cmd_plan)

    run_parser = subparsers.add_parser("run", help="Run a job spec")
    run_parser.add_argument("job_spec")
    run_parser.add_argument("--output-dir", default=str(Path("runs").resolve()))
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", action="store_true", help="Resume an existing run directory instead of failing on recorded progress.")
    run_parser.add_argument("--overwrite-existing", action="store_true", help="Delete and recreate an existing run directory.")
    run_parser.set_defaults(func=cmd_run)

    autonomy_common = argparse.ArgumentParser(add_help=False)
    autonomy_common.add_argument('prompt', nargs='?', help='Natural-language task description for autonomy mode.')
    autonomy_common.add_argument('--prompt-file', help='Path to a text file containing the autonomy prompt.')
    autonomy_common.add_argument('--provider', default='auto', help='auto | local | claude')
    autonomy_common.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
    autonomy_common.add_argument('--workspace-root', default=str(Path('.autonomy').resolve()))
    autonomy_common.add_argument('--mode', choices=['md_only', 'kmc_only', 'md_to_kmc_chain'])
    autonomy_common.add_argument('--template-path')
    autonomy_common.add_argument('--job-id')
    autonomy_common.add_argument('--material-name')

    autonomy_draft_parser = subparsers.add_parser(
        'autonomy-draft',
        parents=[autonomy_common],
        help='Draft a job spec and runnable script workspace from a natural-language task.',
    )
    autonomy_draft_parser.set_defaults(func=cmd_autonomy_draft)

    autonomy_run_parser = subparsers.add_parser(
        'autonomy-run',
        parents=[autonomy_common],
        help='Draft a workspace from a natural-language task, validate it with a dry-run, and optionally launch the real job.',
    )
    autonomy_run_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    autonomy_run_parser.add_argument('--dry-run-only', action='store_true')
    autonomy_run_parser.add_argument('--resume-existing', action='store_true')
    autonomy_run_parser.add_argument('--overwrite-existing', action='store_true')
    autonomy_run_parser.set_defaults(func=cmd_autonomy_run)

    chat_parser = subparsers.add_parser(
        'chat',
        parents=[autonomy_common],
        help='Start the terminal-first mietclaw agent session.',
    )
    chat_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    chat_parser.add_argument('--model', help='Local model id or alias, for example 27b or 122b.')
    chat_parser.add_argument('--once', help='Run one prompt and exit.')
    chat_parser.add_argument('--json', action='store_true', help='When used with --once, print a JSON payload instead of the terminal transcript.')
    chat_parser.add_argument('--history-file', help='Optional JSON file containing prior user/assistant messages for the one-shot chat call.')
    chat_parser.add_argument('--ui', choices=['auto', 'plain', 'tui'], default='plain')
    chat_parser.set_defaults(func=cmd_chat)

    runs_parser = subparsers.add_parser('runs', help='List recent runs')
    runs_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    runs_parser.add_argument('--limit', type=int, default=12)
    runs_parser.set_defaults(func=cmd_runs)

    inspect_parser = subparsers.add_parser('inspect', help='Inspect one run directory')
    inspect_parser.add_argument('run_dir', nargs='?')
    inspect_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    inspect_parser.set_defaults(func=cmd_inspect)

    artifacts_parser = subparsers.add_parser('artifacts', help='List artifacts for one run directory')
    artifacts_parser.add_argument('run_dir', nargs='?')
    artifacts_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    artifacts_parser.add_argument('--limit', type=int, default=80)
    artifacts_parser.set_defaults(func=cmd_artifacts)

    logs_parser = subparsers.add_parser('logs', help='Read a log excerpt for one run directory')
    logs_parser.add_argument('run_dir', nargs='?')
    logs_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    logs_parser.add_argument('--target', choices=['auto', 'md', 'kmc', 'summary'], default='auto')
    logs_parser.add_argument('--max-lines', type=int, default=60)
    logs_parser.set_defaults(func=cmd_logs)

    bridge_parser = subparsers.add_parser('bridge', help='Bridge event.json + neb.txt into KMC lookup files')
    bridge_parser.add_argument('event_json')
    bridge_parser.add_argument('--neb-txt')
    bridge_parser.add_argument('--barrier', type=float)
    bridge_parser.add_argument('--workdir')
    bridge_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    bridge_parser.add_argument('--validate', action='store_true')
    bridge_parser.set_defaults(func=cmd_bridge)

    moire_run_parser = subparsers.add_parser('moire-run', help='Run a MoRe LAMMPS NEB case on this computer, generate or reuse a KMC seed event, then continue into KMC')
    moire_run_parser.add_argument('case_dir')
    moire_run_parser.add_argument('--event-json')
    moire_run_parser.add_argument('--workdir')
    moire_run_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    moire_run_parser.add_argument('--validate', action='store_true')
    moire_run_parser.add_argument('--kmc-seed', type=int, help='Run repo KMC with one explicit random seed.')
    moire_run_parser.add_argument('--kmc-seeds', help='Run repo KMC multiple times with a comma-separated seed list, for example 3401,3402,3403.')
    moire_run_parser.add_argument('--kmc-retries', type=int, help='Automatically add this many retry seeds when only one KMC seed is requested.')
    moire_run_parser.add_argument('--ovito', action='store_true', help='Try to render an OVITO snapshot from each completed KMC dump.')
    moire_run_parser.add_argument('--ovito-python', help='Optional Python/ovitos executable that can import the ovito module.')
    moire_run_parser.set_defaults(func=cmd_moire_run)

    moire_lammps_parser = subparsers.add_parser('moire-lammps', help='Run only the local MoRe LAMMPS NEB case and return the parsed barrier.')
    moire_lammps_parser.add_argument('case_dir')
    moire_lammps_parser.add_argument('--event-json')
    moire_lammps_parser.add_argument('--workdir')
    moire_lammps_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    moire_lammps_parser.add_argument('--ovito', action='store_true', help='Try to render OVITO snapshots for the LAMMPS stage.')
    moire_lammps_parser.add_argument('--ovito-python', help='Optional Python/ovitos executable that can import the ovito module.')
    moire_lammps_parser.set_defaults(func=cmd_moire_lammps)

    moire_kmc_parser = subparsers.add_parser('moire-kmc', help='Run only the repo KMC stage from a MoRe barrier and event/data file.')
    moire_kmc_parser.add_argument('barrier_eV', type=float)
    moire_kmc_parser.add_argument('--event-json')
    moire_kmc_parser.add_argument('--data-lmp')
    moire_kmc_parser.add_argument('--workdir')
    moire_kmc_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    moire_kmc_parser.add_argument('--kmc-seed', type=int, help='Run repo KMC with one explicit random seed.')
    moire_kmc_parser.add_argument('--kmc-seeds', help='Run repo KMC multiple times with a comma-separated seed list, for example 3401,3402,3403.')
    moire_kmc_parser.add_argument('--kmc-retries', type=int, help='Automatically add this many retry seeds when only one KMC seed is requested.')
    moire_kmc_parser.add_argument('--ovito', action='store_true', help='Try to render an OVITO snapshot from each completed KMC dump.')
    moire_kmc_parser.add_argument('--ovito-python', help='Optional Python/ovitos executable that can import the ovito module.')
    moire_kmc_parser.set_defaults(func=cmd_moire_kmc)

    moire_compare_parser = subparsers.add_parser('moire-compare', help='Compare multiple MoRe event.json files on one case, optionally running the repo KMC stage for each event.')
    moire_compare_parser.add_argument('case_dir')
    moire_compare_parser.add_argument('event_jsons', nargs='+')
    moire_compare_parser.add_argument('--workdir')
    moire_compare_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    moire_compare_parser.add_argument('--validate', action='store_true')
    moire_compare_parser.add_argument('--kmc-seed', type=int, help='Run repo KMC with one explicit random seed for every compared event.')
    moire_compare_parser.add_argument('--kmc-seeds', help='Run repo KMC multiple times for every compared event, for example 3401,3402,3403.')
    moire_compare_parser.add_argument('--kmc-retries', type=int, help='Automatically add this many retry seeds when only one KMC seed is requested.')
    moire_compare_parser.add_argument('--ovito', action='store_true', help='Try to render OVITO snapshots for the compared events.')
    moire_compare_parser.add_argument('--ovito-python', help='Optional Python/ovitos executable that can import the ovito module.')
    moire_compare_parser.add_argument('--lammps-only', action='store_true', help='Only compare LAMMPS barriers and skip the repo KMC stage.')
    moire_compare_parser.set_defaults(func=cmd_moire_compare)

    moire_diffusion_parser = subparsers.add_parser('moire-diffusion-sweep', help='Run one MoRe barrier, then sweep repo KMC across temperatures and summarize diffusion coefficient vs temperature.')
    moire_diffusion_parser.add_argument('event_json')
    moire_diffusion_parser.add_argument('case_dir')
    moire_diffusion_parser.add_argument('--workdir')
    moire_diffusion_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    moire_diffusion_parser.add_argument('--temperatures', help='Comma-separated temperatures in K, for example 700,800,900,1000,1100,1200.')
    moire_diffusion_parser.add_argument('--validate', action='store_true')
    moire_diffusion_parser.add_argument('--kmc-seed', type=int, help='Use one explicit KMC random seed for every temperature.')
    moire_diffusion_parser.add_argument('--kmc-seeds', help='Use multiple KMC random seeds for every temperature, for example 3401,3402,3403.')
    moire_diffusion_parser.add_argument('--kmc-retries', type=int, help='Automatically add this many retry seeds when only one KMC seed is requested.')
    moire_diffusion_parser.add_argument('--run-time', help='KMC simulated time for each temperature point.')
    moire_diffusion_parser.add_argument('--stats-step', help='KMC stats and dump interval for each temperature point.')
    moire_diffusion_parser.add_argument('--ovito', action='store_true', help='Try to render OVITO snapshots for the LAMMPS stage and each temperature point.')
    moire_diffusion_parser.add_argument('--ovito-python', help='Optional Python/ovitos executable that can import the ovito module.')
    moire_diffusion_parser.set_defaults(func=cmd_moire_diffusion_sweep)

    tools_parser = subparsers.add_parser('tools', help='List the built-in shell tools and what they do')
    tools_parser.add_argument('--json', action='store_true')
    tools_parser.set_defaults(func=cmd_tools)

    router_eval_parser = subparsers.add_parser('router-golden-eval', help='Run golden eval cases against the heuristic tool router')
    router_eval_parser.add_argument('--golden-file', default='examples/evals/router_golden.json')
    router_eval_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    router_eval_parser.add_argument('--current-run-dir')
    router_eval_parser.set_defaults(func=cmd_router_golden_eval)

    runtime_eval_parser = subparsers.add_parser('runtime-golden-eval', help='Run golden eval cases against runtime health checks')
    runtime_eval_parser.add_argument('--golden-file', default='examples/evals/runtime_health_golden.json')
    runtime_eval_parser.set_defaults(func=cmd_runtime_golden_eval)

    doctor_parser = subparsers.add_parser('doctor', help='Check whether the local model, LAMMPS runtime, and KMC binary are ready')
    doctor_parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
    doctor_parser.add_argument('--json', action='store_true')
    doctor_parser.set_defaults(func=cmd_doctor)

    mcp_parser = subparsers.add_parser('mcp-server', help='Start the local mietclaw MCP server over stdio')
    mcp_parser.add_argument('--provider', default='local')
    mcp_parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
    mcp_parser.add_argument('--workspace-root', default=str(Path('.autonomy-mcp').resolve()))
    mcp_parser.add_argument('--output-dir', default=str(Path('runs').resolve()))
    mcp_parser.set_defaults(func=cmd_mcp_server)

    local_status_parser = subparsers.add_parser('local-status', help='Show the effective local-model profile used by mietclaw')
    local_status_parser.set_defaults(func=cmd_local_status)

    local_self_check_parser = subparsers.add_parser('local-self-check', help='Probe the local model through mietclaw and print the resolved model')
    local_self_check_parser.add_argument('--model', help='Optional local model id or alias, for example 27b.')
    local_self_check_parser.add_argument('--prompt', default='请只回复：mietclaw local ok', help='Short probe prompt sent to the local model.')
    local_self_check_parser.set_defaults(func=cmd_local_self_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
