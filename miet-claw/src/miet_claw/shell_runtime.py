from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .local_profile import get_local_model_settings, get_runtime_settings
from .runtime.approval import approval_policy_name
from .runtime.shell_command_registry import shell_command_summaries
from .runtime.tool_registry import shell_tool_summaries


SHELL_COMMANDS: List[Dict[str, str]] = shell_command_summaries()


DOMAIN_TOOLS: List[Dict[str, str]] = shell_tool_summaries() + [
    {"tool": "local-model-chat", "entrypoint": "chat.py", "summary": "Use the local 27B model for chat, routing, and agent-loop decisions."},
    {"tool": "tool-router", "entrypoint": "tool_router.py", "summary": "Decide whether a prompt should chat, inspect, bridge, draft, or run."},
    {"tool": "mcp-server", "entrypoint": "mcp_server.py", "summary": "Expose the same core tools over MCP for an external client."},
]


def _sibling_kmc_ml_root(project_root: Path) -> Path:
    return project_root.parent / "kmc-ml"


def _default_moire_case(project_root: Path) -> Path:
    return _sibling_kmc_ml_root(project_root) / "soap-KMC" / "NEB_new_data" / "MoRe" / "Re_0.07" / "model_4"


def _kmc_binary(project_root: Path) -> Path:
    return project_root / "crystalkmc-fix-diffusion-coef" / "build" / "bin" / "misa-kmc"


def _run_probe(command: List[str], timeout: float = 25.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "command": command, "error": str(exc)}
    output = (proc.stdout or "").strip()
    return {
        "ok": proc.returncode == 0,
        "command": command,
        "returncode": proc.returncode,
        "output": output,
    }


def collect_runtime_doctor(project_root: Path, local_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    settings = get_local_model_settings()
    runtime = get_runtime_settings(root)
    kmc_binary = Path(runtime["kmc_binary"])
    moire_case = Path(runtime["moire_case_dir"])
    conda_exec = Path(runtime["conda_exec"])
    conda_env = str(runtime["conda_env"])

    doctor: Dict[str, Any] = {
        "agent_name": settings["agent_name"],
        "project_root": str(root),
        "profile_path": settings["profile_path"],
        "preferred_model": settings["preferred_model"],
        "local_model": local_status or {},
        "paths": {
            "kmc_binary": str(kmc_binary),
            "conda_exec": str(conda_exec),
            "moire_case_dir": str(moire_case),
            "kmc_ml_root": str(_sibling_kmc_ml_root(root)),
            "eam_file": runtime["eam_file"],
            "bridge_script": runtime["bridge_script"],
        },
        "checks": {
            "profile_exists": Path(settings["profile_path"]).exists(),
            "kmc_binary_exists": kmc_binary.exists(),
            "conda_exec_exists": conda_exec.exists(),
            "moire_case_exists": moire_case.exists(),
            "eam_file_exists": Path(runtime["eam_file"]).exists(),
            "bridge_script_exists": Path(runtime["bridge_script"]).exists(),
        },
        "runtime": {
            "conda_env": conda_env,
            "mpi_procs": runtime["mpi_procs"],
            "neb_input": runtime["neb_input"],
            "post_script": runtime["post_script"],
            "kmc_retry_attempts": runtime["kmc_retry_attempts"],
        },
    }

    if conda_exec.exists():
        doctor["probes"] = {
            "lmp": _run_probe([str(conda_exec), "run", "-n", conda_env, "which", "lmp"]),
            "mpirun": _run_probe([str(conda_exec), "run", "-n", conda_env, "which", "mpirun"]),
        }
    else:
        doctor["probes"] = {
            "lmp": {"ok": False, "error": "conda executable not found"},
            "mpirun": {"ok": False, "error": "conda executable not found"},
        }
    return doctor


def format_shell_help() -> str:
    lines = ["命令："]
    for item in SHELL_COMMANDS:
        lines.append(f"{item['command']}")
        lines.append(f"  {item['summary']}")
    return "\n".join(lines)


def format_shell_tools() -> str:
    lines = ["Built-in tools"]
    for item in DOMAIN_TOOLS:
        lines.append(f"- {item['tool']} · {item['summary']}")
    return "\n".join(lines)


def build_shell_status(
    *,
    project_root: Path,
    workspace_root: Path,
    output_dir: Path,
    provider: str,
    selected_model: Optional[str],
    local_status: Dict[str, Any],
    current_run_dir: Optional[Path],
    active_turn_id: Optional[str] = None,
    queued_followup_count: int = 0,
    runnable_followup_count: int = 0,
    auto_followup_count: int = 0,
    aborted_turn_count: int = 0,
) -> Dict[str, Any]:
    latest_run = None
    if output_dir.exists():
        candidates = [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and ((path / "state.json").exists() or (path / "summary.json").exists())
        ]
        if candidates:
            latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            latest_run = str(latest)

    return {
        "project_root": str(project_root),
        "workspace_root": str(workspace_root),
        "output_dir": str(output_dir),
        "provider": provider,
        "selected_model": selected_model or local_status.get("default_model"),
        "local_model_healthy": bool(local_status.get("healthy")),
        "tool_approval_policy": approval_policy_name(),
        "current_run_dir": str(current_run_dir) if current_run_dir else None,
        "latest_run_dir": latest_run,
        "active_turn_id": active_turn_id,
        "queued_followup_count": queued_followup_count,
        "runnable_followup_count": runnable_followup_count,
        "auto_followup_count": auto_followup_count,
        "aborted_turn_count": aborted_turn_count,
        "tool_count": len(DOMAIN_TOOLS),
        "command_count": len(SHELL_COMMANDS),
    }


def format_shell_status(payload: Dict[str, Any]) -> str:
    lines = [
        "Shell status",
        f"- project: {payload['project_root']}",
        f"- workspace: {payload['workspace_root']}",
        f"- runs: {payload['output_dir']}",
        f"- provider: {payload['provider']}",
        f"- model: {payload.get('selected_model') or '—'}",
        f"- local model healthy: {payload.get('local_model_healthy')}",
        f"- tool approval policy: {payload.get('tool_approval_policy') or 'allow_all'}",
        f"- current run: {payload.get('current_run_dir') or '—'}",
        f"- latest run: {payload.get('latest_run_dir') or '—'}",
        f"- active turn: {payload.get('active_turn_id') or '—'}",
        f"- queued follow-ups: {payload.get('queued_followup_count', 0)}",
        f"- runnable follow-ups: {payload.get('runnable_followup_count', 0)}",
        f"- auto follow-ups: {payload.get('auto_followup_count', 0)}",
        f"- aborted turns: {payload.get('aborted_turn_count', 0)}",
        f"- built-in tools: {payload['tool_count']}",
        f"- slash commands: {payload['command_count']}",
    ]
    return "\n".join(lines)


def format_runtime_doctor(payload: Dict[str, Any]) -> str:
    local = payload.get("local_model") or {}
    probes = payload.get("probes") or {}
    checks = payload.get("checks") or {}
    paths = payload.get("paths") or {}
    lines = [
        "Runtime doctor",
        f"- profile: {payload.get('profile_path')}",
        f"- preferred model: {payload.get('preferred_model')}",
        f"- local model healthy: {local.get('healthy', False)}",
        f"- local model default: {local.get('default_model') or '—'}",
        f"- kmc binary exists: {checks.get('kmc_binary_exists')}",
        f"- conda exec exists: {checks.get('conda_exec_exists')}",
        f"- MoRe case exists: {checks.get('moire_case_exists')}",
        f"- lmp available: {(probes.get('lmp') or {}).get('ok', False)}",
        f"- mpirun available: {(probes.get('mpirun') or {}).get('ok', False)}",
        f"- kmc binary: {paths.get('kmc_binary')}",
        f"- MoRe case: {paths.get('moire_case_dir')}",
    ]
    if (probes.get("lmp") or {}).get("output"):
        lines.append(f"- lmp path: {probes['lmp']['output']}")
    if (probes.get("mpirun") or {}).get("output"):
        lines.append(f"- mpirun path: {probes['mpirun']['output']}")
    return "\n".join(lines)


def dump_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
