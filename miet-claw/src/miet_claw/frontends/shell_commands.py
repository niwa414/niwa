from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..runtime.shell_command_registry import canonical_shell_command, shell_command_names
from ..shell_runtime import format_shell_help, format_shell_tools
from ..tool_router import ToolIntent

ShellCommandHandler = Callable[[Any, list[str], str, Callable[[str, list[str]], Optional[str]]], str]


def _parse_turn_ref_and_prompt(args: list[str]) -> tuple[Optional[str], Optional[str]]:
    if not args:
        return None, None
    head = str(args[0] or "").strip()
    if head in {"latest", "active"} or head.startswith("turn-"):
        prompt = " ".join(args[1:]).strip()
        return head, prompt or None
    return None, " ".join(args).strip() or None


def _handle_exit(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    raise EOFError


def _handle_help(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return "\n".join([session.banner(), "", format_shell_help()])


def _handle_status(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return session._format_shell_status()


def _handle_doctor(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return session._format_runtime_doctor()


def _handle_tools(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return format_shell_tools()


def _handle_clear(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    session._runtime_state.clear_conversation()
    return "会话已清空。"


def _handle_provider(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if not args:
        return f"当前 provider: {session.provider}"
    if args[0] != session.provider:
        session._reset_mcp_client()
    session.provider = args[0]
    return f"provider 已切到: {session.provider}"


def _handle_model(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    status = session._refresh_local_model_status()
    if not args:
        lines = [
            f"endpoint: {status.get('base_url')}",
            f"healthy: {status.get('healthy')}",
            f"selected: {session.selected_model or status.get('default_model') or '—'}",
            f"preferred default: {status.get('default_model') or '—'}",
        ]
        if status.get("models"):
            lines.append("available:")
            lines.extend([f"- {item}" for item in status["models"]])
            lines.append("aliases:")
            lines.append("- 27b")
            lines.append("- 122b")
        if status.get("error"):
            lines.append(f"error: {status['error']}")
        return "\n".join(lines)
    requested = args[0]
    resolved = resolve_model_alias(requested, status.get("models") or [])
    if status.get("models") and not resolved:
        return "没有找到这个 model。先用 `/model` 看可用列表。"
    session.selected_model = resolved or requested
    return f"已切换到 local model: {session.selected_model}"


def _handle_runs(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return session._run_manual_tool_intent(ToolIntent(action="runs"), "/runs")


def _handle_compare(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    mode = args[0] if args else None
    params = {"mode": mode} if mode else {}
    return session._run_manual_tool_intent(ToolIntent(action="compare_runs", params=params), command_line)


def _handle_inspect(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    target = args[0] if args else (str(session.current_run_dir) if session.current_run_dir else None)
    params = {"run": target} if target else {}
    return session._run_manual_tool_intent(ToolIntent(action="inspect", params=params), command_line)


def _handle_artifacts(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    target = args[0] if args else (str(session.current_run_dir) if session.current_run_dir else None)
    params = {"run": target} if target else {}
    return session._run_manual_tool_intent(ToolIntent(action="artifacts", params=params), command_line)


def _handle_logs(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    target = None
    log_target = "auto"
    if args:
        if len(args) == 1 and args[0] in {"md", "kmc", "summary", "auto"}:
            log_target = args[0]
        else:
            target = args[0]
            if len(args) > 1:
                log_target = args[1]
    params = {"target": log_target}
    if target:
        params["run"] = target
    return session._run_manual_tool_intent(ToolIntent(action="logs", params=params), command_line)


def _handle_followups(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    return session._get_query_engine().format_queued_followups()


def _handle_continue(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    engine = session._get_query_engine()
    followup_id = str(args[0] or "").strip() if args else None
    if followup_id and not followup_id.startswith("followup-"):
        return "用法：`/continue [followup-id]`"
    return engine.continue_queued_followup(followup_id or None)


def _handle_continue_all(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    engine = session._get_query_engine()
    limit = 3
    if args:
        try:
            limit = int(args[0])
        except ValueError:
            return "用法：`/continue-all [limit]`，其中 limit 必须是整数。"
    return engine.drain_queued_followups(limit=limit)


def _handle_open(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if not args:
        return "可用子命令：`/open web [port]`"
    if args[0] == "web":
        port = 4174
        if len(args) > 1:
            try:
                port = int(args[1])
            except ValueError:
                return "端口必须是整数。"
        return session._run_manual_tool_intent(ToolIntent(action="open_web", params={"port": port}), command_line)
    return f"未知 open 子命令：{' '.join(args)}"


def _handle_draft(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if not args:
        return "请在 `/draft` 后面跟一段任务描述。"
    return session._run_manual_tool_intent(ToolIntent(action="draft", params={"prompt": " ".join(args)}), command_line)


def _handle_run(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if not args:
        return "请在 `/run` 后面跟一段任务描述。"
    resume_existing = False
    overwrite_existing = False
    dry_run_only = False
    prompt_parts = []
    for item in args:
        if item == "--resume":
            resume_existing = True
            continue
        if item == "--overwrite":
            overwrite_existing = True
            continue
        if item in {"--dry-run", "--preview"}:
            dry_run_only = True
            continue
        prompt_parts.append(item)
    if not prompt_parts:
        return "请在 `/run` 后面跟一段任务描述。"
    return session._run_manual_tool_intent(
        ToolIntent(
            action="run",
            params={
                "prompt": " ".join(prompt_parts),
                "resume_existing": resume_existing,
                "overwrite_existing": overwrite_existing,
                "dry_run_only": dry_run_only,
            },
        ),
        command_line,
    )


def _handle_resume(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    engine = session._get_query_engine()
    turn_ref, prompt_override = _parse_turn_ref_and_prompt(args)
    resolved_turn_id = engine.resolve_turn_reference(turn_ref)
    if not resolved_turn_id:
        return "没有找到可恢复的 turn。先用 `/followups` 或 `/status` 看当前状态。"
    return engine.resume_turn(resolved_turn_id, prompt=prompt_override).reply


def _handle_retry(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    engine = session._get_query_engine()
    turn_ref, prompt_override = _parse_turn_ref_and_prompt(args)
    resolved_turn_id = engine.resolve_turn_reference(turn_ref)
    if not resolved_turn_id:
        return "没有找到可重试的 turn。先用 `/followups` 或 `/status` 看当前状态。"
    return engine.retry_turn(resolved_turn_id, prompt=prompt_override).reply


def _handle_bridge(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if len(args) < 2:
        return "用法：`/bridge <event.json> <neb.txt> [workdir]`"
    event_json = args[0]
    neb_txt = args[1]
    workdir = args[2] if len(args) > 2 else str(session.output_dir / f"bridge-{int(time.time())}")
    return session._run_manual_tool_intent(
        ToolIntent(
            action="bridge_kmc_lookup",
            params={"event_json": event_json, "neb_txt": neb_txt, "workdir": workdir, "validate": True},
        ),
        command_line,
    )


def _handle_moire_run(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if not args:
        return "用法：`/moire-run <MoRe-case-dir> [workdir]` 或 `/moire-run <event.json> <MoRe-case-dir> [workdir]`"
    event_json = None
    if len(args) >= 2 and Path(args[0]).suffix == ".json":
        event_json = args[0]
        case_dir = args[1]
        workdir = args[2] if len(args) > 2 else str(session.output_dir / f"moire-run-{int(time.time())}")
    else:
        case_dir = args[0]
        workdir = args[1] if len(args) > 1 else str(session.output_dir / f"moire-run-{int(time.time())}")
    params = {"case_dir": case_dir, "workdir": workdir, "validate": True}
    if event_json:
        params["event_json"] = event_json
    return session._run_manual_tool_intent(ToolIntent(action="moire_run", params=params), command_line)


def _handle_moire_compare(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if len(args) < 3:
        return "用法：`/moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]`"
    case_dir = args[0]
    trailing = args[1:]
    workdir = None
    if trailing and Path(trailing[-1]).suffix != ".json":
        workdir = trailing[-1]
        trailing = trailing[:-1]
    event_jsons = [item for item in trailing if Path(item).suffix == ".json"]
    if len(event_jsons) < 2:
        return "用法：`/moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]`"
    params = {
        "case_dir": case_dir,
        "event_jsons": event_jsons,
        "workdir": workdir or str(session.output_dir / f"moire-compare-{int(time.time())}"),
        "validate": True,
    }
    return session._run_manual_tool_intent(ToolIntent(action="moire_compare", params=params), command_line)


def _handle_moire_diffusion_sweep(session: Any, args: list[str], command_line: str, resolve_model_alias: Callable[[str, list[str]], Optional[str]]) -> str:
    if len(args) < 2:
        return "用法：`/moire-diffusion-sweep <event.json> <MoRe-case-dir> [workdir]`"
    event_json = args[0]
    case_dir = args[1]
    workdir = args[2] if len(args) > 2 else str(session.output_dir / f"moire-diffusion-{int(time.time())}")
    params = {
        "event_json": event_json,
        "case_dir": case_dir,
        "workdir": workdir,
        "validate": True,
    }
    return session._run_manual_tool_intent(ToolIntent(action="moire_diffusion_sweep", params=params), command_line)


SHELL_COMMAND_HANDLERS: Dict[str, ShellCommandHandler] = {
    "/exit": _handle_exit,
    "/help": _handle_help,
    "/status": _handle_status,
    "/doctor": _handle_doctor,
    "/tools": _handle_tools,
    "/clear": _handle_clear,
    "/provider": _handle_provider,
    "/model": _handle_model,
    "/runs": _handle_runs,
    "/compare": _handle_compare,
    "/inspect": _handle_inspect,
    "/artifacts": _handle_artifacts,
    "/logs": _handle_logs,
    "/followups": _handle_followups,
    "/continue": _handle_continue,
    "/continue-all": _handle_continue_all,
    "/open": _handle_open,
    "/draft": _handle_draft,
    "/run": _handle_run,
    "/resume": _handle_resume,
    "/retry": _handle_retry,
    "/bridge": _handle_bridge,
    "/moire-run": _handle_moire_run,
    "/moire-compare": _handle_moire_compare,
    "/moire-diffusion-sweep": _handle_moire_diffusion_sweep,
}


def missing_shell_command_handlers() -> list[str]:
    return [name for name in shell_command_names() if name not in SHELL_COMMAND_HANDLERS]


def handle_shell_command(
    session: Any,
    command_line: str,
    *,
    resolve_model_alias: Callable[[str, list[str]], Optional[str]],
) -> str:
    parts = shlex.split(command_line)
    if not parts:
        return ""
    raw_command = parts[0]
    args = parts[1:]
    command = canonical_shell_command(raw_command)
    if command is None:
        return f"未知命令：{command_line}"
    handler = SHELL_COMMAND_HANDLERS.get(command)
    if handler is None:
        return f"命令 `{raw_command}` 已登记，但还没有可执行处理器。"
    return handler(session, args, command_line, resolve_model_alias)
