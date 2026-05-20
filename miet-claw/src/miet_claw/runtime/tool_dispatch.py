from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from ..tool_router import ToolIntent
from .tool_registry import ToolDefinition, get_chat_tool_definition, get_tool_definition
from .types import ToolExecutionOutcome


def _chat_draft(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    output = session._run_draft(str(intent.params.get("prompt") or original_prompt))
    return ToolExecutionOutcome(
        output=output,
        metadata={"report": session.current_report},
    )


def _chat_run(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    prompt = str(intent.params.get("prompt") or original_prompt)
    if bool(intent.params.get("dry_run_only", False)):
        output = session._run_job(
                prompt,
                dry_run_only=True,
                resume_existing=bool(intent.params.get("resume_existing", False)),
                overwrite_existing=bool(intent.params.get("overwrite_existing", False)),
            )
        return ToolExecutionOutcome(
            output=output,
            metadata={"execution": (session.current_report or {}).get("execution", {})},
        )
    if session._preview_before_run_enabled():
        output = session._run_preview(prompt)
        return ToolExecutionOutcome(output=output, metadata={"report": session.current_report})
    output = session._run_job(
            prompt,
            resume_existing=bool(intent.params.get("resume_existing", False)),
            overwrite_existing=bool(intent.params.get("overwrite_existing", False)),
        )
    return ToolExecutionOutcome(
        output=output,
        metadata={"execution": (session.current_report or {}).get("execution", {})},
    )


def _chat_moire_run(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    params = intent.params
    event_json = params.get("event_json")
    case_dir = params.get("case_dir")
    if not case_dir:
        return ToolExecutionOutcome(
            output=intent.reply or "要直接运行 MoRe 的 LAMMPS→KMC，我至少需要 case_dir。",
            ok=False,
        )
    workdir = str(params.get("workdir") or (session.output_dir / f"moire-run-{int(time.time())}"))
    validate = bool(params.get("validate", True))
    result = session._run_moire_workflow(
        event_json=event_json,
        case_dir=case_dir,
        workdir=workdir,
        validate=validate,
        kmc_seed=params.get("kmc_seed"),
        kmc_seeds=params.get("kmc_seeds"),
        ovito=bool(params.get("ovito", False)),
    )
    return ToolExecutionOutcome(output=result, ok=not result.startswith("MoRe LAMMPS→KMC 失败："))


def _chat_moire_compare(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    params = intent.params
    case_dir = params.get("case_dir")
    event_jsons = params.get("event_jsons") or []
    if not case_dir or len(event_jsons) < 2:
        return ToolExecutionOutcome(
            output=intent.reply or "要比较多个 MoRe event，我至少需要一个 case_dir 和两个 event.json 路径。",
            ok=False,
        )
    workdir = str(params.get("workdir") or (session.output_dir / f"moire-compare-{int(time.time())}"))
    validate = bool(params.get("validate", True))
    result = session._run_moire_compare_workflow(
        case_dir=case_dir,
        event_jsons=event_jsons,
        workdir=workdir,
        validate=validate,
        kmc_seed=params.get("kmc_seed"),
        kmc_seeds=params.get("kmc_seeds"),
        ovito=bool(params.get("ovito", False)),
        lammps_only=bool(params.get("lammps_only", False)),
    )
    return ToolExecutionOutcome(output=result, ok=not result.startswith("MoRe 多事件比较失败："))


def _chat_moire_diffusion_sweep(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    params = intent.params
    event_json = params.get("event_json")
    case_dir = params.get("case_dir")
    if not event_json or not case_dir:
        return ToolExecutionOutcome(
            output=intent.reply or "要做扩散系数-温度关系算例，我至少需要 event.json 和 case_dir。",
            ok=False,
        )
    workdir = str(params.get("workdir") or (session.output_dir / f"moire-diffusion-{int(time.time())}"))
    validate = bool(params.get("validate", True))
    result = session._run_moire_diffusion_workflow(
        event_json=event_json,
        case_dir=case_dir,
        workdir=workdir,
        validate=validate,
        temperatures_k=params.get("temperatures_k"),
        kmc_seed=params.get("kmc_seed"),
        kmc_seeds=params.get("kmc_seeds"),
        run_time=params.get("run_time"),
        stats_step=params.get("stats_step"),
        ovito=bool(params.get("ovito", False)),
    )
    return ToolExecutionOutcome(output=result, ok=not result.startswith("MoRe 扩散扫温失败："))


def _chat_list_runs(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    cache_key = f"runs::{session.output_dir.resolve()}"
    cached = session._read_only_cache_get(cache_key)
    if cached is not None:
        return ToolExecutionOutcome(output=cached)
    runs = api["list_runs"](session.output_dir)
    return ToolExecutionOutcome(output=session._read_only_cache_set(cache_key, api["format_run_list"](runs)))


def _chat_compare_runs(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    mode = intent.params.get("mode")
    cache_key = f"compare_runs::{session.output_dir.resolve()}::{mode or ''}"
    cached = session._read_only_cache_get(cache_key)
    if cached is not None:
        return ToolExecutionOutcome(output=cached)
    try:
        report = api["compare_recent_runs"](session.output_dir, mode=str(mode) if mode else None)
    except RuntimeError as exc:
        return ToolExecutionOutcome(output=str(exc), ok=False)
    session.current_run_dir = Path(report["left"]["path"])
    return ToolExecutionOutcome(output=session._read_only_cache_set(cache_key, api["format_compare_report"](report)))


def _chat_inspect_run(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    params = intent.params
    target = params.get("run")
    mode = str(params.get("mode")) if params.get("mode") else None
    run_dir = session._current_or_target_run(target, mode=mode) or session._latest_run_dir(mode=mode)
    if not run_dir:
        return ToolExecutionOutcome(output="还没有可 inspect 的 run。", ok=False)
    session.current_run_dir = run_dir
    cache_key = f"inspect::{run_dir.resolve()}"
    cached = session._read_only_cache_get(cache_key)
    if cached is not None:
        return ToolExecutionOutcome(output=cached)
    return ToolExecutionOutcome(
        output=session._read_only_cache_set(cache_key, api["format_inspect_report"](api["inspect_run"](run_dir)))
    )


def _chat_list_artifacts(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    params = intent.params
    target = params.get("run")
    mode = str(params.get("mode")) if params.get("mode") else None
    run_dir = session._current_or_target_run(target, mode=mode) or session._latest_run_dir(mode=mode)
    if not run_dir:
        return ToolExecutionOutcome(output="还没有可查看 artifact 的 run。", ok=False)
    session.current_run_dir = run_dir
    cache_key = f"artifacts::{run_dir.resolve()}"
    cached = session._read_only_cache_get(cache_key)
    if cached is not None:
        return ToolExecutionOutcome(output=cached)
    return ToolExecutionOutcome(output=session._read_only_cache_set(cache_key, api["format_artifact_report"](run_dir)))


def _chat_get_logs(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    params = intent.params
    target = params.get("run")
    log_target = str(params.get("target") or "auto")
    mode = str(params.get("mode")) if params.get("mode") else None
    run_dir = session._current_or_target_run(target, mode=mode) or session._latest_run_dir(mode=mode)
    if not run_dir:
        return ToolExecutionOutcome(output="还没有可查看日志的 run。", ok=False)
    session.current_run_dir = run_dir
    cache_key = f"logs::{run_dir.resolve()}::{log_target}"
    cached = session._read_only_cache_get(cache_key)
    if cached is not None:
        return ToolExecutionOutcome(output=cached)
    return ToolExecutionOutcome(
        output=session._read_only_cache_set(cache_key, api["format_log_report"](run_dir, target=log_target))
    )


def _chat_open_web(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    port = int(intent.params.get("port") or 4174)
    info = api["ensure_web_console"](Path(session.project_root), port=port)
    return ToolExecutionOutcome(
        output=(
            "Web console ready\n"
            f"- url: {info['url']}\n"
            f"- running: {info['running']}\n"
            f"- opened: {info['opened']}\n"
            f"- log: {info['log_path']}"
        )
    )


def _chat_kmc_bridge(session: Any, intent: ToolIntent, original_prompt: str, *, api: Mapping[str, Any]) -> ToolExecutionOutcome:
    session._invalidate_read_only_cache()
    params = intent.params
    event_json = params.get("event_json")
    neb_txt = params.get("neb_txt")
    if not event_json or not neb_txt:
        return ToolExecutionOutcome(
            output=intent.reply or "要做 KMC bridge，至少需要 event.json 和 neb.txt 路径。",
            ok=False,
        )
    workdir = str(params.get("workdir") or (session.output_dir / f"bridge-{int(time.time())}"))
    validate = bool(params.get("validate", True))
    result = session._run_bridge(event_json=event_json, neb_txt=neb_txt, workdir=workdir, validate=validate)
    return ToolExecutionOutcome(output=result, ok=not result.startswith("KMC bridge 失败："))


_CHAT_EXECUTORS: Dict[str, Callable[[Any, ToolIntent, str], ToolExecutionOutcome]] = {
    "miet_autonomy_draft": _chat_draft,
    "miet_autonomy_run": _chat_run,
    "miet_moire_run": _chat_moire_run,
    "miet_moire_compare": _chat_moire_compare,
    "miet_moire_diffusion_sweep": _chat_moire_diffusion_sweep,
    "miet_list_runs": _chat_list_runs,
    "miet_compare_runs": _chat_compare_runs,
    "miet_inspect_run": _chat_inspect_run,
    "miet_list_artifacts": _chat_list_artifacts,
    "miet_get_logs": _chat_get_logs,
    "miet_open_web": _chat_open_web,
    "miet_kmc_bridge": _chat_kmc_bridge,
}


def execute_chat_tool_intent_outcome(
    session: Any,
    intent: ToolIntent,
    original_prompt: str,
    *,
    api: Mapping[str, Any],
) -> ToolExecutionOutcome:
    if intent.action == "chat":
        return ToolExecutionOutcome(output=intent.reply or "")

    tool = get_chat_tool_definition(intent.action)
    if tool is None:
        return ToolExecutionOutcome(output=f"未知工具动作：{intent.action}", ok=False)

    return tool.execute_chat(
        session,
        intent,
        original_prompt,
        handlers=_CHAT_EXECUTORS,
        api=api,
    )


def _mcp_tool_result(api: Mapping[str, Any], text: str, structured: Dict[str, Any]) -> Dict[str, Any]:
    return api["_tool_result"](text, structured)


def _mcp_runtime_doctor(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    payload = api["collect_runtime_doctor"](
        Path(arguments.get("project_root") or server.project_root).resolve(),
        local_status=api["get_local_model_status"](),
    )
    return _mcp_tool_result(api, api["format_runtime_doctor"](payload), payload)


def _mcp_list_runs(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or server.output_dir).resolve()
    runs = api["list_runs"](output_dir, limit=int(arguments.get("limit") or 12))
    return _mcp_tool_result(api, api["format_run_list"](runs), {"runs": runs})


def _resolve_mcp_run_dir(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Path:
    output_dir = Path(arguments.get("output_dir") or server.output_dir).resolve()
    return api["_resolve_run_dir"](
        output_dir,
        run_dir=arguments.get("run_dir"),
        run_name=arguments.get("run_name"),
    )


def _mcp_inspect_run(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = _resolve_mcp_run_dir(server, arguments, api=api)
    info = api["inspect_run"](run_dir)
    return _mcp_tool_result(api, api["format_inspect_report"](info), info)


def _mcp_get_logs(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = _resolve_mcp_run_dir(server, arguments, api=api)
    target = str(arguments.get("target") or "auto")
    max_lines = int(arguments.get("max_lines") or 60)
    info = api["get_log_excerpt"](run_dir, target=target, max_lines=max_lines)
    return _mcp_tool_result(api, api["format_log_report"](run_dir, target=target), info)


def _mcp_list_artifacts(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = _resolve_mcp_run_dir(server, arguments, api=api)
    limit = int(arguments.get("limit") or 80)
    artifacts = api["list_artifacts"](run_dir, limit=limit)
    return _mcp_tool_result(api, api["format_artifact_report"](run_dir), {"run_dir": str(run_dir), "artifacts": artifacts})


def _mcp_autonomy_draft(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    payload = api["materialize_autonomy_workspace"](
        prompt=str(arguments["prompt"]),
        project_root=server.project_root,
        workspace_root=arguments.get("workspace_root") or server.workspace_root,
        provider=str(arguments.get("provider") or server.provider),
        mode_hint=arguments.get("mode"),
        template_path=arguments.get("template_path"),
        job_id=arguments.get("job_id"),
        material_name=arguments.get("material_name"),
    )
    return _mcp_tool_result(api, api["format_draft_report"](payload), payload)


def _mcp_autonomy_run(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    payload = api["run_autonomy_job"](
        prompt=str(arguments["prompt"]),
        project_root=server.project_root,
        workspace_root=arguments.get("workspace_root") or server.workspace_root,
        provider=str(arguments.get("provider") or server.provider),
        mode_hint=arguments.get("mode"),
        template_path=arguments.get("template_path"),
        job_id=arguments.get("job_id"),
        material_name=arguments.get("material_name"),
        output_dir=arguments.get("output_dir") or server.output_dir,
        dry_run_only=bool(arguments.get("dry_run_only", False)),
        resume_existing=bool(arguments.get("resume_existing", False)),
        overwrite_existing=bool(arguments.get("overwrite_existing", False)),
    )
    return _mcp_tool_result(api, api["format_run_report"](payload), payload)


def _mcp_plan_job(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    spec = api["load_job_spec"](str(Path(str(arguments["job_spec_path"])).expanduser().resolve()))
    plan = api["build_plan_payload"](spec)
    payload = {"job_id": spec["job_id"], "mode": spec["mode"], "plan": plan}
    return _mcp_tool_result(api, api["_json_text"](payload), payload)


def _mcp_run_job(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = api["run_job"](
        str(Path(str(arguments["job_spec_path"])).expanduser().resolve()),
        str(Path(arguments.get("output_dir") or server.output_dir).resolve()),
        dry_run=bool(arguments.get("dry_run", False)),
    )
    info = api["inspect_run"](run_dir)
    return _mcp_tool_result(api, api["format_inspect_report"](info), info)


def _mcp_kmc_bridge(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        summary = api["run_kmc_lookup_bridge"](
            event_json=str(arguments["event_json"]),
            neb_txt=arguments.get("neb_txt"),
            barrier=arguments.get("barrier"),
            workdir=str(arguments["workdir"]),
            validate=bool(arguments.get("validate", True)),
        )
    except api["BridgeError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_bridge_report"](summary), summary)


def _mcp_moire_run(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = api["get_runtime_settings"](Path(server.project_root))
    try:
        summary = api["run_moire_lammps_to_kmc"](
            event_json=str(arguments["event_json"]) if arguments.get("event_json") else None,
            case_dir=str(arguments["case_dir"]),
            workdir=str(arguments["workdir"]),
            validate=bool(arguments.get("validate", True)),
            conda_exec=Path(runtime["conda_exec"]),
            conda_env=runtime["conda_env"],
            neb_input=runtime["neb_input"],
            post_script=runtime["post_script"],
            mpi_procs=int(runtime["mpi_procs"]),
            kmc_seed=int(arguments["kmc_seed"]) if arguments.get("kmc_seed") is not None else None,
            kmc_seeds=arguments.get("kmc_seeds"),
            kmc_retry_attempts=int(runtime.get("kmc_retry_attempts") or 0),
            misa_kmc_binary=Path(runtime["kmc_binary"]),
            eam_file=Path(runtime["eam_file"]),
            kmc_temperature=float(runtime["kmc_temperature"]),
            kmc_stats_step=runtime["kmc_stats_step"],
            kmc_run_time=runtime["kmc_run_time"],
            render_ovito=bool(arguments.get("ovito", False)),
            ovito_python=arguments.get("ovito_python"),
        )
    except api["MoReWorkflowError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_moire_workflow_report"](summary), summary)


def _mcp_moire_compare(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = api["get_runtime_settings"](Path(server.project_root))
    try:
        summary = api["run_moire_event_compare"](
            case_dir=str(arguments["case_dir"]),
            event_jsons=arguments.get("event_jsons") or [],
            workdir=str(arguments["workdir"]),
            validate=bool(arguments.get("validate", True)),
            conda_exec=Path(runtime["conda_exec"]),
            conda_env=runtime["conda_env"],
            neb_input=runtime["neb_input"],
            post_script=runtime["post_script"],
            mpi_procs=int(runtime["mpi_procs"]),
            kmc_seed=int(arguments["kmc_seed"]) if arguments.get("kmc_seed") is not None else None,
            kmc_seeds=arguments.get("kmc_seeds"),
            kmc_retry_attempts=int(runtime.get("kmc_retry_attempts") or 0),
            misa_kmc_binary=Path(runtime["kmc_binary"]),
            eam_file=Path(runtime["eam_file"]),
            kmc_temperature=float(runtime["kmc_temperature"]),
            kmc_stats_step=runtime["kmc_stats_step"],
            kmc_run_time=runtime["kmc_run_time"],
            run_kmc=not bool(arguments.get("lammps_only", False)),
            render_ovito=bool(arguments.get("ovito", False)),
            ovito_python=arguments.get("ovito_python"),
        )
    except api["MoReWorkflowError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_moire_compare_report"](summary), summary)


def _mcp_moire_diffusion_sweep(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = api["get_runtime_settings"](Path(server.project_root))
    try:
        summary = api["run_moire_diffusion_sweep"](
            event_json=str(arguments["event_json"]),
            case_dir=str(arguments["case_dir"]),
            workdir=str(arguments["workdir"]),
            temperatures_k=arguments.get("temperatures_k") or runtime.get("diffusion_temperatures"),
            validate=bool(arguments.get("validate", True)),
            kmc_seed=int(arguments["kmc_seed"]) if arguments.get("kmc_seed") is not None else None,
            kmc_seeds=arguments.get("kmc_seeds"),
            kmc_retry_attempts=int(runtime.get("kmc_retry_attempts") or 0),
            misa_kmc_binary=Path(runtime["kmc_binary"]),
            eam_file=Path(runtime["eam_file"]),
            render_ovito=bool(arguments.get("ovito", False)),
            ovito_python=arguments.get("ovito_python"),
            run_time=str(arguments["run_time"]) if arguments.get("run_time") is not None else runtime["diffusion_run_time"],
            stats_step=str(arguments["stats_step"]) if arguments.get("stats_step") is not None else runtime["diffusion_stats_step"],
        )
    except api["MoReWorkflowError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_moire_diffusion_sweep_report"](summary), summary)


def _mcp_moire_lammps(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = api["get_runtime_settings"](Path(server.project_root))
    try:
        summary = api["run_moire_lammps_case"](
            event_json=str(arguments["event_json"]) if arguments.get("event_json") else None,
            case_dir=str(arguments["case_dir"]),
            workdir=str(arguments["workdir"]),
            conda_exec=Path(runtime["conda_exec"]),
            conda_env=runtime["conda_env"],
            neb_input=runtime["neb_input"],
            post_script=runtime["post_script"],
            mpi_procs=int(runtime["mpi_procs"]),
            render_ovito=bool(arguments.get("ovito", False)),
            ovito_python=arguments.get("ovito_python"),
        )
    except api["MoReWorkflowError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_moire_lammps_report"](summary), summary)


def _mcp_moire_kmc(server: Any, arguments: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = api["get_runtime_settings"](Path(server.project_root))
    try:
        summary = api["run_moire_repo_kmc"](
            event_json=str(arguments["event_json"]) if arguments.get("event_json") else None,
            barrier_eV=float(arguments["barrier_eV"]),
            workdir=str(arguments["workdir"]),
            data_lmp=arguments.get("data_lmp"),
            misa_kmc_binary=Path(runtime["kmc_binary"]),
            eam_file=Path(runtime["eam_file"]),
            temperature=float(runtime["kmc_temperature"]),
            stats_step=runtime["kmc_stats_step"],
            run_time=runtime["kmc_run_time"],
            kmc_seed=int(arguments["kmc_seed"]) if arguments.get("kmc_seed") is not None else None,
            kmc_seeds=arguments.get("kmc_seeds"),
            retry_attempts=int(runtime.get("kmc_retry_attempts") or 0),
            render_ovito=bool(arguments.get("ovito", False)),
            ovito_python=arguments.get("ovito_python"),
        )
    except api["MoReWorkflowError"] as exc:
        raise api["MCPServerError"](str(exc)) from exc
    return _mcp_tool_result(api, api["format_moire_kmc_report"](summary), summary)


_MCP_EXECUTORS: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    "miet_runtime_doctor": _mcp_runtime_doctor,
    "miet_list_runs": _mcp_list_runs,
    "miet_inspect_run": _mcp_inspect_run,
    "miet_get_logs": _mcp_get_logs,
    "miet_list_artifacts": _mcp_list_artifacts,
    "miet_autonomy_draft": _mcp_autonomy_draft,
    "miet_autonomy_run": _mcp_autonomy_run,
    "miet_plan_job": _mcp_plan_job,
    "miet_run_job": _mcp_run_job,
    "miet_kmc_bridge": _mcp_kmc_bridge,
    "miet_moire_run": _mcp_moire_run,
    "miet_moire_compare": _mcp_moire_compare,
    "miet_moire_diffusion_sweep": _mcp_moire_diffusion_sweep,
    "miet_moire_lammps": _mcp_moire_lammps,
    "miet_moire_kmc": _mcp_moire_kmc,
}


def dispatch_mcp_tool(
    server: Any,
    name: str,
    arguments: Dict[str, Any],
    *,
    api: Mapping[str, Any],
) -> Dict[str, Any]:
    tool = get_tool_definition(name)
    error_cls = api["MCPServerError"]
    if tool is None:
        raise error_cls(f"Unknown tool: {name}")

    return tool.execute_mcp(
        server,
        arguments,
        handlers=_MCP_EXECUTORS,
        api=api,
    )
