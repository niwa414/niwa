from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .types import ToolExecutionOutcome


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "miet_runtime_doctor",
        "shell_name": "runtime-doctor",
        "entrypoint": "shell_runtime.py",
        "description": "Check whether the local model, LAMMPS runtime, MoRe case path, and misa-kmc binary are ready.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "project_root": {"type": "string"},
            },
        },
    },
    {
        "name": "miet_list_runs",
        "shell_name": "runs-inspect",
        "entrypoint": "chat.py",
        "description": "List recent mietclaw run directories and their status.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "output_dir": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        },
    },
    {
        "name": "miet_inspect_run",
        "shell_name": "runs-inspect",
        "entrypoint": "chat.py",
        "description": "Inspect one mietclaw run and summarize its current state.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "run_dir": {"type": "string"},
                "run_name": {"type": "string"},
                "output_dir": {"type": "string"},
            },
        },
    },
    {
        "name": "miet_get_logs",
        "shell_name": "runs-inspect",
        "entrypoint": "chat.py",
        "description": "Read the MD, KMC, or summary log excerpt for a run.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "run_dir": {"type": "string"},
                "run_name": {"type": "string"},
                "output_dir": {"type": "string"},
                "target": {"type": "string", "enum": ["auto", "md", "kmc", "summary"]},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
            },
        },
    },
    {
        "name": "miet_list_artifacts",
        "shell_name": "runs-inspect",
        "entrypoint": "chat.py",
        "description": "List archived artifacts for a run.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "run_dir": {"type": "string"},
                "run_name": {"type": "string"},
                "output_dir": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
        },
    },
    {
        "name": "miet_autonomy_draft",
        "shell_name": "autonomy-draft",
        "entrypoint": "autonomy.py",
        "description": "Turn a natural-language MD/KMC task into a generated draft workspace.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string"},
                "provider": {"type": "string"},
                "workspace_root": {"type": "string"},
                "mode": {"type": "string", "enum": ["md_only", "kmc_only", "md_to_kmc_chain"]},
                "template_path": {"type": "string"},
                "job_id": {"type": "string"},
                "material_name": {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "miet_autonomy_run",
        "shell_name": "autonomy-run",
        "entrypoint": "autonomy.py",
        "description": "Draft, validate, and optionally run a natural-language MD/KMC task.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string"},
                "provider": {"type": "string"},
                "workspace_root": {"type": "string"},
                "output_dir": {"type": "string"},
                "mode": {"type": "string", "enum": ["md_only", "kmc_only", "md_to_kmc_chain"]},
                "template_path": {"type": "string"},
                "job_id": {"type": "string"},
                "material_name": {"type": "string"},
                "dry_run_only": {"type": "boolean"},
                "resume_existing": {"type": "boolean"},
                "overwrite_existing": {"type": "boolean"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "miet_plan_job",
        "shell_name": "plan-job",
        "entrypoint": "planner.py",
        "description": "Load an existing job spec and return the deterministic execution plan.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "job_spec_path": {"type": "string"},
            },
            "required": ["job_spec_path"],
        },
    },
    {
        "name": "miet_run_job",
        "shell_name": "run-job",
        "entrypoint": "executor.py",
        "description": "Run an existing job spec.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "job_spec_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["job_spec_path"],
        },
    },
    {
        "name": "miet_kmc_bridge",
        "shell_name": "kmc-bridge",
        "entrypoint": "bridge.py",
        "description": "Turn event.json plus neb.txt or a barrier value into a KMC lookup file and optionally validate it with misa-kmc.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_json": {"type": "string"},
                "neb_txt": {"type": "string"},
                "barrier": {"type": "number"},
                "workdir": {"type": "string"},
                "validate": {"type": "boolean"},
            },
            "required": ["event_json", "workdir"],
        },
    },
    {
        "name": "miet_moire_run",
        "shell_name": "moire-runtime",
        "entrypoint": "moire_runtime.py",
        "description": "Run a real MoRe LAMMPS NEB case on this computer, auto-generate a KMC seed event if needed, then write a repo-compatible KMC input and continue the simulation with the repo misa-kmc binary.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_json": {"type": "string"},
                "case_dir": {"type": "string"},
                "workdir": {"type": "string"},
                "validate": {"type": "boolean"},
                "kmc_seed": {"type": "integer"},
                "kmc_seeds": {"type": "array", "items": {"type": "integer"}},
                "ovito": {"type": "boolean"},
                "ovito_python": {"type": "string"},
            },
            "required": ["case_dir", "workdir"],
        },
    },
    {
        "name": "miet_moire_compare",
        "shell_name": "moire-runtime",
        "entrypoint": "moire_runtime.py",
        "description": "Compare multiple MoRe event.json files on one local case, and optionally continue each event into the repo misa-kmc stage.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "case_dir": {"type": "string"},
                "event_jsons": {"type": "array", "items": {"type": "string"}},
                "workdir": {"type": "string"},
                "validate": {"type": "boolean"},
                "kmc_seed": {"type": "integer"},
                "kmc_seeds": {"type": "array", "items": {"type": "integer"}},
                "ovito": {"type": "boolean"},
                "ovito_python": {"type": "string"},
                "lammps_only": {"type": "boolean"},
            },
            "required": ["case_dir", "event_jsons", "workdir"],
        },
    },
    {
        "name": "miet_moire_diffusion_sweep",
        "shell_name": "moire-runtime",
        "entrypoint": "moire_runtime.py",
        "description": "Run one MoRe LAMMPS barrier from an event.json, then sweep repo misa-kmc across temperatures and summarize diffusion coefficient vs temperature.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_json": {"type": "string"},
                "case_dir": {"type": "string"},
                "workdir": {"type": "string"},
                "temperatures_k": {"type": "array", "items": {"type": "number"}},
                "validate": {"type": "boolean"},
                "kmc_seed": {"type": "integer"},
                "kmc_seeds": {"type": "array", "items": {"type": "integer"}},
                "run_time": {"type": "string"},
                "stats_step": {"type": "string"},
                "ovito": {"type": "boolean"},
                "ovito_python": {"type": "string"},
            },
            "required": ["event_json", "case_dir", "workdir"],
        },
    },
    {
        "name": "miet_moire_lammps",
        "shell_name": "moire-runtime",
        "entrypoint": "moire_runtime.py",
        "description": "Run only the local MoRe LAMMPS NEB case on this computer and return the resulting neb.txt plus parsed barrier.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_json": {"type": "string"},
                "case_dir": {"type": "string"},
                "workdir": {"type": "string"},
                "ovito": {"type": "boolean"},
                "ovito_python": {"type": "string"},
            },
            "required": ["case_dir", "workdir"],
        },
    },
    {
        "name": "miet_moire_kmc",
        "shell_name": "moire-runtime",
        "entrypoint": "moire_runtime.py",
        "description": "Generate a repo-compatible KMC initial state directly from MoRe event.json or, if none is provided, auto-generate a seed event from data.lmp; then write a transparent repo misa-kmc input from a MoRe LAMMPS barrier and run local repo KMC.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_json": {"type": "string"},
                "barrier_eV": {"type": "number"},
                "workdir": {"type": "string"},
                "data_lmp": {"type": "string"},
                "kmc_seed": {"type": "integer"},
                "kmc_seeds": {"type": "array", "items": {"type": "integer"}},
                "ovito": {"type": "boolean"},
                "ovito_python": {"type": "string"},
            },
            "required": ["barrier_eV", "workdir"],
        },
    },
]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    shell_name: str
    entrypoint: str
    description: str
    input_schema: Dict[str, Any]
    chat_actions: Tuple[str, ...] = ()
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
    concurrency_safe: bool = False
    manual_only: bool = False
    permission_scope: str = "read"
    default_response_strategy: str = "default"
    expose_shell: bool = True
    expose_mcp: bool = True
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    chat_executor_key: Optional[str] = None
    mcp_executor_key: Optional[str] = None

    def to_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shell_name": self.shell_name,
            "entrypoint": self.entrypoint,
            "description": self.description,
            "inputSchema": deepcopy(self.input_schema),
        }

    def to_mcp_definition(self) -> Dict[str, Any]:
        item = self.to_spec()
        item.pop("shell_name", None)
        item.pop("entrypoint", None)
        return item

    def to_shell_summary(self) -> Dict[str, str]:
        return {
            "tool": self.shell_name,
            "entrypoint": self.entrypoint,
            "summary": self.description,
            "mcp_name": self.name,
        }

    def permission_profile_payload(self) -> Dict[str, Any]:
        return {
            "tool_name": self.name,
            "scope": self.permission_scope,
            "read_only": self.read_only,
            "mutating": self.mutating,
            "destructive": self.destructive,
            "manual_only": self.manual_only,
            "concurrency_safe": self.concurrency_safe,
        }

    def _schema_properties(self) -> Dict[str, Any]:
        properties = self.input_schema.get("properties")
        if isinstance(properties, dict):
            return properties
        return {}

    def _schema_required(self) -> Tuple[str, ...]:
        required = self.input_schema.get("required")
        if isinstance(required, list):
            return tuple(str(item) for item in required)
        return ()

    def _validate_payload(self, payload: Any, *, source: str, allow_additional: Optional[bool] = None) -> Optional[str]:
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return f"工具 `{self.name}` 的{source}参数必须是对象。"
        properties = self._schema_properties()
        required = self._schema_required()
        additional_allowed = self.input_schema.get("additionalProperties", True) is not False
        if allow_additional is not None:
            additional_allowed = allow_additional
        for field_name in required:
            value = payload.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                return f"工具 `{self.name}` 缺少必需参数 `{field_name}`。"
        if not additional_allowed:
            unknown = sorted(set(payload) - set(properties))
            if unknown:
                return f"工具 `{self.name}` 不接受这些额外参数：{', '.join(unknown)}。"
        for field_name, schema in properties.items():
            if field_name not in payload:
                continue
            value = payload[field_name]
            expected_type = schema.get("type") if isinstance(schema, dict) else None
            if expected_type == "string" and not isinstance(value, str):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是字符串。"
            if expected_type == "boolean" and not isinstance(value, bool):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是布尔值。"
            if expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是整数。"
            if expected_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是数字。"
            if expected_type == "array" and not isinstance(value, list):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是数组。"
            if expected_type == "object" and not isinstance(value, dict):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是对象。"
            if isinstance(schema, dict) and isinstance(schema.get("enum"), list) and value not in schema.get("enum"):
                return f"工具 `{self.name}` 的参数 `{field_name}` 必须是 {', '.join(map(str, schema.get('enum') or []))} 之一。"
        return None

    def validate_chat_intent(self, intent: Any) -> Optional[str]:
        action = getattr(intent, "action", "")
        if self.chat_actions and action not in self.chat_actions:
            return f"工具 `{self.name}` 不接受 chat action `{action}`。"
        params = getattr(intent, "params", {}) or {}
        return self._validate_payload(params, source="chat", allow_additional=True)

    def validate_mcp_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        return self._validate_payload(arguments, source="MCP")

    def render_chat_result(self, intent: Any, outcome: ToolExecutionOutcome) -> ToolExecutionOutcome:
        strategy_payload = self._result_strategy_payload(intent, outcome)
        if outcome.output:
            strategy_lines = self._render_chat_strategy_lines(strategy_payload)
            if strategy_lines:
                merged = outcome.output.rstrip()
                if merged:
                    merged = f"{merged}\n" + "\n".join(strategy_lines)
                else:
                    merged = "\n".join(strategy_lines)
                return ToolExecutionOutcome(output=merged, ok=outcome.ok, metadata=outcome.metadata)
            return outcome
        fallback = f"工具 `{getattr(intent, 'action', self.name)}` 已执行，但没有返回可展示文本。"
        strategy_lines = self._render_chat_strategy_lines(strategy_payload)
        if strategy_lines:
            fallback = f"{fallback}\n" + "\n".join(strategy_lines)
        return ToolExecutionOutcome(output=fallback, ok=outcome.ok, metadata=outcome.metadata)

    def render_mcp_result(self, result: Dict[str, Any], *, api: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = result.get("structuredContent") if isinstance(result, dict) else None
        strategy_payload = self._result_strategy_payload(None, ToolExecutionOutcome(output="", metadata=metadata or {}))
        if isinstance(result, dict) and isinstance(result.get("content"), list) and result["content"]:
            if strategy_payload and isinstance(result.setdefault("structuredContent", {}), dict):
                result["structuredContent"].setdefault("tool_result_strategy", strategy_payload)
            return result
        text = "工具已执行，但没有返回可展示内容。"
        structured = result if isinstance(result, dict) else {"result": result}
        if isinstance(structured, dict) and strategy_payload:
            structured.setdefault("tool_result_strategy", strategy_payload)
        return api["_tool_result"](text, structured)

    def _result_strategy_payload(self, intent: Any, outcome: ToolExecutionOutcome) -> Dict[str, Any]:
        metadata = outcome.metadata if isinstance(outcome.metadata, dict) else {}
        payload: Dict[str, Any] = {
            "strategy": self.default_response_strategy,
            "tool_name": self.name,
        }
        if self.default_response_strategy == "draft":
            payload["next_steps"] = ["- 如果草案方向对，我可以继续把它真正运行起来。"]
            return payload
        if self.default_response_strategy != "execution":
            return payload
        execution = metadata.get("execution")
        if isinstance(execution, dict):
            if execution.get("dry_run_only") is not None:
                payload["dry_run_only"] = bool(execution.get("dry_run_only"))
            if execution.get("resume_existing") is not None:
                payload["resume_existing"] = bool(execution.get("resume_existing"))
            if execution.get("overwrite_existing") is not None:
                payload["overwrite_existing"] = bool(execution.get("overwrite_existing"))
            if execution.get("final_recovery") is not None:
                payload["final_recovery"] = execution.get("final_recovery")
            if execution.get("validation_recovery") is not None:
                payload["validation_recovery"] = execution.get("validation_recovery")
        return payload

    def _render_chat_strategy_lines(self, strategy_payload: Dict[str, Any]) -> List[str]:
        if not strategy_payload:
            return []
        strategy = strategy_payload.get("strategy")
        if strategy == "draft":
            return list(strategy_payload.get("next_steps") or [])
        if strategy != "execution":
            return []
        lines: List[str] = []
        if strategy_payload.get("dry_run_only"):
            lines.append("- 如果这份 dry-run/preview 结果没问题，我可以直接去掉 `--dry-run` 帮你真正启动计算。")
        if strategy_payload.get("resume_existing"):
            lines.append("- 如果你想确认恢复后的状态，我可以继续 inspect 这次 run，或者直接展开关键日志。")
        if strategy_payload.get("overwrite_existing"):
            lines.append("- 这是一次覆盖式重跑；如果你愿意，我可以继续帮你确认新结果有没有替换干净。")
        recovery = strategy_payload.get("final_recovery") or strategy_payload.get("validation_recovery") or {}
        recovery_plan = recovery.get("recovery_plan") if isinstance(recovery, dict) else None
        if isinstance(recovery_plan, dict):
            restarted = [
                item.get("step_id")
                for item in recovery_plan.get("steps") or []
                if item.get("action") in {"restart_resumable_step", "rebuild_from_checkpoint", "rerun_step"}
            ]
            if restarted:
                lines.append(f"- 这次恢复重点处理了这些步骤：{', '.join(restarted[:4])}。")
            missing_outputs = [
                f"{item.get('step_id')}[{', '.join((item.get('missing_outputs') or [])[:2])}]"
                for item in recovery_plan.get("steps") or []
                if item.get("missing_outputs")
            ]
            if missing_outputs:
                lines.append(f"- 系统检测到这些缺失产物并自动调整了恢复策略：{', '.join(missing_outputs[:3])}。")
            drifted_outputs = [
                f"{item.get('step_id')}[{', '.join((item.get('drifted_outputs') or [])[:2])}]"
                for item in recovery_plan.get("steps") or []
                if item.get("drifted_outputs")
            ]
            if drifted_outputs:
                lines.append(f"- 系统还检测到这些产物内容已经变化，因此没有直接复用：{', '.join(drifted_outputs[:3])}。")
            cascaded = [
                f"{item.get('step_id')}←{item.get('invalidated_by')}"
                for item in recovery_plan.get("steps") or []
                if item.get("invalidated_by")
            ]
            if cascaded:
                lines.append(f"- 下游步骤也会跟着重新处理：{', '.join(cascaded[:4])}。")
        return lines

    def execute_chat(
        self,
        session: Any,
        intent: Any,
        original_prompt: str,
        *,
        handlers: Mapping[str, Callable[..., ToolExecutionOutcome]],
        api: Mapping[str, Any],
    ) -> ToolExecutionOutcome:
        validation_error = self.validate_chat_intent(intent)
        if validation_error:
            return ToolExecutionOutcome(output=validation_error, ok=False)
        handler = handlers.get(self.chat_executor_key or self.name)
        if handler is None:
            return ToolExecutionOutcome(output=f"工具 `{self.name}` 还没有绑定 chat handler。", ok=False)
        try:
            outcome = handler(session, intent, original_prompt, api=api)
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionOutcome(output=f"工具 {intent.action} 执行失败：{exc}", ok=False)
        return self.render_chat_result(intent, outcome)

    def execute_mcp(
        self,
        server: Any,
        arguments: Dict[str, Any],
        *,
        handlers: Mapping[str, Callable[..., Dict[str, Any]]],
        api: Mapping[str, Any],
    ) -> Dict[str, Any]:
        error_cls = api["MCPServerError"]
        if not self.expose_mcp:
            raise error_cls(f"Unknown tool: {self.name}")
        validation_error = self.validate_mcp_arguments(arguments)
        if validation_error:
            raise error_cls(validation_error)
        handler = handlers.get(self.mcp_executor_key or self.name)
        if handler is None:
            raise error_cls(f"Tool `{self.name}` is missing an MCP handler")
        result = handler(server, arguments, api=api)
        return self.render_mcp_result(result, api=api)


_CHAT_ACTIONS_BY_NAME = {
    "miet_list_runs": ("runs",),
    "miet_compare_runs": ("compare_runs",),
    "miet_inspect_run": ("inspect",),
    "miet_list_artifacts": ("artifacts",),
    "miet_get_logs": ("logs",),
    "miet_autonomy_draft": ("draft",),
    "miet_autonomy_run": ("run",),
    "miet_kmc_bridge": ("bridge_kmc_lookup",),
    "miet_moire_run": ("moire_run",),
    "miet_moire_compare": ("moire_compare",),
    "miet_moire_diffusion_sweep": ("moire_diffusion_sweep",),
    "miet_open_web": ("open_web",),
}

_READ_ONLY_TOOL_NAMES = {
    "miet_runtime_doctor",
    "miet_list_runs",
    "miet_compare_runs",
    "miet_inspect_run",
    "miet_get_logs",
    "miet_list_artifacts",
    "miet_plan_job",
}

_MUTATING_TOOL_NAMES = {
    "miet_autonomy_draft",
    "miet_autonomy_run",
    "miet_run_job",
    "miet_kmc_bridge",
    "miet_moire_run",
    "miet_moire_compare",
    "miet_moire_diffusion_sweep",
    "miet_moire_lammps",
    "miet_moire_kmc",
    "miet_open_web",
}

_DESTRUCTIVE_TOOL_NAMES: set[str] = set()
_CONCURRENCY_SAFE_TOOL_NAMES = {
    "miet_runtime_doctor",
    "miet_list_runs",
    "miet_compare_runs",
    "miet_inspect_run",
    "miet_get_logs",
    "miet_list_artifacts",
    "miet_plan_job",
}
_MANUAL_ONLY_TOOL_NAMES: set[str] = set()

_PERMISSION_SCOPE_BY_NAME = {
    "miet_runtime_doctor": "read",
    "miet_list_runs": "read",
    "miet_compare_runs": "read",
    "miet_inspect_run": "read",
    "miet_get_logs": "read",
    "miet_list_artifacts": "read",
    "miet_plan_job": "plan",
    "miet_autonomy_draft": "plan",
    "miet_autonomy_run": "run",
    "miet_run_job": "run",
    "miet_kmc_bridge": "run",
    "miet_moire_run": "run",
    "miet_moire_compare": "run",
    "miet_moire_diffusion_sweep": "run",
    "miet_moire_lammps": "run",
    "miet_moire_kmc": "run",
    "miet_open_web": "external",
}

_DEFAULT_RESPONSE_STRATEGY_BY_NAME = {
    "miet_autonomy_draft": "draft",
    "miet_autonomy_run": "execution",
    "miet_run_job": "execution",
    "miet_kmc_bridge": "execution",
    "miet_moire_run": "execution",
    "miet_moire_compare": "execution",
    "miet_moire_diffusion_sweep": "execution",
    "miet_moire_lammps": "execution",
    "miet_moire_kmc": "execution",
}

_INTERNAL_TOOL_DEFINITIONS = (
    ToolDefinition(
        name="miet_compare_runs",
        shell_name="runs-inspect",
        entrypoint="chat.py",
        description="Compare the two most relevant recent runs and summarize their differences.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"mode": {"type": "string"}},
        },
        chat_actions=("compare_runs",),
        read_only=True,
        concurrency_safe=True,
        permission_scope="read",
        default_response_strategy="inspect",
        expose_shell=False,
        expose_mcp=False,
    ),
    ToolDefinition(
        name="miet_open_web",
        shell_name="open-web",
        entrypoint="chat.py",
        description="Open the local web console for this agent.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"port": {"type": "integer", "minimum": 1, "maximum": 65535}},
        },
        chat_actions=("open_web",),
        mutating=True,
        permission_scope="external",
        default_response_strategy="execution",
        expose_shell=False,
        expose_mcp=False,
    ),
)


def _build_tool_definition(spec: Dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=str(spec["name"]),
        shell_name=str(spec["shell_name"]),
        entrypoint=str(spec["entrypoint"]),
        description=str(spec["description"]),
        input_schema=deepcopy(spec["inputSchema"]),
        chat_actions=_CHAT_ACTIONS_BY_NAME.get(str(spec["name"]), ()),
        read_only=str(spec["name"]) in _READ_ONLY_TOOL_NAMES,
        mutating=str(spec["name"]) in _MUTATING_TOOL_NAMES,
        destructive=str(spec["name"]) in _DESTRUCTIVE_TOOL_NAMES,
        concurrency_safe=str(spec["name"]) in _CONCURRENCY_SAFE_TOOL_NAMES,
        manual_only=str(spec["name"]) in _MANUAL_ONLY_TOOL_NAMES,
        permission_scope=_PERMISSION_SCOPE_BY_NAME.get(
            str(spec["name"]),
            "read" if str(spec["name"]) in _READ_ONLY_TOOL_NAMES else "run",
        ),
        default_response_strategy=_DEFAULT_RESPONSE_STRATEGY_BY_NAME.get(str(spec["name"]), "inspect"),
    )


TOOL_DEFINITIONS: Tuple[ToolDefinition, ...] = tuple(_build_tool_definition(spec) for spec in TOOL_SPECS) + _INTERNAL_TOOL_DEFINITIONS
TOOL_DEFINITIONS_BY_NAME: Dict[str, ToolDefinition] = {tool.name: tool for tool in TOOL_DEFINITIONS}
CHAT_TOOL_DEFINITIONS_BY_ACTION: Dict[str, ToolDefinition] = {
    action: tool
    for tool in TOOL_DEFINITIONS
    for action in tool.chat_actions
}


def tool_definitions(*, include_internal: bool = False) -> List[ToolDefinition]:
    items = list(TOOL_DEFINITIONS)
    if include_internal:
        return items
    return [tool for tool in items if tool.expose_mcp or tool.expose_shell]


def iter_chat_tool_definitions() -> Iterable[ToolDefinition]:
    return CHAT_TOOL_DEFINITIONS_BY_ACTION.values()


def get_tool_definition(name: str) -> Optional[ToolDefinition]:
    return TOOL_DEFINITIONS_BY_NAME.get(name)


def get_chat_tool_definition(action: str) -> Optional[ToolDefinition]:
    return CHAT_TOOL_DEFINITIONS_BY_ACTION.get(action)


def mcp_tool_definitions() -> List[Dict[str, Any]]:
    return [tool.to_mcp_definition() for tool in TOOL_DEFINITIONS if tool.expose_mcp]


def shell_tool_summaries() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()
    for tool in TOOL_DEFINITIONS:
        if not tool.expose_shell:
            continue
        key = tool.shell_name
        if key in seen:
            continue
        seen.add(key)
        items.append(tool.to_shell_summary())
    return items
