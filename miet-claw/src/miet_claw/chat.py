import curses
import json
import os
import queue
import re
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .autonomy import detect_project_root, materialize_autonomy_workspace, run_autonomy_job
from .bridge import BridgeError, run_kmc_lookup_bridge
from .frontends.shell_commands import handle_shell_command
from .mcp_client import LocalMCPClient, MCPClientError
from .moire_runtime import MoReWorkflowError, run_moire_diffusion_sweep, run_moire_event_compare, run_moire_lammps_to_kmc
from .runtime.approval import ToolApprovalContext, decide_tool_approval
from .runtime.context import (
    build_engine_context,
    build_current_context_for_chat,
    collect_tool_context_history,
    remember_tool_context_blocks,
    remember_tool_turn,
    tool_backed_response_style,
    tool_context_limit,
    tool_evidence_for_chat,
    tool_history_payload,
)
from .runtime.permissions import (
    build_tool_budget,
    is_mutating_intent,
    read_bounded_int_env,
    record_tool_outcome,
    tool_budget_block_note,
    tool_budget_context,
)
from .runtime.query_loop import (
    build_agent_loop_messages,
    build_local_model_messages,
    build_tool_plan_messages,
    build_tool_router_messages,
    build_tool_summary_messages,
    maybe_extend_tool_turn as runtime_maybe_extend_tool_turn,
    run_agent_query_loop,
    run_engine_turn,
    run_tool_event_loop,
    tool_result_summary_entries,
    trace_tool_result_blocks,
)
from .runtime.query_engine import MietQueryEngine, QueryTurnResult
from .runtime.session import ChatRuntimeState
from .runtime.tool_dispatch import execute_chat_tool_intent_outcome
from .runtime.transcript import append_message, append_progress, append_tool_trace
from .runtime.types import (
    AssistantActionBlock,
    PermissionDecisionEvent,
    ToolBudget,
    ToolExecutionOutcome,
    ToolResponseStrategy,
    ToolResultBlock,
    ToolResultBlockEvent,
    ToolResultEvent,
    ToolTurnState,
    ToolUseEvent,
)
from .shell_runtime import (
    build_shell_status,
    collect_runtime_doctor,
    format_runtime_doctor,
    format_shell_status,
)
from .tool_router import (
    AGENT_LOOP_SYSTEM_PROMPT,
    AgentDecision,
    TOOL_PLAN_SYSTEM_PROMPT,
    TOOL_ROUTER_SYSTEM_PROMPT,
    ToolPlan,
    ToolIntent,
    heuristic_tool_plan,
    heuristic_tool_intent,
    infer_run_mode_hint,
    parse_tool_plan,
    parse_tool_intent,
    should_try_agent_loop,
    should_try_tool_plan,
    should_skip_tool_router,
)

from .run_inspection import (
    compare_recent_runs,
    get_log_excerpt,
    inspect_run,
    list_artifacts,
    list_runs,
    _read_run_mode,
    _resolve_run_dir,
    _shorten,
    _wrap_lines,
)
from .chat_strategy import ChatEvidencePlanner
from .chat_reports import (
    format_artifact_report,
    format_bridge_report,
    format_compare_report,
    format_draft_report,
    format_inspect_report,
    format_log_report,
    format_moire_compare_report,
    format_moire_diffusion_sweep_report,
    format_moire_kmc_report,
    format_moire_lammps_report,
    format_moire_workflow_report,
    format_progress_event,
    format_run_list,
    format_run_report,
)




RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

LOCAL_MODEL_SYSTEM_PROMPT = (
    "You are mietclaw, a concise local-model chat assistant for a multiscale materials simulation agent. "
    "Reply clearly and naturally, defaulting to Chinese when the user speaks Chinese. "
    "Do not claim that a simulation has run unless tool output explicitly says so. "
    "When tool output is available, use it faithfully. Keep answers short, practical, and specific to MD/KMC operations."
)


def stylize(text: str, *styles: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(styles) + text + RESET


from .local_model_client import (
    _compact_local_history,
    _model_for_purpose,
    _resolve_model_alias,
    _truncate_for_model,
    chat_with_local_model,
    ensure_web_console,
    get_local_model_status,
)


class MietClawChatSession:
    _STATE_FIELDS = {
        "current_report",
        "current_run_dir",
        "current_bridge_summary",
        "current_moire_summary",
        "current_moire_compare_summary",
        "current_moire_diffusion_summary",
        "history",
        "selected_model",
        "local_model_status",
        "_read_only_tool_cache",
        "_tool_context_outputs",
        "_tool_context_blocks",
        "session_dir",
        "transcript_path",
        "last_tool_turn_state",
        "turns",
        "active_turn_id",
        "pending_tool_requests",
        "permission_denials",
        "usage_stats",
        "queued_followups",
    }

    def __getattr__(self, name: str) -> Any:
        runtime_state = self.__dict__.get("_runtime_state")
        if runtime_state is not None and name in self._STATE_FIELDS:
            if name == "_read_only_tool_cache":
                return runtime_state.read_only_tool_cache
            if name == "_tool_context_outputs":
                return runtime_state.tool_context_outputs
            if name == "_tool_context_blocks":
                return runtime_state.tool_context_blocks
            return getattr(runtime_state, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        runtime_state = self.__dict__.get("_runtime_state")
        if runtime_state is not None and name in self._STATE_FIELDS:
            if name == "_read_only_tool_cache":
                runtime_state.read_only_tool_cache = value
                return
            if name == "_tool_context_outputs":
                runtime_state.tool_context_outputs = value
                return
            if name == "_tool_context_blocks":
                runtime_state.tool_context_blocks = value
                return
            setattr(runtime_state, name, value)
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        project_root: str,
        workspace_root: str,
        output_dir: str,
        provider: str = "auto",
        mode_hint: Optional[str] = None,
        initial_model: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.project_root = str(detect_project_root(Path(project_root)))
        self.workspace_root = str(Path(workspace_root).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.provider = provider
        self.mode_hint = mode_hint
        self.progress_callback = progress_callback
        self._mcp_client: Optional[LocalMCPClient] = None
        self._query_engine: Optional[MietQueryEngine] = None
        self._last_query_turn_result: Optional[QueryTurnResult] = None
        self._runtime_state = ChatRuntimeState.create(project_root=self.project_root, initial_model=initial_model)

    def _append_transcript(self, text: str) -> None:
        self._runtime_state.append_transcript(text)

    def _record_chat_message(self, role: str, content: str) -> None:
        self.history.append((role, content))
        append_message(self._runtime_state, "mietclaw" if role == "assistant" else role, content)

    def _append_tool_trace_if_new(self, previous_tool_turn: Optional[ToolTurnState]) -> None:
        if self.last_tool_turn_state is not None and self.last_tool_turn_state is not previous_tool_turn:
            self._append_tool_trace(self.last_tool_turn_state)

    def _emit_progress(self, line: str) -> None:
        append_progress(self._runtime_state, line)
        if self.progress_callback:
            self.progress_callback(line)

    def _append_tool_trace(self, state: Optional[ToolTurnState]) -> None:
        append_tool_trace(self._runtime_state, state)

    def _refresh_local_model_status(self) -> Dict[str, Any]:
        self.local_model_status = get_local_model_status()
        preferred = _model_for_purpose(self.local_model_status, purpose="chat", selected_model=self.selected_model)
        if preferred:
            self.selected_model = preferred
        return self.local_model_status

    def _get_query_engine(self) -> MietQueryEngine:
        if self._query_engine is None:
            self._query_engine = MietQueryEngine(self)
        return self._query_engine

    def close(self) -> None:
        if self._mcp_client is not None:
            self._mcp_client.close()
            self._mcp_client = None

    def _reset_mcp_client(self) -> None:
        self.close()

    def _get_local_mcp_client(self) -> LocalMCPClient:
        if self._mcp_client is None:
            self._mcp_client = LocalMCPClient(
                project_root=self.project_root,
                workspace_root=self.workspace_root,
                output_dir=str(self.output_dir),
                provider=self.provider,
            )
        self._mcp_client.connect()
        return self._mcp_client

    def _intent_signature(self, intent: ToolIntent) -> str:
        return json.dumps(
            {
                "action": intent.action,
                "params": {key: intent.params[key] for key in sorted(intent.params)},
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _tool_cache_ttl_seconds(self) -> float:
        raw = os.environ.get("MIETCLAW_READ_CACHE_TTL", "5")
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 5.0

    def _read_only_cache_get(self, key: str) -> Optional[str]:
        cached = self._read_only_tool_cache.get(key)
        if not cached:
            return None
        expires_at, value = cached
        if time.time() > expires_at:
            self._read_only_tool_cache.pop(key, None)
            return None
        return value

    def _read_only_cache_set(self, key: str, value: str) -> str:
        ttl = self._tool_cache_ttl_seconds()
        if ttl <= 0:
            return value
        self._read_only_tool_cache[key] = (time.time() + ttl, value)
        return value

    def _invalidate_read_only_cache(self) -> None:
        self._read_only_tool_cache.clear()

    def _is_mutating_intent(self, intent: ToolIntent) -> bool:
        return is_mutating_intent(intent)

    def _build_tool_budget(self, *, default_steps: int, step_env: Optional[str] = None) -> ToolBudget:
        return build_tool_budget(default_steps=default_steps, step_env=step_env)

    def _new_tool_turn_state(self, *, default_steps: int, step_env: Optional[str] = None) -> ToolTurnState:
        return ToolTurnState(budget=self._build_tool_budget(default_steps=default_steps, step_env=step_env))

    def _add_tool_turn_note(self, state: ToolTurnState, note: Optional[str]) -> None:
        if note and note not in state.notes:
            state.notes.append(note)

    def _tool_turn_note_text(self, state: ToolTurnState) -> Optional[str]:
        notes = list(state.notes)
        if state.duplicate_steps:
            notes.append(f"已跳过 {state.duplicate_steps} 个重复工具步骤，避免重复执行同一动作。")
        if not notes:
            return None
        return "\n\n".join(notes)

    def _tool_context_limit(self) -> int:
        return tool_context_limit()

    def _tool_context_history(
        self,
        current_outputs: Optional[List[Tuple[ToolIntent, str]]] = None,
        *,
        current_state: Optional[ToolTurnState] = None,
        output_limit: int,
    ) -> List[Dict[str, Any]]:
        current_blocks = trace_tool_result_blocks(current_state) if current_state is not None else None
        return collect_tool_context_history(
            existing_outputs=self._tool_context_outputs,
            current_outputs=current_outputs,
            existing_blocks=self._tool_context_blocks,
            current_blocks=current_blocks,
            output_limit=output_limit,
            intent_signature=self._intent_signature,
            truncate_output=_truncate_for_model,
        )

    def _current_context_kind(self) -> Optional[str]:
        if self.current_run_dir is not None:
            return "run"
        if self.current_moire_diffusion_summary is not None:
            return "moire_diffusion"
        if self.current_moire_compare_summary is not None:
            return "moire_compare"
        if self.current_moire_summary is not None:
            return "moire"
        if self.current_bridge_summary is not None:
            return "bridge"
        if self.current_report is not None:
            return "draft"
        return None

    def _remember_tool_turn(self, state: ToolTurnState) -> None:
        self._tool_context_outputs = remember_tool_turn(
            existing_outputs=self._tool_context_outputs,
            new_outputs=state.outputs,
        )
        self._tool_context_blocks = remember_tool_context_blocks(
            existing_blocks=self._tool_context_blocks,
            new_blocks=trace_tool_result_blocks(state),
        )
        self.last_tool_turn_state = state

    def _tool_budget_context(self, budget: ToolBudget) -> Dict[str, int]:
        return tool_budget_context(budget)

    def _tool_budget_block_note(self, intent: ToolIntent, budget: ToolBudget) -> Optional[str]:
        return tool_budget_block_note(intent, budget)

    def _tool_approval_context(self, intent: ToolIntent) -> ToolApprovalContext:
        requested_dry_run = bool(intent.params.get("dry_run_only", False)) if isinstance(intent.params, dict) else False
        preview_before_run = intent.action == "run" and self._preview_before_run_enabled()
        return ToolApprovalContext(
            active_run_dir=str(self.current_run_dir) if self.current_run_dir else None,
            current_context_kind=self._current_context_kind(),
            preview_before_run=preview_before_run,
            requested_dry_run=requested_dry_run,
        )

    def _record_tool_outcome(
        self,
        budget: ToolBudget,
        intent: ToolIntent,
        outcome: ToolExecutionOutcome,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> Optional[str]:
        return record_tool_outcome(budget, intent, outcome, outputs)

    def _apply_tool_intent_to_turn(
        self,
        state: ToolTurnState,
        intent: ToolIntent,
        original_prompt: str,
        *,
        manual: bool = False,
        source: str = "unknown",
    ) -> Tuple[bool, bool]:
        signature = self._intent_signature(intent)
        if signature in state.seen_signatures:
            state.duplicate_steps += 1
            return False, False
        state.trace.add(ToolUseEvent(intent=intent, source=source, manual=manual))
        approval = decide_tool_approval(intent, manual=manual, context=self._tool_approval_context(intent))
        state.trace.add(
            PermissionDecisionEvent(
                intent=intent,
                source=source,
                decision=approval.action,
                reason=approval.reason,
                manual=manual,
            )
        )
        if approval.action != "allow":
            self._add_tool_turn_note(state, approval.reason)
            return False, True
        blocked_note = self._tool_budget_block_note(intent, state.budget)
        if blocked_note:
            self._add_tool_turn_note(state, blocked_note)
            return False, True
        state.seen_signatures.add(signature)
        outcome = self._execute_tool_intent_outcome(intent, original_prompt)
        state.trace.add(ToolResultEvent(intent=intent, outcome=outcome, source=source))
        state.trace.add(
            ToolResultBlockEvent(
                block=ToolResultBlock(
                    request_id=signature,
                    intent=intent,
                    output=outcome.output or "",
                    ok=outcome.ok,
                    source=source,
                )
            )
        )
        stop_note = self._record_tool_outcome(state.budget, intent, outcome, state.outputs)
        if stop_note:
            self._add_tool_turn_note(state, stop_note)
            return True, True
        return True, False

    def _format_model_status(self) -> str:
        status = self._refresh_local_model_status()
        if status.get("healthy"):
            return f"local model connected · {self.selected_model or status.get('default_model') or 'unknown'}"
        return f"local model offline · {status.get('error', 'unavailable')}"

    def banner(self) -> str:
        status_text = self._format_model_status()
        status_style = DIM if self.local_model_status and self.local_model_status.get("healthy") else RED
        return "\n".join(
            [
                stylize("mietclaw", BOLD, CYAN),
                stylize(status_text, status_style),
                stylize("这是一个 Claude Code 风格的本地 agent 壳：自然语言会优先自动调工具，也可以手动用 /status、/doctor、/tools、/run。", DIM),
            ]
        )

    def load_history(self, messages: Union[List[Dict[str, Any]], List[Tuple[str, str]]]) -> None:
        loaded: List[Tuple[str, str]] = []
        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            elif isinstance(item, dict):
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "")
            else:
                continue
            if role not in {"user", "assistant"} or not content.strip():
                continue
            loaded.append((role, content.strip()))
        self.history.extend(loaded)

    def _format_shell_status(self) -> str:
        payload = build_shell_status(
            project_root=Path(self.project_root),
            workspace_root=Path(self.workspace_root),
            output_dir=self.output_dir,
            provider=self.provider,
            selected_model=self.selected_model,
            local_status=self._refresh_local_model_status(),
            current_run_dir=self.current_run_dir,
            active_turn_id=self.active_turn_id,
            queued_followup_count=len(self._runtime_state.queued_followup_items()),
            runnable_followup_count=len(
                [item for item in self._runtime_state.queued_followup_items() if bool(item.get("runnable"))]
            ),
            auto_followup_count=len(
                [item for item in self._runtime_state.queued_followup_items() if bool(item.get("auto_continue"))]
            ),
            aborted_turn_count=len(self._runtime_state.aborted_turns),
        )
        return format_shell_status(payload)

    def _format_runtime_doctor(self) -> str:
        payload = collect_runtime_doctor(Path(self.project_root), local_status=self._refresh_local_model_status())
        return format_runtime_doctor(payload)

    def handle_line(self, line: str) -> str:
        return self._get_query_engine().handle_line(line)

    def _current_or_target_run(self, target: Optional[str], mode: Optional[str] = None) -> Optional[Path]:
        if target in {"latest", ":latest"}:
            return self._latest_run_dir(mode=mode)
        if target in {"current", ":current"}:
            if self.current_run_dir and (not mode or _read_run_mode(self.current_run_dir) == mode):
                return self.current_run_dir
            return self._latest_run_dir(mode=mode)
        if target:
            return _resolve_run_dir(self.output_dir, target)
        if self.current_run_dir and (not mode or _read_run_mode(self.current_run_dir) == mode):
            return self.current_run_dir
        return self._latest_run_dir(mode=mode)

    def _current_context_for_chat(self) -> Optional[str]:
        return build_current_context_for_chat(
            current_run_dir=self.current_run_dir,
            current_report=self.current_report,
            current_bridge_summary=self.current_bridge_summary,
            current_moire_summary=self.current_moire_summary,
            current_moire_compare_summary=self.current_moire_compare_summary,
            current_moire_diffusion_summary=self.current_moire_diffusion_summary,
            api=globals(),
        )

    def _tool_evidence_for_chat(self) -> Optional[str]:
        return tool_evidence_for_chat(evidence=self._tool_context_history(output_limit=1600))

    def _tool_backed_response_style(self) -> str:
        return tool_backed_response_style()

    def _available_runs_for_mode(self, mode: Optional[str], *, limit: int = 12) -> List[Dict[str, Any]]:
        items = list_runs(self.output_dir, limit=limit)
        if mode:
            items = [item for item in items if item.get("mode") == mode]
        return items

    def _run_target_hint(self, mode: Optional[str] = None) -> Optional[str]:
        current = self._current_or_target_run("current", mode=mode)
        if current:
            return "current"
        latest = self._latest_run_dir(mode=mode)
        if latest:
            return "latest"
        return None

    def _chat_evidence_planner(self) -> ChatEvidencePlanner:
        return ChatEvidencePlanner(
            current_run_dir=self.current_run_dir,
            current_run_mode=_read_run_mode(self.current_run_dir) if self.current_run_dir else None,
            available_runs_for_mode=self._available_runs_for_mode,
            run_target_hint=self._run_target_hint,
            forced_log_target=self._forced_log_target,
            intent_signature=self._intent_signature,
            suggest_next_steps=self._suggest_next_steps,
        )

    def _tool_evidence_tags(self, outputs: List[Tuple[ToolIntent, str]]) -> Set[str]:
        return self._chat_evidence_planner().tool_evidence_tags(outputs)

    def _goal_first_position(self, prompt: str, tokens: List[str]) -> int:
        return self._chat_evidence_planner().goal_first_position(prompt, tokens)

    def _goal_tokens(self, goal: str) -> List[str]:
        return self._chat_evidence_planner().goal_tokens(goal)

    def _goal_priority_bonus(self, goal: str, prompt: str) -> int:
        return self._chat_evidence_planner().goal_priority_bonus(goal, prompt)

    def _goal_optional_bonus(self, goal: str, prompt: str) -> int:
        return self._chat_evidence_planner().goal_optional_bonus(goal, prompt)

    def _goal_value_score(
        self,
        goal: str,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> int:
        return self._chat_evidence_planner().goal_value_score(goal, prompt, outputs)

    def _optional_goals_for_prompt(self, goals: List[str], prompt: str) -> Set[str]:
        return self._chat_evidence_planner().optional_goals_for_prompt(goals, prompt)

    def _evidence_goals_for_prompt(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> List[str]:
        return self._chat_evidence_planner().evidence_goals_for_prompt(prompt, outputs)

    def _required_evidence_tags(self, goals: List[str]) -> Set[str]:
        return self._chat_evidence_planner().required_evidence_tags(goals)

    def _goal_label(self, goal: str) -> str:
        return self._chat_evidence_planner().goal_label(goal)

    def _goal_labels_if_satisfied(self, goals: List[str], current_tags: Set[str]) -> List[str]:
        return self._chat_evidence_planner().goal_labels_if_satisfied(goals, current_tags)

    def _run_mode_label(self, mode: Optional[str]) -> Optional[str]:
        return self._chat_evidence_planner().run_mode_label(mode)

    def _extract_bulleted_value(self, text: str, key: str) -> Optional[str]:
        return self._chat_evidence_planner().extract_bulleted_value(text, key)

    def _context_run_name(self, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        return self._chat_evidence_planner().context_run_name(outputs)

    def _context_run_mode(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        return self._chat_evidence_planner().context_run_mode(prompt, outputs)

    def _context_log_target(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> str:
        return self._chat_evidence_planner().context_log_target(prompt, outputs)

    def _log_target_label(self, target: str) -> Optional[str]:
        return self._chat_evidence_planner().log_target_label(target)

    def _goal_followup_prompt(
        self,
        goal: str,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> Optional[str]:
        return self._chat_evidence_planner().goal_followup_prompt(goal, prompt, outputs)

    def _candidate_evidence_action_for_goal(
        self,
        goal: str,
        current_tags: Set[str],
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        run_mode: Optional[str],
    ) -> Optional[ToolIntent]:
        return self._chat_evidence_planner().candidate_evidence_action_for_goal(
            goal,
            current_tags,
            prompt,
            outputs,
            run_mode,
        )

    def _simulate_evidence_tags(self, current_tags: Set[str], intent: ToolIntent) -> Set[str]:
        return self._chat_evidence_planner().simulate_evidence_tags(current_tags, intent)

    def _plan_evidence_followups(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> List[ToolIntent]:
        return self._chat_evidence_planner().plan_evidence_followups(prompt, outputs)

    def _describe_evidence_path(self, path: List[ToolIntent]) -> List[str]:
        return self._chat_evidence_planner().describe_evidence_path(path)

    def _unavailable_evidence_strategy(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        goals: List[str],
    ) -> ToolResponseStrategy:
        return self._chat_evidence_planner().unavailable_evidence_strategy(prompt, outputs, goals)

    def _response_strategy_for_prompt(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> ToolResponseStrategy:
        return self._chat_evidence_planner().response_strategy_for_prompt(prompt, outputs)

    def _latest_run_dir(self, mode: Optional[str] = None) -> Optional[Path]:
        items = list_runs(self.output_dir, limit=80)
        if mode:
            items = [item for item in items if item.get("mode") == mode]
        if not items:
            return None
        return Path(items[0]["path"])

    def _chat_with_local_model(self, messages: List[Dict[str, str]], *, purpose: str) -> Dict[str, Any]:
        reply = chat_with_local_model(messages, model=self.selected_model, purpose=purpose)
        if reply.get("model"):
            self.selected_model = reply["model"]
        return reply

    def _build_engine_context(
        self,
        prompt: str = "",
        *,
        current_state: Optional[ToolTurnState] = None,
    ) -> Dict[str, Any]:
        current_blocks = trace_tool_result_blocks(current_state) if current_state is not None else None
        current_outputs = current_state.outputs if current_state is not None else None
        return build_engine_context(
            current_run_dir=self.current_run_dir,
            current_report=self.current_report,
            current_bridge_summary=self.current_bridge_summary,
            current_moire_summary=self.current_moire_summary,
            current_moire_compare_summary=self.current_moire_compare_summary,
            current_moire_diffusion_summary=self.current_moire_diffusion_summary,
            existing_outputs=self._tool_context_outputs,
            existing_blocks=self._tool_context_blocks,
            current_outputs=current_outputs,
            current_blocks=current_blocks,
            output_limit=1600,
            intent_signature=self._intent_signature,
            truncate_output=_truncate_for_model,
            api=globals(),
            active_turn_id=self.active_turn_id,
            current_turn=self._runtime_state.current_turn(),
            pending_tool_requests=self.pending_tool_requests,
            permission_denials=self.permission_denials,
            queued_followups=self.queued_followups,
            usage_stats=self.usage_stats,
            current_state=current_state,
            session_state=self._runtime_state,
        )

    def _build_local_model_messages(self) -> List[Dict[str, str]]:
        engine_context = self._build_engine_context()
        return build_local_model_messages(
            system_prompt=LOCAL_MODEL_SYSTEM_PROMPT,
            compact_history=_compact_local_history(self.history),
            current_context=engine_context.get("current_context"),
            memory_context=(engine_context.get("memory_context") or {}).get("summary"),
            tool_evidence=engine_context.get("tool_evidence"),
            tool_response_style=self._tool_backed_response_style(),
        )

    def _tool_history_payload(self, outputs: List[Tuple[ToolIntent, str]], *, output_limit: int) -> List[Dict[str, Any]]:
        return tool_history_payload(outputs, output_limit=output_limit, truncate_output=_truncate_for_model)

    def _build_tool_router_messages(self, prompt: str, state: Optional[ToolTurnState] = None) -> List[Dict[str, str]]:
        return self._get_query_engine().build_tool_router_messages(prompt, state=state)

    def _build_tool_plan_messages(self, prompt: str, state: Optional[ToolTurnState] = None) -> List[Dict[str, str]]:
        return self._get_query_engine().build_tool_plan_messages(prompt, state=state)

    def _build_agent_loop_messages(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        budget: ToolBudget,
        state: Optional[ToolTurnState] = None,
    ) -> List[Dict[str, str]]:
        return self._get_query_engine().build_agent_loop_messages(prompt, outputs, budget, state=state)

    def _build_tool_summary_messages(
        self,
        original_prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        note: Optional[str] = None,
        state: Optional[ToolTurnState] = None,
    ) -> List[Dict[str, str]]:
        return self._get_query_engine().build_tool_summary_messages(
            original_prompt,
            outputs,
            note=note,
            state=state,
        )

    def _suggest_next_steps(self, outputs: List[Tuple[ToolIntent, str]]) -> List[str]:
        actions = [intent.action for intent, _ in outputs]
        if not actions:
            return ["- 暂时不需要额外动作。"]
        if any(action in {"run", "moire_run", "bridge_kmc_lookup"} for action in actions):
            return [
                "- 如果你愿意，我可以继续检查这次执行的日志、产物，或者直接判断它是否正常结束。",
            ]
        if "draft" in actions:
            return [
                "- 如果这份草案方向对，我可以继续把它真正运行起来，或者先帮你检查草案里是否缺步骤。",
            ]
        if any(action in {"inspect", "logs", "artifacts", "compare_runs", "runs"} for action in actions):
            return [
                "- 如果你愿意，我可以继续深入看日志、对比其他 run，或者把异常点整理成更短的结论。",
            ]
        return ["- 暂时不需要额外动作。"]

    def _maybe_extend_tool_turn(self, prompt: str, state: ToolTurnState) -> ToolResponseStrategy:
        return runtime_maybe_extend_tool_turn(self, prompt, state)

    def _render_plan_outputs(
        self,
        original_prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        note: Optional[str] = None,
    ) -> str:
        return self._get_query_engine().render_plan_outputs(original_prompt, outputs, note=note)

    def _summarize_tool_outputs(
        self,
        original_prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        note: Optional[str] = None,
        state: Optional[ToolTurnState] = None,
    ) -> str:
        return self._get_query_engine().summarize_tool_outputs(
            original_prompt,
            outputs,
            note=note,
            state=state,
        )

    def _finalize_tool_turn(
        self,
        original_prompt: str,
        state: ToolTurnState,
        *,
        summarize: bool = False,
    ) -> Optional[str]:
        return self._get_query_engine().finalize_tool_turn(original_prompt, state, summarize=summarize)

    def _resolve_tool_plan(self, prompt: str, state: Optional[ToolTurnState] = None) -> Optional[ToolPlan]:
        return self._get_query_engine().resolve_tool_plan(prompt, state=state)

    def _resolve_tool_intent(self, prompt: str, state: Optional[ToolTurnState] = None) -> Optional[ToolIntent]:
        return self._get_query_engine().resolve_tool_intent(prompt, state=state)

    def _execute_tool_plan(
        self,
        plan: ToolPlan,
        original_prompt: str,
        state: Optional[ToolTurnState] = None,
        finalize: bool = True,
        raw_plan_content: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_query_engine().execute_tool_plan(
            plan,
            original_prompt,
            state=state,
            finalize=finalize,
            raw_plan_content=raw_plan_content,
        )

    def _handle_engine_plan_branch(
        self,
        prompt: str,
        state: ToolTurnState,
    ) -> Optional[str]:
        return self._get_query_engine().handle_plan_branch(prompt, state)

    def _execute_legacy_router_step(
        self,
        prompt: str,
        state: ToolTurnState,
        intent: ToolIntent,
        *,
        raw_router_content: str,
    ) -> Optional[str]:
        return self._get_query_engine().execute_legacy_router_step(
            prompt,
            state,
            intent,
            raw_router_content=raw_router_content,
        )

    def _parse_legacy_agent_reply(
        self,
        state: ToolTurnState,
        raw_content: str,
    ) -> Optional[AssistantActionBlock]:
        return self._get_query_engine().parse_legacy_agent_reply(state, raw_content)

    def _execute_legacy_agent_block(
        self,
        prompt: str,
        state: ToolTurnState,
        block: AssistantActionBlock,
    ) -> Tuple[str, Optional[str]]:
        return self._get_query_engine().execute_legacy_agent_block(prompt, state, block)

    def _finalize_engine_turn(
        self,
        prompt: str,
        state: ToolTurnState,
    ) -> Tuple[Optional[str], bool]:
        return self._get_query_engine().finalize_engine_turn(prompt, state)

    def _run_engine_turn(
        self,
        prompt: str,
        state: Optional[ToolTurnState] = None,
    ) -> Tuple[Optional[str], ToolTurnState, bool]:
        return self._get_query_engine().execute_engine_turn(prompt, state=state)

    def _heuristic_agent_first_step(self, prompt: str) -> Optional[ToolIntent]:
        lower = prompt.lower()
        run_mode = infer_run_mode_hint(prompt)
        if any(token in prompt or token in lower for token in ["最新", "latest", "最近"]) and any(
            token in prompt or token in lower for token in ["run", "任务", "日志", "异常", "正常", "结束", "退出"]
        ):
            params: Dict[str, Any] = {"run": "latest"}
            if run_mode:
                params["mode"] = run_mode
            return ToolIntent(action="inspect", params=params)
        if any(token in prompt or token in lower for token in ["日志", "log"]) and any(
            token in prompt or token in lower for token in ["检查", "判断", "分析", "总结"]
        ):
            target = "kmc" if "kmc" in lower else "auto"
            params = {"run": "latest", "target": target}
            if run_mode:
                params["mode"] = run_mode
            return ToolIntent(action="logs", params=params)
        return None

    def _forced_log_target(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        lower = prompt.lower()
        asks_reason = any(
            token in prompt or token in lower
            for token in ["为什么", "原因", "根因", "报错", "失败", "why", "reason", "root cause", "error", "failed"]
        )
        asks_normality = any(
            token in prompt or token in lower
            for token in ["是否正常", "正常吗", "异常", "异常退出", "正常结束", "跑崩", "跑挂", "判断", "健康吗"]
        )
        if not asks_reason and not asks_normality:
            return None
        if any(step.action == "logs" for step, _ in outputs):
            return None
        inspect_outputs = [result for step, result in outputs if step.action == "inspect"]
        if not inspect_outputs:
            return None
        merged = "\n".join(inspect_outputs)
        lowered = merged.lower()

        failed_steps = {name.lower() for name, status in re.findall(r"-\s+([a-zA-Z0-9_.-]+):\s+([a-zA-Z0-9_.-]+)", merged) if status.lower() == "failed"}
        if any(step.startswith("kmc.") for step in failed_steps):
            return "kmc"
        if any(step.startswith("md.") for step in failed_steps):
            return "md"

        if asks_reason:
            if any(token in lowered for token in ["workflow: md_only", "mode: md_only", "md.run"]):
                if not any(token in lowered for token in ["workflow: kmc_only", "mode: kmc_only", "kmc.run", "barrier source", "k m c"]):
                    return "md"
            if any(token in merged for token in ["kmc.run", "mode: kmc_only", "KMC", "barrier source"]):
                return "kmc"

        if asks_normality and any(token in merged for token in ["kmc.run", "mode: kmc_only", "KMC", "barrier source"]):
            return "kmc"
        return None

    def _run_agent_loop(self, prompt: str, state: Optional[ToolTurnState] = None) -> Optional[str]:
        return self._get_query_engine().run_agent_loop(prompt, state=state)

    def _run_turn(self, prompt: str) -> QueryTurnResult:
        return self._get_query_engine().run_turn(prompt)

    def _run_tool_turn(self, prompt: str) -> Optional[str]:
        result = self._get_query_engine().run_tool_turn(prompt)
        return result.reply or None

    def _run_manual_tool_intent(self, intent: ToolIntent, original_prompt: str) -> str:
        turn = self._new_tool_turn_state(default_steps=1)
        self._apply_tool_intent_to_turn(turn, intent, original_prompt, manual=True, source="manual_command")
        self._remember_tool_turn(turn)
        return self._finalize_tool_turn(original_prompt, turn, summarize=False) or ""

    def _execute_tool_intent_outcome(self, intent: ToolIntent, original_prompt: str) -> ToolExecutionOutcome:
        return execute_chat_tool_intent_outcome(self, intent, original_prompt, api=globals())

    def _execute_tool_intent(self, intent: ToolIntent, original_prompt: str) -> Optional[str]:
        return self._execute_tool_intent_outcome(intent, original_prompt).output

    def _handle_command(self, command_line: str) -> str:
        return handle_shell_command(self, command_line, resolve_model_alias=_resolve_model_alias)

    def _handle_prompt(self, prompt: str) -> str:
        return self._get_query_engine().handle_prompt_turn(prompt).reply

    def _progress_hook(self, stage: str, payload: Dict[str, Any]) -> None:
        self._emit_progress(format_progress_event(stage, payload))

    def _use_internal_mcp(self) -> bool:
        return os.environ.get("MIETCLAW_INTERNAL_MCP", "1") != "0"

    def _preview_before_run_enabled(self) -> bool:
        return os.environ.get("MIETCLAW_PREVIEW_RUNS", "0") == "1"

    def _call_local_mcp_tool(self, tool_name: str, arguments: Dict[str, Any], purpose: str) -> Dict[str, Any]:
        self._progress_hook("mcp.tool.start", {"tool": tool_name, "purpose": purpose})
        client = self._get_local_mcp_client()
        result = client.call_tool(tool_name, arguments)
        self._progress_hook("mcp.tool.complete", {"tool": tool_name, "purpose": purpose})
        return result

    def _run_draft(self, prompt: str) -> str:
        report = materialize_autonomy_workspace(
            prompt=prompt,
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            provider=self.provider,
            mode_hint=self.mode_hint,
            progress_callback=self._progress_hook,
        )
        self.current_report = report
        return format_draft_report(report)

    def _run_preview(self, prompt: str) -> str:
        report = materialize_autonomy_workspace(
            prompt=prompt,
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            provider=self.provider,
            mode_hint=self.mode_hint,
            progress_callback=self._progress_hook,
        )
        self.current_report = report
        draft_lines = format_draft_report(report).splitlines()
        remainder = "\n".join(draft_lines[1:]) if len(draft_lines) > 1 else ""
        return "\n".join(
            [
                stylize("Execution preview ready", BOLD, GREEN),
                "- nothing has run yet",
                remainder,
            ]
        ).strip()

    def _run_job(
        self,
        prompt: str,
        *,
        resume_existing: bool = False,
        overwrite_existing: bool = False,
        dry_run_only: bool = False,
    ) -> str:
        report = run_autonomy_job(
            prompt=prompt,
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            provider=self.provider,
            mode_hint=self.mode_hint,
            output_dir=str(self.output_dir),
            dry_run_only=dry_run_only,
            resume_existing=resume_existing,
            overwrite_existing=overwrite_existing,
            progress_callback=self._progress_hook,
        )
        self.current_report = report
        final_run_dir = report.get("execution", {}).get("final_run_dir")
        if final_run_dir:
            self.current_run_dir = Path(final_run_dir)
        return format_run_report(report)

    def _run_bridge(self, *, event_json: str, neb_txt: str, workdir: str, validate: bool = True) -> str:
        try:
            if self._use_internal_mcp():
                result = self._call_local_mcp_tool(
                    "miet_kmc_bridge",
                    {
                        "event_json": event_json,
                        "neb_txt": neb_txt,
                        "workdir": workdir,
                        "validate": validate,
                    },
                    "根据 LAMMPS 的迁移能垒生成 KMC 输入并运行本地 misa-kmc",
                )
                summary = dict((result.get("structuredContent") or {}))
                summary["dispatch"] = {
                    "transport": "local stdio MCP",
                    "tool": "miet_kmc_bridge",
                }
            else:
                summary = run_kmc_lookup_bridge(
                    event_json=event_json,
                    neb_txt=neb_txt,
                    workdir=workdir,
                    validate=validate,
                    progress_callback=self._progress_hook,
                )
        except (BridgeError, MCPClientError) as exc:
            return f"KMC bridge 失败：{exc}"
        self.current_bridge_summary = summary
        self.current_moire_diffusion_summary = None
        return format_bridge_report(summary)

    def _run_moire_workflow(
        self,
        *,
        event_json: Optional[str],
        case_dir: str,
        workdir: str,
        validate: bool = True,
        kmc_seed: Optional[int] = None,
        kmc_seeds: Optional[List[int]] = None,
        ovito: bool = False,
        ovito_python: Optional[str] = None,
    ) -> str:
        try:
            if self._use_internal_mcp():
                tool_args: Dict[str, Any] = {
                    "case_dir": case_dir,
                    "workdir": workdir,
                    "validate": validate,
                }
                if event_json:
                    tool_args["event_json"] = event_json
                if kmc_seed is not None:
                    tool_args["kmc_seed"] = kmc_seed
                if kmc_seeds:
                    tool_args["kmc_seeds"] = kmc_seeds
                if ovito:
                    tool_args["ovito"] = True
                if ovito_python:
                    tool_args["ovito_python"] = ovito_python
                result = self._call_local_mcp_tool(
                    "miet_moire_run",
                    tool_args,
                    "调用本地 MoRe LAMMPS/NEB 并继续进入 repo KMC",
                )
                summary = dict((result.get("structuredContent") or {}))
                mcp_dispatch = {
                    "transport": "local stdio MCP",
                    "tool": "miet_moire_run",
                }
                summary["dispatch"] = {
                    "workflow": mcp_dispatch,
                    "lammps": mcp_dispatch,
                    "kmc": mcp_dispatch,
                }
                if (summary.get("runtime_health") or {}).get("status") == "failed":
                    raise MCPClientError("repo KMC runtime health failed.")
            else:
                summary = run_moire_lammps_to_kmc(
                    event_json=event_json,
                    case_dir=case_dir,
                    workdir=workdir,
                    validate=validate,
                    kmc_seed=kmc_seed,
                    kmc_seeds=kmc_seeds,
                    render_ovito=ovito,
                    ovito_python=ovito_python,
                    progress_callback=self._progress_hook,
                )
        except (MoReWorkflowError, MCPClientError, KeyError) as exc:
            return f"MoRe LAMMPS→KMC 失败：{exc}"
        self.current_moire_summary = summary
        self.current_moire_compare_summary = None
        self.current_moire_diffusion_summary = None
        self.current_bridge_summary = summary.get("kmc")
        return format_moire_workflow_report(summary)

    def _run_moire_compare_workflow(
        self,
        *,
        case_dir: str,
        event_jsons: List[str],
        workdir: str,
        validate: bool = True,
        kmc_seed: Optional[int] = None,
        kmc_seeds: Optional[List[int]] = None,
        ovito: bool = False,
        ovito_python: Optional[str] = None,
        lammps_only: bool = False,
    ) -> str:
        try:
            if self._use_internal_mcp():
                tool_args: Dict[str, Any] = {
                    "case_dir": case_dir,
                    "event_jsons": event_jsons,
                    "workdir": workdir,
                    "validate": validate,
                    "lammps_only": lammps_only,
                }
                if kmc_seed is not None:
                    tool_args["kmc_seed"] = kmc_seed
                if kmc_seeds:
                    tool_args["kmc_seeds"] = kmc_seeds
                if ovito:
                    tool_args["ovito"] = True
                if ovito_python:
                    tool_args["ovito_python"] = ovito_python
                result = self._call_local_mcp_tool(
                    "miet_moire_compare",
                    tool_args,
                    "比较多个 MoRe 事件在本地 LAMMPS/NEB 和 repo KMC 上的结果",
                )
                summary = dict((result.get("structuredContent") or {}))
                summary["dispatch"] = {
                    "transport": "local stdio MCP",
                    "tool": "miet_moire_compare",
                }
            else:
                summary = run_moire_event_compare(
                    case_dir=case_dir,
                    event_jsons=event_jsons,
                    workdir=workdir,
                    validate=validate,
                    kmc_seed=kmc_seed,
                    kmc_seeds=kmc_seeds,
                    run_kmc=not lammps_only,
                    render_ovito=ovito,
                    ovito_python=ovito_python,
                    progress_callback=self._progress_hook,
                )
        except (MoReWorkflowError, MCPClientError, KeyError) as exc:
            return f"MoRe 多事件比较失败：{exc}"
        self.current_moire_compare_summary = summary
        self.current_moire_summary = None
        self.current_moire_diffusion_summary = None
        self.current_bridge_summary = None
        return format_moire_compare_report(summary)

    def _run_moire_diffusion_workflow(
        self,
        *,
        event_json: str,
        case_dir: str,
        workdir: str,
        validate: bool = True,
        temperatures_k: Optional[List[float]] = None,
        kmc_seed: Optional[int] = None,
        kmc_seeds: Optional[List[int]] = None,
        run_time: Optional[str] = None,
        stats_step: Optional[str] = None,
        ovito: bool = False,
        ovito_python: Optional[str] = None,
    ) -> str:
        try:
            if self._use_internal_mcp():
                tool_args: Dict[str, Any] = {
                    "event_json": event_json,
                    "case_dir": case_dir,
                    "workdir": workdir,
                    "validate": validate,
                }
                if temperatures_k:
                    tool_args["temperatures_k"] = temperatures_k
                if kmc_seed is not None:
                    tool_args["kmc_seed"] = kmc_seed
                if kmc_seeds:
                    tool_args["kmc_seeds"] = kmc_seeds
                if run_time:
                    tool_args["run_time"] = run_time
                if stats_step:
                    tool_args["stats_step"] = stats_step
                if ovito:
                    tool_args["ovito"] = True
                if ovito_python:
                    tool_args["ovito_python"] = ovito_python
                result = self._call_local_mcp_tool(
                    "miet_moire_diffusion_sweep",
                    tool_args,
                    "先用本地 MoRe LAMMPS/NEB 算 barrier，再扫温度得到扩散系数-温度关系",
                )
                summary = dict((result.get("structuredContent") or {}))
                summary["dispatch"] = {
                    "transport": "local stdio MCP",
                    "tool": "miet_moire_diffusion_sweep",
                }
            else:
                summary = run_moire_diffusion_sweep(
                    event_json=event_json,
                    case_dir=case_dir,
                    workdir=workdir,
                    temperatures_k=temperatures_k,
                    validate=validate,
                    kmc_seed=kmc_seed,
                    kmc_seeds=kmc_seeds,
                    render_ovito=ovito,
                    ovito_python=ovito_python,
                    run_time=run_time or "1e-6",
                    stats_step=stats_step or "1e-7",
                    progress_callback=self._progress_hook,
                )
        except (MoReWorkflowError, MCPClientError, KeyError) as exc:
            return f"MoRe 扩散扫温失败：{exc}"
        self.current_moire_diffusion_summary = summary
        self.current_moire_summary = None
        self.current_moire_compare_summary = None
        self.current_bridge_summary = None
        return format_moire_diffusion_sweep_report(summary)

    def side_panel_text(self) -> str:
        lines = [
            "mietclaw",
            self._format_model_status(),
            f"provider: {self.provider}",
            f"workspace: {self.workspace_root}",
            f"transcript: {self.transcript_path.name}",
            "",
        ]
        if self.current_run_dir:
            info = inspect_run(self.current_run_dir)
            lines.extend(
                [
                    "Current run",
                    f"- job_id: {info['job_id']}",
                    f"- workflow: {info.get('workflow_kind')}",
                    f"- barrier source: {info.get('barrier_source_mode')}",
                ]
            )
            if info.get("neb_images") is not None:
                lines.append(f"- NEB images: {info['neb_images']}")
            for event in info.get("events", [])[:4]:
                lines.append(f"- {event['species']} {float(event['barrier_ev']):.4f} eV")
            lines.append("")
        elif self.current_moire_compare_summary:
            lines.extend(
                [
                    "Current MoRe compare",
                    f"- case: {self.current_moire_compare_summary.get('case_dir')}",
                    f"- completed: {self.current_moire_compare_summary.get('completed_count')}/{self.current_moire_compare_summary.get('event_count')}",
                ]
            )
            for item in (self.current_moire_compare_summary.get("barrier_ranking") or [])[:4]:
                lines.append(f"- {item['label']} {float(item['barrier_eV']):.4f} eV")
            lines.append("")
        elif self.current_moire_diffusion_summary:
            lines.extend(
                [
                    "Current diffusion sweep",
                    f"- case: {self.current_moire_diffusion_summary.get('case_dir')}",
                    f"- barrier: {float(self.current_moire_diffusion_summary.get('barrier_eV') or 0.0):.4f} eV",
                    f"- temperatures: {self.current_moire_diffusion_summary.get('completed_count')}/{len(self.current_moire_diffusion_summary.get('temperature_runs') or [])}",
                ]
            )
            for item in (self.current_moire_diffusion_summary.get("temperature_runs") or [])[:4]:
                coeff = item.get("diffusion_coefficient")
                coeff_text = "—" if coeff is None else f"{float(coeff):.4g}"
                lines.append(f"- {item.get('label')} D={coeff_text}")
            lines.append("")
        elif self.current_report:
            lines.extend(
                [
                    "Current draft",
                    f"- job_id: {self.current_report['job_id']}",
                    f"- mode: {self.current_report['mode']}",
                    f"- template: {self.current_report.get('selected_template', {}).get('file_name')}",
                    "",
                ]
            )
        elif self.current_moire_summary:
            lines.extend(["Current MoRe workflow", format_moire_workflow_report(self.current_moire_summary), ""])
        elif self.current_bridge_summary:
            if self.current_bridge_summary.get("parsed_run") is not None:
                parsed_run = self.current_bridge_summary.get("parsed_run") or {}
                lines.extend(
                    [
                        "Current repo KMC",
                        f"- barrier: {float(self.current_bridge_summary.get('barrier_eV', 0.0)):.4f} eV",
                        f"- accepted: {parsed_run.get('accepted_events')}",
                        f"- final time: {parsed_run.get('final_time')}",
                        "",
                    ]
                )
            else:
                validation = self.current_bridge_summary.get("validation") or {}
                lines.extend(
                    [
                        "Current bridge",
                        f"- barrier: {float(self.current_bridge_summary.get('barrier_eV', 0.0)):.4f} eV",
                        f"- hits: {validation.get('lookup_hits')}",
                        f"- misses: {validation.get('live_ml_misses')}",
                        "",
                    ]
                )
        lines.extend(
            [
                "Commands",
                "/status",
                "/doctor",
                "/tools",
                "/model",
                "/runs",
                "/inspect <run>",
                "/draft <prompt>",
                "/run <prompt>",
                "/bridge ...",
                "/moire-run ...",
                "/moire-compare ...",
                "/moire-diffusion-sweep ...",
                "/exit",
            ]
        )
        return "\n".join(lines)


def _draw_box(stdscr: "curses._CursesWindow", y: int, x: int, h: int, w: int, title: str) -> None:
    if h < 3 or w < 4:
        return
    try:
        stdscr.addstr(y, x, "┌" + "─" * (w - 2) + "┐")
        for row in range(y + 1, y + h - 1):
            stdscr.addstr(row, x, "│")
            stdscr.addstr(row, x + w - 1, "│")
        stdscr.addstr(y + h - 1, x, "└" + "─" * (w - 2) + "┘")
        stdscr.addstr(y, x + 2, f" {title} "[: max(0, w - 4)])
    except curses.error:
        pass


def _draw_text_block(stdscr: "curses._CursesWindow", y: int, x: int, h: int, w: int, text: str) -> None:
    lines = _wrap_lines(text, max(8, w - 2))
    visible = lines[-max(0, h - 2):]
    for idx, line in enumerate(visible, start=1):
        try:
            stdscr.addstr(y + idx, x + 1, line[: max(0, w - 2)])
        except curses.error:
            pass


def _render_history(entries: List[Tuple[str, str]], width: int) -> str:
    lines: List[str] = []
    for role, text in entries[-14:]:
        prefix = "you>" if role == "user" else "miet>"
        lines.append(f"{prefix} {text}")
        lines.append("")
    return "\n".join(lines)


def render_tui_snapshot(
    session: MietClawChatSession,
    width: int = 140,
    height: int = 28,
    status: str = "Ready",
    input_buffer: str = "",
) -> str:
    left_w = max(28, min(36, width // 4))
    right_w = max(34, min(44, width // 3))
    center_w = width - left_w - right_w
    top_h = height - 4

    def box(title: str, content: str, w: int, h: int) -> List[str]:
        inner_w = max(1, w - 2)
        lines = [f"┌{'─' * (w - 2)}┐", f"│{(' ' + title + ' ')[:inner_w]:<{inner_w}}│"]
        wrapped = _wrap_lines(content, inner_w)
        for line in wrapped[: max(0, h - 3)]:
            lines.append(f"│{line[:inner_w]:<{inner_w}}│")
        while len(lines) < h - 1:
            lines.append(f"│{'':<{inner_w}}│")
        lines.append(f"└{'─' * (w - 2)}┘")
        return lines

    run_lines = []
    for item in list_runs(session.output_dir, limit=12):
        marker = "*" if session.current_run_dir and Path(item["path"]) == session.current_run_dir else " "
        run_lines.append(f"{marker} {item['job_id']} [{item['status']}] {item['completed_steps']}/{item['total_steps']}")
        run_lines.append(f"  {item['mode']} · {item['updated_at']}")
        run_lines.append(f"  {_shorten(item['material_name'], max(12, left_w - 4))}")
        run_lines.append("")

    left_box = box("Runs", "\n".join(run_lines), left_w, top_h)
    center_box = box("Conversation", _render_history(session.history or [("assistant", session.banner())], center_w - 2), center_w, top_h)
    right_box = box("Context", session.side_panel_text(), right_w, top_h)
    input_box = box("Input", f"{status}\n\nmietclaw> {input_buffer}", width, 4)

    rows = []
    for idx in range(top_h):
        rows.append(left_box[idx] + center_box[idx] + right_box[idx])
    rows.extend(input_box)
    return "\n".join(rows)


def run_chat_tui(session: MietClawChatSession) -> int:
    event_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    busy = {"value": False}
    status = {"value": "Ready"}
    input_buffer = {"value": ""}

    def progress_sink(line: str) -> None:
        event_queue.put(("progress", line))

    session.progress_callback = progress_sink

    def worker(command: str) -> None:
        try:
            result = session.handle_line(command)
            event_queue.put(("result", result))
        except EOFError:
            event_queue.put(("exit", ""))
        except Exception as exc:  # noqa: BLE001
            event_queue.put(("error", f"出错了：{exc}"))

    def app(stdscr: "curses._CursesWindow") -> int:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.nodelay(True)
        stdscr.timeout(100)
        spinner_index = 0
        chat_messages: List[Tuple[str, str]] = [("assistant", session.banner())]

        while True:
            while True:
                try:
                    kind, payload = event_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    chat_messages.append(("assistant", payload))
                    status["value"] = payload
                elif kind == "result":
                    chat_messages.append(("assistant", payload))
                    busy["value"] = False
                    status["value"] = "Ready"
                elif kind == "error":
                    chat_messages.append(("assistant", payload))
                    busy["value"] = False
                    status["value"] = "Error"
                elif kind == "exit":
                    return 0

            stdscr.erase()
            height, width = stdscr.getmaxyx()
            if height < 18 or width < 90:
                try:
                    stdscr.addstr(0, 0, "Terminal too small for mietclaw TUI. Resize or run with --ui plain.")
                except curses.error:
                    pass
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (3, 4):
                    return 0
                continue

            left_w = max(28, min(36, width // 4))
            right_w = max(34, min(44, width // 3))
            center_w = width - left_w - right_w
            top_h = height - 4

            _draw_box(stdscr, 0, 0, top_h, left_w, "Runs")
            _draw_box(stdscr, 0, left_w, top_h, center_w, "Conversation")
            _draw_box(stdscr, 0, left_w + center_w, top_h, right_w, "Context")
            _draw_box(stdscr, top_h, 0, 4, width, "Input")

            run_lines = []
            for item in list_runs(session.output_dir, limit=12):
                marker = "*" if session.current_run_dir and Path(item["path"]) == session.current_run_dir else " "
                run_lines.append(f"{marker} {item['job_id']} [{item['status']}] {item['completed_steps']}/{item['total_steps']}")
                run_lines.append(f"  {item['mode']} · {item['updated_at']}")
                run_lines.append(f"  {_shorten(item['material_name'], max(12, left_w - 4))}")
                run_lines.append("")
            _draw_text_block(stdscr, 0, 0, top_h, left_w, "\n".join(run_lines))

            _draw_text_block(stdscr, 0, left_w, top_h, center_w, _render_history(chat_messages + session.history, max(12, center_w - 2)))
            _draw_text_block(stdscr, 0, left_w + center_w, top_h, right_w, session.side_panel_text())

            spinner = SPINNER[spinner_index % len(SPINNER)] if busy["value"] else "•"
            spinner_index += 1
            status_line = f"{spinner} {status['value']}"
            try:
                stdscr.addstr(top_h + 1, 2, status_line[: width - 4])
                stdscr.addstr(top_h + 2, 2, f"mietclaw> {input_buffer['value']}"[: width - 4])
                stdscr.move(top_h + 2, min(width - 3, len("mietclaw> ") + 2 + len(input_buffer["value"])))
            except curses.error:
                pass
            stdscr.refresh()

            ch = stdscr.getch()
            if ch == -1:
                continue
            if ch in (3, 4):
                return 0
            if ch in (10, 13):
                command = input_buffer["value"].strip()
                input_buffer["value"] = ""
                if not command or busy["value"]:
                    continue
                if command in ("/exit", "/quit"):
                    return 0
                chat_messages.append(("user", command))
                busy["value"] = True
                status["value"] = "Working..."
                threading.Thread(target=worker, args=(command,), daemon=True).start()
                continue
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                input_buffer["value"] = input_buffer["value"][:-1]
                continue
            if ch == 12:
                chat_messages.clear()
                continue
            if 32 <= ch <= 126:
                input_buffer["value"] += chr(ch)
        return 0

    return curses.wrapper(app)


def run_chat(
    project_root: str,
    workspace_root: str,
    output_dir: str,
    provider: str = "auto",
    mode_hint: Optional[str] = None,
    model: Optional[str] = None,
    once: Optional[str] = None,
    ui: str = "plain",
) -> int:
    def plain_progress_sink(line: str) -> None:
        print(line)

    session = MietClawChatSession(
        project_root=project_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        provider=provider,
        mode_hint=mode_hint,
        initial_model=model,
    )
    try:
        if once:
            session.progress_callback = plain_progress_sink
            print(session.banner())
            print()
            print(stylize("user>", BOLD), once)
            print()
            print(session.handle_line(once))
            return 0

        if ui == "tui" and sys.stdin.isatty() and sys.stdout.isatty():
            return run_chat_tui(session)

        session.progress_callback = plain_progress_sink
        print(session.banner())
        while True:
            try:
                line = input(stylize("mietclaw> ", BOLD, CYAN))
            except EOFError:
                print()
                break
            try:
                output = session.handle_line(line)
            except EOFError:
                print(stylize("会话已结束。", DIM))
                break
            except Exception as exc:  # noqa: BLE001
                output = stylize(f"出错了：{exc}", RED)
            if output:
                print()
                print(output)
                print()
        return 0
    finally:
        session.close()


def run_chat_once_payload(
    *,
    project_root: str,
    workspace_root: str,
    output_dir: str,
    prompt: str,
    provider: str = "auto",
    mode_hint: Optional[str] = None,
    model: Optional[str] = None,
    history_messages: Optional[Union[List[Dict[str, Any]], List[Tuple[str, str]]]] = None,
) -> Dict[str, Any]:
    progress_lines: List[str] = []

    session = MietClawChatSession(
        project_root=project_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        provider=provider,
        mode_hint=mode_hint,
        initial_model=model,
    )
    if history_messages:
        session.load_history(history_messages)
    session.progress_callback = progress_lines.append
    try:
        return session._get_query_engine().run_once_payload(prompt, progress_lines)
    finally:
        session.close()
