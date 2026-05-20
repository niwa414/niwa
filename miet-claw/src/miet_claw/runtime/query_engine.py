from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tool_router import (
    AGENT_LOOP_SYSTEM_PROMPT,
    TOOL_PLAN_SYSTEM_PROMPT,
    TOOL_ROUTER_SYSTEM_PROMPT,
    ToolIntent,
    ToolPlan,
    heuristic_tool_intent,
    heuristic_tool_plan,
    should_skip_tool_router,
    should_try_agent_loop,
    should_try_tool_plan,
)
from .query_loop import (
    build_agent_loop_messages,
    build_tool_plan_messages,
    build_tool_router_messages,
    build_tool_summary_messages,
    legacy_reply_to_blocks,
    tool_result_summary_entries,
    trace_tool_result_blocks,
)
from .query_blocks import (
    blocks_to_tool_intent,
    blocks_to_tool_plan,
    parse_native_action_blocks,
    synthetic_router_blocks,
)
from .query_followups import (
    can_auto_continue_followup_item,
    describe_followup_intent,
    format_queued_followups as render_queued_followups,
    intent_from_followup_item,
    queued_followup_prompt,
    resume_source_for_followup,
)
from .approval import approval_policy_name
from .permissions import read_bounded_int_env
from .snapshot import (
    build_chat_once_payload,
    build_snapshot_from_turn_result,
    build_tool_trace_replay,
    build_tool_trace_summary,
)
from .types import (
    AssistantActionBlock,
    AssistantActionBlockEvent,
    AssistantTurnEvent,
    FinalAnswerBlock,
    ToolRequestBlock,
    TurnFinishEvent,
    ToolTurnState,
)


@dataclass
class QueryTurnResult:
    turn_id: str
    reply: str
    used_tools: bool
    status: str
    tool_turn_state: Optional[ToolTurnState] = None
    assistant_blocks: List[AssistantActionBlock] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    progress_lines: List[str] = field(default_factory=list)
    followups: List[Dict[str, Any]] = field(default_factory=list)
    turn_usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


class MietQueryEngine:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.state = session._runtime_state

    def _recent_run_names(self, limit: int = 8) -> List[str]:
        output_dir = Path(self.session.output_dir)
        if not output_dir.exists():
            return []
        candidates = [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and ((path / "state.json").exists() or (path / "summary.json").exists())
        ]
        ordered = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        return [path.name for path in ordered[:limit]]

    def _truncate_for_model(self, text: str, limit: int = 3500) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def submit_prompt(self, prompt: str) -> str:
        return self.run_turn(prompt).reply

    def _latest_turn_id(self) -> Optional[str]:
        if self.state.active_turn_id:
            return self.state.active_turn_id
        if not self.state.turns:
            return None
        return self.state.turns[-1].get("turn_id")

    def resolve_turn_reference(self, turn_ref: Optional[str] = None) -> Optional[str]:
        text = str(turn_ref or "").strip()
        if not text or text == "latest":
            return self._latest_turn_id()
        if text == "active":
            return self.state.active_turn_id or self._latest_turn_id()
        for item in reversed(self.state.turns):
            if str(item.get("turn_id") or "") == text:
                return text
        return None

    def list_queued_followups(
        self,
        *,
        limit: int = 8,
        include_consumed: bool = False,
    ) -> List[Dict[str, Any]]:
        getter = getattr(self.state, "queued_followup_items", None)
        if not callable(getter):
            return []
        return getter(limit=limit, include_consumed=include_consumed)

    def format_queued_followups(self, *, limit: int = 8) -> str:
        return render_queued_followups(self.list_queued_followups(limit=limit))

    def _resume_source_for_followup(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return resume_source_for_followup(item)

    def _queued_followup_prompt(self, item: Dict[str, Any], *, intent: Optional[ToolIntent] = None) -> str:
        return queued_followup_prompt(item, intent=intent)

    def _intent_from_followup_item(self, item: Optional[Dict[str, Any]]) -> Optional[ToolIntent]:
        return intent_from_followup_item(item)

    def _describe_followup_intent(self, intent: ToolIntent) -> str:
        return describe_followup_intent(intent)

    def _can_auto_continue_followup(self, item: Dict[str, Any]) -> bool:
        max_attempts = read_bounded_int_env("MIETCLAW_AUTO_FOLLOWUP_MAX_ATTEMPTS", 2, 1, 4)
        return can_auto_continue_followup_item(
            item,
            intent=self._intent_from_followup_item(item),
            approval_policy=approval_policy_name(),
            has_pending_tool_requests=bool(self.state.pending_tool_requests),
            max_attempts=max_attempts,
        )

    def _auto_followup_limit(self) -> int:
        return read_bounded_int_env("MIETCLAW_AUTO_FOLLOWUP_MAX_TURNS", 2, 0, 6)

    def _run_followup_intent(
        self,
        item: Dict[str, Any],
        *,
        prompt: str,
        turn_id: str,
        source: str,
    ) -> QueryTurnResult:
        previous_tool_turn = self.session.last_tool_turn_state
        self._record_turn_message(turn_id, "user", prompt)
        self.session._record_chat_message("user", prompt)
        intent = self._intent_from_followup_item(item)
        if intent is None:
            return QueryTurnResult(
                turn_id=turn_id,
                reply="这个 follow-up 没有可执行的 intent。",
                used_tools=False,
                status="error",
                finish_reason="missing followup intent",
            )
        turn = self.session._new_tool_turn_state(default_steps=2, step_env="MIETCLAW_AGENT_MAX_STEPS")
        self.session._apply_tool_intent_to_turn(
            turn,
            intent,
            prompt,
            manual=False,
            source=source,
        )
        reply, used_tools = self.finalize_engine_turn(prompt, turn)
        result = self._finalize_turn(
            prompt=prompt,
            turn_id=turn_id,
            reply=reply or "",
            status="tool" if (used_tools or turn.outputs or turn.notes or turn.trace.events) else "empty",
            used_tools=bool(used_tools or turn.outputs or turn.notes or turn.trace.events),
            tool_turn_state=turn if (used_tools or turn.outputs or turn.notes or turn.trace.events) else None,
        )
        if result.reply or result.used_tools or result.status != "empty":
            self.commit_turn(result)
        self.session._append_tool_trace_if_new(previous_tool_turn)
        self._record_turn_message(turn_id, "assistant", result.reply)
        self.session._record_chat_message("assistant", result.reply)
        return result

    def run_queued_followup(
        self,
        followup_id: Optional[str] = None,
        *,
        auto_background: bool = True,
        started_via: str = "continue",
    ) -> Optional[QueryTurnResult]:
        suspended_command_turn_id: Optional[str] = None
        if self.state.active_turn_id:
            current_turn = self.state.current_turn() or {}
            current_prompt = str(current_turn.get("prompt") or "").strip()
            if current_prompt.startswith("/"):
                suspended_command_turn_id = self.state.active_turn_id
                self.state.active_turn_id = None
            else:
                return QueryTurnResult(
                    turn_id=self.state.active_turn_id,
                    reply="当前已经有一个进行中的 turn，先等它结束后再继续 follow-up。",
                    used_tools=False,
                    status="busy",
                    finish_reason="active turn in progress",
                )
        if followup_id:
            item = next(
                (
                    dict(entry)
                    for entry in self.state.queued_followups
                    if str(entry.get("followup_id") or "") == str(followup_id)
                    and not bool(entry.get("consumed"))
                ),
                None,
            )
        else:
            getter = getattr(self.state, "next_queued_followup", None)
            item = getter(runnable_only=True) if callable(getter) else None
        if not item:
            if suspended_command_turn_id and not self.state.active_turn_id:
                self.state.active_turn_id = suspended_command_turn_id
            return None
        intent = self._intent_from_followup_item(item)
        prompt = self._queued_followup_prompt(item, intent=intent)
        if not prompt:
            if suspended_command_turn_id and not self.state.active_turn_id:
                self.state.active_turn_id = suspended_command_turn_id
            return None
        try:
            turn_id = self._start_turn(prompt, resume_source=self._resume_source_for_followup(item))
            started_item = self.state.consume_queued_followup(
                str(item.get("followup_id") or ""),
                turn_id=turn_id,
                status="running",
                attempt=True,
                extra={"started_via": started_via, "started_at": item.get("started_at") or time.time()},
            )
            if intent is not None:
                result = self._run_followup_intent(
                    item,
                    prompt=prompt,
                    turn_id=turn_id,
                    source="background_followup" if started_via == "background_auto" else "queued_followup",
                )
            else:
                result = self.handle_prompt_turn(prompt, auto_background=auto_background)
            self.state.update_queued_followup(
                str(item.get("followup_id") or ""),
                status="completed",
                completed_at=(started_item or {}).get("consumed_at"),
                completed_turn_id=result.turn_id,
                completed_status=result.status,
                completed_finish_reason=result.finish_reason or result.status,
                completed_reply_preview=(result.reply or "")[:240],
            )
            return result
        finally:
            if suspended_command_turn_id and not self.state.active_turn_id:
                self.state.active_turn_id = suspended_command_turn_id

    def continue_queued_followup(self, followup_id: Optional[str] = None) -> str:
        result = self.run_queued_followup(followup_id)
        if result is None:
            return "当前没有可继续执行的 follow-up。"
        return result.reply

    def run_background_followups(
        self,
        *,
        source_turn_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[QueryTurnResult]:
        if getattr(self, "_background_followups_running", False):
            return []
        max_turns = self._auto_followup_limit() if limit is None else max(0, limit)
        if max_turns <= 0:
            return []
        results: List[QueryTurnResult] = []
        self._background_followups_running = True
        try:
            for _ in range(max_turns):
                next_item = next(
                    (
                        item
                        for item in self.state.queued_followup_items(
                            runnable_only=True,
                            auto_only=True,
                            source_turn_id=source_turn_id,
                        )
                        if self._can_auto_continue_followup(item)
                    ),
                    None,
                )
                if not next_item:
                    break
                result = self.run_queued_followup(
                    str(next_item.get("followup_id") or ""),
                    auto_background=False,
                    started_via="background_auto",
                )
                if result is None:
                    break
                results.append(result)
            return results
        finally:
            self._background_followups_running = False

    def drain_queued_followups(self, *, limit: int = 3) -> str:
        if limit <= 0:
            limit = 1
        results: List[QueryTurnResult] = []
        for _ in range(limit):
            next_item = self.state.next_queued_followup(runnable_only=True)
            if not next_item:
                break
            result = self.run_queued_followup(
                str(next_item.get("followup_id") or ""),
                auto_background=False,
                started_via="continue_all",
            )
            if result is None:
                break
            results.append(result)
        if not results:
            return "当前没有可自动继续处理的 follow-up。"
        lines = [f"已连续处理 {len(results)} 个 follow-up："]
        for index, result in enumerate(results, start=1):
            preview = (result.reply or "").strip().replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:177] + "..."
            lines.append(f"{index}. {result.turn_id} · {result.status}")
            lines.append(f"   {preview or '—'}")
        remaining = len(self.list_queued_followups())
        lines.append(f"剩余待处理 follow-up：{remaining}")
        return "\n".join(lines)

    def _start_turn(
        self,
        prompt: str,
        *,
        resume_source: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.state.active_turn_id:
            return self.state.active_turn_id
        if resume_source:
            return self.state.begin_resumed_turn(
                prompt,
                source_turn_id=str(resume_source.get("source_turn_id") or ""),
                mode=str(resume_source.get("mode") or "resume"),
                payload=resume_source,
            )
        return self.state.start_turn(prompt)

    def _resume_source_for_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        matcher = getattr(self.state, "match_queued_followup", None)
        if not callable(matcher):
            return None
        matched = matcher(prompt)
        if not matched:
            return None
        return {
            "mode": "followup",
            "source_turn_id": matched.get("source_turn_id"),
            "followup_id": matched.get("followup_id"),
            "followup_kind": matched.get("kind"),
            "followup_text": matched.get("text"),
            "queued_at": matched.get("queued_at"),
        }

    def _consume_followup_if_needed(self, turn_id: str, resume_source: Optional[Dict[str, Any]]) -> None:
        if not turn_id or not resume_source:
            return
        followup_id = resume_source.get("followup_id")
        if not followup_id:
            return
        consumer = getattr(self.state, "consume_queued_followup", None)
        if callable(consumer):
            consumer(str(followup_id), turn_id=turn_id)

    def _record_turn_message(self, turn_id: Optional[str], role: str, content: str) -> None:
        if turn_id:
            self.state.record_turn_message(turn_id, role, content)

    def _normalize_assistant_block(self, block: AssistantActionBlock) -> Dict[str, Any]:
        return {
            "type": "assistant_action",
            "source": block.source,
            "rawContent": block.raw_content,
            "toolRequests": [
                {
                    "requestId": request.request_id,
                    "action": request.intent.action,
                    "params": dict(request.intent.params),
                    "source": request.source,
                }
                for request in (block.tool_requests or [])
            ],
            "finalAnswer": block.final_answer.reply if block.final_answer else None,
            "metadata": dict(block.metadata or {}),
        }

    def _normalize_tool_result_block(self, block: Any) -> Dict[str, Any]:
        return {
            "type": "tool_result",
            "requestId": block.request_id,
            "action": block.intent.action,
            "params": dict(block.intent.params),
            "output": str(block.output or ""),
            "ok": bool(block.ok),
            "source": block.source,
        }

    def _build_turn_status_detail(
        self,
        *,
        status: str,
        used_tools: bool,
        reply: str,
        tool_turn_state: Optional[ToolTurnState] = None,
        assistant_blocks: Optional[List[AssistantActionBlock]] = None,
        notes: Optional[List[str]] = None,
        progress_lines: Optional[List[str]] = None,
        followups: Optional[List[Dict[str, Any]]] = None,
        turn_usage: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        input_kind: str = "prompt",
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {
            "status": status,
            "usedTools": used_tools,
            "replyPreview": reply[:400] if reply else "",
            "inputKind": input_kind,
            "assistantBlockCount": len(assistant_blocks or []),
            "noteCount": len(notes or []),
            "progressLineCount": len(progress_lines or []),
            "followupCount": len(followups or []),
        }
        if turn_usage:
            detail["turnUsage"] = dict(turn_usage)
        if finish_reason:
            detail["finishReason"] = finish_reason
        if tool_turn_state is not None:
            detail["duplicateSteps"] = tool_turn_state.duplicate_steps
            detail["toolStepCount"] = len(tool_turn_state.outputs)
            detail["traceSummary"] = build_tool_trace_summary(tool_turn_state)
        return detail

    def _refresh_session_memory(self) -> None:
        rebuilder = getattr(self.state, "rebuild_memory_summary", None)
        if callable(rebuilder):
            rebuilder()
        pruner = getattr(self.state, "prune_stale_runtime_state", None)
        if callable(pruner):
            pruner()

    def _record_turn_trace_state(
        self,
        turn_id: str,
        *,
        tool_turn_state: Optional[ToolTurnState],
        assistant_blocks: Optional[List[AssistantActionBlock]] = None,
    ) -> None:
        if not turn_id:
            return
        if assistant_blocks:
            for block in assistant_blocks:
                self.state.record_turn_block(turn_id, self._normalize_assistant_block(block))
        if tool_turn_state is None:
            return
        for block in trace_tool_result_blocks(tool_turn_state):
            self.state.record_turn_block(turn_id, self._normalize_tool_result_block(block))
        for event in build_tool_trace_replay(tool_turn_state):
            self.state.record_turn_event(turn_id, event)

    def _turn_finish_payload(
        self,
        tool_turn_state: Optional[ToolTurnState],
        *,
        status: str,
        reply: str,
    ) -> Dict[str, Any]:
        if tool_turn_state is not None:
            for event in reversed(tool_turn_state.trace.events):
                if getattr(event, "kind", "") == "turn_finish":
                    return {
                        "status": getattr(event, "status", status),
                        "reason": getattr(event, "reason", status),
                        "reply": getattr(event, "reply", reply) or reply,
                    }
        return {"status": status, "reason": status, "reply": reply}

    def _followups_for_turn(
        self,
        prompt: str,
        tool_turn_state: Optional[ToolTurnState],
    ) -> List[Dict[str, Any]]:
        if tool_turn_state is None or not tool_turn_state.outputs:
            return []
        strategy = self.session._response_strategy_for_prompt(prompt, tool_turn_state.outputs)
        followups: List[Dict[str, Any]] = []
        if strategy.reason:
            followups.append({"kind": "reason", "text": strategy.reason, "status": strategy.status})
        for item in strategy.next_steps or []:
            followups.append({"kind": "next_step", "text": item, "status": strategy.status})
        if strategy.followup_intent is not None:
            followups.append(
                {
                    "kind": "followup_intent",
                    "text": self._describe_followup_intent(strategy.followup_intent),
                    "status": strategy.status,
                    "action": strategy.followup_intent.action,
                    "params": dict(strategy.followup_intent.params or {}),
                    "reply": strategy.followup_intent.reply,
                    "runnable": True,
                    "auto_continue": strategy.status == "needs_more_evidence",
                }
            )
        for item in strategy.followup_prompts or []:
            followups.append({"kind": "followup_prompt", "text": item, "status": strategy.status})
        return followups

    def _turn_usage_payload(
        self,
        *,
        used_tools: bool,
        tool_turn_state: Optional[ToolTurnState],
        status: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if used_tools:
            payload["tool_turns"] = 1
            payload["tool_steps"] = len(tool_turn_state.outputs) if tool_turn_state is not None else 0
        elif status == "chat":
            payload["chat_turns"] = 1
            payload["model_calls"] = 1
        return payload

    def parse_reply_to_blocks(
        self,
        raw_content: str,
        *,
        source: str,
        mode: str,
    ) -> List[AssistantActionBlock]:
        native_blocks = parse_native_action_blocks(
            raw_content,
            source=source,
            intent_signature=self.session._intent_signature,
        )
        if native_blocks:
            return native_blocks
        return legacy_reply_to_blocks(self.session, raw_content, source=source, mode=mode)

    def _synthetic_router_blocks(
        self,
        intent: ToolIntent,
        *,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[AssistantActionBlock]:
        return synthetic_router_blocks(
            intent,
            source=source,
            intent_signature=self.session._intent_signature,
            metadata=metadata,
        )

    def _blocks_to_tool_plan(self, blocks: List[AssistantActionBlock]) -> Optional[ToolPlan]:
        return blocks_to_tool_plan(blocks)

    def _blocks_to_tool_intent(self, blocks: List[AssistantActionBlock]) -> Optional[ToolIntent]:
        return blocks_to_tool_intent(blocks)

    def _record_model_reply_blocks(
        self,
        state: ToolTurnState,
        *,
        source: str,
        raw_content: str,
        blocks: List[AssistantActionBlock],
    ) -> None:
        state.trace.add(
            AssistantTurnEvent(
                source=source,
                raw_content=raw_content,
                parsed={
                    "block_count": len(blocks),
                    "tool_actions": [
                        request.intent.action
                        for block in blocks
                        for request in (block.tool_requests or [])
                    ],
                    "has_final_answer": any(block.final_answer is not None for block in blocks),
                    "native": any(bool((block.metadata or {}).get("native")) for block in blocks),
                },
            )
        )

    def execute_tool_request_block(
        self,
        prompt: str,
        state: ToolTurnState,
        request: ToolRequestBlock,
        *,
        source: str,
        repeated_behavior: str = "continue",
    ) -> tuple[str, Optional[str]]:
        signature = request.request_id
        if signature in state.seen_signatures:
            if repeated_behavior == "stop":
                self.session._add_tool_turn_note(
                    state,
                    "Agent 已拿到一些证据，但开始重复请求同一步工具。我已停止重复执行，并基于现有结果给出结论。",
                )
                return "repeated", None
            state.duplicate_steps += 1
            return "continue", None
        _, should_stop = self.session._apply_tool_intent_to_turn(
            state,
            request.intent,
            prompt,
            source=source,
        )
        if should_stop:
            return "tool_stop", None
        return "continue", None

    def execute_block_sequence(
        self,
        prompt: str,
        state: ToolTurnState,
        blocks: List[AssistantActionBlock],
        *,
        source: str,
        repeated_behavior: str = "continue",
    ) -> tuple[str, Optional[str]]:
        if not blocks:
            return "break", None
        for block in blocks:
            state.trace.add(AssistantActionBlockEvent(block=block))
            if block.final_answer is not None:
                return "final_answer", block.final_answer.reply or None
            if not block.tool_requests:
                continue
            for request in block.tool_requests:
                action, reply = self.execute_tool_request_block(
                    prompt,
                    state,
                    request,
                    source=source,
                    repeated_behavior=repeated_behavior,
                )
                if action != "continue":
                    return action, reply
        return "continue", None

    def handle_line(self, line: str) -> str:
        message = line.strip()
        if not message:
            return ""
        self.session._last_query_turn_result = None
        if message.startswith("/"):
            previous_tool_turn = self.session.last_tool_turn_state
            turn_id = self.ensure_turn(message)
            self._record_turn_message(turn_id, "user", message)
            self.session._record_chat_message("user", message)
            result = self.session._handle_command(message)
            self.session._append_tool_trace_if_new(previous_tool_turn)
            self._record_turn_message(turn_id, "assistant", result)
            self.session._record_chat_message("assistant", result)
            used_tools = self.session.last_tool_turn_state is not None and self.session.last_tool_turn_state is not previous_tool_turn
            self._record_turn_trace_state(
                turn_id,
                tool_turn_state=self.session.last_tool_turn_state if used_tools else None,
            )
            self.state.set_turn_status_detail(
                turn_id,
                self._build_turn_status_detail(
                    status="command",
                    used_tools=used_tools,
                    reply=result,
                    tool_turn_state=self.session.last_tool_turn_state if used_tools else None,
                    input_kind="command",
                ),
            )
            self.state.set_turn_finish_reason(
                turn_id,
                status="command",
                reason="command",
                reply=result,
            )
            self.state.finish_turn(turn_id, reply=result, used_tools=used_tools, status="command")
            return result
        return self.handle_prompt_turn(message).reply

    def handle_prompt_turn(self, prompt: str, *, auto_background: bool = True) -> QueryTurnResult:
        self.session._last_query_turn_result = None
        previous_tool_turn = self.session.last_tool_turn_state
        turn_id = self.ensure_turn(prompt)
        self._record_turn_message(turn_id, "user", prompt)
        self.session._record_chat_message("user", prompt)
        result = self.run_turn(prompt)
        self.session._append_tool_trace_if_new(previous_tool_turn)
        self._record_turn_message(turn_id, "assistant", result.reply)
        self.session._record_chat_message("assistant", result.reply)
        if auto_background:
            background_results = self.run_background_followups(source_turn_id=turn_id)
            if background_results:
                self.state.set_turn_status_detail(
                    turn_id,
                    {
                        "autoFollowupCount": len(background_results),
                        "autoFollowupTurns": [item.turn_id for item in background_results if item.turn_id],
                        "autoFollowupReplies": [item.reply[:160] for item in background_results if item.reply],
                    },
                )
            self.session._last_query_turn_result = result
        return result

    def run_once_payload(self, prompt: str, progress_lines: Optional[List[str]] = None) -> Dict[str, Any]:
        lines = progress_lines if progress_lines is not None else []
        reply = self.handle_line(prompt)
        if self.session._last_query_turn_result is not None:
            return build_snapshot_from_turn_result(
                turn_result=self.session._last_query_turn_result,
                session_state=self.state,
                progress_lines=lines,
                transcript_path=self.session.transcript_path,
                selected_model=self.session.selected_model,
                history_length=len(self.session.history),
                current_run_dir=self.session.current_run_dir,
                current_report=self.session.current_report,
                current_bridge_summary=self.session.current_bridge_summary,
                current_moire_summary=self.session.current_moire_summary,
                current_moire_compare_summary=self.session.current_moire_compare_summary,
                current_moire_diffusion_summary=self.session.current_moire_diffusion_summary,
                turn_count=len(self.session.turns),
                active_turn_id=self.session.active_turn_id,
                permission_denial_count=len(self.session.permission_denials),
                usage_stats=self.session.usage_stats,
            )
        turn_id = self._latest_turn_id()
        return build_chat_once_payload(
            reply=reply,
            progress_lines=lines,
            transcript_path=self.session.transcript_path,
            selected_model=self.session.selected_model,
            history_length=len(self.session.history),
            current_run_dir=self.session.current_run_dir,
            current_report=self.session.current_report,
            current_bridge_summary=self.session.current_bridge_summary,
            current_moire_summary=self.session.current_moire_summary,
            current_moire_compare_summary=self.session.current_moire_compare_summary,
            current_moire_diffusion_summary=self.session.current_moire_diffusion_summary,
            last_tool_turn_state=self.session.last_tool_turn_state,
            turn_count=len(self.session.turns),
            active_turn_id=self.session.active_turn_id,
            permission_denial_count=len(self.session.permission_denials),
            usage_stats=self.session.usage_stats,
            session_state=self.state,
            turn_id=turn_id,
        )

    def run_turn(self, prompt: str) -> QueryTurnResult:
        turn_id = self.ensure_turn(prompt)
        tool_result = self.run_tool_turn(prompt, turn_id=turn_id)
        if tool_result.reply:
            return tool_result
        plain_result = self.run_plain_chat_turn(prompt, turn_id=turn_id)
        self.commit_turn(plain_result)
        return plain_result

    def ensure_turn(self, prompt: str) -> str:
        if self.state.active_turn_id:
            return self.state.active_turn_id
        resume_source = self._resume_source_for_prompt(prompt)
        turn_id = self._start_turn(prompt, resume_source=resume_source)
        self._consume_followup_if_needed(turn_id, resume_source)
        return turn_id

    def run_tool_turn(self, prompt: str, *, turn_id: Optional[str] = None) -> QueryTurnResult:
        turn_id = turn_id or self.ensure_turn(prompt)
        reply, turn_state, used_tools = self.execute_engine_turn(prompt)
        has_turn_activity = bool(
            used_tools
            or reply
            or turn_state.outputs
            or turn_state.notes
            or turn_state.trace.events
        )
        result = self._finalize_turn(
            prompt=prompt,
            turn_id=turn_id,
            reply=reply or "",
            status="tool" if has_turn_activity else "empty",
            used_tools=has_turn_activity,
            tool_turn_state=turn_state if has_turn_activity else None,
        )
        if result.reply:
            self.commit_turn(result)
        return result

    def execute_engine_turn(
        self,
        prompt: str,
        *,
        state: Optional[ToolTurnState] = None,
    ) -> tuple[Optional[str], ToolTurnState, bool]:
        if should_skip_tool_router(prompt):
            turn = state or self.session._new_tool_turn_state(default_steps=4, step_env="MIETCLAW_AGENT_MAX_STEPS")
            return None, turn, False

        turn = state or self.session._new_tool_turn_state(default_steps=4, step_env="MIETCLAW_AGENT_MAX_STEPS")

        loop_result = self.run_agent_loop(prompt, state=turn)
        if loop_result:
            self.session._remember_tool_turn(turn)
            return loop_result, turn, True

        plan_reply = self.handle_plan_branch(prompt, turn)
        if plan_reply:
            return plan_reply, turn, bool(turn.outputs)

        if not turn.outputs and not turn.notes:
            router_blocks = self.resolve_router_blocks(prompt, state=turn)
            if router_blocks:
                router_reply, used_tools = self.execute_router_blocks(
                    prompt,
                    turn,
                    router_blocks,
                )
                if router_reply:
                    return router_reply, turn, used_tools

        reply, used_tools = self.finalize_engine_turn(prompt, turn)
        return reply, turn, used_tools

    def run_plain_chat_turn(self, prompt: str, *, turn_id: Optional[str] = None) -> QueryTurnResult:
        turn_id = turn_id or self.ensure_turn(prompt)
        status = self.session._refresh_local_model_status()
        if not status.get("healthy"):
            return self._finalize_turn(
                prompt=prompt,
                turn_id=turn_id,
                reply=(
                    f"本地模型当前不可用：{status.get('error', 'unavailable')}\n"
                    "你仍然可以使用 /status、/doctor、/tools、/runs、/inspect、/draft、/run、/bridge、/moire-run、/moire-compare、/moire-diffusion-sweep。"
                ),
                status="unavailable",
                used_tools=False,
            )
        try:
            reply = self.session._chat_with_local_model(self.session._build_local_model_messages(), purpose="chat")
        except Exception as exc:  # noqa: BLE001
            return self._finalize_turn(
                prompt=prompt,
                turn_id=turn_id,
                reply=f"本地模型调用失败：{exc}",
                status="error",
                used_tools=False,
            )
        return self._finalize_turn(
            prompt=prompt,
            turn_id=turn_id,
            reply=reply.get("content") or "本地模型没有返回内容。",
            status="chat",
            used_tools=False,
        )

    def build_turn_context(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> Dict[str, Any]:
        return self.session._build_engine_context(prompt, current_state=state)

    def build_tool_router_messages(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> List[Dict[str, str]]:
        engine_context = self.build_turn_context(prompt, state=state)
        context = {
            "current_run": self.session.current_run_dir.name if self.session.current_run_dir else None,
            "recent_runs": self._recent_run_names(limit=8),
            "workspace_root": self.session.workspace_root,
            "output_dir": str(self.session.output_dir),
            "turn_context": engine_context.get("turn_context"),
            "work_context": engine_context.get("work_context"),
            "memory_context": engine_context.get("memory_context"),
        }
        if state is not None:
            context["tool_budget"] = self.session._tool_budget_context(state.budget)
        tool_context = engine_context.get("tool_history") or self.session._tool_context_history(
            state.outputs if state is not None else None,
            output_limit=1800,
        )
        if tool_context:
            context["completed_tool_steps"] = tool_context
        return build_tool_router_messages(
            system_prompt=TOOL_ROUTER_SYSTEM_PROMPT,
            context=context,
            prompt=prompt,
        )

    def build_tool_plan_messages(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> List[Dict[str, str]]:
        budget = state.budget if state is not None else self.session._build_tool_budget(default_steps=4)
        engine_context = self.build_turn_context(prompt, state=state)
        context = {
            "current_run": self.session.current_run_dir.name if self.session.current_run_dir else None,
            "recent_runs": self._recent_run_names(limit=8),
            "workspace_root": self.session.workspace_root,
            "output_dir": str(self.session.output_dir),
            "tool_budget": self.session._tool_budget_context(budget),
            "turn_context": engine_context.get("turn_context"),
            "work_context": engine_context.get("work_context"),
            "followup_context": engine_context.get("followup_context"),
            "memory_context": engine_context.get("memory_context"),
        }
        tool_context = engine_context.get("tool_history") or self.session._tool_context_history(
            state.outputs if state is not None else None,
            output_limit=1800,
        )
        if tool_context:
            context["completed_tool_steps"] = tool_context
        return build_tool_plan_messages(
            system_prompt=TOOL_PLAN_SYSTEM_PROMPT,
            context=context,
            prompt=prompt,
        )

    def build_agent_loop_messages(
        self,
        prompt: str,
        outputs: List[Any],
        budget: Any,
        *,
        state: Optional[ToolTurnState] = None,
    ) -> List[Dict[str, str]]:
        engine_context = self.build_turn_context(prompt, state=state)
        context = {
            "current_run": self.session.current_run_dir.name if self.session.current_run_dir else None,
            "recent_runs": self._recent_run_names(limit=8),
            "workspace_root": self.session.workspace_root,
            "output_dir": str(self.session.output_dir),
            "tool_budget": self.session._tool_budget_context(budget),
            "turn_context": engine_context.get("turn_context"),
            "work_context": engine_context.get("work_context"),
            "followup_context": engine_context.get("followup_context"),
            "memory_context": engine_context.get("memory_context"),
        }
        tool_history = engine_context.get("tool_history") or self.session._tool_context_history(
            outputs,
            current_state=state,
            output_limit=2600,
        )
        return build_agent_loop_messages(
            system_prompt=AGENT_LOOP_SYSTEM_PROMPT,
            context=context,
            prompt=prompt,
            tool_history=tool_history,
        )

    def build_tool_summary_messages(
        self,
        original_prompt: str,
        outputs: List[Any],
        *,
        note: Optional[str] = None,
        state: Optional[ToolTurnState] = None,
    ) -> List[Dict[str, str]]:
        strategy = self.session._response_strategy_for_prompt(original_prompt, outputs)
        traced_blocks = trace_tool_result_blocks(state) if state is not None else []
        if traced_blocks:
            tool_blocks = tool_result_summary_entries(traced_blocks, truncate_output=self._truncate_for_model)
        else:
            tool_blocks = []
            for idx, (intent, result) in enumerate(outputs, start=1):
                tool_blocks.append(
                    f"[step {idx}] action={intent.action}\n"
                    f"params={json.dumps(intent.params, ensure_ascii=False)}\n"
                    f"output=\n{self._truncate_for_model(result, 2800)}"
                )
        return build_tool_summary_messages(
            original_prompt=original_prompt,
            tool_blocks=tool_blocks,
            host_strategy={
                "status": strategy.status,
                "reason": strategy.reason,
                "next_steps": strategy.next_steps,
                "answered_goals": strategy.answered_goals,
                "deferred_goals": strategy.deferred_goals,
                "followup_prompts": strategy.followup_prompts,
            },
            tool_response_style=self.session._tool_backed_response_style(),
            note=note,
        )

    def render_plan_outputs(
        self,
        original_prompt: str,
        outputs: List[Any],
        *,
        note: Optional[str] = None,
    ) -> str:
        strategy = self.session._response_strategy_for_prompt(original_prompt, outputs)
        latest_action = outputs[-1][0].action if outputs else "未知工具"
        conclusion_lines = [
            "结论",
            f"- 我已经完成了 {len(outputs)} 个工具步骤。",
            f"- 最近一次工具动作是 `{latest_action}`。",
        ]
        if strategy.status == "needs_more_evidence":
            conclusion_lines.append("- 当前还不能下最终结论，建议再补一轮关键证据。")
        elif strategy.status == "provisional":
            conclusion_lines.append("- 现在可以给初步判断，但还不是最稳的最终结论。")
        else:
            conclusion_lines.append("- 当前证据已经足够支撑这轮回答。")
        conclusion_lines.append(f"- 判断依据：{strategy.reason}")
        if strategy.answered_goals:
            conclusion_lines.append(f"- 已回答：{', '.join(strategy.answered_goals)}。")
        if strategy.deferred_goals:
            conclusion_lines.append(f"- 暂未展开：{', '.join(strategy.deferred_goals)}。")
        if note:
            conclusion_lines.append(f"- 备注：{note}")
        evidence_lines: List[str] = ["", "证据"]
        for idx, (intent, result) in enumerate(outputs, start=1):
            evidence_lines.append(f"- step {idx}: {intent.action}")
            evidence_lines.append(textwrap.indent(self._truncate_for_model(result, 1600), "  "))
        next_step_lines = ["", "下一步"]
        next_step_lines.extend(strategy.next_steps or self.session._suggest_next_steps(outputs))
        if strategy.followup_prompts:
            next_step_lines.append("- 可直接继续问：")
            next_step_lines.extend([f"  - {prompt}" for prompt in strategy.followup_prompts])
        return "\n".join(conclusion_lines + evidence_lines + next_step_lines)

    def summarize_tool_outputs(
        self,
        original_prompt: str,
        outputs: List[Any],
        *,
        note: Optional[str] = None,
        state: Optional[ToolTurnState] = None,
    ) -> str:
        if not outputs:
            return note or "没有可总结的工具输出。"
        status = self.session._refresh_local_model_status()
        fallback = self.render_plan_outputs(original_prompt, outputs, note=note)
        if not status.get("healthy"):
            return fallback
        try:
            reply = self.session._chat_with_local_model(
                self.build_tool_summary_messages(original_prompt, outputs, note=note, state=state),
                purpose="summary",
            )
        except Exception:
            return fallback
        content = (reply.get("content") or "").strip()
        if not content:
            return fallback
        if note and note not in content:
            return f"{note}\n\n{content}"
        return content

    def finalize_tool_turn(
        self,
        original_prompt: str,
        state: ToolTurnState,
        *,
        summarize: bool = False,
    ) -> Optional[str]:
        note = self.session._tool_turn_note_text(state)
        if not state.outputs:
            return note
        if len(state.outputs) == 1 and not summarize and not note:
            return state.outputs[0][1]
        return self.session._summarize_tool_outputs(original_prompt, state.outputs, note=note, state=state)

    def _synthetic_plan_blocks(
        self,
        plan: ToolPlan,
        *,
        source: str,
        raw_plan_content: Optional[str] = None,
    ) -> List[AssistantActionBlock]:
        if raw_plan_content:
            blocks = self.parse_reply_to_blocks(
                raw_plan_content,
                source=source,
                mode="plan",
            )
            if blocks:
                return blocks
        if plan.steps:
            rebuilt: List[AssistantActionBlock] = []
            for step in plan.steps:
                rebuilt.append(
                    AssistantActionBlock(
                        source=source,
                        tool_requests=[
                            ToolRequestBlock(
                                request_id=self.session._intent_signature(step),
                                intent=step,
                                source=source,
                            )
                        ],
                        metadata={"synthetic": True, "step_action": step.action, "plan_summarize": bool(plan.summarize)},
                    )
                )
            return rebuilt
        if plan.reply:
            return [
                AssistantActionBlock(
                    source=source,
                    final_answer=FinalAnswerBlock(reply=plan.reply, source=source),
                    metadata={"synthetic": True, "plan_summarize": bool(plan.summarize)},
                )
            ]
        return []

    def execute_tool_plan(
        self,
        plan: ToolPlan,
        original_prompt: str,
        *,
        state: Optional[ToolTurnState] = None,
        finalize: bool = True,
        raw_plan_content: Optional[str] = None,
    ) -> Optional[str]:
        if not plan.steps and not plan.reply:
            return None
        turn = state or self.session._new_tool_turn_state(default_steps=4)
        if not plan.steps:
            return plan.reply
        plan_blocks = self._synthetic_plan_blocks(
            plan,
            source="compat_planner_model",
            raw_plan_content=raw_plan_content,
        )
        block_action, block_reply = self.execute_block_sequence(
            original_prompt,
            turn,
            plan_blocks,
            source="compat_plan",
            repeated_behavior="continue",
        )
        if block_action == "final_answer":
            return block_reply
        if not finalize:
            return None
        return self.finalize_tool_turn(original_prompt, turn, summarize=plan.summarize)

    def handle_plan_branch(
        self,
        prompt: str,
        state: ToolTurnState,
    ) -> Optional[str]:
        blocks = self.resolve_tool_plan_blocks(prompt, state=state)
        if not blocks:
            return None
        block_action, block_reply = self.execute_block_sequence(
            prompt,
            state,
            blocks,
            source="planner_model",
            repeated_behavior="continue",
        )
        if block_action == "final_answer" and block_reply:
            state.trace.add(
                TurnFinishEvent(
                    status="finish",
                    reason="planner branch replied directly",
                    reply=block_reply,
                )
            )
            return block_reply
        return None

    def resolve_tool_plan_blocks(
        self,
        prompt: str,
        *,
        state: Optional[ToolTurnState] = None,
    ) -> List[AssistantActionBlock]:
        if should_skip_tool_router(prompt):
            return []
        heuristic = heuristic_tool_plan(prompt, self.session.output_dir, self.session.current_run_dir)
        if heuristic:
            return self._synthetic_plan_blocks(heuristic, source="heuristic_planner")
        direct_intent = heuristic_tool_intent(prompt, self.session.output_dir, self.session.current_run_dir)
        if direct_intent and direct_intent.action != "chat":
            return []
        if not should_try_tool_plan(prompt):
            return []
        status = self.session._refresh_local_model_status()
        if not status.get("healthy"):
            return []
        try:
            reply = self.session._chat_with_local_model(self.build_tool_plan_messages(prompt, state=state), purpose="plan")
        except Exception:  # noqa: BLE001
            return []
        raw_content = reply.get("content", "")
        blocks = self.parse_reply_to_blocks(
            raw_content,
            source="planner_model",
            mode="plan",
        )
        if state is not None:
            self._record_model_reply_blocks(
                state=state,
                source="planner_model",
                raw_content=raw_content,
                blocks=blocks,
            )
        return blocks

    def resolve_tool_plan(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> Optional[ToolPlan]:
        return self._blocks_to_tool_plan(self.resolve_tool_plan_blocks(prompt, state=state))

    def resolve_router_blocks(
        self,
        prompt: str,
        *,
        state: Optional[ToolTurnState] = None,
    ) -> List[AssistantActionBlock]:
        if should_skip_tool_router(prompt):
            return []
        heuristic = heuristic_tool_intent(prompt, self.session.output_dir, self.session.current_run_dir)
        if heuristic:
            return self._synthetic_router_blocks(
                heuristic,
                source="heuristic_router",
                metadata={"heuristic": True},
            )
        status = self.session._refresh_local_model_status()
        if not status.get("healthy"):
            return []
        try:
            reply = self.session._chat_with_local_model(self.build_tool_router_messages(prompt, state=state), purpose="router")
        except Exception:  # noqa: BLE001
            return []
        raw_content = reply.get("content", "")
        blocks = self.parse_reply_to_blocks(
            raw_content,
            source="router_model",
            mode="router",
        )
        if state is not None:
            self._record_model_reply_blocks(
                state,
                source="router_model",
                raw_content=raw_content,
                blocks=blocks,
            )
        return blocks

    def resolve_tool_intent(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> Optional[ToolIntent]:
        parsed = self._blocks_to_tool_intent(self.resolve_router_blocks(prompt, state=state))
        if parsed and parsed.action == "chat":
            return None
        return parsed

    def execute_router_blocks(
        self,
        prompt: str,
        state: ToolTurnState,
        blocks: List[AssistantActionBlock],
    ) -> tuple[Optional[str], bool]:
        block_action, block_reply = self.execute_block_sequence(
            prompt,
            state,
            blocks,
            source="router_model",
            repeated_behavior="continue",
        )
        if block_action == "final_answer":
            state.trace.add(
                TurnFinishEvent(
                    status="finish",
                    reason="router branch replied directly",
                    reply=block_reply,
                )
            )
            return block_reply, bool(state.outputs)
        if block_action == "tool_stop":
            reply = self.session._finalize_tool_turn(prompt, state, summarize=False)
            self.session._remember_tool_turn(state)
            state.trace.add(
                TurnFinishEvent(
                    status="stopped",
                    reason="router branch requested a stop",
                    reply=reply,
                )
            )
            return reply, bool(state.outputs)
        return None, bool(state.outputs)

    def execute_legacy_router_step(
        self,
        prompt: str,
        state: ToolTurnState,
        intent: ToolIntent,
        *,
        raw_router_content: str,
    ) -> Optional[str]:
        if raw_router_content:
            blocks = self.parse_reply_to_blocks(
                raw_router_content,
                source="legacy_router_model",
                mode="router",
            )
        else:
            blocks = []
        if not blocks:
            blocks = [AssistantActionBlock(
                source="legacy_router_model",
                tool_requests=[
                    ToolRequestBlock(
                        request_id=self.session._intent_signature(intent),
                        intent=intent,
                        source="legacy_router_model",
                    )
                ],
                metadata={"synthetic": True, "action": intent.action, "params": intent.params, "reply": intent.reply},
            )]
        block_action, _ = self.execute_block_sequence(
            prompt,
            state,
            blocks,
            source="legacy_router",
            repeated_behavior="continue",
        )
        if block_action != "tool_stop":
            return None
        reply = self.session._finalize_tool_turn(prompt, state, summarize=False)
        self.session._remember_tool_turn(state)
        state.trace.add(
            TurnFinishEvent(
                status="stopped",
                reason="legacy router step requested a stop",
                reply=reply,
            )
        )
        return reply

    def parse_legacy_agent_reply(
        self,
        state: ToolTurnState,
        raw_content: str,
    ) -> Optional[AssistantActionBlock]:
        blocks = self.parse_reply_to_blocks(
            raw_content,
            source="legacy_agent_model",
            mode="agent",
        )
        block = blocks[0] if blocks else None
        state.trace.add(
            AssistantTurnEvent(
                source="legacy_agent_model",
                raw_content=raw_content,
                parsed=block.metadata if block else {},
            )
        )
        return block

    def execute_legacy_agent_block(
        self,
        prompt: str,
        state: ToolTurnState,
        block: AssistantActionBlock,
    ) -> tuple[str, Optional[str]]:
        return self.execute_block_sequence(
            prompt,
            state,
            [block],
            source="legacy_agent_model",
            repeated_behavior="stop",
        )

    def finalize_engine_turn(
        self,
        prompt: str,
        state: ToolTurnState,
    ) -> tuple[Optional[str], bool]:
        if state.outputs:
            strategy = self.session._maybe_extend_tool_turn(prompt, state)
            if strategy.reason and strategy.reason not in (self.session._tool_turn_note_text(state) or "") and strategy.status == "provisional":
                self.session._add_tool_turn_note(state, strategy.reason)

        if state.outputs or state.notes:
            reply = self.session._finalize_tool_turn(prompt, state, summarize=False)
            self.session._remember_tool_turn(state)
            state.trace.add(
                TurnFinishEvent(
                    status="finish",
                    reason="tool event loop produced a final answer",
                    reply=reply,
                )
            )
            return reply, bool(state.outputs)

        state.trace.add(
            TurnFinishEvent(
                status="empty",
                reason="tool event loop found no actionable tool step",
                reply=None,
            )
        )
        return None, False

    def _finalize_agent_loop_reply(
        self,
        prompt: str,
        state: ToolTurnState,
        *,
        summarize: bool,
        status: str,
        reason: str,
        remember: bool = False,
        reply: Optional[str] = None,
    ) -> Optional[str]:
        final_reply = reply if reply is not None else self.session._finalize_tool_turn(prompt, state, summarize=summarize)
        if remember:
            self.session._remember_tool_turn(state)
        state.trace.add(
            TurnFinishEvent(
                status=status,
                reason=reason,
                reply=final_reply,
            )
        )
        return final_reply

    def _execute_forced_log_followup(
        self,
        prompt: str,
        state: ToolTurnState,
    ) -> tuple[bool, Optional[str]]:
        forced_target = self.session._forced_log_target(prompt, state.outputs)
        if not forced_target:
            return False, None
        forced_logs = ToolIntent(action="logs", params={"run": "current", "target": forced_target})
        _, should_stop = self.session._apply_tool_intent_to_turn(
            state,
            forced_logs,
            prompt,
            source="forced_log_followup",
        )
        if should_stop:
            return True, self._finalize_agent_loop_reply(
                prompt,
                state,
                summarize=True,
                status="stopped",
                reason="forced log followup requested a stop",
            )
        return True, None

    def run_agent_loop(self, prompt: str, *, state: Optional[ToolTurnState] = None) -> Optional[str]:
        if should_skip_tool_router(prompt) or not should_try_agent_loop(prompt):
            return None
        status = self.session._refresh_local_model_status()
        if not status.get("healthy"):
            return None

        turn = state or self.session._new_tool_turn_state(default_steps=4, step_env="MIETCLAW_AGENT_MAX_STEPS")
        bootstrap = self.session._heuristic_agent_first_step(prompt)
        if bootstrap:
            _, should_stop = self.session._apply_tool_intent_to_turn(
                turn,
                bootstrap,
                prompt,
                source="heuristic_agent_bootstrap",
            )
            if should_stop:
                return self._finalize_agent_loop_reply(
                    prompt,
                    turn,
                    summarize=True,
                    status="stopped",
                    reason="bootstrap tool step requested a stop",
                )

        _, forced_reply = self._execute_forced_log_followup(prompt, turn)
        if forced_reply:
            return forced_reply

        while turn.budget.remaining_steps > 0:
            try:
                reply = self.session._chat_with_local_model(
                    self.build_agent_loop_messages(prompt, turn.outputs, turn.budget, state=turn),
                    purpose="agent",
                )
            except Exception:
                if not turn.outputs:
                    turn.trace.add(
                        TurnFinishEvent(
                            status="error",
                            reason="agent model call failed",
                            reply=None,
                        )
                    )
                    return None
                return self._finalize_agent_loop_reply(
                    prompt,
                    turn,
                    summarize=True,
                    status="error",
                    reason="agent model call failed",
                )

            raw_content = reply.get("content", "")
            block = self.parse_legacy_agent_reply(turn, raw_content)
            if not block:
                if turn.outputs:
                    return None
                turn.trace.add(
                    TurnFinishEvent(
                        status="invalid",
                        reason="agent model returned invalid JSON",
                        reply=None,
                    )
                )
                return None

            forced_executed, forced_reply = self._execute_forced_log_followup(prompt, turn)
            if forced_reply:
                return forced_reply
            if forced_executed:
                continue

            block_action, block_reply = self.execute_legacy_agent_block(prompt, turn, block)
            if block_action == "final_answer":
                return self._finalize_agent_loop_reply(
                    prompt,
                    turn,
                    summarize=True,
                    status="finish",
                    reason="agent model chose to finish",
                    reply=block_reply,
                )
            if block_action == "repeated":
                return self._finalize_agent_loop_reply(
                    prompt,
                    turn,
                    summarize=True,
                    status="stopped",
                    reason="agent loop repeated the same tool step",
                )
            if block_action == "tool_stop":
                return self._finalize_agent_loop_reply(
                    prompt,
                    turn,
                    summarize=True,
                    status="stopped",
                    reason="tool execution requested a stop",
                )
            if block_action == "break":
                break

        if not turn.outputs:
            turn.trace.add(
                TurnFinishEvent(
                    status="empty",
                    reason="agent loop exited without usable tool evidence",
                    reply=None,
                )
            )
            return None
        self.session._add_tool_turn_note(
            turn,
            f"Agent 已达到本轮最多 {turn.budget.max_steps} 个工具步骤。我已基于当前证据先给出结论。",
        )
        return self._finalize_agent_loop_reply(
            prompt,
            turn,
            summarize=True,
            status="budget_exhausted",
            reason="agent loop reached the step budget",
        )

    def commit_turn(self, result: QueryTurnResult) -> None:
        if result.turn_id:
            self.state.finish_turn(
                result.turn_id,
                reply=result.reply,
                used_tools=result.used_tools,
                status=result.status,
            )
            for note in result.notes:
                self.state.append_turn_note(result.turn_id, note)
            self._record_turn_trace_state(
                result.turn_id,
                tool_turn_state=result.tool_turn_state,
                assistant_blocks=result.assistant_blocks,
            )
            self.state.set_turn_status_detail(
                result.turn_id,
                self._build_turn_status_detail(
                    status=result.status,
                    used_tools=result.used_tools,
                    reply=result.reply,
                    tool_turn_state=result.tool_turn_state,
                    assistant_blocks=result.assistant_blocks,
                    notes=result.notes,
                    progress_lines=result.progress_lines,
                    followups=result.followups,
                    turn_usage=result.turn_usage,
                    finish_reason=result.finish_reason,
                ),
            )
            for followup in result.followups:
                followup_payload = dict(followup)
                followup_payload.setdefault("source_status", result.status)
                followup_payload.setdefault("source_finish_reason", result.finish_reason or result.status)
                if result.reply:
                    followup_payload.setdefault("source_reply_preview", result.reply[:240])
                self.state.record_turn_followup(result.turn_id, followup_payload)
                if followup_payload.get("kind") in {"next_step", "followup_prompt"}:
                    self.state.queue_followup(result.turn_id, followup_payload)
            if result.turn_usage:
                self.state.record_turn_usage(result.turn_id, result.turn_usage)
                self.state.record_usage(result.turn_usage)
            self.state.set_turn_finish_reason(
                result.turn_id,
                status=result.status,
                reason=result.finish_reason or result.status,
                reply=result.reply,
            )
            self._refresh_session_memory()

        if result.tool_turn_state is not None:
            self.state.last_tool_turn_state = result.tool_turn_state
            pending_requests: List[Dict[str, Any]] = []
            for event in result.tool_turn_state.trace.events:
                kind = getattr(event, "kind", "")
                if kind == "assistant_action_block":
                    block = getattr(event, "block", None)
                    if block is not None:
                        for request in block.tool_requests:
                            pending_requests.append(
                                {
                                    "request_id": request.request_id,
                                    "action": request.intent.action,
                                    "params": dict(request.intent.params),
                                    "source": request.source,
                                }
                            )
                elif kind == "permission_decision" and getattr(event, "decision", "allow") != "allow":
                    denial_payload = {
                        "action": event.intent.action,
                        "source": event.source,
                        "decision": event.decision,
                        "reason": event.reason,
                        "manual": event.manual,
                    }
                    self.state.record_denial(denial_payload)
                    if result.turn_id:
                        self.state.record_turn_denial(result.turn_id, denial_payload)
            self.state.pending_tool_requests = pending_requests
        else:
            self.state.pending_tool_requests = []

        self.session._last_query_turn_result = result

    def abort_turn(self, turn_id: Optional[str] = None, *, reason: str = "aborted by host") -> Optional[Dict[str, Any]]:
        resolved_turn_id = turn_id or self.state.active_turn_id or self._latest_turn_id()
        if not resolved_turn_id:
            return None
        payload = self.state.abort_turn(resolved_turn_id, reason)
        self._refresh_session_memory()
        return payload

    def retry_turn(self, turn_id: str, *, prompt: Optional[str] = None) -> QueryTurnResult:
        snapshot = self.state.resume_turn_state(turn_id) or {}
        source_turn = snapshot.get("turn") or {}
        retry_prompt = prompt or source_turn.get("prompt") or ""
        new_turn_id = self._start_turn(
            retry_prompt,
            resume_source={
                "mode": "retry",
                "source_turn_id": turn_id,
                "source_status": source_turn.get("status"),
                "source_finish_reason": (snapshot.get("finish_reason") or {}).get("reason"),
            },
        )
        result = self.run_turn(retry_prompt)
        if not result.turn_id:
            result.turn_id = new_turn_id
        return result

    def resume_turn(self, turn_id: str, *, prompt: Optional[str] = None) -> QueryTurnResult:
        snapshot = self.state.resume_turn_state(turn_id) or {}
        source_turn = snapshot.get("turn") or {}
        resume_prompt = prompt or source_turn.get("prompt") or ""
        new_turn_id = self._start_turn(
            resume_prompt,
            resume_source={
                "mode": "resume",
                "source_turn_id": turn_id,
                "source_status": source_turn.get("status"),
                "source_finish_reason": (snapshot.get("finish_reason") or {}).get("reason"),
            },
        )
        result = self.run_turn(resume_prompt)
        if not result.turn_id:
            result.turn_id = new_turn_id
        return result

    def _assistant_blocks_from_state(self, state: Optional[ToolTurnState]) -> List[AssistantActionBlock]:
        if state is None:
            return []
        return [
            event.block
            for event in state.trace.events
            if getattr(event, "kind", "") == "assistant_action_block"
        ]

    def _finalize_turn(
        self,
        *,
        prompt: str,
        turn_id: Optional[str],
        reply: str,
        status: str,
        used_tools: bool,
        tool_turn_state: Optional[ToolTurnState] = None,
    ) -> QueryTurnResult:
        followups = self._followups_for_turn(prompt, tool_turn_state)
        turn_usage = self._turn_usage_payload(
            used_tools=used_tools,
            tool_turn_state=tool_turn_state,
            status=status,
        )
        finish_payload = self._turn_finish_payload(tool_turn_state, status=status, reply=reply)
        return QueryTurnResult(
            turn_id=turn_id or self.state.active_turn_id or "",
            reply=reply,
            used_tools=used_tools,
            status=status,
            tool_turn_state=tool_turn_state,
            assistant_blocks=self._assistant_blocks_from_state(tool_turn_state),
            notes=list(tool_turn_state.notes) if tool_turn_state is not None else [],
            followups=followups,
            turn_usage=turn_usage,
            finish_reason=str(finish_payload.get("reason") or status),
        )
