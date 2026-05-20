from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..tool_router import ToolIntent


if TYPE_CHECKING:
    from .types import ToolTurnState


@dataclass
class ChatRuntimeState:
    session_dir: Path
    transcript_path: Path
    current_report: Optional[Dict[str, Any]] = None
    current_run_dir: Optional[Path] = None
    current_bridge_summary: Optional[Dict[str, Any]] = None
    current_moire_summary: Optional[Dict[str, Any]] = None
    current_moire_compare_summary: Optional[Dict[str, Any]] = None
    current_moire_diffusion_summary: Optional[Dict[str, Any]] = None
    history: List[Tuple[str, str]] = field(default_factory=list)
    selected_model: Optional[str] = None
    local_model_status: Optional[Dict[str, Any]] = None
    read_only_tool_cache: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    tool_context_outputs: List[Tuple[ToolIntent, str]] = field(default_factory=list)
    tool_context_blocks: List[Dict[str, Any]] = field(default_factory=list)
    last_tool_turn_state: Optional["ToolTurnState"] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    active_turn_id: Optional[str] = None
    pending_tool_requests: List[Dict[str, Any]] = field(default_factory=list)
    permission_denials: List[Dict[str, Any]] = field(default_factory=list)
    usage_stats: Dict[str, Any] = field(default_factory=dict)
    queued_followups: List[Dict[str, Any]] = field(default_factory=list)
    turn_messages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    turn_blocks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    turn_events: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    turn_followups: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    turn_denials: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    turn_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    turn_finish_reason: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    turn_resume_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    turn_status_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    turn_children: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    memory_records: List[Dict[str, Any]] = field(default_factory=list)
    memory_summary: Dict[str, Any] = field(default_factory=dict)
    compact_boundary_turn_id: Optional[str] = None
    aborted_turns: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, *, project_root: str, initial_model: Optional[str]) -> "ChatRuntimeState":
        session_dir = Path(project_root) / ".sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = session_dir / f"chat-{int(time.time())}.md"
        state = cls(
            session_dir=session_dir,
            transcript_path=transcript_path,
            selected_model=initial_model,
        )
        state.append_transcript("# mietclaw terminal session\n")
        return state

    def append_transcript(self, text: str) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def clear_conversation(self) -> None:
        self.history.clear()
        self.tool_context_outputs.clear()
        self.tool_context_blocks.clear()
        self.last_tool_turn_state = None
        self.turns.clear()
        self.active_turn_id = None
        self.pending_tool_requests.clear()
        self.permission_denials.clear()
        self.queued_followups.clear()
        self.usage_stats.clear()
        self.turn_messages.clear()
        self.turn_blocks.clear()
        self.turn_events.clear()
        self.turn_followups.clear()
        self.turn_denials.clear()
        self.turn_usage.clear()
        self.turn_finish_reason.clear()
        self.turn_resume_source.clear()
        self.turn_status_detail.clear()
        self.turn_children.clear()
        self.memory_records.clear()
        self.memory_summary.clear()
        self.compact_boundary_turn_id = None
        self.aborted_turns.clear()

    def start_turn(self, prompt: str) -> str:
        turn_id = f"turn-{int(time.time() * 1000)}-{len(self.turns) + 1}"
        self.active_turn_id = turn_id
        self.turns.append(
            {
                "turn_id": turn_id,
                "prompt": prompt,
                "status": "running",
                "reply": None,
                "used_tools": False,
                "notes": [],
            }
        )
        self.turn_messages.setdefault(turn_id, [])
        self.turn_blocks.setdefault(turn_id, [])
        self.turn_events.setdefault(turn_id, [])
        self.turn_followups.setdefault(turn_id, [])
        self.turn_denials.setdefault(turn_id, [])
        self.turn_usage.setdefault(turn_id, {})
        self.turn_finish_reason.setdefault(turn_id, {})
        self.turn_resume_source.setdefault(turn_id, {})
        self.turn_status_detail.setdefault(
            turn_id,
            {
                "prompt": prompt,
                "status": "running",
                "used_tools": False,
            },
        )
        self.turn_children.setdefault(turn_id, [])
        return turn_id

    def finish_turn(self, turn_id: str, *, reply: str, used_tools: bool, status: str) -> None:
        for item in reversed(self.turns):
            if item.get("turn_id") == turn_id:
                item["reply"] = reply
                item["used_tools"] = used_tools
                item["status"] = status
                break
        detail = self.turn_status_detail.setdefault(turn_id, {})
        detail.update(
            {
                "reply": reply,
                "used_tools": used_tools,
                "status": status,
            }
        )
        if self.active_turn_id == turn_id:
            self.active_turn_id = None

    def append_turn_note(self, turn_id: str, note: str) -> None:
        if not note:
            return
        for item in reversed(self.turns):
            if item.get("turn_id") == turn_id:
                notes = item.setdefault("notes", [])
                if note not in notes:
                    notes.append(note)
                break

    def current_turn(self) -> Optional[Dict[str, Any]]:
        if not self.active_turn_id:
            return None
        for item in reversed(self.turns):
            if item.get("turn_id") == self.active_turn_id:
                return item
        return None

    def record_denial(self, payload: Dict[str, Any]) -> None:
        self.permission_denials.append(dict(payload))

    def record_usage(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                self.usage_stats[key] = self.usage_stats.get(key, 0) + value
            else:
                self.usage_stats[key] = value

    def record_turn_message(self, turn_id: str, role: str, content: str) -> None:
        if not turn_id or not content:
            return
        self.turn_messages.setdefault(turn_id, []).append(
            {
                "role": role,
                "content": content,
            }
        )

    def record_turn_block(self, turn_id: str, block: Dict[str, Any]) -> None:
        if not turn_id or not block:
            return
        self.turn_blocks.setdefault(turn_id, []).append(dict(block))

    def record_turn_event(self, turn_id: str, event: Dict[str, Any]) -> None:
        if not turn_id or not event:
            return
        self.turn_events.setdefault(turn_id, []).append(dict(event))

    def record_turn_followup(self, turn_id: str, payload: Dict[str, Any]) -> None:
        if not turn_id or not payload:
            return
        self.turn_followups.setdefault(turn_id, []).append(dict(payload))

    def queued_followup_items(
        self,
        *,
        limit: Optional[int] = None,
        include_consumed: bool = False,
        runnable_only: bool = False,
        auto_only: bool = False,
        source_turn_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items = [
            dict(item)
            for item in self.queued_followups
            if (include_consumed or not bool(item.get("consumed")))
            and (not runnable_only or bool(item.get("runnable")))
            and (not auto_only or bool(item.get("auto_continue")))
            and (source_turn_id is None or str(item.get("source_turn_id") or "") == source_turn_id)
        ]
        if limit is not None and limit >= 0:
            return items[-limit:]
        return items

    def queue_followup(self, turn_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not turn_id or not payload:
            return None
        text = str(payload.get("text") or "").strip()
        kind = str(payload.get("kind") or "").strip()
        if not text or not kind:
            return None
        deduped: List[Dict[str, Any]] = []
        for item in self.queued_followups:
            if (
                str(item.get("source_turn_id") or "") == turn_id
                and str(item.get("kind") or "") == kind
                and str(item.get("text") or "") == text
                and str(item.get("action") or "") == str(payload.get("action") or "")
                and dict(item.get("params") or {}) == dict(payload.get("params") or {})
                and not bool(item.get("consumed"))
            ):
                continue
            deduped.append(item)
        self.queued_followups = deduped
        entry = dict(payload)
        entry.setdefault("followup_id", f"followup-{int(time.time() * 1000)}-{len(self.queued_followups) + 1}")
        entry.setdefault("source_turn_id", turn_id)
        entry.setdefault("queued_at", time.time())
        entry.setdefault("consumed", False)
        entry.setdefault("status", "queued")
        entry.setdefault("attempt_count", 0)
        entry.setdefault("runnable", kind in {"followup_prompt", "followup_intent"})
        entry.setdefault("auto_continue", False)
        self.queued_followups.append(entry)
        return dict(entry)

    @staticmethod
    def _normalize_followup_text(text: str) -> str:
        normalized = " ".join(str(text or "").strip().lower().split())
        return normalized.rstrip("。！？!?.,；;：:")

    def match_queued_followup(self, prompt: str) -> Optional[Dict[str, Any]]:
        normalized_prompt = self._normalize_followup_text(prompt)
        if not normalized_prompt:
            return None
        for item in reversed(self.queued_followups):
            if bool(item.get("consumed")):
                continue
            if self._normalize_followup_text(str(item.get("text") or "")) == normalized_prompt:
                return dict(item)
        return None

    def consume_queued_followup(
        self,
        followup_id: str,
        *,
        turn_id: Optional[str] = None,
        status: str = "consumed",
        attempt: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not followup_id:
            return None
        for item in self.queued_followups:
            if str(item.get("followup_id") or "") != followup_id:
                continue
            item["consumed"] = True
            item["consumed_at"] = time.time()
            item["status"] = status
            if turn_id:
                item["consumed_by_turn_id"] = turn_id
            if attempt:
                item["attempt_count"] = int(item.get("attempt_count") or 0) + 1
                item["last_attempt_at"] = time.time()
            if extra:
                item.update(dict(extra))
            return dict(item)
        return None

    def update_queued_followup(self, followup_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        if not followup_id:
            return None
        for item in self.queued_followups:
            if str(item.get("followup_id") or "") != followup_id:
                continue
            item.update(fields)
            return dict(item)
        return None

    def next_queued_followup(
        self,
        *,
        runnable_only: bool = False,
        auto_only: bool = False,
        source_turn_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        for item in self.queued_followups:
            if bool(item.get("consumed")):
                continue
            if runnable_only and not bool(item.get("runnable")):
                continue
            if auto_only and not bool(item.get("auto_continue")):
                continue
            if source_turn_id is not None and str(item.get("source_turn_id") or "") != source_turn_id:
                continue
            return dict(item)
        return None

    def record_turn_denial(self, turn_id: str, payload: Dict[str, Any]) -> None:
        if not turn_id or not payload:
            return
        self.turn_denials.setdefault(turn_id, []).append(dict(payload))

    def record_turn_usage(self, turn_id: str, payload: Dict[str, Any]) -> None:
        if not turn_id or not payload:
            return
        current = self.turn_usage.setdefault(turn_id, {})
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                current[key] = current.get(key, 0) + value
            else:
                current[key] = value

    def set_turn_finish_reason(
        self,
        turn_id: str,
        *,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        reply: Optional[str] = None,
    ) -> None:
        if not turn_id:
            return
        payload = self.turn_finish_reason.setdefault(turn_id, {})
        if status is not None:
            payload["status"] = status
        if reason is not None:
            payload["reason"] = reason
        if reply is not None:
            payload["reply"] = reply

    def set_turn_resume_source(self, turn_id: str, payload: Dict[str, Any]) -> None:
        if not turn_id or not payload:
            return
        current = self.turn_resume_source.setdefault(turn_id, {})
        current.update(dict(payload))

    def set_turn_status_detail(self, turn_id: str, detail: Dict[str, Any]) -> None:
        if not turn_id or not detail:
            return
        current = self.turn_status_detail.setdefault(turn_id, {})
        current.update(detail)

    def record_turn_child(
        self,
        source_turn_id: str,
        child_turn_id: str,
        *,
        mode: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not source_turn_id or not child_turn_id:
            return
        children = self.turn_children.setdefault(source_turn_id, [])
        for item in children:
            if str(item.get("turn_id") or "") == child_turn_id:
                item.update(dict(payload or {}))
                item["mode"] = mode
                return
        children.append(
            {
                "turn_id": child_turn_id,
                "mode": mode,
                "created_at": time.time(),
                **dict(payload or {}),
            }
        )

    def mark_turn_aborted(self, turn_id: str, reason: str) -> None:
        if not turn_id:
            return
        self.aborted_turns.append({"turn_id": turn_id, "reason": reason})
        self.set_turn_status_detail(turn_id, {"status": "aborted", "reason": reason})
        if self.active_turn_id == turn_id:
            self.active_turn_id = None

    def abort_turn(self, turn_id: str, reason: str, *, clear_pending_requests: bool = True) -> Optional[Dict[str, Any]]:
        if not turn_id:
            return None
        self.mark_turn_aborted(turn_id, reason)
        self.set_turn_finish_reason(turn_id, status="aborted", reason=reason, reply="")
        if clear_pending_requests:
            self.pending_tool_requests = []
        for item in reversed(self.turns):
            if item.get("turn_id") == turn_id:
                item["status"] = "aborted"
                item["reply"] = item.get("reply") or ""
                break
        return self.resume_turn_state(turn_id)

    def begin_resumed_turn(
        self,
        prompt: str,
        *,
        source_turn_id: str,
        mode: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        turn_id = self.start_turn(prompt)
        resume_payload = {
            "mode": mode,
            "source_turn_id": source_turn_id,
            **dict(payload or {}),
        }
        self.set_turn_resume_source(turn_id, resume_payload)
        self.set_turn_status_detail(turn_id, {"resumeMode": mode, "resumeSource": resume_payload})
        if source_turn_id:
            self.record_turn_child(
                source_turn_id,
                turn_id,
                mode=mode,
                payload={
                    "prompt": prompt,
                    "source_status": resume_payload.get("source_status"),
                    "source_finish_reason": resume_payload.get("source_finish_reason"),
                },
            )
        return turn_id

    def resume_turn_state(self, turn_id: str) -> Optional[Dict[str, Any]]:
        if not turn_id:
            return None
        return {
            "turn": next((item for item in reversed(self.turns) if item.get("turn_id") == turn_id), None),
            "messages": list(self.turn_messages.get(turn_id, [])),
            "blocks": list(self.turn_blocks.get(turn_id, [])),
            "events": list(self.turn_events.get(turn_id, [])),
            "followups": list(self.turn_followups.get(turn_id, [])),
            "denials": list(self.turn_denials.get(turn_id, [])),
            "usage": dict(self.turn_usage.get(turn_id, {})),
            "finish_reason": dict(self.turn_finish_reason.get(turn_id, {})),
            "resume_source": dict(self.turn_resume_source.get(turn_id, {})),
            "status_detail": dict(self.turn_status_detail.get(turn_id, {})),
            "child_turns": list(self.turn_children.get(turn_id, [])),
        }

    def _build_memory_record(self, turn_id: str) -> Optional[Dict[str, Any]]:
        turn = next((item for item in self.turns if item.get("turn_id") == turn_id), None)
        if not turn:
            return None
        finish = dict(self.turn_finish_reason.get(turn_id, {}))
        followups = list(self.turn_followups.get(turn_id, []))
        denials = list(self.turn_denials.get(turn_id, []))
        usage = dict(self.turn_usage.get(turn_id, {}))
        children = list(self.turn_children.get(turn_id, []))
        latest_child = dict(children[-1]) if children else None
        prompt_preview = str(turn.get("prompt") or "").strip()[:160]
        reply_preview = str(turn.get("reply") or "").strip()[:200]
        summary_parts: List[str] = []
        if prompt_preview:
            summary_parts.append(f"用户在这一轮主要想做：{prompt_preview}")
        if reply_preview:
            summary_parts.append(f"当时得到的结果是：{reply_preview}")
        if finish.get("reason"):
            summary_parts.append(f"结束原因：{finish.get('reason')}")
        if followups:
            summary_parts.append(f"留下了 {len(followups)} 个后续动作。")
        if denials:
            summary_parts.append(f"当时有 {len(denials)} 次权限拦截。")
        stale = latest_child is not None and str(latest_child.get("mode") or "") in {"resume", "retry"}
        return {
            "memory_id": f"memory-{turn_id}",
            "turn_id": turn_id,
            "status": turn.get("status"),
            "prompt_preview": prompt_preview,
            "reply_preview": reply_preview,
            "finish_reason": finish.get("reason"),
            "followup_count": len(followups),
            "denial_count": len(denials),
            "child_turn_count": len(children),
            "superseded_by": latest_child.get("turn_id") if stale and latest_child else None,
            "stale": stale,
            "usage": usage,
            "summary": " ".join(part for part in summary_parts if part).strip(),
        }

    def rebuild_memory_summary(
        self,
        *,
        live_turn_window: int = 6,
        max_records: int = 24,
    ) -> Dict[str, Any]:
        completed_turns = [
            item
            for item in self.turns
            if str(item.get("status") or "") not in {"running", ""}
        ]
        if live_turn_window < 0:
            live_turn_window = 0
        archived_turns = completed_turns[:-live_turn_window] if live_turn_window else completed_turns
        records = [
            record
            for record in (self._build_memory_record(str(item.get("turn_id") or "")) for item in archived_turns)
            if record is not None
        ]
        self.memory_records = records[-max_records:]
        if not self.memory_records:
            self.memory_summary = {}
            self.compact_boundary_turn_id = None
            return {}
        self.compact_boundary_turn_id = str(self.memory_records[-1].get("turn_id") or "") or None
        fresh_records = [item for item in self.memory_records if not bool(item.get("stale"))]
        stale_records = [item for item in self.memory_records if bool(item.get("stale"))]
        recent_records = fresh_records[-4:] if fresh_records else self.memory_records[-4:]
        summary_lines = ["Earlier working memory from archived turns:"]
        for item in recent_records:
            label = str(item.get("prompt_preview") or item.get("reply_preview") or item.get("turn_id") or "—")
            status = str(item.get("status") or "unknown")
            summary_lines.append(f"- {item.get('turn_id')} · {status} · {label}")
        self.memory_summary = {
            "archived_turn_count": len(self.memory_records),
            "fresh_memory_count": len(fresh_records),
            "stale_memory_count": len(stale_records),
            "compact_boundary_turn_id": self.compact_boundary_turn_id,
            "summary": "\n".join(summary_lines),
            "recent_records": recent_records,
        }
        return dict(self.memory_summary)

    def prune_stale_runtime_state(
        self,
        *,
        max_consumed_followups: int = 12,
        max_aborted_turns: int = 12,
    ) -> None:
        active_items = [item for item in self.queued_followups if not bool(item.get("consumed"))]
        consumed_items = [item for item in self.queued_followups if bool(item.get("consumed"))]
        self.queued_followups = active_items + consumed_items[-max_consumed_followups:]
        self.aborted_turns = self.aborted_turns[-max_aborted_turns:]
