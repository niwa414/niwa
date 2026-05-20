from __future__ import annotations

import json
from typing import Optional

from .session import ChatRuntimeState
from .snapshot import build_tool_trace_id, build_tool_trace_replay, build_tool_trace_summary
from .types import ToolTurnState


def append_progress(state: ChatRuntimeState, line: str) -> None:
    state.append_transcript(f"\n### progress\n\n{line}\n")


def append_message(state: ChatRuntimeState, speaker: str, content: str) -> None:
    state.append_transcript(f"\n## {speaker}\n\n{content}\n")


def append_tool_trace(state: ChatRuntimeState, turn_state: Optional[ToolTurnState]) -> None:
    trace_summary = build_tool_trace_summary(turn_state)
    trace_replay = build_tool_trace_replay(turn_state)
    trace_id = build_tool_trace_id(turn_state)
    if not trace_summary and not trace_replay:
        return
    payload = {
        "traceId": trace_id,
        "summary": trace_summary,
        "replay": trace_replay,
    }
    state.append_transcript(
        "\n### tool trace\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
