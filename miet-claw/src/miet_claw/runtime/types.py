from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..tool_router import ToolIntent


@dataclass
class ToolExecutionOutcome:
    output: str
    ok: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolBudget:
    max_steps: int
    max_mutating_steps: int
    max_failures: int
    steps_used: int = 0
    mutating_steps_used: int = 0
    failures: int = 0

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    @property
    def remaining_mutating_steps(self) -> int:
        return max(0, self.max_mutating_steps - self.mutating_steps_used)


@dataclass(frozen=True)
class ToolRequestBlock:
    request_id: str
    intent: ToolIntent
    source: str


@dataclass(frozen=True)
class ToolResultBlock:
    request_id: str
    intent: ToolIntent
    output: str
    ok: bool
    source: str


@dataclass(frozen=True)
class FinalAnswerBlock:
    reply: str
    source: str


@dataclass(frozen=True)
class AssistantActionBlock:
    source: str
    raw_content: str = ""
    tool_requests: List[ToolRequestBlock] = field(default_factory=list)
    final_answer: Optional[FinalAnswerBlock] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantTurnEvent:
    source: str
    raw_content: str = ""
    parsed: Dict[str, Any] = field(default_factory=dict)
    kind: str = field(default="assistant_turn", init=False)


@dataclass
class AssistantActionBlockEvent:
    block: AssistantActionBlock
    kind: str = field(default="assistant_action_block", init=False)


@dataclass
class ToolUseEvent:
    intent: ToolIntent
    source: str
    manual: bool = False
    kind: str = field(default="tool_use", init=False)


@dataclass
class PermissionDecisionEvent:
    intent: ToolIntent
    source: str
    decision: str
    reason: str = ""
    manual: bool = False
    kind: str = field(default="permission_decision", init=False)


@dataclass
class ToolResultEvent:
    intent: ToolIntent
    outcome: ToolExecutionOutcome
    source: str
    kind: str = field(default="tool_result", init=False)


@dataclass
class ToolResultBlockEvent:
    block: ToolResultBlock
    kind: str = field(default="tool_result_block", init=False)


@dataclass
class TurnFinishEvent:
    status: str
    reason: str
    reply: Optional[str] = None
    kind: str = field(default="turn_finish", init=False)


@dataclass
class TurnExecutionTrace:
    events: List[Any] = field(default_factory=list)

    def add(self, event: Any) -> None:
        self.events.append(event)


@dataclass
class ToolTurnState:
    budget: ToolBudget
    outputs: List[Tuple[ToolIntent, str]] = field(default_factory=list)
    seen_signatures: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    duplicate_steps: int = 0
    trace: TurnExecutionTrace = field(default_factory=TurnExecutionTrace)


@dataclass
class ToolResponseStrategy:
    status: str
    reason: str
    followup_intent: Optional[ToolIntent] = None
    next_steps: List[str] = field(default_factory=list)
    answered_goals: List[str] = field(default_factory=list)
    deferred_goals: List[str] = field(default_factory=list)
    followup_prompts: List[str] = field(default_factory=list)
