"""Evidence-planning helpers for chat tool turns."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from .runtime.types import ToolResponseStrategy
from .tool_router import ToolIntent, infer_run_mode_hint


class AvailableRunsForMode(Protocol):
    def __call__(self, mode: Optional[str], *, limit: int = 12) -> List[Dict[str, Any]]:
        ...


class RunTargetHint(Protocol):
    def __call__(self, mode: Optional[str] = None) -> Optional[str]:
        ...


class ForcedLogTarget(Protocol):
    def __call__(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        ...


class IntentSignature(Protocol):
    def __call__(self, intent: ToolIntent) -> str:
        ...


class SuggestNextSteps(Protocol):
    def __call__(self, outputs: List[Tuple[ToolIntent, str]]) -> List[str]:
        ...


@dataclass(frozen=True)
class ChatEvidencePlanner:
    """Plan whether a chat answer has enough tool evidence, and what to fetch next."""

    current_run_dir: Optional[Path]
    current_run_mode: Optional[str]
    available_runs_for_mode: AvailableRunsForMode
    run_target_hint: RunTargetHint
    forced_log_target: ForcedLogTarget
    intent_signature: IntentSignature
    suggest_next_steps: SuggestNextSteps

    def tool_evidence_tags(self, outputs: List[Tuple[ToolIntent, str]]) -> Set[str]:
        tags: Set[str] = set()
        for intent, _ in outputs:
            if intent.action == "runs":
                tags.update({"runs"})
            elif intent.action == "compare_runs":
                tags.update({"runs", "compare"})
            elif intent.action == "moire_compare":
                tags.update({"compare"})
            elif intent.action == "inspect":
                tags.update({"runs", "inspect"})
            elif intent.action == "logs":
                tags.update({"runs", "inspect", "logs"})
            elif intent.action == "artifacts":
                tags.update({"runs", "artifacts"})
        return tags

    def goal_first_position(self, prompt: str, tokens: List[str]) -> int:
        lower = prompt.lower()
        positions: List[int] = []
        for token in tokens:
            token_lower = token.lower()
            idx = lower.find(token_lower)
            if idx >= 0:
                positions.append(idx)
            idx = prompt.find(token)
            if idx >= 0:
                positions.append(idx)
        return min(positions) if positions else 10**6

    def goal_tokens(self, goal: str) -> List[str]:
        mapping = {
            "diagnostic": ["为什么", "原因", "根因", "失败", "报错", "异常", "是否正常", "正常吗", "why", "reason", "error"],
            "compare": ["比较", "对比", "compare", "versus", "vs", "差异", "区别"],
            "artifacts": ["artifact", "artifacts", "产物", "输出文件", "生成了什么文件", "文件在哪", "生成了哪些文件"],
        }
        return mapping.get(goal, [])

    def goal_priority_bonus(self, goal: str, prompt: str) -> int:
        tokens = self.goal_tokens(goal)
        if not tokens:
            return 0
        lower = prompt.lower()
        explicit_markers: List[Tuple[str, int, int]] = [
            ("优先", 120, 12),
            ("首先", 120, 12),
            ("first", 120, 18),
            ("重点", 80, 12),
            ("先", 60, 8),
            ("先看", 60, 8),
            ("先查", 60, 8),
        ]
        best = 0
        for marker, bonus, window in explicit_markers:
            marker_lower = marker.lower()
            search_start = 0
            while True:
                idx = lower.find(marker_lower, search_start)
                if idx < 0:
                    break
                for token in tokens:
                    token_idx = lower.find(token.lower(), idx)
                    if token_idx >= idx and token_idx - idx <= window:
                        best = max(best, bonus)
                search_start = idx + len(marker_lower)
        return best

    def goal_optional_bonus(self, goal: str, prompt: str) -> int:
        tokens = self.goal_tokens(goal)
        if not tokens:
            return 0
        lower = prompt.lower()
        optional_markers: List[Tuple[str, int, int]] = [
            ("顺便", 120, 12),
            ("顺带", 120, 12),
            ("如果方便", 120, 18),
            ("如果可以", 120, 18),
            ("有空的话", 120, 18),
            ("if helpful", 120, 28),
            ("if convenient", 120, 28),
            ("if possible", 120, 28),
        ]
        best = 0
        for marker, bonus, window in optional_markers:
            marker_lower = marker.lower()
            search_start = 0
            while True:
                idx = lower.find(marker_lower, search_start)
                if idx < 0:
                    break
                for token in tokens:
                    token_idx = lower.find(token.lower(), idx)
                    if token_idx >= idx and token_idx - idx <= window:
                        best = max(best, bonus)
                search_start = idx + len(marker_lower)
        return best

    def goal_value_score(
        self,
        goal: str,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> int:
        lower = prompt.lower()
        base = {
            "diagnostic": 100,
            "compare": 70,
            "artifacts": 45,
        }.get(goal, 0)
        if goal == "diagnostic":
            if any(
                token in lower or token in prompt
                for token in ["根因", "root cause", "异常退出", "报错", "失败", "failed", "debug", "diagnose"]
            ):
                base += 25
            merged = "\n".join(result for _, result in outputs).lower()
            if any(token in merged for token in ["failed", "mpi_abort", "error", "异常", "报错"]):
                base += 15
        elif goal == "compare":
            if any(token in lower or token in prompt for token in ["变化", "趋势", "差异", "区别", "trend"]):
                base += 10
        elif goal == "artifacts":
            if any(token in lower or token in prompt for token in ["文件在哪", "路径", "where", "path"]):
                base += 10
        return base + self.goal_priority_bonus(goal, prompt)

    def optional_goals_for_prompt(self, goals: List[str], prompt: str) -> Set[str]:
        if len(goals) <= 1:
            return set()
        optional_goals: Set[str] = set()
        for goal in goals:
            if self.goal_priority_bonus(goal, prompt) >= 60:
                continue
            if self.goal_optional_bonus(goal, prompt) >= 80:
                optional_goals.add(goal)
        return optional_goals

    def evidence_goals_for_prompt(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> List[str]:
        lower = prompt.lower()
        followup_target = self.forced_log_target(prompt, outputs)
        asks_diagnostic = followup_target is not None or any(
            token in lower or token in prompt
            for token in [
                "为什么",
                "原因",
                "根因",
                "失败",
                "报错",
                "异常",
                "是否正常",
                "正常吗",
                "why",
                "reason",
                "error",
            ]
        )
        asks_compare = any(
            token in lower or token in prompt
            for token in ["比较", "对比", "compare", "versus", "vs", "差异", "区别"]
        )
        asks_artifacts = any(
            token in lower or token in prompt
            for token in ["artifact", "artifacts", "产物", "输出文件", "生成了什么文件", "文件在哪", "生成了哪些文件"]
        )

        ranked: List[Tuple[int, int, int, str]] = []
        if asks_compare:
            ranked.append(
                (
                    -self.goal_value_score("compare", prompt, outputs),
                    self.goal_first_position(prompt, self.goal_tokens("compare")),
                    0,
                    "compare",
                )
            )
        if asks_diagnostic:
            ranked.append(
                (
                    -self.goal_value_score("diagnostic", prompt, outputs),
                    self.goal_first_position(prompt, self.goal_tokens("diagnostic")),
                    1,
                    "diagnostic",
                )
            )
        if asks_artifacts:
            ranked.append(
                (
                    -self.goal_value_score("artifacts", prompt, outputs),
                    self.goal_first_position(prompt, self.goal_tokens("artifacts")),
                    2,
                    "artifacts",
                )
            )
        ranked.sort()
        return [goal for _, _, _, goal in ranked]

    def required_evidence_tags(self, goals: List[str]) -> Set[str]:
        mapping = {
            "compare": "compare",
            "diagnostic": "logs",
            "artifacts": "artifacts",
        }
        return {mapping[goal] for goal in goals if goal in mapping}

    def goal_label(self, goal: str) -> str:
        mapping = {
            "compare": "run 差异",
            "diagnostic": "失败原因/异常判断",
            "artifacts": "产物文件",
        }
        return mapping.get(goal, goal)

    def goal_labels_if_satisfied(self, goals: List[str], current_tags: Set[str]) -> List[str]:
        labels: List[str] = []
        for goal in goals:
            required = self.required_evidence_tags([goal])
            if not required or required.issubset(current_tags):
                labels.append(self.goal_label(goal))
        return labels

    def run_mode_label(self, mode: Optional[str]) -> Optional[str]:
        mapping = {
            "md_to_kmc_chain": "MD→KMC chain",
            "kmc_only": "KMC-only",
            "md_only": "MD-only",
        }
        return mapping.get(mode or "")

    def extract_bulleted_value(self, text: str, key: str) -> Optional[str]:
        pattern = rf"(?m)^\s*-\s+{re.escape(key)}:\s+(.+)$"
        match = re.search(pattern, text)
        if not match:
            return None
        return match.group(1).strip()

    def context_run_name(self, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        if self.current_run_dir:
            return self.current_run_dir.name
        for intent, output in reversed(outputs):
            if intent.action == "inspect":
                value = self.extract_bulleted_value(output, "job_id") or self.extract_bulleted_value(output, "path")
                if value:
                    return Path(value).name
            if intent.action == "artifacts":
                value = self.extract_bulleted_value(output, "run")
                if value:
                    return Path(value).name
            if intent.action == "compare_runs":
                value = self.extract_bulleted_value(output, "newer run")
                if value:
                    return Path(value).name
            if intent.action == "logs":
                value = self.extract_bulleted_value(output, "path")
                if value:
                    path = Path(value)
                    try:
                        return path.parents[2].name
                    except IndexError:
                        return path.parent.name
        return None

    def context_run_mode(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> Optional[str]:
        mode = infer_run_mode_hint(prompt)
        if mode:
            return mode
        if self.current_run_mode:
            return self.current_run_mode
        for intent, output in reversed(outputs):
            mode_hint = intent.params.get("mode")
            if isinstance(mode_hint, str) and mode_hint:
                return mode_hint
            value = self.extract_bulleted_value(output, "mode")
            if value:
                return value
        return None

    def context_log_target(self, prompt: str, outputs: List[Tuple[ToolIntent, str]]) -> str:
        target = self.forced_log_target(prompt, outputs)
        if target:
            return target
        lower = prompt.lower()
        if "kmc" in lower:
            return "kmc"
        if "lammps" in lower or "md" in lower:
            return "md"
        if "summary" in lower or "总结" in prompt:
            return "summary"
        for intent, output in reversed(outputs):
            if intent.action == "logs":
                param_target = intent.params.get("target")
                if isinstance(param_target, str) and param_target:
                    return param_target
                match = re.search(r"Log excerpt \(([^)]+)\)", output)
                if match:
                    return match.group(1).strip()
        return "auto"

    def log_target_label(self, target: str) -> Optional[str]:
        mapping = {
            "kmc": "KMC",
            "md": "MD",
            "summary": "summary",
        }
        return mapping.get(target)

    def goal_followup_prompt(
        self,
        goal: str,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> Optional[str]:
        run_name = self.context_run_name(outputs)
        mode_label = self.run_mode_label(self.context_run_mode(prompt, outputs))
        log_target_label = self.log_target_label(self.context_log_target(prompt, outputs))

        mapping = {
            "compare": (
                f"那再帮我比较最近两次 {mode_label} run 的差异。"
                if mode_label
                else "那再帮我比较最近两次 run 的差异。"
            ),
            "diagnostic": (
                f"那再继续帮我看一下 `{run_name}` 的 {log_target_label} 日志，找出失败根因。"
                if run_name and log_target_label
                else (
                    f"那再继续帮我看一下 {log_target_label} 日志，找出失败根因。"
                    if log_target_label
                    else (
                        f"那再继续帮我看一下 `{run_name}` 的日志，找出失败根因。"
                        if run_name
                        else "那再继续帮我找一下这次失败的根因。"
                    )
                )
            ),
            "artifacts": (
                f"那再帮我列一下 `{run_name}` 这个 run 生成的产物文件。"
                if run_name
                else (
                    f"那再帮我列一下这个 {mode_label} run 生成的产物文件。"
                    if mode_label
                    else "那再帮我列一下这个 run 生成的产物文件。"
                )
            ),
        }
        return mapping.get(goal)

    def candidate_evidence_action_for_goal(
        self,
        goal: str,
        current_tags: Set[str],
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        run_mode: Optional[str],
    ) -> Optional[ToolIntent]:
        available_runs = self.available_runs_for_mode(run_mode, limit=3)

        if goal == "compare":
            if "compare" in current_tags:
                return None
            params: Dict[str, Any] = {}
            if run_mode:
                params["mode"] = run_mode
            if len(available_runs) >= 2:
                return ToolIntent(action="compare_runs", params=params)
            if "runs" not in current_tags and available_runs:
                return ToolIntent(action="runs", params={})
            return None

        if goal == "diagnostic":
            if "logs" in current_tags:
                return None
            if "inspect" in current_tags:
                log_target = self.forced_log_target(prompt, outputs) or "auto"
                params = {"run": "current", "target": log_target}
                if run_mode:
                    params["mode"] = run_mode
                return ToolIntent(action="logs", params=params)
            target_hint = self.run_target_hint(run_mode)
            if target_hint:
                params = {"run": target_hint}
                if run_mode:
                    params["mode"] = run_mode
                return ToolIntent(action="inspect", params=params)
            if "runs" not in current_tags and available_runs:
                return ToolIntent(action="runs", params={})
            return None

        if goal == "artifacts":
            if "artifacts" in current_tags:
                return None
            target_hint = self.run_target_hint(run_mode)
            if target_hint:
                params = {"run": target_hint}
                if run_mode:
                    params["mode"] = run_mode
                return ToolIntent(action="artifacts", params=params)
            if "runs" not in current_tags and available_runs:
                return ToolIntent(action="runs", params={})
            return None

        return None

    def simulate_evidence_tags(self, current_tags: Set[str], intent: ToolIntent) -> Set[str]:
        next_tags = set(current_tags)
        if intent.action == "runs":
            next_tags.update({"runs"})
        elif intent.action == "compare_runs":
            next_tags.update({"runs", "compare"})
        elif intent.action == "inspect":
            next_tags.update({"runs", "inspect"})
        elif intent.action == "logs":
            next_tags.update({"runs", "inspect", "logs"})
        elif intent.action == "artifacts":
            next_tags.update({"runs", "artifacts"})
        return next_tags

    def plan_evidence_followups(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> List[ToolIntent]:
        goals = self.evidence_goals_for_prompt(prompt, outputs)
        if not goals:
            return []

        optional_goals = self.optional_goals_for_prompt(goals, prompt)
        active_goals = [goal for goal in goals if goal not in optional_goals] or goals[:1]
        required_tags = self.required_evidence_tags(active_goals)
        start_tags = frozenset(self.tool_evidence_tags(outputs))
        if required_tags.issubset(start_tags):
            return []

        run_mode = infer_run_mode_hint(prompt)
        queue: deque[Tuple[frozenset[str], List[ToolIntent]]] = deque([(start_tags, [])])
        visited: Set[frozenset[str]] = {start_tags}

        while queue:
            tags_frozen, path = queue.popleft()
            tags = set(tags_frozen)
            if required_tags.issubset(tags):
                return path
            if len(path) >= 4:
                continue

            emitted: Set[str] = set()
            existing_signatures = {self.intent_signature(intent) for intent in path}
            for goal in active_goals:
                intent = self.candidate_evidence_action_for_goal(goal, tags, prompt, outputs, run_mode)
                if intent is None:
                    continue
                signature = self.intent_signature(intent)
                if signature in existing_signatures or signature in emitted:
                    continue
                emitted.add(signature)
                next_tags = frozenset(self.simulate_evidence_tags(tags, intent))
                if next_tags in visited:
                    continue
                visited.add(next_tags)
                queue.append((next_tags, path + [intent]))
        return []

    def describe_evidence_path(self, path: List[ToolIntent]) -> List[str]:
        if not path:
            return []
        labels = {
            "runs": "列出相关 runs",
            "compare_runs": "直接对比最近两次 run",
            "inspect": "先深入 inspect 具体 run",
            "logs": "再补看关键日志",
            "artifacts": "直接列出产物文件",
        }
        rendered = [labels.get(intent.action, intent.action) for intent in path]
        if len(rendered) == 1:
            return [f"- 我会优先补最关键的证据；当前最短路径是：{rendered[0]}。"]
        return [
            "- 我会优先补最关键的证据；当前最短路径是：" + " → ".join(rendered) + "。",
            f"- 我会先执行：{rendered[0]}。",
        ]

    def unavailable_evidence_strategy(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
        goals: List[str],
    ) -> ToolResponseStrategy:
        run_mode = infer_run_mode_hint(prompt)
        available_runs = self.available_runs_for_mode(run_mode, limit=3)
        current_tags = self.tool_evidence_tags(outputs)
        missing_tags = self.required_evidence_tags(goals) - current_tags

        if "compare" in missing_tags and len(available_runs) < 2:
            return ToolResponseStrategy(
                status="insufficient",
                reason="当前可比较的相关 run 不足两个，所以还没法直接做差异对比。",
                next_steps=["- 先再产出一个同模式 run，之后我可以立刻帮你 compare。"],
            )
        if "logs" in missing_tags and not self.run_target_hint(run_mode):
            return ToolResponseStrategy(
                status="insufficient",
                reason="当前还没有定位到可 inspect / 看日志的 run，所以没法继续诊断根因。",
                next_steps=["- 先确认要调查哪个 run，或者先跑出一个 run。"],
            )
        if "artifacts" in missing_tags and not self.run_target_hint(run_mode):
            return ToolResponseStrategy(
                status="insufficient",
                reason="当前还没有定位到可查看产物的 run，所以暂时列不出输出文件。",
                next_steps=["- 先确认 run 或先执行一次任务，之后我就能列 artifacts。"],
            )
        return ToolResponseStrategy(
            status="provisional",
            reason="当前已经拿到一些证据，但还不够覆盖你这轮真正关心的问题。",
            next_steps=["- 目前没有更短的自动补证据路径了；如果你愿意，我可以先基于现有信息给初步判断。"],
        )

    def response_strategy_for_prompt(
        self,
        prompt: str,
        outputs: List[Tuple[ToolIntent, str]],
    ) -> ToolResponseStrategy:
        if not outputs:
            return ToolResponseStrategy(
                status="insufficient",
                reason="当前还没有拿到任何工具证据。",
                next_steps=["- 先执行一个相关工具步骤，再继续判断。"],
            )
        actions = [intent.action for intent, _ in outputs]
        goals = self.evidence_goals_for_prompt(prompt, outputs)
        optional_goals = self.optional_goals_for_prompt(goals, prompt)
        required_goals = [goal for goal in goals if goal not in optional_goals] or goals[:1]
        followup_path = self.plan_evidence_followups(prompt, outputs)
        current_tags = self.tool_evidence_tags(outputs)
        required_tags = self.required_evidence_tags(required_goals)
        missing_tags = required_tags - current_tags
        has_execution = any(action in {"run", "moire_run", "moire_diffusion_sweep", "bridge_kmc_lookup"} for action in actions)
        has_draft = any(action == "draft" for action in actions)

        if goals:
            if followup_path:
                first_intent = followup_path[0]
                reason_map = {
                    "runs": "当前先需要确认有哪些相关 run，才能继续走最短调查路径。",
                    "compare_runs": "当前还缺少最近两次 run 的直接对比结果。",
                    "inspect": "当前还缺少足够细的 run 细节，没法直接下判断。",
                    "logs": "当前还缺少能解释异常原因的关键日志证据。",
                    "artifacts": "当前还没把这次 run 的产物文件列出来。",
                }
                return ToolResponseStrategy(
                    status="needs_more_evidence",
                    reason=reason_map.get(first_intent.action, "当前还需要补一轮关键证据。"),
                    followup_intent=first_intent,
                    next_steps=self.describe_evidence_path(followup_path),
                    answered_goals=self.goal_labels_if_satisfied(required_goals, current_tags),
                )
            if not missing_tags:
                answered_required = self.goal_labels_if_satisfied(required_goals, current_tags)
                optional_missing = [
                    goal
                    for goal in optional_goals
                    if self.required_evidence_tags([goal]) - current_tags
                ]
                if optional_missing:
                    deferred_labels = [self.goal_label(goal) for goal in optional_missing]
                    followup_prompts = [
                        prompt_text
                        for goal in optional_missing
                        if (prompt_text := self.goal_followup_prompt(goal, prompt, outputs))
                    ]
                    return ToolResponseStrategy(
                        status="sufficient",
                        reason=(
                            "主要问题已经回答；我先停在这里，避免为了顺带问题继续补证据。"
                            f"暂时没继续展开的次要问题：{', '.join(deferred_labels)}。"
                        ),
                        next_steps=[f"- 如果你还想继续，我可以再补查这些次要问题：{', '.join(deferred_labels)}。"],
                        answered_goals=answered_required,
                        deferred_goals=deferred_labels,
                        followup_prompts=followup_prompts,
                    )
                satisfied_reason: List[str] = []
                if "compare" in goals:
                    satisfied_reason.append("最近两次 run 的直接对比结果已经拿到")
                if "diagnostic" in goals:
                    satisfied_reason.append("关键日志证据已经拿到")
                if "artifacts" in goals:
                    satisfied_reason.append("产物文件列表已经拿到")
                reason = "；".join(satisfied_reason) + "，可以直接回答。" if satisfied_reason else "当前证据已经足够回答这轮问题。"
                return ToolResponseStrategy(
                    status="sufficient",
                    reason=reason,
                    next_steps=["- 如果你愿意，我可以把这些证据再压缩成一句更短的结论。"],
                    answered_goals=self.goal_labels_if_satisfied(goals, current_tags),
                )
            return self.unavailable_evidence_strategy(prompt, outputs, goals)

        if has_execution:
            return ToolResponseStrategy(
                status="sufficient",
                reason="这轮的核心目标是执行动作，当前已经拿到执行结果。",
                next_steps=["- 如果你愿意，我可以继续检查这次执行是否正常完成。"],
            )
        if has_draft:
            return ToolResponseStrategy(
                status="sufficient",
                reason="当前已经拿到草案结果，可以先确认方向再决定是否执行。",
                next_steps=["- 如果草案方向对，我可以继续把它运行起来。"],
            )
        return ToolResponseStrategy(
            status="sufficient",
            reason="当前工具结果已经足够回答这轮问题。",
            next_steps=self.suggest_next_steps(outputs),
        )
