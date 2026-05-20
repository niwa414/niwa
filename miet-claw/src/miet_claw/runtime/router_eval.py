from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..tool_router import ToolIntent, ToolPlan, heuristic_tool_intent, heuristic_tool_plan


def _intent_payload(intent: Optional[ToolIntent]) -> Optional[Dict[str, Any]]:
    if intent is None:
        return None
    return {"action": intent.action, "params": dict(intent.params), "reply": intent.reply}


def _plan_payload(plan: Optional[ToolPlan]) -> Optional[Dict[str, Any]]:
    if plan is None:
        return None
    return {
        "steps": [_intent_payload(step) for step in plan.steps],
        "actions": [step.action for step in plan.steps],
        "summarize": bool(plan.summarize),
        "reply": plan.reply,
    }


def _format_case_string(value: str, replacements: Dict[str, str]) -> str:
    formatted = value
    for key, replacement in replacements.items():
        formatted = formatted.replace("{" + key + "}", replacement)
    return formatted


def _format_case_value(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return _format_case_string(value, replacements)
    if isinstance(value, list):
        return [_format_case_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _format_case_value(item, replacements) for key, item in value.items()}
    return value


def _prepare_case_fixtures(case: Dict[str, Any], replacements: Dict[str, str]) -> None:
    fixtures = case.get("fixtures") or {}
    for raw_dir in fixtures.get("dirs") or []:
        Path(_format_case_value(raw_dir, replacements)).mkdir(parents=True, exist_ok=True)
    for raw_path, raw_content in (fixtures.get("files") or {}).items():
        path = Path(_format_case_value(raw_path, replacements))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = _format_case_value(raw_content, replacements)
        path.write_text(str(content), encoding="utf-8")


def _matches_expected(actual: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if expected.get("action") is not None:
        if actual is None or actual.get("action") != expected["action"]:
            errors.append(f"action expected {expected['action']!r}, got {(actual or {}).get('action')!r}")
    expected_params = expected.get("params") or {}
    actual_params = (actual or {}).get("params") or {}
    for key, value in expected_params.items():
        if actual_params.get(key) != value:
            errors.append(f"params.{key} expected {value!r}, got {actual_params.get(key)!r}")
    return not errors, errors


def _matches_plan(actual: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    expected_actions = expected.get("actions")
    if expected_actions is not None:
        actual_actions = (actual or {}).get("actions") or []
        if actual_actions != expected_actions:
            errors.append(f"plan actions expected {expected_actions!r}, got {actual_actions!r}")
    if expected.get("summarize") is not None:
        actual_summarize = (actual or {}).get("summarize") if actual is not None else None
        if actual_summarize != expected["summarize"]:
            errors.append(f"plan summarize expected {expected['summarize']!r}, got {actual_summarize!r}")
    return not errors, errors


def run_router_golden_eval(golden_file: str | Path, *, output_dir: str | Path, current_run_dir: str | Path | None = None) -> Dict[str, Any]:
    path = Path(golden_file).expanduser().resolve()
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("router golden file must contain a list of cases")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    current = Path(current_run_dir).expanduser().resolve() if current_run_dir else None
    replacements = {
        "output_dir": str(output_root),
        "current_run_dir": str(current) if current else "",
    }

    results = []
    passed = 0
    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"router golden case {index} must be an object")
        _prepare_case_fixtures(raw_case, replacements)
        case = _format_case_value(raw_case, replacements)
        prompt = str(case.get("prompt") or "")
        mode = str(case.get("mode") or "intent")
        expected = case.get("expected") or {}
        if mode == "plan":
            actual = _plan_payload(heuristic_tool_plan(prompt, output_root, current_run_dir=current))
            ok, errors = _matches_plan(actual, expected)
        else:
            actual = _intent_payload(heuristic_tool_intent(prompt, output_root, current_run_dir=current))
            ok, errors = _matches_expected(actual, expected)
        if ok:
            passed += 1
        results.append(
            {
                "index": index,
                "name": case.get("name") or f"case-{index}",
                "mode": mode,
                "prompt": prompt,
                "ok": ok,
                "errors": errors,
                "expected": expected,
                "actual": actual,
            }
        )
    total = len(results)
    return {
        "golden_file": str(path),
        "output_dir": str(output_root),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "ok": passed == total,
        "results": results,
    }
