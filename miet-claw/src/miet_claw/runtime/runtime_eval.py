from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..moire_health import build_repo_kmc_runtime_health, parse_repo_kmc_run_output


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "actual"


def _matches_subset(actual: Any, expected: Any, path: tuple[str, ...] = ()) -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{_format_path(path)} expected object, got {type(actual).__name__}"]
        for key, expected_value in expected.items():
            if key not in actual:
                errors.append(f"{_format_path(path + (str(key),))} missing")
                continue
            errors.extend(_matches_subset(actual[key], expected_value, path + (str(key),)))
        return errors
    if actual != expected:
        errors.append(f"{_format_path(path)} expected {expected!r}, got {actual!r}")
    return errors


def run_runtime_health_golden_eval(golden_file: str | Path) -> Dict[str, Any]:
    path = Path(golden_file).expanduser().resolve()
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("runtime health golden file must contain a list of cases")

    results = []
    passed = 0
    for index, case in enumerate(cases, start=1):
        run_text = str(case.get("run_text") or "")
        returncode = int(case.get("returncode", 0))
        parsed = parse_repo_kmc_run_output(run_text)
        status, runtime_health = build_repo_kmc_runtime_health(
            returncode=returncode,
            parsed=parsed,
            run_text=run_text,
        )
        actual = {
            "status": status,
            "parsed": parsed,
            "runtime_health": runtime_health,
        }
        expected = case.get("expected") or {}
        errors = _matches_subset(actual, expected)
        ok = not errors
        if ok:
            passed += 1
        results.append(
            {
                "index": index,
                "name": case.get("name") or f"case-{index}",
                "ok": ok,
                "errors": errors,
                "expected": expected,
                "actual": actual,
            }
        )

    total = len(results)
    return {
        "golden_file": str(path),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "ok": passed == total,
        "results": results,
    }
