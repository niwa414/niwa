from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .local_profile import get_local_model_settings


LOCAL_MODEL_ALIASES = {
    "27b": ["27b", "huihui-qwen3.5-27b"],
    "122b": ["122b", "qwen3.5-122b"],
}


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _local_model_base_url() -> str:
    return get_local_model_settings()["base_url"]


def _local_model_api_key() -> str:
    return get_local_model_settings()["api_key"]


def _resolve_model_alias(requested: Optional[str], models: List[str]) -> Optional[str]:
    if not requested:
        return None
    text = requested.strip()
    if not text:
        return None
    lower = text.lower()
    for item in models:
        if item == text or item.lower() == lower:
            return item
    alias_tokens = LOCAL_MODEL_ALIASES.get(lower)
    if alias_tokens:
        for item in models:
            item_lower = item.lower()
            if any(token in item_lower for token in alias_tokens):
                return item
    for item in models:
        if lower in item.lower():
            return item
    return None


def _preferred_local_model(models: List[str], requested: Optional[str] = None, fallback: Optional[str] = None) -> Optional[str]:
    resolved_requested = _resolve_model_alias(requested, models)
    if resolved_requested:
        return resolved_requested
    preferred_env = get_local_model_settings()["preferred_model"]
    resolved_env = _resolve_model_alias(preferred_env, models)
    if resolved_env:
        return resolved_env
    preferred_27b = _resolve_model_alias("27b", models)
    if preferred_27b:
        return preferred_27b
    resolved_fallback = _resolve_model_alias(fallback, models)
    if resolved_fallback:
        return resolved_fallback
    return models[0] if models else fallback


def _model_for_purpose(status: Dict[str, Any], purpose: str, selected_model: Optional[str] = None) -> Optional[str]:
    models = status.get("models") or []
    purpose_env = {
        "chat": os.environ.get("MIETCLAW_CHAT_MODEL"),
        "router": os.environ.get("MIETCLAW_ROUTER_MODEL"),
        "plan": os.environ.get("MIETCLAW_PLAN_MODEL"),
        "agent": os.environ.get("MIETCLAW_AGENT_MODEL"),
        "summary": os.environ.get("MIETCLAW_SUMMARY_MODEL"),
    }.get(purpose)
    fallback = status.get("default_model")
    return _preferred_local_model(models, requested=selected_model or purpose_env, fallback=fallback)


def _local_model_request(resource_path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 90.0) -> Dict[str, Any]:
    url = f"{_local_model_base_url()}{resource_path}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {_local_model_api_key()}",
    }
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        error_payload = parsed.get("error") if isinstance(parsed, dict) else {}
        message = error_payload.get("message") if isinstance(error_payload, dict) else None
        message = message or parsed.get("detail") if isinstance(parsed, dict) else None
        raise RuntimeError(message or raw or str(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Local model returned non-JSON content: {raw[:200]}") from exc


def get_local_model_status() -> Dict[str, Any]:
    profile = get_local_model_settings()
    base_url = profile["base_url"]
    try:
        health = _local_model_request("/health")
        models_payload = _local_model_request("/v1/models")
        models = [item.get("id") for item in models_payload.get("data", []) if item.get("id")]
        default_model = _preferred_local_model(models, fallback=health.get("default_model"))
        return {
            "healthy": health.get("status") == "healthy",
            "default_model": default_model,
            "models": models,
            "base_url": base_url,
            "profile_path": profile["profile_path"],
            "preferred_model": profile["preferred_model"],
            "agent_name": profile["agent_name"],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "healthy": False,
            "default_model": None,
            "models": [],
            "base_url": base_url,
            "profile_path": profile["profile_path"],
            "preferred_model": profile["preferred_model"],
            "agent_name": profile["agent_name"],
            "error": str(exc),
        }


def _coerce_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _compact_local_history(history: List[Tuple[str, str]], limit: int = 10) -> List[Dict[str, str]]:
    compact: List[Dict[str, str]] = []
    for role, content in history[-limit:]:
        compact.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": content[:5000],
            }
        )
    return compact


def _truncate_for_model(text: str, limit: int = 3500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def chat_with_local_model(messages: List[Dict[str, str]], model: Optional[str] = None, purpose: str = "chat") -> Dict[str, Any]:
    status = get_local_model_status()
    if not status.get("healthy"):
        raise RuntimeError(status.get("error") or "Local model is unavailable")
    chosen_model = _model_for_purpose(status, purpose=purpose, selected_model=model)
    payload = {
        "model": chosen_model,
        "temperature": 0.2,
        "stream": False,
        "messages": messages,
    }
    response = _local_model_request("/v1/chat/completions", payload=payload, timeout=180.0)
    choice = ((response.get("choices") or [{}])[0]).get("message", {})
    return {
        "content": _coerce_message_content(choice.get("content")),
        "model": response.get("model") or payload["model"],
        "reasoning": _coerce_message_content(choice.get("reasoning_content")),
    }


def ensure_web_console(project_root: Path, port: int = 4174) -> Dict[str, Any]:
    runtime_dir = project_root / ".runtime-checks"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}/"
    log_path = runtime_dir / "mietclaw-web-preview.log"
    pid_path = runtime_dir / "mietclaw-web-preview.pid"

    running = _http_ok(url)
    if not running:
        stdout = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["npm", "run", "preview", "--workspace", "@miet-claw/web", "--", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(project_root),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_path.write_text(str(process.pid), encoding="utf-8")
        deadline = time.time() + 12
        while time.time() < deadline:
            if _http_ok(url):
                running = True
                break
            time.sleep(0.4)

    opener = shutil.which("open") or shutil.which("xdg-open")
    opened = False
    if opener:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        opened = True

    return {
        "url": url,
        "running": running,
        "opened": opened,
        "log_path": str(log_path),
        "pid_path": str(pid_path),
    }
