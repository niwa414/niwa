import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .local_profile import get_runtime_settings


TOOL_ROUTER_SYSTEM_PROMPT = """You are the tool router for mietclaw.
Choose exactly one action and return strict JSON only.

Allowed actions:
- chat
- draft
- run
- moire_run
- moire_compare
- moire_diffusion_sweep
- runs
- compare_runs
- inspect
- artifacts
- logs
- open_web
- bridge_kmc_lookup

Rules:
- Prefer chat unless the user clearly wants a real tool action.
- Do not invent file paths.
- Respect the tool_budget object from the system context. If remaining_mutating_steps is 0, do not choose draft, run, moire_run, moire_compare, moire_diffusion_sweep, or bridge_kmc_lookup.
- If completed_tool_steps already contain enough evidence, prefer action=chat with a short reply instead of repeating a tool.
- Prefer the highest-value evidence first. When the user mixes multiple asks, prioritize the step that most quickly rules out failure or unsafe state, unless the user explicitly says to do another step first.
- Prefer the shortest evidence path: comparison questions should usually go straight to compare_runs, output-file questions should usually go straight to artifacts, and root-cause questions should usually go inspect -> logs.
- If a lower-priority ask is clearly optional (for example "顺便" or "if helpful"), do not choose extra tools for it once the main question is already answerable.
- Use draft when the user wants to generate or sketch a workflow but not execute it.
- Use run when the user wants to actually launch a workflow or simulation.
- Use bridge_kmc_lookup when the user wants to turn an event.json plus a neb.txt or barrier into a KMC lookup file and optionally validate it.
- Use moire_run when the user wants to run a real local MoRe LAMMPS case on this computer and then continue into KMC.
- Use moire_compare when the user wants to compare multiple MoRe event.json files on one case, optionally including KMC.
- Use moire_diffusion_sweep when the user wants one MoRe barrier first and then a temperature sweep that summarizes diffusion coefficient vs temperature or an Arrhenius-style result.
- If required data is missing, return action=chat and explain briefly in reply.
- Prefer the native block envelope below. The legacy flat action JSON is still accepted as a fallback, but it is not preferred.

Preferred JSON schema:
{"blocks":[{"toolRequests":[{"action":"draft|run|moire_run|moire_compare|moire_diffusion_sweep|runs|compare_runs|inspect|artifacts|logs|open_web|bridge_kmc_lookup","params":{"prompt":null,"run":null,"target":null,"mode":null,"port":4174,"event_json":null,"event_jsons":null,"neb_txt":null,"case_dir":null,"workdir":null,"temperatures_k":null,"run_time":null,"stats_step":null,"validate":true,"kmc_seed":null,"kmc_seeds":null,"ovito":false,"lammps_only":false}}],"finalAnswer":{"reply":null},"metadata":{"mode":"router"}}]}

If the best action is chat, return:
{"blocks":[{"finalAnswer":{"reply":"..."}}]}
"""


ALLOWED_ACTIONS = {
    "chat",
    "draft",
    "run",
    "moire_run",
    "moire_compare",
    "moire_diffusion_sweep",
    "runs",
    "compare_runs",
    "inspect",
    "artifacts",
    "logs",
    "open_web",
    "bridge_kmc_lookup",
}

ALLOWED_PARAM_KEYS = {
    "prompt",
    "run",
    "target",
    "mode",
    "port",
    "event_json",
    "event_jsons",
    "neb_txt",
    "case_dir",
    "workdir",
    "temperatures_k",
    "run_time",
    "stats_step",
    "validate",
    "kmc_seed",
    "kmc_seeds",
    "ovito",
    "lammps_only",
}


TOOL_PLAN_SYSTEM_PROMPT = """You are the multi-step tool planner for mietclaw.
When the user clearly needs more than one tool step, return strict JSON only.

Allowed actions inside steps:
- runs
- compare_runs
- inspect
- artifacts
- logs
- draft
- run
- moire_run
- moire_compare
- moire_diffusion_sweep
- bridge_kmc_lookup

Rules:
- Return at most 4 steps.
- Respect the tool_budget object from the system context and keep the plan within that budget.
- Use run="latest" when the user refers to the latest or newest run.
- Use inspect before logs or artifacts when the user asks to check a run and then explain it.
- Prefer the highest-value evidence first. If multiple asks are mixed together, prioritize the step that most quickly rules out failure or unsafe state, unless the user explicitly orders the steps.
- Prefer the shortest evidence chain that still answers the question: compare_runs for direct comparisons, artifacts for output-file questions, and inspect -> logs for root-cause questions.
- If a side ask is clearly optional (for example "顺便" or "if helpful"), stop after the main question is answerable instead of extending the plan.
- Prefer run over draft+run when the user wants real execution.
- Prefer moire_compare when the user explicitly wants multiple MoRe events compared on one case.
- Prefer moire_diffusion_sweep when the user explicitly wants diffusion coefficient vs temperature or an Arrhenius-style MoRe→KMC sweep.
- Prefer a single bridge_kmc_lookup step when bridge validation already answers the request.
- Prefer the native block envelope below. The legacy {"steps":[...]} planner JSON is still accepted as a fallback, but it is not preferred.
- If the request does not need multiple tool steps, return {"blocks":[]}.

Preferred JSON schema:
{"blocks":[{"toolRequests":[{"action":"runs|compare_runs|inspect|artifacts|logs|draft|run|moire_run|moire_compare|moire_diffusion_sweep|bridge_kmc_lookup","params":{"prompt":null,"run":null,"target":null,"mode":null,"event_json":null,"event_jsons":null,"neb_txt":null,"case_dir":null,"workdir":null,"temperatures_k":null,"run_time":null,"stats_step":null,"validate":true,"kmc_seed":null,"kmc_seeds":null,"ovito":false,"lammps_only":false}}],"metadata":{"plan_summarize":true}}]}

If you should answer directly instead of requesting more tools, return:
{"blocks":[{"finalAnswer":{"reply":"..."}}]}
"""


AGENT_LOOP_SYSTEM_PROMPT = """You are the iterative tool-using agent for mietclaw.
At each turn, decide either:
- one next tool step to run
- or the final answer for the user

Allowed actions:
- runs
- compare_runs
- inspect
- artifacts
- logs
- draft
- run
- moire_run
- moire_compare
- moire_diffusion_sweep
- bridge_kmc_lookup

Rules:
- Return strict JSON only.
- Return exactly one next step when more evidence is needed.
- Use run="latest" when the user refers to the latest or newest run.
- If the user asks whether a run is normal, inspect first; if the inspect output is inconclusive, then read logs.
- Prefer the highest-value evidence first. If multiple asks are mixed together, prioritize the step that most quickly rules out failure or unsafe state, unless the user explicitly orders the steps.
- Prefer the shortest evidence path that can close the question: compare_runs for direct comparisons, artifacts for output-file questions, and inspect -> logs for root-cause questions.
- If a side ask is clearly optional (for example "顺便" or "if helpful"), finish once the main question is answerable instead of asking for more tools.
- Respect the tool_budget object from the system context. If remaining_steps is 0, finish instead of requesting another tool.
- If remaining_mutating_steps is 0, do not ask for draft, run, moire_run, moire_compare, moire_diffusion_sweep, or bridge_kmc_lookup.
- If failure_count is already high, prefer finishing with the best explanation supported by current evidence.
- If you already have enough evidence, finish instead of asking for more tools.
- Never invent file paths.
- Keep step params minimal.
- Prefer the native block envelope below. The legacy {"status":"continue|finish",...} JSON is still accepted as a fallback, but it is not preferred.

Preferred JSON schema:
{"blocks":[{"toolRequests":[{"action":"runs|compare_runs|inspect|artifacts|logs|draft|run|moire_run|moire_compare|moire_diffusion_sweep|bridge_kmc_lookup","params":{"prompt":null,"run":null,"target":null,"mode":null,"event_json":null,"event_jsons":null,"neb_txt":null,"case_dir":null,"workdir":null,"temperatures_k":null,"run_time":null,"stats_step":null,"validate":true,"kmc_seed":null,"kmc_seeds":null,"ovito":false,"lammps_only":false}}],"finalAnswer":{"reply":null}}]}

If you should finish, return:
{"blocks":[{"finalAnswer":{"reply":"..."}}]}
"""


@dataclass
class ToolIntent:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reply: Optional[str] = None


@dataclass
class ToolPlan:
    steps: List[ToolIntent] = field(default_factory=list)
    summarize: bool = False
    reply: Optional[str] = None


@dataclass
class AgentDecision:
    status: str
    step: Optional[ToolIntent] = None
    reply: Optional[str] = None


def _sanitize_params(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    params: Dict[str, Any] = {}
    for key in ALLOWED_PARAM_KEYS:
        if key in raw and raw[key] is not None:
            params[key] = raw[key]
    return params


def _make_tool_intent(action: Any, params: Any = None, reply: Optional[str] = None) -> Optional[ToolIntent]:
    if not isinstance(action, str):
        return None
    normalized_action = action.strip()
    if normalized_action not in ALLOWED_ACTIONS:
        return None
    return ToolIntent(action=normalized_action, params=_sanitize_params(params), reply=reply)


def _extract_abs_paths(prompt: str) -> List[str]:
    matches = re.findall(r'(/[^\s"\'`]+)', prompt)
    cleaned: List[str] = []
    for item in matches:
        cleaned_item = item.rstrip(".,;:!?)】）}>")
        cleaned.append(cleaned_item)
    return cleaned


def _recent_run_names(output_dir: Path, limit: int = 20) -> List[str]:
    if not output_dir.exists():
        return []
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and ((path / "state.json").exists() or (path / "summary.json").exists())
    ]
    ordered = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return [path.name for path in ordered[:limit]]


def _default_bridge_workdir(output_dir: Path) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str((output_dir / f"bridge_{stamp}").resolve())


def _guess_project_root_from_output_dir(output_dir: Path) -> Path:
    return output_dir.resolve().parent


def _default_moire_case_dir(output_dir: Path) -> Optional[str]:
    override = os.environ.get("MIETCLAW_MOIRE_CASE_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
    project_root = _guess_project_root_from_output_dir(output_dir)
    candidate = Path(get_runtime_settings(project_root).get("moire_case_dir") or "")
    if candidate.exists():
        return str(candidate)
    return None


def _default_moire_event_json(output_dir: Path) -> Optional[str]:
    override = os.environ.get("MIETCLAW_MOIRE_EVENT_JSON")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
    return None


def _default_seed_sequence(count: int, start: int = 3401) -> List[int]:
    return [start + index for index in range(count)]


def _parse_seed_numbers(raw: str) -> List[int]:
    numbers = [int(item) for item in re.split(r"[\s,，、]+", raw.strip()) if item]
    seen = set()
    ordered: List[int] = []
    for value in numbers:
        if value <= 0 or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _extract_moire_seed_params(prompt: str) -> Dict[str, Any]:
    explicit_patterns = [
        r"(?:随机种子|种子列表|seeds?|seed)\s*(?:[:=：]|是|为)?\s*([0-9][0-9,\s，、]+)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        seeds = _parse_seed_numbers(match.group(1))
        if not seeds:
            continue
        if len(seeds) == 1:
            return {"kmc_seed": seeds[0]}
        return {"kmc_seeds": seeds}

    count_match = re.search(r"(?:随机种子|seeds?|seed)\D{0,12}(\d+)\s*(?:个|组|次)", prompt, re.IGNORECASE)
    if count_match:
        count = int(count_match.group(1))
        if count > 1:
            return {"kmc_seeds": _default_seed_sequence(count)}

    single_match = re.search(r"(?:随机种子|种子|seed)\s*(?:[:=：]|是|为)?\s*(\d+)", prompt, re.IGNORECASE)
    if single_match:
        return {"kmc_seed": int(single_match.group(1))}
    return {}


def _extract_temperature_values(prompt: str) -> List[float]:
    values: List[float] = []
    seen = set()
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*K\b", prompt, re.IGNORECASE):
        value = float(match.group(1))
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    if values:
        return values

    explicit = re.search(
        r"(?:温度(?:列表|点|扫描)?|temperatures?|temperature)\s*(?:[:=：]|是|为)?\s*([0-9][0-9,\s，、.]+)",
        prompt,
        re.IGNORECASE,
    )
    if explicit:
        for item in re.split(r"[\s,，、]+", explicit.group(1).strip()):
            if not item:
                continue
            try:
                value = float(item)
            except ValueError:
                continue
            if value in seen:
                continue
            values.append(value)
            seen.add(value)
    return values


def _detect_log_target(prompt: str) -> str:
    lower = prompt.lower()
    if "kmc" in lower:
        return "kmc"
    if "md" in lower:
        return "md"
    if "summary" in lower or "总结" in prompt:
        return "summary"
    return "auto"


def _looks_like_lammps_barrier_to_kmc_request(prompt: str) -> bool:
    lower = prompt.lower()
    md_tokens = ["lammps", "neb", "ci-neb", "迁移能垒", "扩散能垒", "migration barrier", "barrier"]
    kmc_tokens = ["kmc", "misa-kmc", "spparks"]
    chain_tokens = [
        "传给",
        "传递",
        "继续",
        "继续模拟",
        "接着",
        "工作流",
        "workflow",
        "chain",
        "实现",
        "搭建",
        "接起来",
        "feed into",
        "continue",
    ]
    has_md = any(token in prompt or token in lower for token in md_tokens)
    has_kmc = any(token in prompt or token in lower for token in kmc_tokens)
    has_chain = any(token in prompt or token in lower for token in chain_tokens)
    return has_md and has_kmc and has_chain


def _should_launch_native_chain(prompt: str) -> bool:
    lower = prompt.lower()
    run_tokens = [
        "直接运行",
        "帮我运行",
        "现在运行",
        "现在就跑",
        "启动",
        "执行",
        "launch",
        "run it",
        "please run",
    ]
    return any(token in prompt or token in lower for token in run_tokens)


def _wants_summary(prompt: str) -> bool:
    lower = prompt.lower()
    tokens = [
        "总结",
        "概括",
        "分析",
        "解释",
        "说明一下",
        "说一下",
        "判断",
        "是否正常",
        "正常吗",
        "看起来正常",
        "summarize",
        "summary",
        "analyze",
        "explain",
        "is it normal",
    ]
    return any(token in prompt or token in lower for token in tokens)


def _looks_like_compare_runs_request(prompt: str) -> bool:
    lower = prompt.lower()
    compare_tokens = ["比较", "对比", "compare", "versus", "vs"]
    run_tokens = ["run", "runs", "任务"]
    scope_tokens = ["最近两个", "最近两次", "两个", "两次", "latest two", "last two", "recent two", "最新两个"]
    metric_tokens = ["barrier", "diffusion", "jump frequency", "温度", "扩散", "系数"]
    return (
        any(token in prompt or token in lower for token in compare_tokens)
        and any(token in prompt or token in lower for token in run_tokens)
        and (
            any(token in prompt or token in lower for token in scope_tokens)
            or any(token in prompt or token in lower for token in metric_tokens)
        )
    )


def infer_run_mode_hint(prompt: str) -> Optional[str]:
    lower = prompt.lower()
    if any(token in prompt or token in lower for token in ["kmc only", "kmc-only", "仅 kmc", "只有 kmc", "仅看 kmc"]):
        return "kmc_only"
    if any(token in prompt or token in lower for token in ["md only", "仅 md", "只有 md"]):
        return "md_only"
    if any(token in prompt or token in lower for token in ["md to kmc", "md→kmc", "md->kmc", "lammps", "迁移能垒"]) and any(
        token in prompt or token in lower for token in ["kmc", "继续模拟", "扩散", "diffusion", "链路", "chain", "workflow"]
    ):
        return "md_to_kmc_chain"
    return None


def _looks_like_run_diagnosis_request(prompt: str) -> bool:
    lower = prompt.lower()
    context_tokens = ["run", "runs", "任务", "日志", "log", "执行记录", "completed", "状态"]
    diagnostic_tokens = [
        "检查",
        "判断",
        "分析",
        "排查",
        "诊断",
        "正常结束",
        "真的正常",
        "是否正常",
        "completed",
        "异常",
        "异常痕迹",
        "异常退出",
        "报错",
        "失败",
        "why",
        "reason",
        "root cause",
        "diagnose",
        "normal",
        "abnormal",
        "completed status",
    ]
    return any(token in prompt or token in lower for token in context_tokens) and any(
        token in prompt or token in lower for token in diagnostic_tokens
    )


def _requested_log_targets(prompt: str) -> List[str]:
    lower = prompt.lower()
    wants_md = any(token in prompt or token in lower for token in ["lammps", "md", "md.run", "执行记录"])
    wants_kmc = any(token in prompt or token in lower for token in ["kmc", "kmc.run", "spparks", "执行记录"])
    if wants_md and wants_kmc:
        return ["md", "kmc"]
    return [_detect_log_target(prompt)]


def should_try_tool_plan(prompt: str) -> bool:
    lower = prompt.lower()
    toolish = [
        "run",
        "runs",
        "compare",
        "比较",
        "对比",
        "inspect",
        "artifact",
        "log",
        "日志",
        "输出文件",
        "起草",
        "草拟",
        "生成工作流",
        "启动",
        "执行",
        "bridge",
        "lookup",
        "event.json",
        "neb.txt",
        "检查",
        "看一下",
        "看下",
    ]
    multi = ["然后", "再", "并", "并且", "同时", "之后", "先", "latest", "最新", "最近"]
    return any(token in prompt or token in lower for token in toolish) and (
        any(token in prompt or token in lower for token in multi) or _wants_summary(prompt)
    )


def should_try_agent_loop(prompt: str) -> bool:
    lower = prompt.lower()
    diagnostic = [
        "如果有必要",
        "如果需要",
        "必要的话",
        "视情况",
        "判断",
        "判断一下",
        "为什么",
        "原因",
        "根因",
        "报错",
        "失败",
        "有没有异常",
        "是否正常",
        "异常退出",
        "跑崩",
        "跑挂",
        "排查",
        "调查",
        "why",
        "reason",
        "root cause",
        "what happened",
        "what failed",
        "error",
        "failed",
        "debug",
        "diagnose",
        "if needed",
        "if necessary",
        "whether",
        "is it normal",
        "is it healthy",
    ]
    context = [
        "run",
        "runs",
        "日志",
        "log",
        "artifact",
        "产物",
        "输出文件",
        "event.json",
        "neb.txt",
        "bridge",
    ]
    return any(token in prompt or token in lower for token in diagnostic) and any(
        token in prompt or token in lower for token in context
    )


def should_skip_tool_router(prompt: str) -> bool:
    text = prompt.strip()
    lower = text.lower()
    chat_markers = [
        "你好",
        "您好",
        "hello",
        "hi ",
        "hi，",
        "hi,",
        "你是谁",
        "介绍一下你自己",
        "简单介绍一下你自己",
        "what can you do",
        "who are you",
        "introduce yourself",
        "你能做什么",
    ]
    return any(marker in text or marker in lower for marker in chat_markers)


def _detect_bridge_intent(prompt: str, output_dir: Path) -> Optional[ToolIntent]:
    lower = prompt.lower()
    keywords = ["bridge", "lookup", "barriers.tsv", "neb.txt", "能垒表", "lookup 文件"]
    if not any(keyword in lower or keyword in prompt for keyword in keywords):
        return None

    abs_paths = _extract_abs_paths(prompt)
    event_json = None
    neb_txt = None
    workdir = None
    for item in abs_paths:
        if item.endswith(".json") and event_json is None:
            event_json = item
        elif item.endswith("neb.txt") and neb_txt is None:
            neb_txt = item
        elif workdir is None and not Path(item).suffix:
            workdir = item

    if not event_json or not neb_txt:
        return ToolIntent(
            action="chat",
            reply="要做 KMC bridge，我至少需要 event.json 和 neb.txt 的绝对路径。",
        )

    validate = not any(token in prompt for token in ["不要验证", "不验证", "no validate", "without validate"])
    return ToolIntent(
        action="bridge_kmc_lookup",
        params={
            "event_json": event_json,
            "neb_txt": neb_txt,
            "workdir": workdir or _default_bridge_workdir(output_dir),
            "validate": validate,
        },
    )


def _detect_moire_run_intent(prompt: str, output_dir: Path) -> Optional[ToolIntent]:
    lower = prompt.lower()
    if not any(token in prompt or token in lower for token in ["more", "mo re", "MoRe", "moire", "run lammps", "运行 lammps", "跑 lammps"]):
        return None
    if not any(token in prompt or token in lower for token in ["kmc", "继续模拟", "继续 kmc", "bridge", "接到 kmc", "传给 kmc"]):
        return None

    abs_paths = _extract_abs_paths(prompt)
    event_json = None
    case_dir = None
    workdir = None
    for item in abs_paths:
        if item.endswith(".json") and event_json is None:
            event_json = item
        elif case_dir is None and not Path(item).suffix:
            path_obj = Path(item)
            lowered_path = item.lower()
            if any(token in lowered_path for token in ["more", "neb_new_data", "model_"]) and path_obj.exists():
                case_dir = item
            elif workdir is None:
                workdir = item
        elif workdir is None and not Path(item).suffix:
            workdir = item

    if event_json is None:
        event_json = _default_moire_event_json(output_dir)
    if case_dir is None:
        case_dir = _default_moire_case_dir(output_dir)

    if not case_dir:
        return ToolIntent(
            action="chat",
            reply=(
                "我没在本机找到可用的默认 MoRe case 目录。"
                "如果你要我直接跑，请给我 MoRe case 目录的绝对路径。"
            ),
        )

    validate = not any(token in prompt for token in ["不要验证", "不验证", "no validate", "without validate"])
    params = {
        "case_dir": case_dir,
        "workdir": workdir or str((output_dir / f"moire_run_{time.strftime('%Y%m%d_%H%M%S')}").resolve()),
        "validate": validate,
    }
    params.update(_extract_moire_seed_params(prompt))
    if any(token in prompt or token in lower for token in ["ovito", "可视化", "snapshot", "渲染"]):
        params["ovito"] = True
    if event_json:
        params["event_json"] = event_json
    return ToolIntent(action="moire_run", params=params)


def _detect_moire_compare_intent(prompt: str, output_dir: Path) -> Optional[ToolIntent]:
    lower = prompt.lower()
    compare_tokens = ["比较", "对比", "compare", "versus", "vs"]
    moire_tokens = ["more", "mo re", "moire", "MoRe", "lammps", "迁移能垒", "barrier"]
    if not any(token in prompt or token in lower for token in compare_tokens):
        return None
    if not any(token in prompt or token in lower for token in moire_tokens):
        return None

    abs_paths = _extract_abs_paths(prompt)
    event_jsons: List[str] = []
    case_dir = None
    workdir = None
    for item in abs_paths:
        if item.endswith(".json"):
            event_jsons.append(item)
            continue
        if Path(item).suffix:
            continue
        path_obj = Path(item)
        lowered_path = item.lower()
        if case_dir is None and any(token in lowered_path for token in ["more", "neb_new_data", "model_"]) and path_obj.exists():
            case_dir = item
        elif workdir is None:
            workdir = item

    deduped_events: List[str] = []
    seen_events = set()
    for item in event_jsons:
        if item in seen_events:
            continue
        deduped_events.append(item)
        seen_events.add(item)

    if len(deduped_events) < 2:
        return None
    if case_dir is None:
        case_dir = _default_moire_case_dir(output_dir)
    if not case_dir:
        return ToolIntent(
            action="chat",
            reply=(
                "要比较多个 MoRe event，我至少需要一个可用的 MoRe case 目录绝对路径。"
            ),
        )

    validate = not any(token in prompt for token in ["不要验证", "不验证", "no validate", "without validate"])
    params: Dict[str, Any] = {
        "case_dir": case_dir,
        "event_jsons": deduped_events,
        "workdir": workdir or str((output_dir / f"moire_compare_{time.strftime('%Y%m%d_%H%M%S')}").resolve()),
        "validate": validate,
    }
    params.update(_extract_moire_seed_params(prompt))
    if any(token in prompt or token in lower for token in ["ovito", "可视化", "snapshot", "渲染"]):
        params["ovito"] = True
    if any(token in prompt or token in lower for token in ["只算 barrier", "只看 barrier", "只看能垒", "只跑 lammps", "lammps only", "skip kmc", "不跑 kmc"]):
        params["lammps_only"] = True
    return ToolIntent(action="moire_compare", params=params)


def _detect_moire_diffusion_sweep_intent(prompt: str, output_dir: Path) -> Optional[ToolIntent]:
    lower = prompt.lower()
    diffusion_tokens = [
        "扩散系数",
        "温度关系",
        "温度扫描",
        "扫温",
        "arrhenius",
        "diffusion coefficient",
        "diffusion vs temperature",
        "temperature sweep",
    ]
    moire_tokens = ["more", "mo re", "moire", "MoRe", "lammps", "kmc", "迁移能垒", "barrier"]
    if not any(token in prompt or token in lower for token in diffusion_tokens):
        return None
    if not any(token in prompt or token in lower for token in moire_tokens):
        return None

    abs_paths = _extract_abs_paths(prompt)
    event_json = None
    case_dir = None
    workdir = None
    for item in abs_paths:
        if item.endswith(".json") and event_json is None:
            event_json = item
            continue
        if Path(item).suffix:
            continue
        path_obj = Path(item)
        lowered_path = item.lower()
        if case_dir is None and any(token in lowered_path for token in ["more", "neb_new_data", "model_"]) and path_obj.exists():
            case_dir = item
        elif workdir is None:
            workdir = item

    if event_json is None:
        event_json = _default_moire_event_json(output_dir)
    if case_dir is None:
        case_dir = _default_moire_case_dir(output_dir)
    if not event_json or not case_dir:
        return None

    validate = not any(token in prompt for token in ["不要验证", "不验证", "no validate", "without validate"])
    params: Dict[str, Any] = {
        "event_json": event_json,
        "case_dir": case_dir,
        "workdir": workdir or str((output_dir / f"moire_diffusion_{time.strftime('%Y%m%d_%H%M%S')}").resolve()),
        "validate": validate,
    }
    temperatures = _extract_temperature_values(prompt)
    if temperatures:
        params["temperatures_k"] = temperatures
    params.update(_extract_moire_seed_params(prompt))
    if any(token in prompt or token in lower for token in ["ovito", "可视化", "snapshot", "渲染"]):
        params["ovito"] = True
    return ToolIntent(action="moire_diffusion_sweep", params=params)


def heuristic_tool_plan(prompt: str, output_dir: Path, current_run_dir: Optional[Path] = None) -> Optional[ToolPlan]:
    text = prompt.strip()
    lower = text.lower()
    run_hint = "current" if current_run_dir else "latest"

    if (
        _detect_bridge_intent(text, output_dir)
        or _detect_moire_diffusion_sweep_intent(text, output_dir)
        or _detect_moire_compare_intent(text, output_dir)
        or _detect_moire_run_intent(text, output_dir)
    ):
        return None

    if _looks_like_compare_runs_request(text):
        return ToolPlan(
            steps=[
                ToolIntent(
                    action="compare_runs",
                    params={
                        "run": "latest_two",
                        "mode": infer_run_mode_hint(text),
                    },
                )
            ],
            summarize=False,
        )

    if _looks_like_run_diagnosis_request(text):
        run_mode = infer_run_mode_hint(text)
        run_ref = "latest" if any(token in text or token in lower for token in ["最新", "latest", "最近"]) else ("current" if current_run_dir else "latest")
        inspect_params = {"run": run_ref}
        if run_mode:
            inspect_params["mode"] = run_mode
        steps = [ToolIntent(action="inspect", params=inspect_params)]
        for target in _requested_log_targets(text):
            steps.append(ToolIntent(action="logs", params={"run": "current", "target": target}))
        return ToolPlan(steps=steps[:4], summarize=True)

    if any(token in text or token in lower for token in ["日志", "log", "artifact", "产物", "输出文件", "inspect", "检查", "看一下", "看下"]) and _wants_summary(text):
        run_mode = infer_run_mode_hint(text)
        steps: List[ToolIntent] = []
        if any(token in text or token in lower for token in ["日志", "log", "artifact", "产物", "输出文件", "inspect", "检查", "看一下", "看下"]):
            inspect_params = {"run": run_hint}
            if run_mode:
                inspect_params["mode"] = run_mode
            steps.append(ToolIntent(action="inspect", params=inspect_params))
        if any(token in text or token in lower for token in ["日志", "log"]):
            steps.append(ToolIntent(action="logs", params={"run": run_hint, "target": _detect_log_target(text), "mode": run_mode}))
        elif any(token in text or token in lower for token in ["artifact", "产物", "输出文件"]):
            artifact_params = {"run": run_hint}
            if run_mode:
                artifact_params["mode"] = run_mode
            steps.append(ToolIntent(action="artifacts", params=artifact_params))
        if steps:
            return ToolPlan(steps=steps, summarize=True)

    if any(token in text or token in lower for token in ["列出", "最近的 run", "最近 runs", "最新 run", "最新那个 run"]) and any(
        token in text or token in lower for token in ["日志", "log", "inspect", "检查", "看一下", "看下"]
    ):
        steps = [ToolIntent(action="runs")]
        if any(token in text or token in lower for token in ["inspect", "检查", "看一下", "看下", "详情"]):
            steps.append(ToolIntent(action="inspect", params={"run": "latest"}))
        if any(token in text or token in lower for token in ["日志", "log"]):
            steps.append(ToolIntent(action="logs", params={"run": "latest", "target": _detect_log_target(text)}))
        return ToolPlan(steps=steps, summarize=_wants_summary(text) or True)

    return None


def heuristic_tool_intent(prompt: str, output_dir: Path, current_run_dir: Optional[Path] = None) -> Optional[ToolIntent]:
    text = prompt.strip()
    lower = text.lower()
    run_names = _recent_run_names(output_dir)
    current_run_name = current_run_dir.name if current_run_dir else None

    bridge_intent = _detect_bridge_intent(text, output_dir)
    if bridge_intent:
        return bridge_intent

    moire_diffusion_intent = _detect_moire_diffusion_sweep_intent(text, output_dir)
    if moire_diffusion_intent:
        return moire_diffusion_intent

    moire_compare_intent = _detect_moire_compare_intent(text, output_dir)
    if moire_compare_intent:
        return moire_compare_intent

    moire_intent = _detect_moire_run_intent(text, output_dir)
    if moire_intent:
        return moire_intent

    if _looks_like_compare_runs_request(text):
        return ToolIntent(
            action="compare_runs",
            params={
                "run": "latest_two",
                "mode": infer_run_mode_hint(text),
            },
        )

    if _looks_like_run_diagnosis_request(text):
        run_mode = infer_run_mode_hint(text)
        run_hint = "latest" if any(token in text or token in lower for token in ["最新", "latest", "最近"]) else (current_run_name if current_run_name else (run_names[0] if run_names else None))
        params = {"run": run_hint}
        if run_mode:
            params["mode"] = run_mode
        return ToolIntent(action="inspect", params=params)

    if _looks_like_lammps_barrier_to_kmc_request(text):
        action = "run" if _should_launch_native_chain(text) else "draft"
        return ToolIntent(action=action, params={"prompt": text})

    if any(token in lower for token in ["recent runs", "list runs", "列出 run", "最近的 run", "最近 runs", "有哪些 run"]):
        return ToolIntent(action="runs")

    if any(token in lower for token in ["打开 web", "open web", "打开控制台", "web 控制台"]):
        return ToolIntent(action="open_web", params={"port": 4174})

    if any(token in lower for token in ["日志", " logs", "log "]):
        target = "auto"
        if "kmc" in lower:
            target = "kmc"
        elif "md" in lower:
            target = "md"
        elif "summary" in lower or "总结" in text:
            target = "summary"
        run_hint = current_run_name if current_run_name else (run_names[0] if run_names else None)
        params = {"run": run_hint, "target": target}
        run_mode = infer_run_mode_hint(text)
        if run_mode:
            params["mode"] = run_mode
        return ToolIntent(action="logs", params=params)

    if any(token in lower for token in ["artifact", "产物", "生成了什么文件", "输出文件"]):
        run_hint = current_run_name if current_run_name else (run_names[0] if run_names else None)
        params = {"run": run_hint}
        run_mode = infer_run_mode_hint(text)
        if run_mode:
            params["mode"] = run_mode
        return ToolIntent(action="artifacts", params=params)

    if any(token in lower for token in ["inspect", "详情", "详细看", "查看 run", "看一下 run"]):
        run_hint = current_run_name if current_run_name else (run_names[0] if run_names else None)
        params = {"run": run_hint}
        run_mode = infer_run_mode_hint(text)
        if run_mode:
            params["mode"] = run_mode
        return ToolIntent(action="inspect", params=params)

    question_like = any(token in lower for token in ["怎么", "如何", "how to", "what is", "是什么"])
    if not question_like and any(token in lower for token in ["直接运行", "帮我运行", "启动", "执行", "launch", "run "]):
        return ToolIntent(action="run", params={"prompt": text})

    if not question_like and any(
        token in lower for token in ["起草", "草拟", "生成工作流", "draft", "create a", "create an", "帮我生成"]
    ):
        return ToolIntent(action="draft", params={"prompt": text})

    return None


def parse_tool_intent(content: str) -> Optional[ToolIntent]:
    text = content.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return _make_tool_intent(payload.get("action"), payload, payload.get("reply"))


def parse_tool_plan(content: str) -> Optional[ToolPlan]:
    text = content.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    raw_steps = payload.get("steps")
    if raw_steps is None:
        return None
    steps: List[ToolIntent] = []
    if isinstance(raw_steps, list):
        for item in raw_steps[:4]:
            if not isinstance(item, dict):
                continue
            intent = _make_tool_intent(item.get("action"), item.get("params"))
            if intent is not None:
                steps.append(intent)
    return ToolPlan(
        steps=steps,
        summarize=bool(payload.get("summarize", False)),
        reply=payload.get("reply"),
    )


def parse_agent_decision(content: str) -> Optional[AgentDecision]:
    text = content.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    if not isinstance(status, str):
        return None
    normalized_status = status.strip()
    if normalized_status not in {"continue", "finish"}:
        return None
    step = None
    raw_step = payload.get("step")
    if isinstance(raw_step, dict):
        step = _make_tool_intent(raw_step.get("action"), raw_step.get("params"))
    if normalized_status == "continue" and step is None:
        return None
    return AgentDecision(status=normalized_status, step=step, reply=payload.get("reply"))
