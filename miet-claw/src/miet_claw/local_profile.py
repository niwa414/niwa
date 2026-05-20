from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


DEFAULT_LOCAL_AGENT_PROFILE: Dict[str, Any] = {
    "agent_name": "mietclaw",
    "local_model": {
        "base_url": "http://127.0.0.1:8000",
        "api_key": "omlx-local",
        "preferred_model": "27b",
    },
    "runtime": {
        "conda_exec": "conda",
        "conda_env": "miet-stack",
        "mpi_procs": 5,
        "neb_input": "in.neb.mosia",
        "post_script": "neb_right.sh",
    },
    "moire": {
        "case_dir": "/path/to/MoRe/Re_0.07/model_4",
        "kmc_binary": "/path/to/misa-kmc",
        "eam_file": "/path/to/MoRe.eam.fs",
        "kmc_temperature": 1100.0,
        "kmc_stats_step": "1e-10",
        "kmc_run_time": "1e-10",
        "kmc_seed": 3401,
        "kmc_retry_attempts": 0,
        "diffusion_temperatures": [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0],
        "diffusion_stats_step": "1e-7",
        "diffusion_run_time": "1e-6",
        "diffusion_seeds": [3401, 3402, 3403],
    },
    "bridge": {
        "python": sys.executable,
        "script": "/path/to/run_lammps_kmc_bridge.py",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_agent_profile_path() -> Path:
    configured = os.environ.get("MIETCLAW_LOCAL_PROFILE_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root() / "config" / "local-agent.json").resolve()


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_profile_path(value: Any, *, root: Path) -> Path:
    raw = str(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _resolve_profile_command(value: Any, *, root: Path) -> str:
    raw = str(value)
    if "/" not in raw and not raw.startswith("."):
        return raw
    return str(_resolve_profile_path(raw, root=root))


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_list(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, list):
        return list(default)
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out or list(default)


def _int_list(value: Any, default: list[int]) -> list[int]:
    if not isinstance(value, list):
        return list(default)
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out or list(default)


def load_local_agent_profile() -> Dict[str, Any]:
    path = local_agent_profile_path()
    if not path.exists():
        return dict(DEFAULT_LOCAL_AGENT_PROFILE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_LOCAL_AGENT_PROFILE)
    if not isinstance(payload, dict):
        return dict(DEFAULT_LOCAL_AGENT_PROFILE)
    return _deep_merge(DEFAULT_LOCAL_AGENT_PROFILE, payload)


def get_local_model_settings() -> Dict[str, Any]:
    profile = load_local_agent_profile()
    local_model = profile.get("local_model") if isinstance(profile.get("local_model"), dict) else {}
    base_url = os.environ.get("MIETCLAW_LOCAL_MODEL_BASE_URL") or local_model.get("base_url") or DEFAULT_LOCAL_AGENT_PROFILE["local_model"]["base_url"]
    api_key = os.environ.get("MIETCLAW_LOCAL_MODEL_API_KEY") or local_model.get("api_key") or DEFAULT_LOCAL_AGENT_PROFILE["local_model"]["api_key"]
    preferred_model = os.environ.get("MIETCLAW_LOCAL_MODEL") or local_model.get("preferred_model") or DEFAULT_LOCAL_AGENT_PROFILE["local_model"]["preferred_model"]
    return {
        "agent_name": profile.get("agent_name") or DEFAULT_LOCAL_AGENT_PROFILE["agent_name"],
        "profile_path": str(local_agent_profile_path()),
        "base_url": str(base_url).rstrip("/"),
        "api_key": str(api_key),
        "preferred_model": str(preferred_model),
    }


def get_runtime_settings(project_root_path: Path | None = None) -> Dict[str, Any]:
    root = Path(project_root_path).resolve() if project_root_path is not None else project_root()
    profile = load_local_agent_profile()
    runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
    moire = profile.get("moire") if isinstance(profile.get("moire"), dict) else {}
    bridge = profile.get("bridge") if isinstance(profile.get("bridge"), dict) else {}
    defaults = DEFAULT_LOCAL_AGENT_PROFILE
    default_runtime = defaults["runtime"]
    default_moire = defaults["moire"]
    default_bridge = defaults["bridge"]

    conda_exec = os.environ.get("MIETCLAW_CONDA_EXEC") or runtime.get("conda_exec") or default_runtime["conda_exec"]
    conda_env = os.environ.get("MIETCLAW_CONDA_ENV") or runtime.get("conda_env") or default_runtime["conda_env"]
    mpi_procs = _int_value(os.environ.get("MIETCLAW_MPI_PROCS") or runtime.get("mpi_procs"), int(default_runtime["mpi_procs"]))
    neb_input = os.environ.get("MIETCLAW_NEB_INPUT") or runtime.get("neb_input") or default_runtime["neb_input"]
    post_script = os.environ.get("MIETCLAW_POST_SCRIPT") or runtime.get("post_script") or default_runtime["post_script"]

    kmc_binary = os.environ.get("MIETCLAW_MOIRE_KMC_BINARY") or moire.get("kmc_binary") or default_moire["kmc_binary"]
    eam_file = os.environ.get("MIETCLAW_MOIRE_EAM_FILE") or moire.get("eam_file") or default_moire["eam_file"]
    moire_case_dir = os.environ.get("MIETCLAW_MOIRE_CASE_DIR") or moire.get("case_dir") or default_moire["case_dir"]
    kmc_retry_attempts = _int_value(
        os.environ.get("MIETCLAW_KMC_RETRY_ATTEMPTS") or moire.get("kmc_retry_attempts"),
        int(default_moire["kmc_retry_attempts"]),
    )

    return {
        "profile_path": str(local_agent_profile_path()),
        "project_root": str(root),
        "conda_exec": _resolve_profile_command(conda_exec, root=root),
        "conda_env": str(conda_env),
        "mpi_procs": mpi_procs,
        "neb_input": str(neb_input),
        "post_script": str(post_script),
        "moire_case_dir": str(_resolve_profile_path(moire_case_dir, root=root)),
        "kmc_binary": str(_resolve_profile_path(kmc_binary, root=root)),
        "eam_file": str(_resolve_profile_path(eam_file, root=root)),
        "kmc_temperature": _float_value(moire.get("kmc_temperature"), float(default_moire["kmc_temperature"])),
        "kmc_stats_step": str(moire.get("kmc_stats_step") or default_moire["kmc_stats_step"]),
        "kmc_run_time": str(moire.get("kmc_run_time") or default_moire["kmc_run_time"]),
        "kmc_seed": _int_value(moire.get("kmc_seed"), int(default_moire["kmc_seed"])),
        "kmc_retry_attempts": max(0, kmc_retry_attempts),
        "diffusion_temperatures": _float_list(
            moire.get("diffusion_temperatures"),
            [float(item) for item in default_moire["diffusion_temperatures"]],
        ),
        "diffusion_stats_step": str(moire.get("diffusion_stats_step") or default_moire["diffusion_stats_step"]),
        "diffusion_run_time": str(moire.get("diffusion_run_time") or default_moire["diffusion_run_time"]),
        "diffusion_seeds": _int_list(
            moire.get("diffusion_seeds"),
            [int(item) for item in default_moire["diffusion_seeds"]],
        ),
        "bridge_python": _resolve_profile_command(os.environ.get("MIETCLAW_KMC_BRIDGE_PYTHON") or bridge.get("python") or default_bridge["python"], root=root),
        "bridge_script": str(_resolve_profile_path(os.environ.get("MIETCLAW_KMC_BRIDGE_SCRIPT") or bridge.get("script") or default_bridge["script"], root=root)),
    }
