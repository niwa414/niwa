#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Any

import operator
import re


OP_MAP = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

DEFAULT_ARTIFACT_POLICY = {
    "keep_on_fail_outputs": 20,
    "keep_on_fail_outputs_head": 4,
    "keep_on_fail_outputs_uniform": 20,
    "keep_on_fail_athena_steps": 20,
    "keep_on_fail_athena_steps_head": 4,
    "archive_fail": True,
    "archive_pass": False,
    "prune_raw_on_pass": True,
    "prune_raw_on_fail": True,
    "archive_format": "gz",
    "raw_size_warn_gb": 25.0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_case(case_ref: str, root: Path) -> tuple[dict[str, Any], Path, Path]:
    path = Path(case_ref)
    if path.is_file():
        case_path = path
        case_dir = case_path.parent
    else:
        case_dir = root / "cases" / case_ref
        case_path = case_dir / "case.json"
    if not case_path.exists():
        raise SystemExit(f"Case file not found: {case_path}")
    with case_path.open("r", encoding="utf-8") as handle:
        case = json.load(handle)
    return case, case_dir, case_path


def expand_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**mapping)
    if isinstance(value, list):
        return [expand_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: expand_value(val, mapping) for key, val in value.items()}
    return value


def safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def tail_lines(path: Path, max_lines: int = 20) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = deque(handle, maxlen=max_lines)
        if not lines:
            return None
        return "".join(lines).rstrip("\n")
    except Exception:
        return None


def signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    sig_num = -returncode
    try:
        return signal.Signals(sig_num).name
    except Exception:
        return f"SIG{sig_num}"


def run_steps(
    steps: list[dict[str, Any]],
    stage: str,
    mapping: dict[str, str],
    env: dict[str, str],
    root: Path,
    log_dir: Path,
    dry_run: bool,
    default_timeout_s: float | None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    logs = []
    timeouts = []
    results = []
    for idx, step in enumerate(steps, start=1):
        name = step.get("name") or f"{stage}-{idx}"
        cmd = expand_value(step["cmd"], mapping)
        cmd = [str(part) for part in cmd]
        log_path = log_dir / f"{stage}_{safe_name(name)}.log"
        logs.append(str(log_path))
        print(f"[{stage}:{name}] {' '.join(cmd)}")
        if dry_run:
            results.append(
                {
                    "stage": stage,
                    "name": name,
                    "cmd": cmd,
                    "log": str(log_path),
                    "exit_code": None,
                    "timeout_hit": False,
                    "signal": None,
                    "stderr_tail": None,
                    "wall_time_s": None,
                }
            )
            continue
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_s = step.get("timeout_s", default_timeout_s)
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                start_time = time.monotonic()
                completed = subprocess.run(
                    cmd,
                    check=False,
                    cwd=root,
                    env=env,
                    stdout=log_handle,
                    stderr=log_handle,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start_time
                timeouts.append(
                    {
                        "stage": stage,
                        "name": name,
                        "log": str(log_path),
                        "timeout_s": timeout_s,
                    }
                )
                results.append(
                    {
                        "stage": stage,
                        "name": name,
                        "cmd": cmd,
                        "log": str(log_path),
                        "exit_code": None,
                        "timeout_hit": True,
                        "signal": None,
                        "stderr_tail": tail_lines(log_path),
                        "wall_time_s": elapsed,
                    }
                )
                break
        elapsed = time.monotonic() - start_time
        exit_code = completed.returncode
        results.append(
            {
                "stage": stage,
                "name": name,
                "cmd": cmd,
                "log": str(log_path),
                "exit_code": exit_code,
                "timeout_hit": False,
                "signal": signal_name(exit_code),
                "stderr_tail": tail_lines(log_path) if exit_code != 0 else None,
                "wall_time_s": elapsed,
            }
        )
        if exit_code != 0:
            break
    return logs, timeouts, results


def summarize_stage(results: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    stage_results = [res for res in results if res.get("stage") == stage]
    if not stage_results:
        return {"exit_code": None, "timeout_hit": False, "signal": None, "stderr_tail": None}
    for res in stage_results:
        exit_code = res.get("exit_code")
        if res.get("timeout_hit") or (exit_code not in (None, 0)):
            return {
                "exit_code": exit_code,
                "timeout_hit": bool(res.get("timeout_hit")),
                "signal": res.get("signal"),
                "stderr_tail": res.get("stderr_tail"),
            }
    return {"exit_code": 0, "timeout_hit": False, "signal": None, "stderr_tail": None}


def summarize_substage(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    token = f"-{label}".lower()
    alt_token = f"_{label}".lower()
    for res in results:
        name = str(res.get("name", "")).lower()
        if name.endswith(token) or name.endswith(alt_token):
            return {
                "exit_code": res.get("exit_code"),
                "timeout_hit": bool(res.get("timeout_hit")),
                "signal": res.get("signal"),
                "stderr_tail": res.get("stderr_tail"),
            }
    return {"exit_code": None, "timeout_hit": False, "signal": None, "stderr_tail": None}


def select_run_wall_time(results: list[dict[str, Any]]) -> float | None:
    for res in reversed(results):
        name = str(res.get("name", "")).lower()
        if res.get("stage") == "run" and name.startswith("run-"):
            wall_time = res.get("wall_time_s")
            if wall_time is not None:
                return float(wall_time)
    return None


def update_metrics_resources(
    metrics_path: Path | None,
    wall_time_s: float | None,
    archive_paths: list[Path] | None,
) -> None:
    if metrics_path is None or not metrics_path.exists():
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return
    updated = False
    if wall_time_s is not None:
        metrics["wall_time_s"] = float(wall_time_s)
        updated = True
    archive_size_gb = None
    if archive_paths:
        total_bytes = 0
        for path in archive_paths:
            if not path.exists():
                continue
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
        if total_bytes > 0:
            archive_size_gb = total_bytes / (1024 ** 3)
    if archive_size_gb is not None:
        metrics["archive_size_gb"] = float(archive_size_gb)
        updated = True
    if updated:
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def hash_inputs(case_dir: Path, case: dict[str, Any], case_path: Path) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    missing = []

    def add_file(path: Path, rel: str) -> None:
        if not path.exists():
            missing.append(rel)
            return
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())

    add_file(case_path, "case.json")
    for rel in sorted(case.get("inputs", [])):
        add_file(case_dir / rel, rel)
    return digest.hexdigest(), missing


def git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": "unknown", "dirty": "unknown"}
    if not (root / ".git").exists():
        return info
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["commit"] = commit or "unknown"
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


def evaluate_thresholds(
    metrics: dict[str, Any], thresholds: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    results = []
    all_pass = True
    for entry in thresholds:
        metric = entry.get("metric")
        op = entry.get("op")
        target = entry.get("value")
        actual = metrics.get(metric) if metrics else None
        passed = False
        reason = None
        if metric is None or op is None:
            reason = "invalid_threshold"
        elif actual is None:
            reason = "missing_metric"
        else:
            func = OP_MAP.get(op)
            if func is None:
                reason = "unsupported_op"
            else:
                try:
                    passed = bool(func(actual, target))
                except Exception:
                    reason = "compare_failed"
        if not passed:
            all_pass = False
        results.append(
            {
                "metric": metric,
                "op": op,
                "value": target,
                "actual": actual,
                "pass": passed,
                "reason": reason,
            }
        )
    return results, all_pass


def copy_inputs(case_dir: Path, case: dict[str, Any], dest_root: Path, case_path: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case_path, dest_root / "case.json")
    for rel in case.get("inputs", []):
        src = case_dir / rel
        if not src.exists():
            continue
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def parse_athena_tlim(case_dir: Path, inputs: list[str]) -> float | None:
    pattern = re.compile(r"^\s*tlim\s*=\s*([0-9eE+.\-]+)")
    for rel in inputs:
        path = case_dir / rel
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = pattern.match(line)
                if match:
                    return float(match.group(1))
        except Exception:
            continue
    return None


def parse_athena_progress(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"cycle=(\d+)\s+time=([0-9eE+.\-]+)")
    cycle = None
    time_val = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                cycle = int(match.group(1))
                time_val = float(match.group(2))
    except Exception:
        return None
    if cycle is None or time_val is None:
        return None
    return {"cycle_reached": cycle, "time_reached": time_val}


def to_relative(path: str | Path, root: Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(root).as_posix()
    except ValueError:
        return path_obj.as_posix()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def extract_flag_values(cmd: list[str], flag: str) -> list[str]:
    values = []
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            values.append(cmd[idx + 1])
        elif token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
    return values


def preflight_missing_opmd_inputs(
    steps: list[dict[str, Any]],
    mapping: dict[str, str],
    root: Path,
    output_root: Path,
) -> list[str]:
    missing = []
    for step in steps:
        cmd = expand_value(step.get("cmd", []), mapping)
        cmd = [str(part) for part in cmd]
        if not any("warpx_driver" in part and part.endswith(".py") for part in cmd):
            continue
        for flag in ("--opmd-fluid", "--opmd-b"):
            for value in extract_flag_values(cmd, flag):
                path = resolve_path(value, root)
                if path.exists():
                    continue
                if output_root in path.parents or path == output_root:
                    continue
                if not path.exists():
                    missing.append(str(path))
    return sorted(set(missing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def resolve_path(path: str | Path, root: Path) -> Path:
    obj = Path(path)
    if not obj.is_absolute():
        obj = root / obj
    return obj.resolve()


def path_related(path: Path, keep_paths: set[Path]) -> bool:
    for keep in keep_paths:
        if path == keep:
            return True
        if keep in path.parents:
            return True
        if path in keep.parents:
            return True
    return False


def select_tail_warpx_diags(diag_dir: Path, keep_count: int) -> set[Path]:
    if keep_count <= 0 or not diag_dir.exists():
        return set()
    entries = []
    for item in diag_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith("diag"):
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        entries.append((int(suffix), item.resolve()))
    if not entries:
        return set()
    entries.sort(key=lambda pair: pair[0])
    return {path for _, path in entries[-keep_count:]}


def select_head_warpx_diags(diag_dir: Path, keep_count: int) -> set[Path]:
    if keep_count <= 0 or not diag_dir.exists():
        return set()
    entries = []
    for item in diag_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith("diag"):
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        entries.append((int(suffix), item.resolve()))
    if not entries:
        return set()
    entries.sort(key=lambda pair: pair[0])
    return {path for _, path in entries[:keep_count]}


def select_uniform_warpx_diags(diag_dir: Path, keep_count: int) -> set[Path]:
    if keep_count <= 0 or not diag_dir.exists():
        return set()
    entries = []
    for item in diag_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith("diag"):
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        entries.append((int(suffix), item.resolve()))
    if not entries:
        return set()
    entries.sort(key=lambda pair: pair[0])
    if keep_count >= len(entries):
        return {path for _, path in entries}
    step = (len(entries) - 1) / float(keep_count + 1)
    keep = set()
    for idx in range(1, keep_count + 1):
        pick = int(round(idx * step))
        pick = min(max(pick, 0), len(entries) - 1)
        keep.add(entries[pick][1])
    return keep


def select_tail_athena_vtk(run_dir: Path, keep_steps: int) -> set[Path]:
    if keep_steps <= 0 or not run_dir.exists():
        return set()
    step_re = re.compile(r"\.(\d+)\.vtk$")
    steps: dict[int, list[Path]] = {}
    for path in run_dir.glob("*.vtk"):
        match = step_re.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        steps.setdefault(step, []).append(path.resolve())
    if not steps:
        return set()
    keep = set()
    for step in sorted(steps.keys())[-keep_steps:]:
        keep.update(steps[step])
    return keep


def select_head_athena_vtk(run_dir: Path, keep_steps: int) -> set[Path]:
    if keep_steps <= 0 or not run_dir.exists():
        return set()
    step_re = re.compile(r"\.(\d+)\.vtk$")
    steps: dict[int, list[Path]] = {}
    for path in run_dir.glob("*.vtk"):
        match = step_re.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        steps.setdefault(step, []).append(path.resolve())
    if not steps:
        return set()
    keep = set()
    for step in sorted(steps.keys())[:keep_steps]:
        keep.update(steps[step])
    return keep


def select_warpx_meta_files(run_dir: Path) -> set[Path]:
    if not run_dir.exists():
        return set()
    keep = set()
    keep.update(path.resolve() for path in run_dir.glob("warpx_run_*.json"))
    keep.update(path.resolve() for path in run_dir.glob("warpx_heartbeat*.json"))
    drift_meta = run_dir / "drift_meta.json"
    if drift_meta.exists():
        keep.add(drift_meta.resolve())
    return keep


def extract_resistivity_meta(output_raw: Path) -> dict[str, Any] | None:
    run_dir = output_raw / "run"
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.glob("warpx_run_*.json"))
    if not candidates:
        return None
    for path in reversed(candidates):
        try:
            with path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        resistivity = meta.get("resistivity")
        if isinstance(resistivity, dict):
            return resistivity
        hybrid_cfg = (meta.get("args") or {}).get("hybrid")
        if isinstance(hybrid_cfg, dict):
            return {
                "plasma_resistivity_expr": hybrid_cfg.get("eta"),
                "plasma_resistivity_scale": hybrid_cfg.get("eta_scale", 1.0),
                "plasma_hyper_resistivity_expr": hybrid_cfg.get("eta_h"),
                "plasma_hyper_resistivity_scale": hybrid_cfg.get("eta_h_scale", 1.0),
                "eta_source": "input_expr_scale",
            }
    return None


def create_archive(raw_dir: Path, archive_path: Path) -> tuple[bool, str | None]:
    if not raw_dir.exists():
        return False, "raw_dir_missing"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    raw_parent = raw_dir.parent
    raw_name = raw_dir.name
    if archive_path.suffixes[-2:] == [".tar", ".zst"] and shutil.which("zstd"):
        cmd = ["tar", "-I", "zstd", "-cf", str(archive_path), "-C", str(raw_parent), raw_name]
        try:
            subprocess.run(cmd, check=True)
            return True, None
        except Exception as exc:
            return False, f"archive_failed_zstd: {exc}"
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(raw_dir, arcname=raw_name)
        return True, None
    except Exception as exc:
        return False, f"archive_failed_gzip: {exc}"


def prune_raw_outputs(raw_dir: Path, keep_paths: set[Path]) -> tuple[int, int]:
    removed_files = 0
    removed_bytes = 0
    if not raw_dir.exists():
        return removed_files, removed_bytes
    for path in sorted(raw_dir.rglob("*"), reverse=True):
        if path_related(path, keep_paths):
            continue
        if path.is_file() or path.is_symlink():
            try:
                removed_bytes += path.stat().st_size
            except OSError:
                pass
            try:
                path.unlink()
                removed_files += 1
            except OSError:
                continue
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                continue
    return removed_files, removed_bytes


def categorize_path(
    path: Path,
    output_root: Path,
    output_analysis: Path,
    output_plots: Path,
    output_logs: Path,
    output_raw: Path,
) -> str:
    for label, base in (
        ("analysis", output_analysis),
        ("plots", output_plots),
        ("logs", output_logs),
        ("raw", output_raw),
    ):
        try:
            path.relative_to(base)
            return label
        except ValueError:
            continue
    try:
        path.relative_to(output_root)
        return "output"
    except ValueError:
        return "external"


def write_manifest(
    manifest_path: Path,
    output_root: Path,
    root: Path,
    case_id: str,
    status: str,
    policy: dict[str, Any],
    git: dict[str, Any],
    input_hash: str,
    raw_size_before: int,
    raw_size_after: int,
    cleanup_summary: dict[str, Any],
    resistivity_meta: dict[str, Any] | None = None,
) -> None:
    files = []
    output_analysis = output_root / "analysis"
    output_plots = output_root / "plots"
    output_logs = output_root / "logs"
    output_raw = output_root / "raw"
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        if path == manifest_path:
            continue
        rel = to_relative(path, root)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        entry = {
            "path": rel,
            "size_bytes": size,
            "sha256": sha256_file(path),
            "category": categorize_path(
                path, output_root, output_analysis, output_plots, output_logs, output_raw
            ),
        }
        files.append(entry)

    manifest = {
        "case_id": case_id,
        "status": status,
        "generated": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "input_hash": input_hash,
        "policy": policy,
        "raw_size_bytes_before": raw_size_before,
        "raw_size_bytes_after": raw_size_after,
        "cleanup_summary": cleanup_summary,
        "files": files,
    }
    if resistivity_meta is not None:
        manifest["resistivity"] = resistivity_meta
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def manage_artifacts(
    case: dict[str, Any],
    root: Path,
    output_root: Path,
    output_raw: Path,
    output_analysis: Path,
    output_plots: Path,
    output_logs: Path,
    status: str,
    artifacts: list[str],
) -> dict[str, Any]:
    policy = dict(DEFAULT_ARTIFACT_POLICY)
    policy.update(case.get("artifact_policy", {}))
    keep_paths: set[Path] = set()
    for base in (output_analysis, output_plots, output_logs, output_raw / "inputs"):
        keep_paths.add(base.resolve())
    for item in artifacts:
        keep_paths.add(resolve_path(item, root))
    keep_paths.update(select_warpx_meta_files(output_raw / "run"))

    if status == "FAIL":
        keep_paths.update(
            select_tail_warpx_diags(output_raw / "run" / "diag", policy["keep_on_fail_outputs"])
        )
        keep_paths.update(
            select_head_warpx_diags(
                output_raw / "run" / "diag", policy["keep_on_fail_outputs_head"]
            )
        )
        keep_paths.update(
            select_uniform_warpx_diags(
                output_raw / "run" / "diag", policy.get("keep_on_fail_outputs_uniform", 0)
            )
        )
        keep_paths.update(
            select_tail_athena_vtk(output_raw / "run", policy["keep_on_fail_athena_steps"])
        )
        keep_paths.update(
            select_head_athena_vtk(
                output_raw / "run", policy["keep_on_fail_athena_steps_head"]
            )
        )

    raw_size_before = dir_size(output_raw)
    cleanup_warnings = []
    warn_gb = policy.get("raw_size_warn_gb")
    if isinstance(warn_gb, (int, float)) and raw_size_before > 0:
        raw_size_gb = raw_size_before / (1024**3)
        if raw_size_gb >= float(warn_gb):
            cleanup_warnings.append(f"raw_size_exceeded_gb:{raw_size_gb:.2f}")
    archive_paths: list[Path] = []
    cleanup_errors: list[str] = []
    archive_ok = False
    archive_error = None
    archive_attempted = False
    use_zstd = policy.get("archive_format") == "zst" and shutil.which("zstd")
    archive_format_used = "zst" if use_zstd else "gz"
    suffix = ".tar.zst" if use_zstd else ".tar.gz"
    if status == "FAIL" and policy.get("archive_fail"):
        archive_attempted = True
        archive_path = output_root / f"raw_fail{suffix}"
        archive_ok, archive_error = create_archive(output_raw, archive_path)
        if archive_ok:
            try:
                size = archive_path.stat().st_size
            except OSError:
                size = 0
            if size <= 0:
                archive_ok = False
                archive_error = "archive_zero_bytes"
                try:
                    archive_path.unlink()
                except OSError:
                    pass
            else:
                archive_paths.append(archive_path)
        if not archive_ok:
            if archive_path.exists():
                try:
                    archive_path.unlink()
                except OSError:
                    pass
            if archive_error:
                cleanup_errors.append(archive_error)
    elif status == "PASS" and policy.get("archive_pass"):
        archive_attempted = True
        archive_path = output_root / f"raw_pass{suffix}"
        archive_ok, archive_error = create_archive(output_raw, archive_path)
        if archive_ok:
            try:
                size = archive_path.stat().st_size
            except OSError:
                size = 0
            if size <= 0:
                archive_ok = False
                archive_error = "archive_zero_bytes"
                try:
                    archive_path.unlink()
                except OSError:
                    pass
            else:
                archive_paths.append(archive_path)
        if not archive_ok:
            if archive_path.exists():
                try:
                    archive_path.unlink()
                except OSError:
                    pass
            if archive_error:
                cleanup_errors.append(archive_error)

    removed_files = 0
    removed_bytes = 0
    prune_executed = False
    prune_skipped_reason = None
    if status == "PASS" and policy.get("prune_raw_on_pass"):
        prune_executed = True
        removed_files, removed_bytes = prune_raw_outputs(output_raw, keep_paths)
    elif status == "FAIL" and policy.get("prune_raw_on_fail"):
        if policy.get("archive_fail") and not archive_ok:
            prune_skipped_reason = "archive_failed"
            cleanup_errors.append("prune_skipped_archive_failed")
        else:
            prune_executed = True
            removed_files, removed_bytes = prune_raw_outputs(output_raw, keep_paths)
    else:
        prune_skipped_reason = "policy_disabled"

    raw_size_after = dir_size(output_raw)
    cleanup_summary = {
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "archive_paths": [to_relative(path, root) for path in archive_paths],
        "archive_format": archive_format_used,
        "archive_attempted": archive_attempted,
        "archive_ok": archive_ok if archive_attempted else None,
        "archive_error": archive_error if archive_attempted and not archive_ok else None,
        "prune_executed": prune_executed,
        "prune_skipped_reason": prune_skipped_reason,
        "warnings": cleanup_warnings,
        "errors": cleanup_errors,
    }
    return {
        "policy": policy,
        "archive_paths": archive_paths,
        "cleanup_summary": cleanup_summary,
        "raw_size_before": raw_size_before,
        "raw_size_after": raw_size_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a case with standardized outputs.")
    parser.add_argument("--case", required=True, help="Case id or path to case.json")
    parser.add_argument(
        "--stage",
        choices=["run", "analyze", "all"],
        default="all",
        help="Stages to execute.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--update-evidence",
        action="store_true",
        help="Update evidence/index.md after run.",
    )
    args = parser.parse_args()

    root = repo_root()
    case, case_dir, case_path = load_case(args.case, root)
    case_id = case.get("id") or case_dir.name

    output_root = root / "outputs" / case_id
    output_raw = output_root / "raw"
    output_analysis = output_root / "analysis"
    output_plots = output_root / "plots"
    output_logs = output_root / "logs"
    for path in (output_raw, output_analysis, output_plots, output_logs):
        path.mkdir(parents=True, exist_ok=True)

    mapping = {
        "python": sys.executable,
        "repo_root": str(root),
        "case_dir": str(case_dir),
        "output_root": str(output_root),
        "output_raw": str(output_raw),
        "output_analysis": str(output_analysis),
        "output_plots": str(output_plots),
        "output_logs": str(output_logs),
    }

    env = os.environ.copy()
    env.update(
        {
            "FUSION_CASE_ID": case_id,
            "FUSION_CASE_DIR": str(case_dir),
            "FUSION_OUTPUT_ROOT": str(output_root),
            "FUSION_OUTPUT_RAW": str(output_raw),
            "FUSION_OUTPUT_ANALYSIS": str(output_analysis),
            "FUSION_OUTPUT_PLOTS": str(output_plots),
            "FUSION_OUTPUT_LOGS": str(output_logs),
        }
    )

    error_messages = []
    preflight_failed = False
    preflight_missing = []
    logs = []
    timeouts = []
    step_results = []

    try:
        copy_inputs(case_dir, case, output_raw / "inputs", case_path)
    except Exception as exc:
        error_messages.append(f"copy_inputs_failed: {exc}")

    if args.stage in ("run", "all"):
        preflight_missing = preflight_missing_opmd_inputs(
            case.get("run", []), mapping, root, output_root
        )
        if preflight_missing:
            preflight_failed = True
            error_messages.append(f"prereq_missing: {', '.join(preflight_missing)}")
            print(f"[preflight] missing prerequisites: {', '.join(preflight_missing)}")

    if args.stage in ("run", "all") and not preflight_failed:
        try:
            run_logs, run_timeouts, run_results = run_steps(
                case.get("run", []),
                "run",
                mapping,
                env,
                root,
                output_logs,
                args.dry_run,
                case.get("timeout_s"),
            )
            logs.extend(run_logs)
            timeouts.extend(run_timeouts)
            step_results.extend(run_results)
            if run_timeouts:
                error_messages.append(
                    f"timeout: {run_timeouts[0]['stage']}/{run_timeouts[0]['name']}"
                )
            run_failed = next(
                (
                    res
                    for res in run_results
                    if res.get("exit_code") not in (None, 0) and not res.get("timeout_hit")
                ),
                None,
            )
            if run_failed:
                exc = subprocess.CalledProcessError(run_failed["exit_code"], run_failed["cmd"])
                error_messages.append(f"run_failed: {exc}")
        except Exception as exc:
            error_messages.append(f"run_failed: {exc}")

    if args.stage in ("analyze", "all") and not preflight_failed:
        try:
            analyze_logs, analyze_timeouts, analyze_results = run_steps(
                case.get("analyze", []),
                "analyze",
                mapping,
                env,
                root,
                output_logs,
                args.dry_run,
                case.get("timeout_s"),
            )
            logs.extend(analyze_logs)
            timeouts.extend(analyze_timeouts)
            step_results.extend(analyze_results)
            if analyze_timeouts:
                error_messages.append(
                    f"timeout: {analyze_timeouts[0]['stage']}/{analyze_timeouts[0]['name']}"
                )
            analyze_failed = next(
                (
                    res
                    for res in analyze_results
                    if res.get("exit_code") not in (None, 0) and not res.get("timeout_hit")
                ),
                None,
            )
            if analyze_failed:
                exc = subprocess.CalledProcessError(
                    analyze_failed["exit_code"], analyze_failed["cmd"]
                )
                error_messages.append(f"analyze_failed: {exc}")
        except Exception as exc:
            error_messages.append(f"analyze_failed: {exc}")

    metrics = {}
    metrics_file = None
    metrics_path = expand_value(case.get("metrics_file"), mapping)
    if metrics_path:
        metrics_file = Path(metrics_path)
        if metrics_file.exists():
            try:
                with metrics_file.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
            except Exception as exc:
                error_messages.append(f"metrics_read_failed: {exc}")
        else:
            error_messages.append(f"metrics_missing: {metrics_file}")

    thresholds = case.get("thresholds", [])
    threshold_results, thresholds_pass = evaluate_thresholds(metrics, thresholds)
    gate_layers = []
    layer_status = {}
    if case.get("thresholds_layers"):
        for idx, entry in enumerate(case.get("thresholds_layers", []), start=1):
            layer_thresholds = entry.get("thresholds", [])
            name = entry.get("name") or f"layer{idx}"
            layer_results, layer_pass = evaluate_thresholds(metrics, layer_thresholds)
            gate_layers.append(
                {
                    "name": name,
                    "thresholds": layer_thresholds,
                    "threshold_results": layer_results,
                    "pass": layer_pass,
                }
            )
            layer_status[name] = layer_pass
    elif case.get("thresholds_layer1"):
        layer1 = case.get("thresholds_layer1", [])
        layer1_results, layer1_pass = evaluate_thresholds(metrics, layer1)
        gate_layers.append(
            {
                "name": "layer1",
                "thresholds": layer1,
                "threshold_results": layer1_results,
                "pass": layer1_pass,
            }
        )
        layer_status["layer1"] = layer1_pass
        layer2 = case.get("thresholds_layer2") or thresholds
        if layer2:
            layer2_results, layer2_pass = evaluate_thresholds(metrics, layer2)
            gate_layers.append(
                {
                    "name": "layer2",
                    "thresholds": layer2,
                    "threshold_results": layer2_results,
                    "pass": layer2_pass,
                }
            )
            layer_status["layer2"] = layer2_pass

    input_hash, missing_inputs = hash_inputs(case_dir, case, case_path)
    if missing_inputs:
        error_messages.append(f"missing_inputs: {', '.join(missing_inputs)}")

    status = "PASS" if (not error_messages and thresholds_pass) else "FAIL"
    git = git_info(root)
    metrics_file_rel = to_relative(metrics_file, root) if metrics_file else None
    artifacts = expand_value(case.get("artifacts", []), mapping) + logs
    artifacts_rel = []
    seen = set()
    for item in artifacts:
        path = resolve_path(item, root)
        if not path.exists():
            continue
        rel = to_relative(path, root)
        if rel in seen:
            continue
        seen.add(rel)
        artifacts_rel.append(rel)
    failure_reason = None
    progress = None
    if timeouts:
        failure_reason = "TIMEOUT"
        timeout_log = Path(timeouts[0]["log"])
        progress = parse_athena_progress(timeout_log)
        tlim = parse_athena_tlim(case_dir, case.get("inputs", []))
        if progress is None:
            progress = {}
        if tlim is not None:
            progress["tlim"] = tlim

    run_summary = summarize_stage(step_results, "run")
    analyze_summary = summarize_stage(step_results, "analyze")
    pre_summary = summarize_substage(step_results, "pre")
    post_summary = summarize_substage(step_results, "post")
    overall_exit_code = None
    overall_timeout_hit = False
    overall_signal = None
    overall_stderr_tail = None
    for summary in (run_summary, analyze_summary):
        exit_code = summary.get("exit_code")
        if summary.get("timeout_hit") or (exit_code not in (None, 0)):
            overall_exit_code = exit_code
            overall_timeout_hit = bool(summary.get("timeout_hit"))
            overall_signal = summary.get("signal")
            overall_stderr_tail = summary.get("stderr_tail")
            break
    if overall_exit_code is None and any(
        res.get("exit_code") is not None for res in step_results
    ):
        overall_exit_code = 0

    passfail = {
        "case_id": case_id,
        "case_status": case.get("case_status", "active"),
        "case_notes": case.get("case_notes"),
        "known_gaps": case.get("known_gaps")
        or (metrics.get("known_gaps") if metrics else None),
        "known_gap_metrics": case.get("known_gap_metrics")
        or (metrics.get("known_gap_metrics") if metrics else None),
        "gate_class": case.get("gate_class"),
        "physical_validity": case.get("physical_validity"),
        "status": status,
        "result": status,
        "failure_reason": failure_reason,
        "progress": progress,
        "exit_code": overall_exit_code,
        "timeout_hit": overall_timeout_hit,
        "signal": overall_signal,
        "stderr_tail": overall_stderr_tail,
        "exit_code_run": run_summary.get("exit_code"),
        "timeout_hit_run": run_summary.get("timeout_hit"),
        "signal_run": run_summary.get("signal"),
        "stderr_tail_run": run_summary.get("stderr_tail"),
        "exit_code_analyze": analyze_summary.get("exit_code"),
        "timeout_hit_analyze": analyze_summary.get("timeout_hit"),
        "signal_analyze": analyze_summary.get("signal"),
        "stderr_tail_analyze": analyze_summary.get("stderr_tail"),
        "exit_code_pre": pre_summary.get("exit_code"),
        "timeout_hit_pre": pre_summary.get("timeout_hit"),
        "signal_pre": pre_summary.get("signal"),
        "stderr_tail_pre": pre_summary.get("stderr_tail"),
        "exit_code_post": post_summary.get("exit_code"),
        "timeout_hit_post": post_summary.get("timeout_hit"),
        "signal_post": post_summary.get("signal"),
        "stderr_tail_post": post_summary.get("stderr_tail"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "git_commit": git.get("commit", "unknown"),
        "git_dirty": git.get("dirty", "unknown"),
        "input_hash": input_hash,
        "env": env_info(),
        "run_env": env_info(),
        "metrics": metrics,
        "metrics_file": metrics_file_rel,
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "gate_layers": gate_layers,
        "layer_status": layer_status,
        "artifacts": artifacts_rel,
        "artifact_policy": None,
        "artifact_cleanup": None,
        "errors": error_messages,
    }

    if status == "FAIL" and isinstance(metrics, dict):
        metric_reason = metrics.get("fail_reason") or metrics.get("FAIL_REASON")
        if metric_reason:
            if passfail.get("failure_reason") != "TIMEOUT":
                passfail["failure_reason"] = metric_reason

    passfail_path = output_analysis / "PASSFAIL.json"
    if not args.dry_run:
        write_json_atomic(passfail_path, passfail)
        print(f"[passfail] {passfail_path}")

        if not preflight_failed:
            artifact_manager = manage_artifacts(
                case,
                root,
                output_root,
                output_raw,
                output_analysis,
                output_plots,
                output_logs,
                status,
                artifacts,
            )
            if artifact_manager is not None:
                resistivity_meta = extract_resistivity_meta(output_raw)
                write_manifest(
                    output_root / "manifest.json",
                    output_root,
                    root,
                    case_id,
                    status,
                    artifact_manager.get("policy", {}),
                    git,
                    input_hash,
                    artifact_manager.get("raw_size_before", 0),
                    artifact_manager.get("raw_size_after", 0),
                    artifact_manager.get("cleanup_summary", {}),
                    resistivity_meta,
                )
                update_metrics_resources(
                    metrics_file,
                    select_run_wall_time(step_results),
                    artifact_manager.get("archive_paths", []),
                )
                artifacts_rel = []
                seen = set()
                for item in artifacts:
                    path = resolve_path(item, root)
                    if not path.exists():
                        continue
                    rel = to_relative(path, root)
                    if rel in seen:
                        continue
                    seen.add(rel)
                    artifacts_rel.append(rel)
                manifest_path = output_root / "manifest.json"
                if manifest_path.exists():
                    rel = to_relative(manifest_path, root)
                    if rel not in seen:
                        seen.add(rel)
                        artifacts_rel.append(rel)
                for archive_path in artifact_manager.get("archive_paths", []):
                    if not archive_path.exists():
                        continue
                    rel = to_relative(archive_path, root)
                    if rel in seen:
                        continue
                    seen.add(rel)
                    artifacts_rel.append(rel)
                passfail["artifacts"] = artifacts_rel
                passfail["artifact_policy"] = artifact_manager.get("policy")
                passfail["artifact_cleanup"] = artifact_manager.get("cleanup_summary")
                write_json_atomic(passfail_path, passfail)

        if args.update_evidence:
            subprocess.run(
                [sys.executable, "tools/build_evidence.py"],
                check=False,
                cwd=root,
                env=env,
            )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
