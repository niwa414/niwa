#!/usr/bin/env python3
"""Simulation operations orchestrator for Slurm or local execution.

Features:
- Submit case runs (tools/run_case.py) to Slurm (sbatch) or local subprocesses.
- Monitor states (squeue/sacct or local process polling).
- Evaluate PASS/FAIL gates from outputs/<case_id>/analysis/PASSFAIL.json.
- Retry failed jobs up to per-job/global retry budget.
- Persist registry/state for resume and audit.
- Generate run summary, work orders, and procurement spec draft.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def sanitize_key(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip())
    return clean.strip("-") or "job"


def normalize_slurm_state(state: str | None) -> str | None:
    if not state:
        return None
    base = str(state).strip().split()[0].split("+")[0].upper()
    return base or None


def parse_exit_code(exit_code: str | None) -> int | None:
    if not exit_code:
        return None
    token = str(exit_code).split(":", 1)[0].strip()
    try:
        return int(token)
    except Exception:
        return None


def resolve_case_file(case_ref: str, repo_root: Path) -> Path:
    candidate = Path(case_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if (repo_root / case_ref).exists():
        p = (repo_root / case_ref)
        if p.is_file():
            return p
    p2 = repo_root / "cases" / case_ref / "case.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot resolve case ref: {case_ref}")


def resolve_case_id(case_ref: str, repo_root: Path) -> str:
    path = resolve_case_file(case_ref, repo_root)
    data = load_json(path)
    cid = data.get("id")
    if isinstance(cid, str) and cid:
        return cid
    return path.parent.name


def run_case_cmd(repo_root: Path, case_ref: str, stage: str) -> list[str]:
    cmd = [sys.executable, "tools/run_case.py", "--case", case_ref]
    if stage and stage != "all":
        cmd.extend(["--stage", stage])
    return cmd


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def merge_slurm_options(global_opts: dict[str, Any], job_opts: dict[str, Any]) -> dict[str, Any]:
    merged = dict(global_opts or {})
    merged.update(job_opts or {})
    return merged


def build_sbatch_command(
    script_path: Path,
    job_name: str,
    stdout_path: Path,
    stderr_path: Path,
    slurm_opts: dict[str, Any],
) -> list[str]:
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(stdout_path),
        "--error",
        str(stderr_path),
    ]

    simple_map = {
        "partition": "--partition",
        "account": "--account",
        "qos": "--qos",
        "constraint": "--constraint",
        "time": "--time",
        "mem": "--mem",
        "nodes": "--nodes",
        "ntasks": "--ntasks",
        "cpus_per_task": "--cpus-per-task",
    }
    for key, flag in simple_map.items():
        value = slurm_opts.get(key)
        if value is None or value == "":
            continue
        cmd.extend([flag, str(value)])

    gpus = slurm_opts.get("gpus")
    if isinstance(gpus, int) and gpus > 0:
        cmd.extend(["--gpus", str(gpus)])

    gres = slurm_opts.get("gres")
    if gres:
        cmd.extend(["--gres", str(gres)])

    for extra in slurm_opts.get("extra_args", []) or []:
        cmd.append(str(extra))

    cmd.append(str(script_path))
    return cmd


def write_sbatch_script(script_path: Path, repo_root: Path, run_cmd: list[str], env: dict[str, str]) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
    ]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(shlex.join(run_cmd))
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def ensure_slurm_tools() -> None:
    required = ["sbatch", "squeue", "sacct"]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing Slurm tools in PATH: {', '.join(missing)}")


def submit_slurm_job(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    if not out:
        raise RuntimeError("sbatch returned empty output")
    # --parsable can return: 12345 or 12345;cluster
    job_id = out.split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(f"Cannot parse job id from sbatch output: {out}")
    return job_id


def query_squeue_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    cmd = ["squeue", "-h", "-o", "%i|%T", "-j", ",".join(job_ids)]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    states: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        jid, state = line.split("|", 1)
        jid = jid.strip()
        state = normalize_slurm_state(state)
        if jid and state:
            states[jid] = state
    return states


def query_sacct_state(job_id: str) -> tuple[str | None, int | None]:
    cmd = ["sacct", "-P", "-n", "-j", str(job_id), "--format", "JobIDRaw,State,ExitCode"]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    best_state = None
    best_exit = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid_raw, state_raw, exit_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if jid_raw != str(job_id):
            continue
        best_state = normalize_slurm_state(state_raw)
        best_exit = parse_exit_code(exit_raw)
        break

    if best_state is None:
        # fallback: first row if exact row absent
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            best_state = normalize_slurm_state(parts[1])
            best_exit = parse_exit_code(parts[2])
            break

    return best_state, best_exit


def is_terminal_state(mode: str, state: str | None) -> bool:
    if not state:
        return False
    if mode == "slurm":
        return state in TERMINAL_SLURM_STATES
    return state in {"COMPLETED", "FAILED", "CANCELLED"}


def load_plan(plan_path: Path) -> dict[str, Any]:
    data = load_json(plan_path)
    if not data:
        raise ValueError(f"Invalid or empty plan: {plan_path}")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Plan must include non-empty 'cases' list")
    return data


def make_run_id(plan_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{sanitize_key(plan_name)}-{stamp}"


def default_run_root(repo_root: Path) -> Path:
    return repo_root / "outputs" / "orchestrator"


def init_state(
    repo_root: Path,
    plan: dict[str, Any],
    plan_path: Path,
    run_id: str,
    mode: str,
    force_stage: str | None,
    retry_override: int | None,
) -> dict[str, Any]:
    jobs = []
    global_retries = int(plan.get("max_retries", 0))
    for idx, entry in enumerate(plan.get("cases", []), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"cases[{idx}] must be object")
        case_ref = str(entry.get("case") or "").strip()
        if not case_ref:
            raise ValueError(f"cases[{idx}] missing 'case'")
        key = sanitize_key(str(entry.get("key") or f"case-{idx}"))
        stage = str(entry.get("stage") or "all")
        if force_stage:
            stage = force_stage
        case_id = resolve_case_id(case_ref, repo_root)
        retries = int(entry.get("retries", global_retries))
        if retry_override is not None:
            retries = int(retry_override)
        jobs.append(
            {
                "key": key,
                "case_ref": case_ref,
                "case_id": case_id,
                "role": entry.get("role"),
                "stage": stage,
                "max_retries": max(0, retries),
                "attempts": [],
                "final_status": "PENDING",
                "notes": entry.get("notes"),
                "slurm": entry.get("slurm") or {},
            }
        )

    return {
        "run_id": run_id,
        "name": plan.get("name") or run_id,
        "description": plan.get("description"),
        "mode": mode,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "repo_root": str(repo_root),
        "plan_path": str(plan_path.resolve()),
        "plan": plan,
        "jobs": jobs,
        "summary": {},
    }


def get_passfail_path(repo_root: Path, case_id: str) -> Path:
    return repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"


def get_metrics_path(repo_root: Path, case_id: str) -> Path:
    return repo_root / "outputs" / case_id / "analysis" / "metrics.json"


def get_metrics_source_path(repo_root: Path, case_id: str) -> Path:
    return repo_root / "outputs" / case_id / "analysis" / "metrics_source.json"


def submit_attempt(
    *,
    repo_root: Path,
    run_dir: Path,
    state: dict[str, Any],
    job: dict[str, Any],
    mode: str,
    local_procs: dict[str, subprocess.Popen[Any]],
) -> None:
    attempt_num = len(job["attempts"]) + 1
    key = job["key"]
    case_ref = job["case_ref"]
    stage = job["stage"]
    run_cmd = run_case_cmd(repo_root, case_ref, stage)

    attempt: dict[str, Any] = {
        "attempt": attempt_num,
        "submitted_at": now_iso(),
        "scheduler_mode": mode,
        "scheduler_id": None,
        "scheduler_state": "SUBMITTED",
        "terminal": False,
        "return_code": None,
        "ended_at": None,
        "passfail_status": None,
        "passfail_path": rel(get_passfail_path(repo_root, job["case_id"]), repo_root),
        "run_case_cmd": run_cmd,
        "error": None,
        "stdout_log": None,
        "stderr_log": None,
        "script_path": None,
    }

    if mode == "slurm":
        ensure_slurm_tools()
        slurm_dir = run_dir / "slurm"
        script_dir = run_dir / "scripts"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        script_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = slurm_dir / f"{key}_a{attempt_num}.out"
        stderr_path = slurm_dir / f"{key}_a{attempt_num}.err"
        script_path = script_dir / f"{key}_a{attempt_num}.sh"
        job_name = sanitize_key(f"{state['run_id']}-{key}-a{attempt_num}")[:120]

        write_sbatch_script(
            script_path,
            repo_root,
            run_cmd,
            {
                "SIM_OPS_RUN_ID": str(state["run_id"]),
                "SIM_OPS_JOB_KEY": str(key),
                "SIM_OPS_ATTEMPT": str(attempt_num),
            },
        )

        slurm_opts = merge_slurm_options(state["plan"].get("slurm", {}), job.get("slurm", {}))
        sbatch_cmd = build_sbatch_command(script_path, job_name, stdout_path, stderr_path, slurm_opts)
        try:
            job_id = submit_slurm_job(sbatch_cmd)
        except Exception as exc:
            attempt["scheduler_state"] = "FAILED"
            attempt["terminal"] = True
            attempt["ended_at"] = now_iso()
            attempt["error"] = f"submit_failed: {exc}"
            attempt["return_code"] = 1
            job["attempts"].append(attempt)
            job["final_status"] = "FAIL"
            return

        attempt["scheduler_id"] = job_id
        attempt["scheduler_state"] = "PENDING"
        attempt["stdout_log"] = rel(stdout_path, repo_root)
        attempt["stderr_log"] = rel(stderr_path, repo_root)
        attempt["script_path"] = rel(script_path, repo_root)
        job["attempts"].append(attempt)
        job["final_status"] = "RUNNING"
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "ts": now_iso(),
                "event": "submit",
                "mode": "slurm",
                "job_key": key,
                "attempt": attempt_num,
                "scheduler_id": job_id,
                "case_id": job["case_id"],
            },
        )
        return

    # local mode
    local_dir = run_dir / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = local_dir / f"{key}_a{attempt_num}.log"
    with stdout_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            run_cmd,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    proc_key = f"{key}#{attempt_num}"
    local_procs[proc_key] = proc

    attempt["scheduler_id"] = str(proc.pid)
    attempt["scheduler_state"] = "RUNNING"
    attempt["stdout_log"] = rel(stdout_path, repo_root)
    job["attempts"].append(attempt)
    job["final_status"] = "RUNNING"

    append_jsonl(
        run_dir / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "submit",
            "mode": "local",
            "job_key": key,
            "attempt": attempt_num,
            "pid": proc.pid,
            "case_id": job["case_id"],
        },
    )


def mark_attempt_terminal(attempt: dict[str, Any], state: str, exit_code: int | None) -> None:
    attempt["scheduler_state"] = state
    attempt["terminal"] = True
    attempt["ended_at"] = now_iso()
    attempt["return_code"] = exit_code


def attempt_scheduler_succeeded(attempt: dict[str, Any]) -> bool:
    state = normalize_slurm_state(str(attempt.get("scheduler_state") or ""))
    if state != "COMPLETED":
        return False
    rc = attempt.get("return_code")
    if rc is None:
        return True
    try:
        return int(rc) == 0
    except Exception:
        return False


def evaluate_attempt_gate(repo_root: Path, job: dict[str, Any], attempt: dict[str, Any]) -> str:
    pf_path = get_passfail_path(repo_root, job["case_id"])
    passfail = load_json(pf_path)
    status = str(passfail.get("status") or passfail.get("result") or "").upper()
    if status == "PASS":
        attempt["passfail_status"] = "PASS"
        return "PASS"
    attempt["passfail_status"] = "FAIL"
    return "FAIL"


def update_slurm_attempts(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> bool:
    changed = False
    active: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for job in state["jobs"]:
        if not job["attempts"]:
            continue
        attempt = job["attempts"][-1]
        if attempt.get("terminal"):
            continue
        jid = attempt.get("scheduler_id")
        if not jid:
            continue
        active.append((job, attempt))

    if not active:
        return changed

    ids = [str(att["scheduler_id"]) for _, att in active]
    try:
        squeue_states = query_squeue_states(ids)
    except Exception as exc:
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "ts": now_iso(),
                "event": "monitor_warning",
                "mode": "slurm",
                "message": f"squeue query failed: {exc}",
            },
        )
        squeue_states = {}

    for job, attempt in active:
        jid = str(attempt["scheduler_id"])
        prev_state = attempt.get("scheduler_state")

        state_now = squeue_states.get(jid)
        exit_code = attempt.get("return_code")
        if state_now is None:
            try:
                state_now, exit_code = query_sacct_state(jid)
            except Exception as exc:
                append_jsonl(
                    run_dir / "events.jsonl",
                    {
                        "ts": now_iso(),
                        "event": "monitor_warning",
                        "mode": "slurm",
                        "job_key": job["key"],
                        "attempt": attempt.get("attempt"),
                        "message": f"sacct query failed: {exc}",
                    },
                )
                continue

        if state_now and state_now != prev_state:
            attempt["scheduler_state"] = state_now
            changed = True

        if is_terminal_state("slurm", state_now):
            mark_attempt_terminal(attempt, state_now or "FAILED", exit_code)
            changed = True
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "ts": now_iso(),
                    "event": "attempt_terminal",
                    "mode": "slurm",
                    "job_key": job["key"],
                    "attempt": attempt.get("attempt"),
                    "scheduler_id": jid,
                    "scheduler_state": state_now,
                    "return_code": exit_code,
                },
            )

    return changed


def update_local_attempts(
    repo_root: Path,
    run_dir: Path,
    state: dict[str, Any],
    local_procs: dict[str, subprocess.Popen[Any]],
) -> bool:
    changed = False
    for job in state["jobs"]:
        if not job["attempts"]:
            continue
        attempt = job["attempts"][-1]
        if attempt.get("terminal"):
            continue
        attempt_num = int(attempt["attempt"])
        proc_key = f"{job['key']}#{attempt_num}"
        proc = local_procs.get(proc_key)
        if proc is None:
            continue
        rc = proc.poll()
        if rc is None:
            continue
        state_now = "COMPLETED" if rc == 0 else "FAILED"
        mark_attempt_terminal(attempt, state_now, int(rc))
        changed = True
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "ts": now_iso(),
                "event": "attempt_terminal",
                "mode": "local",
                "job_key": job["key"],
                "attempt": attempt_num,
                "pid": attempt.get("scheduler_id"),
                "scheduler_state": state_now,
                "return_code": int(rc),
            },
        )
    return changed


def progress_jobs(
    repo_root: Path,
    run_dir: Path,
    state: dict[str, Any],
    mode: str,
    local_procs: dict[str, subprocess.Popen[Any]],
) -> bool:
    changed = False

    for job in state["jobs"]:
        if not job["attempts"]:
            continue
        attempt = job["attempts"][-1]
        if not attempt.get("terminal"):
            continue
        if job.get("final_status") in {"PASS", "FAIL", "CANCELLED"}:
            continue

        if not attempt_scheduler_succeeded(attempt):
            attempt["passfail_status"] = "FAIL_SCHEDULER"
            gate = "FAIL"
        else:
            gate = evaluate_attempt_gate(repo_root, job, attempt)
        if gate == "PASS":
            job["final_status"] = "PASS"
            changed = True
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "ts": now_iso(),
                    "event": "job_pass",
                    "job_key": job["key"],
                    "attempt": attempt.get("attempt"),
                    "case_id": job["case_id"],
                },
            )
            continue

        used_retries = len(job["attempts"]) - 1
        retry_budget = int(job.get("max_retries", 0))
        if used_retries < retry_budget:
            job["final_status"] = "RETRYING"
            submit_attempt(
                repo_root=repo_root,
                run_dir=run_dir,
                state=state,
                job=job,
                mode=mode,
                local_procs=local_procs,
            )
            changed = True
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "ts": now_iso(),
                    "event": "job_retry",
                    "job_key": job["key"],
                    "retry_used": used_retries + 1,
                    "retry_budget": retry_budget,
                },
            )
        else:
            job["final_status"] = "FAIL"
            changed = True
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "ts": now_iso(),
                    "event": "job_fail",
                    "job_key": job["key"],
                    "attempt": attempt.get("attempt"),
                    "case_id": job["case_id"],
                },
            )

    return changed


def all_jobs_terminal(state: dict[str, Any]) -> bool:
    terminal = {"PASS", "FAIL", "CANCELLED"}
    return all(job.get("final_status") in terminal for job in state.get("jobs", []))


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in state.get("jobs", []):
        st = str(job.get("final_status") or "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return {
        "job_count": len(state.get("jobs", [])),
        "status_counts": counts,
    }


def collect_metrics(repo_root: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    m = load_json(get_metrics_path(repo_root, case_id))
    ms = load_json(get_metrics_source_path(repo_root, case_id))
    return m, ms


def recommend_job(repo_root: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for job in state.get("jobs", []):
        if job.get("final_status") != "PASS":
            continue
        m, _ = collect_metrics(repo_root, job["case_id"])
        tilt = m.get("tilt_amp_max")
        comp = m.get("compression_ratio")
        try:
            tilt_v = float(tilt)
            comp_v = float(comp)
        except Exception:
            continue
        candidates.append((tilt_v, -comp_v, job, m))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, job, m = candidates[0]
    return {
        "job_key": job["key"],
        "case_id": job["case_id"],
        "role": job.get("role"),
        "knob_name": m.get("knob_name"),
        "knob_value": m.get("knob_value"),
        "tilt_amp_max": m.get("tilt_amp_max"),
        "compression_ratio": m.get("compression_ratio"),
    }


def build_work_orders(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> tuple[Path, Path]:
    recommended = recommend_job(repo_root, state)
    orders: list[dict[str, Any]] = []

    # Global execution summary work order
    orders.append(
        {
            "id": "WO-OPS-001",
            "title": "Review simulation campaign gate results",
            "status": "READY" if all_jobs_terminal(state) else "PENDING",
            "priority": "high",
            "action": "Review PASS/FAIL, logs, and threshold misses before next shot window.",
            "run_id": state["run_id"],
        }
    )

    if recommended:
        knob = recommended.get("knob_name") or "knob"
        orders.append(
            {
                "id": "WO-OPS-002",
                "title": "Apply recommended control knob for next shot batch",
                "status": "READY",
                "priority": "high",
                "action": (
                    f"Set {knob}={recommended.get('knob_value')} for next 3-5 shots; "
                    f"track tilt_amp_max and compression_ratio drift."
                ),
                "recommended_case": recommended,
            }
        )

    for job in state.get("jobs", []):
        if job.get("final_status") == "PASS":
            continue
        pf_path = get_passfail_path(repo_root, job["case_id"])
        passfail = load_json(pf_path)
        failed_metrics = []
        for tr in passfail.get("threshold_results", []) or []:
            if tr.get("pass") is False:
                failed_metrics.append(
                    {
                        "metric": tr.get("metric"),
                        "actual": tr.get("actual"),
                        "op": tr.get("op"),
                        "target": tr.get("value"),
                        "reason": tr.get("reason"),
                    }
                )
        orders.append(
            {
                "id": f"WO-DBG-{sanitize_key(job['key']).upper()}",
                "title": f"Debug failed simulation case: {job['case_id']}",
                "status": "OPEN",
                "priority": "critical",
                "action": "Inspect run/analyze logs, root-cause failed gates, and rerun with corrected input.",
                "case_id": job["case_id"],
                "failed_metrics": failed_metrics,
                "passfail_path": rel(pf_path, repo_root),
            }
        )

    json_path = run_dir / "work_orders.json"
    write_json_atomic(
        json_path,
        {
            "run_id": state["run_id"],
            "generated_at": now_iso(),
            "orders": orders,
        },
    )

    md_lines = [
        "# Work Orders",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- generated_at: `{now_iso()}`",
        "",
        "| id | status | priority | title | action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in orders:
        md_lines.append(
            f"| {o.get('id')} | {o.get('status')} | {o.get('priority')} | {o.get('title')} | {o.get('action')} |"
        )

    md_path = run_dir / "work_orders.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def compute_min_nom_max(values: list[float]) -> tuple[float, float, float]:
    arr = sorted(values)
    return arr[0], float(median(arr)), arr[-1]


def load_internal_parity_metrics(repo_root: Path) -> tuple[dict[str, Any], Path]:
    path = repo_root / "outputs" / "m28-d1-helion-internal-parity-gate" / "analysis" / "metrics.json"
    return load_json(path), path


def build_procurement_spec(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> tuple[Path, Path]:
    datasets: list[dict[str, Any]] = []
    for job in state.get("jobs", []):
        if job.get("final_status") != "PASS":
            continue
        m, ms = collect_metrics(repo_root, job["case_id"])
        datasets.append(
            {
                "case_id": job["case_id"],
                "job_key": job["key"],
                "role": job.get("role"),
                "e_in_J": ms.get("e_in_J"),
                "e_load_J": ms.get("e_load_J"),
                "eta_recaptured": ms.get("eta_recaptured"),
                "dphi_dt_max": ms.get("dphi_dt_max"),
                "compression_ratio": m.get("compression_ratio"),
                "tilt_amp_max": m.get("tilt_amp_max"),
                "energy_residual_rel": m.get("energy_residual_rel"),
            }
        )

    spec_defs = [
        {
            "spec_id": "E-LIVE-01",
            "name": "Pulse input energy capacity (J)",
            "field": "e_in_J",
            "safety_factor": 1.25,
            "metric_binding": "metrics_source.e_in_J",
        },
        {
            "spec_id": "E-LIVE-02",
            "name": "Pulse delivered energy capacity (J)",
            "field": "e_load_J",
            "safety_factor": 1.25,
            "metric_binding": "metrics_source.e_load_J",
        },
        {
            "spec_id": "E-LIVE-03",
            "name": "Recapture efficiency target",
            "field": "eta_recaptured",
            "safety_factor": 1.0,
            "metric_binding": "metrics_source.eta_recaptured",
        },
        {
            "spec_id": "P-LIVE-01",
            "name": "Compression ratio operating window",
            "field": "compression_ratio",
            "safety_factor": 1.0,
            "metric_binding": "metrics.compression_ratio",
        },
        {
            "spec_id": "S-LIVE-01",
            "name": "Tilt amplitude acceptance max",
            "field": "tilt_amp_max",
            "safety_factor": 1.0,
            "metric_binding": "metrics.tilt_amp_max",
        },
        {
            "spec_id": "Q-LIVE-01",
            "name": "Energy accounting residual max",
            "field": "energy_residual_rel",
            "safety_factor": 1.0,
            "metric_binding": "metrics.energy_residual_rel",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in spec_defs:
        vals: list[float] = []
        for ds in datasets:
            value = ds.get(spec["field"])
            if isinstance(value, (int, float)):
                vals.append(float(value))
        if vals:
            min_v, nom_v, max_v = compute_min_nom_max(vals)
            sf = float(spec["safety_factor"])
            rows.append(
                {
                    **spec,
                    "min": min_v,
                    "nom": nom_v,
                    "max": max_v,
                    "max_with_sf": max_v * sf,
                    "status": "READY" if len(vals) >= 2 else "LIMITED_SAMPLE",
                    "procurement_class": "可直接采购" if len(vals) >= 2 else "需先补实验/补模型",
                    "sample_count": len(vals),
                }
            )
        else:
            rows.append(
                {
                    **spec,
                    "min": None,
                    "nom": None,
                    "max": None,
                    "max_with_sf": None,
                    "status": "MISSING",
                    "procurement_class": "需先补实验/补模型",
                    "sample_count": 0,
                }
            )

    direct_rows = [r for r in rows if r.get("procurement_class") == "可直接采购"]
    pending_rows = [r for r in rows if r.get("procurement_class") != "可直接采购"]

    internal_metrics, internal_path = load_internal_parity_metrics(repo_root)
    gap_defs = {
        "gpu_runtime_proven": {
            "action": "补GPU runtime proof（GPU后端构建+最小闭环运行日志+manifest回写）",
            "unblock": "GPU runtime proof",
        },
        "private_shot_dataset_bound": {
            "action": "补private shot dataset绑定与回放校验",
            "unblock": "private shot dataset",
        },
        "private_hardware_model_bound": {
            "action": "补private hardware model绑定并复核V_ind/F映射",
            "unblock": "private hardware model",
        },
    }
    internal_gaps: list[dict[str, Any]] = []
    for gap in internal_metrics.get("internal_only_gaps", []) or []:
        gap_name = str(gap)
        spec = gap_defs.get(gap_name, {"action": "补充对应internal证据", "unblock": gap_name})
        current = internal_metrics.get(gap_name)
        internal_gaps.append(
            {
                "gap": gap_name,
                "current": current,
                "action": spec["action"],
                "unblock": spec["unblock"],
            }
        )

    json_path = run_dir / "procurement_spec.json"
    write_json_atomic(
        json_path,
        {
            "run_id": state["run_id"],
            "generated_at": now_iso(),
            "sample_cases": [ds["case_id"] for ds in datasets],
            "direct_procurement_count": len(direct_rows),
            "needs_more_evidence_count": len(pending_rows),
            "internal_parity": {
                "source_metrics_path": rel(internal_path, repo_root),
                "internal_parity_claimable": internal_metrics.get("internal_parity_claimable"),
                "internal_only_gap_count": internal_metrics.get("internal_only_gap_count"),
                "gaps": internal_gaps,
            },
            "rows": rows,
        },
    )

    md_lines = [
        "# Procurement Spec Draft",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- generated_at: `{now_iso()}`",
        f"- sample_cases: `{len(datasets)}`",
        f"- direct_procurement_count: `{len(direct_rows)}`",
        f"- needs_more_evidence_count: `{len(pending_rows)}`",
        "",
        "## A) 可直接采购",
        "",
        "| spec_id | name | min | nom | max | max_with_sf | sf | status | metric_binding |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    if not direct_rows:
        md_lines.append("| - | - | - | - | - | - | - | - | - |")
    for row in direct_rows:
        md_lines.append(
            "| {spec_id} | {name} | {min} | {nom} | {max} | {max_with_sf} | {safety_factor} | {status} | `{metric_binding}` |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## B) 需先补实验/补模型",
            "",
            "| spec_id | name | min | nom | max | max_with_sf | sf | status | metric_binding |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    if not pending_rows:
        md_lines.append("| - | - | - | - | - | - | - | - | - |")
    for row in pending_rows:
        md_lines.append(
            "| {spec_id} | {name} | {min} | {nom} | {max} | {max_with_sf} | {safety_factor} | {status} | `{metric_binding}` |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## C) Internal-only gap 清单与闭环动作",
            "",
            f"- internal_parity_metrics: `{rel(internal_path, repo_root)}`",
            f"- internal_parity_claimable: `{internal_metrics.get('internal_parity_claimable')}`",
            f"- internal_only_gap_count: `{internal_metrics.get('internal_only_gap_count')}`",
            "",
            "| gap | current | action | unblock |",
            "| --- | --- | --- | --- |",
        ]
    )

    if not internal_gaps:
        md_lines.append("| - | - | - | - |")
    for gap in internal_gaps:
        md_lines.append(
            f"| {gap['gap']} | {gap['current']} | {gap['action']} | {gap['unblock']} |"
        )

    md_lines.extend(
        [
            "",
            "Notes:",
            "- `status=MISSING` or `LIMITED_SAMPLE` means do not issue final PO yet.",
            "- Use this draft with `/Users/ni/Desktop/fusion/outputs/analysis/procurement-ready-spec.md` for release sign-off.",
        ]
    )

    md_path = run_dir / "procurement_spec.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def maybe_generate_trade_report(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> Path | None:
    role_map: dict[str, str] = {}
    for job in state.get("jobs", []):
        role = job.get("role")
        if not role or job.get("final_status") != "PASS":
            continue
        role_map[str(role)] = job.get("case_id")

    needed = ["baseline", "knob_minus", "knob_plus"]
    if not all(name in role_map for name in needed):
        return None

    output = run_dir / "trade_study_report.md"
    cmd = [
        sys.executable,
        "tools/report_helion_demo.py",
        str(repo_root / "outputs" / role_map["baseline"]),
        str(repo_root / "outputs" / role_map["knob_minus"]),
        str(repo_root / "outputs" / role_map["knob_plus"]),
        "--output",
        str(output),
    ]
    try:
        subprocess.run(cmd, cwd=repo_root, check=True)
    except Exception as exc:
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "ts": now_iso(),
                "event": "report_warning",
                "message": f"trade_report_failed: {exc}",
            },
        )
        return None
    return output


def write_summary(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> Path:
    summary = summarize_state(state)
    recommended = recommend_job(repo_root, state)

    lines = [
        "# Simulation Ops Run Summary",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- mode: `{state['mode']}`",
        f"- created_at: `{state.get('created_at')}`",
        f"- finished_at: `{now_iso()}`",
        f"- job_count: `{summary['job_count']}`",
        f"- status_counts: `{summary['status_counts']}`",
        "",
        "| key | case_id | role | final_status | attempts | last_scheduler_id | passfail_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for job in state.get("jobs", []):
        attempts = job.get("attempts", [])
        last = attempts[-1] if attempts else {}
        lines.append(
            f"| {job.get('key')} | {job.get('case_id')} | {job.get('role')} | {job.get('final_status')} | "
            f"{len(attempts)} | {last.get('scheduler_id')} | {last.get('passfail_status')} |"
        )

    lines.append("")
    if recommended:
        lines.extend(
            [
                "## Recommended Operating Point",
                "",
                f"- case_id: `{recommended.get('case_id')}`",
                f"- role: `{recommended.get('role')}`",
                f"- knob: `{recommended.get('knob_name')}={recommended.get('knob_value')}`",
                f"- tilt_amp_max: `{recommended.get('tilt_amp_max')}`",
                f"- compression_ratio: `{recommended.get('compression_ratio')}`",
                "",
            ]
        )

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def finalize_outputs(repo_root: Path, run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    summary_path = write_summary(repo_root, run_dir, state)
    wo_json, wo_md = build_work_orders(repo_root, run_dir, state)
    spec_json, spec_md = build_procurement_spec(repo_root, run_dir, state)
    trade_report = maybe_generate_trade_report(repo_root, run_dir, state)

    result = {
        "summary_md": rel(summary_path, repo_root),
        "work_orders_json": rel(wo_json, repo_root),
        "work_orders_md": rel(wo_md, repo_root),
        "procurement_spec_json": rel(spec_json, repo_root),
        "procurement_spec_md": rel(spec_md, repo_root),
        "trade_report_md": rel(trade_report, repo_root) if trade_report else None,
    }
    return result


def monitor_until_done(
    *,
    repo_root: Path,
    run_dir: Path,
    state: dict[str, Any],
    mode: str,
    poll_interval_s: int,
    local_procs: dict[str, subprocess.Popen[Any]],
    state_path: Path,
) -> None:
    while True:
        changed = False
        if mode == "slurm":
            changed |= update_slurm_attempts(repo_root, run_dir, state)
        else:
            changed |= update_local_attempts(repo_root, run_dir, state, local_procs)

        changed |= progress_jobs(repo_root, run_dir, state, mode, local_procs)

        if changed:
            state["updated_at"] = now_iso()
            state["summary"] = summarize_state(state)
            write_json_atomic(state_path, state)

        if all_jobs_terminal(state):
            break

        time.sleep(max(1, int(poll_interval_s)))


def cmd_start(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = (repo_root / args.plan).resolve() if not Path(args.plan).is_absolute() else Path(args.plan)
    plan = load_plan(plan_path)

    mode = args.mode
    if mode == "slurm":
        ensure_slurm_tools()

    run_id = args.run_id or make_run_id(str(plan.get("name") or "sim-ops"))
    run_root = default_run_root(repo_root)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    state = init_state(
        repo_root=repo_root,
        plan=plan,
        plan_path=plan_path,
        run_id=run_id,
        mode=mode,
        force_stage=args.force_stage,
        retry_override=args.max_retries,
    )

    state_path = run_dir / "state.json"
    write_json_atomic(state_path, state)
    write_json_atomic(run_dir / "plan.snapshot.json", plan)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "run_started",
            "run_id": run_id,
            "mode": mode,
            "plan_path": str(plan_path),
        },
    )

    local_procs: dict[str, subprocess.Popen[Any]] = {}

    for job in state["jobs"]:
        submit_attempt(
            repo_root=repo_root,
            run_dir=run_dir,
            state=state,
            job=job,
            mode=mode,
            local_procs=local_procs,
        )

    state["updated_at"] = now_iso()
    state["summary"] = summarize_state(state)
    write_json_atomic(state_path, state)

    monitor_until_done(
        repo_root=repo_root,
        run_dir=run_dir,
        state=state,
        mode=mode,
        poll_interval_s=args.poll_interval_s,
        local_procs=local_procs,
        state_path=state_path,
    )

    outputs = finalize_outputs(repo_root, run_dir, state)
    state["updated_at"] = now_iso()
    state["finished_at"] = now_iso()
    state["summary"] = summarize_state(state)
    state["artifacts"] = outputs
    write_json_atomic(state_path, state)

    print(f"[sim-ops] run_id={run_id}")
    print(f"[sim-ops] state={state_path}")
    print(f"[sim-ops] summary={run_dir / 'summary.md'}")

    any_fail = any(job.get("final_status") != "PASS" for job in state.get("jobs", []))
    return 1 if any_fail else 0


def cmd_resume(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = default_run_root(repo_root)
    run_dir = (run_root / args.run_id).resolve()
    state_path = run_dir / "state.json"
    state = load_json(state_path)
    if not state:
        raise FileNotFoundError(f"State not found: {state_path}")

    mode = str(state.get("mode") or args.mode or "slurm")
    if args.mode and args.mode != mode:
        raise ValueError(f"Mode mismatch: state={mode}, arg={args.mode}")

    if mode == "local":
        raise RuntimeError("Resume for local mode is not supported (PIDs not persisted).")

    ensure_slurm_tools()

    append_jsonl(
        run_dir / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "run_resumed",
            "run_id": state.get("run_id"),
        },
    )

    local_procs: dict[str, subprocess.Popen[Any]] = {}
    monitor_until_done(
        repo_root=repo_root,
        run_dir=run_dir,
        state=state,
        mode=mode,
        poll_interval_s=args.poll_interval_s,
        local_procs=local_procs,
        state_path=state_path,
    )

    outputs = finalize_outputs(repo_root, run_dir, state)
    state["updated_at"] = now_iso()
    state["finished_at"] = now_iso()
    state["summary"] = summarize_state(state)
    state["artifacts"] = outputs
    write_json_atomic(state_path, state)

    any_fail = any(job.get("final_status") != "PASS" for job in state.get("jobs", []))
    return 1 if any_fail else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulation operations orchestrator (Slurm/local).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Start a new orchestrator run from plan file.")
    p_start.add_argument("--plan", required=True, help="Plan JSON path.")
    p_start.add_argument("--mode", choices=["slurm", "local"], default="slurm")
    p_start.add_argument("--run-id", default=None, help="Optional run id; default derived from plan name.")
    p_start.add_argument("--poll-interval-s", type=int, default=30)
    p_start.add_argument("--force-stage", choices=["all", "run", "analyze"], default=None)
    p_start.add_argument("--max-retries", type=int, default=None, help="Override retry budget for all jobs.")
    p_start.set_defaults(func=cmd_start)

    p_resume = sub.add_parser("resume", help="Resume monitoring an existing Slurm run.")
    p_resume.add_argument("--run-id", required=True, help="Run id under outputs/orchestrator/<run_id>")
    p_resume.add_argument("--mode", choices=["slurm", "local"], default=None)
    p_resume.add_argument("--poll-interval-s", type=int, default=30)
    p_resume.set_defaults(func=cmd_resume)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
