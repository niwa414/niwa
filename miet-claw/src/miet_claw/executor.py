import json
import os
import shutil
import subprocess
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .archive import write_manifest
from .explain import write_summary
from .planner import build_plan
from .specs import load_job_spec
from .state import StateStore
from .transforms import (
    compile_event_table,
    derive_kmc_barrier_map,
    render_kmc_input,
    stage_support_file,
    synthetic_event_rows,
    write_event_table_csv,
    write_json,
)


class ExecutionError(RuntimeError):
    pass


class ExecutionCancelled(ExecutionError):
    pass


ProgressCallback = Callable[[str, Dict[str, Any]], None]
CancelCheck = Callable[[str, Path, Dict[str, Any]], Optional[Any]]
CheckpointCallback = Callable[[str, Dict[str, Any]], None]


def _command_exists(command: str) -> bool:
    candidate = Path(command)
    if candidate.is_absolute() or "/" in command:
        return candidate.exists()
    return shutil.which(command) is not None


def _run_command(
    command: List[str],
    cwd: Path,
    stdin_path: Path = None,
    environment: Dict = None,
    heartbeat=None,
    heartbeat_interval_s: float = 1.5,
    live_log_path: Path = None,
) -> Dict:
    stdin_handle = stdin_path.open("rb") if stdin_path else None
    live_log_handle = live_log_path.open("w", encoding="utf-8") if live_log_path else None
    log_lock = threading.Lock()
    stderr_started = False
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _write_live(chunk: str, is_stderr: bool = False) -> None:
        nonlocal stderr_started
        if not live_log_handle or not chunk:
            return
        with log_lock:
            if is_stderr and not stderr_started:
                live_log_handle.write("\nSTDERR\n")
                stderr_started = True
            live_log_handle.write(chunk)
            live_log_handle.flush()

    def _consume_stream(stream, sink: List[str], is_stderr: bool = False) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                sink.append(text)
                _write_live(text, is_stderr=is_stderr)
        finally:
            stream.close()

    process = None
    stdout_thread = None
    stderr_thread = None
    try:
        env = os.environ.copy()
        if environment:
            env.update(environment)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout_thread = threading.Thread(target=_consume_stream, args=(process.stdout, stdout_chunks, False), daemon=True)
        stderr_thread = threading.Thread(target=_consume_stream, args=(process.stderr, stderr_chunks, True), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        if heartbeat:
            heartbeat(process.pid)
        last_heartbeat = time.monotonic()
        while process.poll() is None:
            if heartbeat and (time.monotonic() - last_heartbeat) >= heartbeat_interval_s:
                heartbeat(process.pid)
                last_heartbeat = time.monotonic()
            time.sleep(0.25)

        stdout_thread.join()
        stderr_thread.join()
        returncode = process.wait()
    except Exception:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        if stdout_thread:
            stdout_thread.join()
        if stderr_thread:
            stderr_thread.join()
        raise
    finally:
        if stdin_handle:
            stdin_handle.close()
        if live_log_handle:
            live_log_handle.close()
    return {
        "returncode": returncode,
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
    }


class JobExecutor:
    def __init__(
        self,
        spec_path: str,
        output_dir: str,
        dry_run: bool = False,
        *,
        resume: bool = False,
        overwrite_existing: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
        checkpoint_callback: Optional[CheckpointCallback] = None,
    ):
        self.spec = load_job_spec(spec_path)
        self.plan = build_plan(self.spec)
        self.run_dir = Path(output_dir).resolve() / self.spec["job_id"]
        self.dry_run = dry_run
        self.resume = resume
        self.overwrite_existing = overwrite_existing
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.checkpoint_callback = checkpoint_callback
        existing_state = self._existing_state()
        self.resume_summary = self._resume_summary(existing_state)
        self.recovery_plan = self._recovery_plan(existing_state)
        has_existing_progress = self._has_existing_progress(existing_state)
        if has_existing_progress and overwrite_existing:
            shutil.rmtree(self.run_dir)
            existing_state = None
            has_existing_progress = False
            self.resume_summary = None
            self.recovery_plan = None
        if has_existing_progress and not resume:
            raise ExecutionError(
                f"Run directory already has recorded progress: {self.run_dir}. "
                "Use resume=True to continue, or overwrite_existing=True to start over."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.state_store = StateStore(
            self.run_dir / "state.json",
            self.spec["job_id"],
            self.spec["mode"],
            [step.id for step in self.plan],
        )
        self.state_store.update_job(
            dry_run=self.dry_run,
            resume=self.resume,
            overwrite_existing=self.overwrite_existing,
            plan_steps=[
                {
                    "id": step.id,
                    "stage": step.stage,
                    "mutating": step.mutating,
                    "resumable": step.resumable,
                    "checkpoint_kind": step.checkpoint_kind,
                }
                for step in self.plan
            ],
            resume_summary=self.resume_summary,
            recovery_plan=self.recovery_plan,
        )
        self._persist_job_spec_copy()
        self.state_store.record_checkpoint(
            "executor.job.initialized",
            payload={
                "dry_run": self.dry_run,
                "resume": self.resume,
                "overwrite_existing": self.overwrite_existing,
                "reused_existing_progress": has_existing_progress,
            },
        )
        self._emit_progress(
            "executor.job.initialized",
            job_id=self.spec["job_id"],
            run_dir=str(self.run_dir),
            dry_run=self.dry_run,
            resume=self.resume,
            overwrite_existing=self.overwrite_existing,
            reused_existing_progress=has_existing_progress,
        )
        if self.resume_summary:
            self.state_store.record_checkpoint("executor.job.resume", payload=self.resume_summary)
            self._emit_progress("executor.job.resume", **self.resume_summary)
            self._emit_checkpoint("executor.job.resume", **self.resume_summary)
        if self.recovery_plan:
            self.state_store.record_checkpoint("executor.job.recovery_plan", payload=self.recovery_plan)
            self._emit_progress("executor.job.recovery_plan", recovery_steps=len(self.recovery_plan.get("steps") or []))
            self._emit_checkpoint("executor.job.recovery_plan", recovery_steps=len(self.recovery_plan.get("steps") or []))

    def _persist_job_spec_copy(self) -> None:
        spec_copy = self.run_dir / "job_spec.resolved.json"
        spec_copy.write_text(json.dumps(self.spec, indent=2, ensure_ascii=False), encoding="utf-8")

    def _existing_state(self) -> Optional[Dict[str, Any]]:
        state_path = self.run_dir / "state.json"
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _has_existing_progress(self, state: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(state, dict):
            return False
        steps = state.get("steps")
        if not isinstance(steps, dict):
            return False
        return any(
            isinstance(record, dict) and str(record.get("status") or "pending") != "pending"
            for record in steps.values()
        )

    def _resume_summary(self, state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.resume or not isinstance(state, dict):
            return None
        steps = state.get("steps")
        if not isinstance(steps, dict) or not steps:
            return None
        buckets = {
            "completed_steps": [],
            "failed_steps": [],
            "running_steps": [],
            "cancelled_steps": [],
            "pending_steps": [],
        }
        for step_id, record in steps.items():
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or "pending")
            if status == "completed":
                buckets["completed_steps"].append(step_id)
            elif status == "failed":
                buckets["failed_steps"].append(step_id)
            elif status == "running":
                buckets["running_steps"].append(step_id)
            elif status == "cancelled":
                buckets["cancelled_steps"].append(step_id)
            else:
                buckets["pending_steps"].append(step_id)
        return {
            "resumed_from_existing": True,
            **buckets,
        }

    def _recovery_plan(self, state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.resume or not isinstance(state, dict):
            return None
        steps = state.get("steps")
        checkpoints = state.get("checkpoints")
        if not isinstance(steps, dict):
            return None
        if not isinstance(checkpoints, list):
            checkpoints = []
        items: List[Dict[str, Any]] = []
        upstream_recovery: Optional[Dict[str, Any]] = None
        for step in self.plan:
            record = steps.get(step.id) or {}
            entry = self._recovery_entry_for_step(step, record, checkpoints, upstream_recovery=upstream_recovery)
            items.append(entry)
            if entry.get("action") != "reuse_completed":
                upstream_recovery = {
                    "step_id": step.id,
                    "action": entry.get("action"),
                }
        return {
            "job_id": self.spec["job_id"],
            "steps": items,
        }

    def _expected_step_artifacts(self, step_id: str, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        expected: List[Dict[str, Any]] = []
        artifact_root = self.run_dir / "artifacts"
        relative_root_by_step = {
            "md.run": artifact_root / "md",
            "chain.compile_event_table": artifact_root / "chain",
            "kmc.prepare_input": artifact_root / "kmc",
            "kmc.run": artifact_root / "kmc",
            "explain.summary": self.run_dir / "explain",
            "archive.results": self.run_dir / "archive",
        }

        def _add(path: Path) -> None:
            resolved = path.resolve()
            if any(item["path"] == resolved for item in expected):
                return
            try:
                relative = resolved.relative_to(self.run_dir)
                label = str(relative)
            except ValueError:
                label = str(resolved)
            expected.append({"label": label, "path": resolved})

        default_by_step = {
            "md.run": [
                artifact_root / "md" / "barriers.json",
                artifact_root / "md" / "md_execution.json",
                artifact_root / "md" / "md_execution.log",
            ],
            "chain.compile_event_table": [
                artifact_root / "chain" / "event_table.csv",
                artifact_root / "chain" / "kmc_barrier_map.json",
            ],
            "kmc.prepare_input": [
                artifact_root / "kmc" / "generated_kmc.in",
            ],
            "kmc.run": [
                artifact_root / "kmc" / "log.spparks",
                artifact_root / "kmc" / "diffusion.csv",
                artifact_root / "kmc" / "kmc_execution.json",
            ],
            "explain.summary": [
                self.run_dir / "explain" / "summary.md",
            ],
            "archive.results": [
                self.run_dir / "archive" / "manifest.json",
            ],
        }
        for path in default_by_step.get(step_id, []):
            _add(path)

        outputs = record.get("outputs")
        if isinstance(outputs, dict):
            for value in outputs.values():
                if not isinstance(value, str) or not value.strip():
                    continue
                looks_like_path = Path(value).is_absolute() or "/" in value or value.endswith(
                    (".json", ".md", ".log", ".csv", ".in", ".xyz", ".tsv")
                )
                if not looks_like_path:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    relative_root = relative_root_by_step.get(step_id, self.run_dir)
                    candidate = (relative_root / candidate).resolve()
                if candidate.exists() or candidate.parent.exists():
                    _add(candidate)
        return expected

    def _artifact_fingerprint(self, label: str, path: Path) -> Dict[str, Any]:
        stat_result = path.stat()
        payload: Dict[str, Any] = {
            "label": label,
            "path": str(path),
            "exists": True,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "fingerprint_mode": "sha256",
        }
        if path.is_file() and stat_result.st_size <= 5 * 1024 * 1024:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            payload["sha256"] = digest.hexdigest()
        else:
            payload["fingerprint_mode"] = "size_mtime"
        return payload

    def _artifact_snapshot(self, step_id: str, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        record = {"outputs": outputs}
        items = []
        for artifact in self._expected_step_artifacts(step_id, record):
            path = artifact["path"]
            label = artifact["label"]
            if path.exists():
                items.append(self._artifact_fingerprint(label, path))
            else:
                items.append({"label": label, "path": str(path), "exists": False})
        return items

    def _recovery_entry_for_step(
        self,
        step: Any,
        record: Dict[str, Any],
        checkpoints: List[Dict[str, Any]],
        *,
        upstream_recovery: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        previous_status = str(record.get("status") or "pending")
        step_checkpoints = [
            item for item in checkpoints
            if isinstance(item, dict) and item.get("step_id") == step.id
        ]
        latest_checkpoint = step_checkpoints[-1] if step_checkpoints else None
        expected_artifacts = self._expected_step_artifacts(step.id, record)
        artifact_snapshot = record.get("artifact_snapshot") if isinstance(record.get("artifact_snapshot"), list) else []
        snapshot_by_label = {
            str(item.get("label")): item
            for item in artifact_snapshot
            if isinstance(item, dict) and item.get("label")
        }
        missing_outputs = [
            item["label"]
            for item in expected_artifacts
            if not item["path"].exists()
        ]
        drifted_outputs: List[str] = []
        for item in expected_artifacts:
            label = item["label"]
            path = item["path"]
            snapshot = snapshot_by_label.get(label)
            if not snapshot or not path.exists():
                continue
            current = self._artifact_fingerprint(label, path)
            if snapshot.get("sha256") and current.get("sha256"):
                if snapshot.get("sha256") != current.get("sha256"):
                    drifted_outputs.append(label)
            elif snapshot.get("size") != current.get("size") or snapshot.get("mtime_ns") != current.get("mtime_ns"):
                drifted_outputs.append(label)
        artifacts_valid = not missing_outputs and not drifted_outputs
        has_checkpoint = bool(step_checkpoints)
        available_checkpoints = sorted(
            {
                str(item.get("kind"))
                for item in step_checkpoints
                if isinstance(item, dict) and item.get("kind")
            }
        )
        invalidated_by = None
        invalidated_by_action = None

        if previous_status == "completed" and artifacts_valid and upstream_recovery is None:
            action = "reuse_completed"
            decision_reason = "completed-state-validated"
        elif previous_status == "completed" and upstream_recovery is not None:
            invalidated_by = upstream_recovery.get("step_id")
            invalidated_by_action = upstream_recovery.get("action")
            if step.resumable and step.checkpoint_kind == "state_store":
                action = "restart_resumable_step"
            elif step.checkpoint_kind and has_checkpoint:
                action = "rebuild_from_checkpoint"
            else:
                action = "rerun_step"
            decision_reason = "upstream-step-replanned"
        elif previous_status == "pending":
            action = "run_pending"
            decision_reason = "step-still-pending"
        elif step.resumable and step.checkpoint_kind == "state_store":
            action = "restart_resumable_step"
            if previous_status == "completed" and missing_outputs:
                decision_reason = "completed-artifacts-missing"
            elif previous_status == "completed" and drifted_outputs:
                decision_reason = "artifact-fingerprint-mismatch"
            else:
                decision_reason = f"{previous_status}-resumable-state"
        elif step.checkpoint_kind and has_checkpoint:
            action = "rebuild_from_checkpoint"
            if previous_status == "completed" and missing_outputs:
                decision_reason = "completed-artifacts-missing"
            elif previous_status == "completed" and drifted_outputs:
                decision_reason = "artifact-fingerprint-mismatch"
            else:
                decision_reason = f"{previous_status}-checkpoint-rebuild"
        else:
            action = "rerun_step"
            if previous_status == "completed" and missing_outputs:
                decision_reason = "completed-artifacts-missing"
            elif previous_status == "completed" and drifted_outputs:
                decision_reason = "artifact-fingerprint-mismatch"
            else:
                decision_reason = f"{previous_status}-rerun"

        return {
            "step_id": step.id,
            "stage": step.stage,
            "previous_status": previous_status,
            "action": action,
            "resumable": step.resumable,
            "checkpoint_kind": step.checkpoint_kind,
            "latest_checkpoint_kind": latest_checkpoint.get("kind") if isinstance(latest_checkpoint, dict) else None,
            "available_checkpoints": available_checkpoints,
            "has_checkpoint": has_checkpoint,
            "artifacts_checked": len(expected_artifacts),
            "artifacts_valid": artifacts_valid,
            "missing_outputs": missing_outputs,
            "drifted_outputs": drifted_outputs,
            "decision_reason": decision_reason,
            "invalidated_by": invalidated_by,
            "invalidated_by_action": invalidated_by_action,
        }

    def _emit_progress(self, stage: str, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            stage,
            {
                "job_id": self.spec["job_id"],
                "run_dir": str(self.run_dir),
                "dry_run": self.dry_run,
                **payload,
            },
        )

    def _emit_checkpoint(self, stage: str, **payload: Any) -> None:
        self.state_store.record_checkpoint(stage, step_id=payload.get("step_id"), payload=payload)
        if self.checkpoint_callback is None:
            return
        self.checkpoint_callback(
            stage,
            {
                "job_id": self.spec["job_id"],
                "run_dir": str(self.run_dir),
                **payload,
            },
        )

    def _check_cancel_requested(self, step_id: str) -> None:
        if self.cancel_check is None:
            return
        decision = self.cancel_check(step_id, self.run_dir, self.state_store.read())
        if not decision:
            return
        if isinstance(decision, str):
            message = decision
        else:
            message = f"Execution cancelled before step `{step_id}`."
        raise ExecutionCancelled(message)

    def _heartbeat(self, step_id: str, *, pid: Optional[int] = None, detail: Optional[str] = None) -> None:
        self._check_cancel_requested(step_id)
        self.state_store.heartbeat(step_id, pid=pid, detail=detail)
        self._emit_progress("executor.step.heartbeat", step_id=step_id, pid=pid, detail=detail)

    def _resume_step_if_needed(self, step_id: str, *, resumable: bool, checkpoint_kind: Optional[str]) -> bool:
        if not self.resume:
            return False
        current_status = self.state_store.step_status(step_id)
        recovery_entry = self._recovery_entry(step_id)
        recovery_action = recovery_entry.get("action") if recovery_entry else None
        decision_reason = recovery_entry.get("decision_reason") if recovery_entry else None
        missing_outputs = list(recovery_entry.get("missing_outputs") or []) if recovery_entry else []
        drifted_outputs = list(recovery_entry.get("drifted_outputs") or []) if recovery_entry else []
        invalidated_by = recovery_entry.get("invalidated_by") if recovery_entry else None
        if current_status == "pending" and recovery_action in {None, "run_pending"}:
            return False
        if current_status == "completed" and recovery_action in {None, "reuse_completed"}:
            return False
        self.state_store.prepare_resume(
            step_id,
            from_status=current_status,
            reason=decision_reason or recovery_action or "resume-requested",
        )
        self._emit_progress(
            "executor.step.resume",
            step_id=step_id,
            previous_status=current_status,
            resumable=resumable,
            checkpoint_kind=checkpoint_kind,
            recovery_action=recovery_action,
            decision_reason=decision_reason,
            missing_outputs=missing_outputs,
            drifted_outputs=drifted_outputs,
            invalidated_by=invalidated_by,
        )
        self._emit_checkpoint(
            "executor.step.resume",
            step_id=step_id,
            previous_status=current_status,
            resumable=resumable,
            checkpoint_kind=checkpoint_kind,
            recovery_action=recovery_action,
            decision_reason=decision_reason,
            missing_outputs=missing_outputs,
            drifted_outputs=drifted_outputs,
            invalidated_by=invalidated_by,
        )
        return True

    def run(self) -> Path:
        self._emit_progress("executor.job.start", plan_steps=len(self.plan))
        for step in self.plan:
            try:
                self._check_cancel_requested(step.id)
                self._resume_step_if_needed(
                    step.id,
                    resumable=step.resumable,
                    checkpoint_kind=step.checkpoint_kind,
                )
                if self.state_store.step_status(step.id) == "completed":
                    self._emit_progress("executor.step.skipped", step_id=step.id, reason="already-completed")
                    continue
                self._emit_progress(
                    "executor.step.start",
                    step_id=step.id,
                    step_stage=step.stage,
                    resumable=step.resumable,
                    checkpoint_kind=step.checkpoint_kind,
                )
                self.state_store.start(
                    step.id,
                    resumable=step.resumable,
                    checkpoint_kind=step.checkpoint_kind,
                    stage=step.stage,
                    recovery_action=self._recovery_action_for_step(step.id),
                    recovery_reason=self._recovery_reason_for_step(step.id),
                    missing_outputs=self._recovery_missing_outputs_for_step(step.id),
                    invalidated_by=self._recovery_invalidated_by_for_step(step.id),
                )
                self._emit_checkpoint(
                    "executor.step.start",
                    step_id=step.id,
                    step_stage=step.stage,
                    resumable=step.resumable,
                    checkpoint_kind=step.checkpoint_kind,
                    recovery_action=self._recovery_action_for_step(step.id),
                    decision_reason=self._recovery_reason_for_step(step.id),
                    missing_outputs=self._recovery_missing_outputs_for_step(step.id),
                    invalidated_by=self._recovery_invalidated_by_for_step(step.id),
                )
                outputs = self._execute_step(step.id)
                artifact_snapshot = self._artifact_snapshot(step.id, outputs)
                self.state_store.complete(step.id, outputs, artifact_snapshot=artifact_snapshot)
                self._emit_progress("executor.step.complete", step_id=step.id, outputs=outputs)
                self._emit_checkpoint(
                    "executor.step.complete",
                    step_id=step.id,
                    outputs=outputs,
                    artifact_snapshot_count=len(artifact_snapshot),
                )
                if step.checkpoint_kind:
                    self._emit_checkpoint(
                        "executor.step.checkpoint_ready",
                        step_id=step.id,
                        checkpoint_kind=step.checkpoint_kind,
                        outputs=outputs,
                        recovery_action=self._recovery_action_for_step(step.id),
                        drifted_outputs=self._recovery_drifted_outputs_for_step(step.id),
                    )
            except ExecutionCancelled as exc:
                self.state_store.cancel(step.id, str(exc))
                self._emit_progress("executor.step.cancelled", step_id=step.id, reason=str(exc))
                self._emit_checkpoint("executor.step.cancelled", step_id=step.id, reason=str(exc))
                self._emit_progress("executor.job.cancelled", step_id=step.id, reason=str(exc))
                self._emit_checkpoint("executor.job.cancelled", step_id=step.id, reason=str(exc))
                raise
            except Exception as exc:  # noqa: BLE001
                self.state_store.fail(step.id, str(exc))
                self._emit_progress("executor.step.failed", step_id=step.id, error=str(exc))
                self._emit_checkpoint("executor.step.failed", step_id=step.id, error=str(exc))
                raise

        final_state = self.state_store.read()
        if (self.run_dir / "explain").exists():
            write_summary(self.run_dir, self.spec, final_state)
        if (self.run_dir / "archive").exists():
            write_manifest(self.run_dir, self.spec, final_state)
        self._emit_progress("executor.job.complete", completed_steps=len(self.plan))
        self._emit_checkpoint("executor.job.complete", completed_steps=len(self.plan))
        return self.run_dir

    def _recovery_action_for_step(self, step_id: str) -> Optional[str]:
        entry = self._recovery_entry(step_id)
        return entry.get("action") if entry else None

    def _recovery_reason_for_step(self, step_id: str) -> Optional[str]:
        entry = self._recovery_entry(step_id)
        return entry.get("decision_reason") if entry else None

    def _recovery_missing_outputs_for_step(self, step_id: str) -> List[str]:
        entry = self._recovery_entry(step_id)
        return list(entry.get("missing_outputs") or []) if entry else []

    def _recovery_drifted_outputs_for_step(self, step_id: str) -> List[str]:
        entry = self._recovery_entry(step_id)
        return list(entry.get("drifted_outputs") or []) if entry else []

    def _recovery_invalidated_by_for_step(self, step_id: str) -> Optional[str]:
        entry = self._recovery_entry(step_id)
        return entry.get("invalidated_by") if entry else None

    def _recovery_entry(self, step_id: str) -> Optional[Dict[str, Any]]:
        if not self.recovery_plan:
            return None
        for item in self.recovery_plan.get("steps") or []:
            if item.get("step_id") == step_id:
                return item
        return None

    def _execute_step(self, step_id: str) -> Dict:
        if step_id == "md.run":
            return self._step_md_run()
        if step_id == "chain.compile_event_table":
            return self._step_compile_event_table()
        if step_id == "kmc.prepare_input":
            return self._step_kmc_prepare_input()
        if step_id == "kmc.run":
            return self._step_kmc_run()
        if step_id == "archive.results":
            manifest_path = write_manifest(self.run_dir, self.spec, self.state_store.read())
            return {"manifest": str(manifest_path)}
        if step_id == "explain.summary":
            summary_path = write_summary(self.run_dir, self.spec, self.state_store.read())
            return {"summary": str(summary_path)}
        raise ExecutionError(f"Unknown step: {step_id}")

    def _step_md_run(self) -> Dict:
        artifact_dir = self.artifacts_dir / "md"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        md = self.spec.get("md", {})
        source = md.get("barriers_source")
        if not source:
            raise ExecutionError("md.barriers_source is required for current MVP")
        command = md.get("command", [])
        working_dir = Path(md.get("working_dir") or self.spec["_meta"]["job_spec_dir"]).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "barriers.json"

        command = self.spec["md"].get("command", [])
        environment = md.get("environment", {})
        result = {
            "engine": md.get("engine", "md"),
            "command": command,
            "environment": environment,
            "workdir": str(working_dir),
            "barriers_file": str(destination),
            "mode": "dry-run" if self.dry_run else "file-backed",
        }

        should_execute = (not self.dry_run) and command and _command_exists(command[0])
        log_path = artifact_dir / "md_execution.log"
        if should_execute:
            run_output = _run_command(
                command,
                working_dir,
                environment=environment,
                heartbeat=lambda pid: self._heartbeat(
                    "md.run",
                    pid=pid,
                    detail=f"MD command is still running in {working_dir}",
                ),
                live_log_path=log_path,
            )
            result.update(run_output)
            if run_output["returncode"] != 0:
                raise ExecutionError(f"MD command failed with return code {run_output['returncode']}")
            result["mode"] = "executed"
        else:
            log_path.write_text(
                "MD command was not executed in this environment.\n",
                encoding="utf-8",
            )
            if command:
                result["reason"] = "dry-run or missing executable"

        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = working_dir / source_path
        if not source_path.exists():
            raise ExecutionError(f"MD barrier file not found after MD stage: {source_path}")
        shutil.copy2(source_path, destination)
        neb_campaign_dir = working_dir / "neb_campaign"
        if neb_campaign_dir.exists() and neb_campaign_dir.is_dir():
            archived_neb_dir = artifact_dir / "neb_campaign"
            if archived_neb_dir.exists():
                shutil.rmtree(archived_neb_dir)
            shutil.copytree(neb_campaign_dir, archived_neb_dir)
            result["neb_campaign"] = str(archived_neb_dir)
        result["log"] = str(log_path)
        write_json(artifact_dir / "md_execution.json", result)
        return result

    def _step_compile_event_table(self) -> Dict:
        barrier_path = self.artifacts_dir / "md" / "barriers.json"
        if not barrier_path.exists():
            raise ExecutionError("MD barrier file not found")
        barrier_payload = json.loads(barrier_path.read_text(encoding="utf-8"))
        temperature_k = float(self.spec["kmc"]["temperature_k"])
        rows = compile_event_table(barrier_payload, temperature_k)
        barrier_map = derive_kmc_barrier_map(rows)

        artifact_dir = self.artifacts_dir / "chain"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        event_table_path = artifact_dir / "event_table.csv"
        barrier_map_path = artifact_dir / "kmc_barrier_map.json"
        write_event_table_csv(rows, event_table_path)
        write_json(barrier_map_path, {"temperature_k": temperature_k, "barriers": barrier_map})
        return {
            "event_table": str(event_table_path),
            "kmc_barrier_map": str(barrier_map_path),
            "events": len(rows),
        }

    def _kmc_barrier_inputs(self) -> Dict:
        kmc = self.spec["kmc"]
        temperature_k = float(kmc["temperature_k"])
        if self.spec["mode"] == "md_to_kmc_chain":
            barrier_map_payload = json.loads((self.artifacts_dir / "chain" / "kmc_barrier_map.json").read_text(encoding="utf-8"))
            barrier_map = barrier_map_payload["barriers"]
            event_table_path = self.artifacts_dir / "chain" / "event_table.csv"
        else:
            rows = synthetic_event_rows(kmc["template"]["precomputed_barriers"], temperature_k)
            event_table_path = self.artifacts_dir / "kmc" / "event_table.csv"
            write_event_table_csv(rows, event_table_path)
            barrier_map = derive_kmc_barrier_map(rows)
            write_json(self.artifacts_dir / "kmc" / "kmc_barrier_map.json", {"temperature_k": temperature_k, "barriers": barrier_map})
        return {"barrier_map": barrier_map, "temperature_k": temperature_k, "event_table_path": str(event_table_path)}

    def _step_kmc_prepare_input(self) -> Dict:
        artifact_dir = self.artifacts_dir / "kmc"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        kmc_inputs = self._kmc_barrier_inputs()
        template = self.spec["kmc"]["template"]

        cluster_ref = None
        if template.get("cluster_xyz"):
            cluster_ref = stage_support_file(template["cluster_xyz"], artifact_dir)

        staged_assets = []
        for asset in template.get("potential_assets", []):
            staged_assets.append(stage_support_file(asset, artifact_dir))

        kmc_text = render_kmc_input(template, kmc_inputs["barrier_map"], kmc_inputs["temperature_k"], cluster_ref=cluster_ref)
        input_path = artifact_dir / "generated_kmc.in"
        input_path.write_text(kmc_text, encoding="utf-8")
        return {
            "generated_input": str(input_path),
            "cluster_ref": cluster_ref,
            "staged_assets": staged_assets,
            "event_table": kmc_inputs["event_table_path"],
        }

    def _step_kmc_run(self) -> Dict:
        artifact_dir = self.artifacts_dir / "kmc"
        input_path = artifact_dir / "generated_kmc.in"
        if not input_path.exists():
            raise ExecutionError("KMC input not prepared")

        command = self.spec["kmc"].get("command", [])
        environment = self.spec["kmc"].get("environment", {})
        result = {
            "engine": self.spec["kmc"].get("engine", "kmc"),
            "command": command,
            "environment": environment,
            "workdir": str(artifact_dir),
            "mode": "dry-run" if self.dry_run else "simulated",
        }

        diffusion_csv = artifact_dir / "diffusion.csv"
        log_path = artifact_dir / "log.spparks"

        should_execute = (not self.dry_run) and command and _command_exists(command[0])
        if should_execute:
            run_output = _run_command(
                command,
                artifact_dir,
                stdin_path=input_path,
                environment=environment,
                heartbeat=lambda pid: self._heartbeat(
                    "kmc.run",
                    pid=pid,
                    detail=f"KMC command is still running in {artifact_dir}",
                ),
                live_log_path=log_path,
            )
            result.update(run_output)
            if run_output["returncode"] != 0:
                raise ExecutionError(f"KMC command failed with return code {run_output['returncode']}")
            result["mode"] = "executed"
            result["diffusion_mode"] = "engine-generated" if diffusion_csv.exists() else "missing-from-engine"
        else:
            diffusion_csv.write_text(
                "No.,jumps,simulation_time,jump frequency,diffusion coefficient\n0,3,0.000366,8196.72,0.00476899\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "Simulated KMC run. Command not executed in this environment.\n",
                encoding="utf-8",
            )
            result["reason"] = "dry-run or missing executable"
            result["diffusion_mode"] = "simulated"

        write_json(artifact_dir / "kmc_execution.json", result)
        return {
            "log": str(log_path),
            "diffusion_csv": str(diffusion_csv) if diffusion_csv.exists() else None,
            "execution": str(artifact_dir / "kmc_execution.json"),
        }


def run_job(
    spec_path: str,
    output_dir: str,
    dry_run: bool = False,
    *,
    resume: bool = False,
    overwrite_existing: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
) -> Path:
    return JobExecutor(
        spec_path=spec_path,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        overwrite_existing=overwrite_existing,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        checkpoint_callback=checkpoint_callback,
    ).run()
