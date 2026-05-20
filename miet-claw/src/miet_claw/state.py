import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path, job_id: str, mode: str, plan_step_ids: List[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "job_id": job_id,
                "mode": mode,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "status": "pending",
                "job": {},
                "checkpoints": [],
                "steps": {step_id: {"status": "pending"} for step_id in plan_step_ids},
            }
            self._write()

        self.state.setdefault("status", "pending")
        self.state.setdefault("job", {})
        self.state.setdefault("checkpoints", [])
        for step_id in plan_step_ids:
            self.state.setdefault("steps", {}).setdefault(step_id, {"status": "pending"})
        self._write()

    def _write(self) -> None:
        self.state["updated_at"] = utc_now()
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def read(self) -> Dict[str, Any]:
        return self.state

    def step_record(self, step_id: str) -> Dict[str, Any]:
        return self.state.setdefault("steps", {}).setdefault(step_id, {"status": "pending"})

    def step_status(self, step_id: str) -> str:
        return self.state["steps"].get(step_id, {}).get("status", "pending")

    def checkpoints_for_step(self, step_id: str) -> List[Dict[str, Any]]:
        return [
            item
            for item in self.state.get("checkpoints", [])
            if isinstance(item, dict)
            and (
                item.get("step_id") == step_id
                or (
                    isinstance(item.get("payload"), dict)
                    and item.get("payload", {}).get("step_id") == step_id
                )
            )
        ]

    def latest_checkpoint(self, step_id: str, *, kind_prefix: Optional[str] = None) -> Optional[Dict[str, Any]]:
        items = self.checkpoints_for_step(step_id)
        if kind_prefix:
            items = [item for item in items if str(item.get("kind") or "").startswith(kind_prefix)]
        if not items:
            return None
        return items[-1]

    def update_job(self, **fields: Any) -> None:
        job = self.state.setdefault("job", {})
        for key, value in fields.items():
            if value is not None:
                job[key] = value
        self._write()

    def record_checkpoint(
        self,
        kind: str,
        *,
        step_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "kind": kind,
            "timestamp": utc_now(),
        }
        if step_id is not None:
            entry["step_id"] = step_id
        if payload:
            entry["payload"] = payload
        self.state.setdefault("checkpoints", []).append(entry)
        self._write()

    def start(self, step_id: str, **fields: Any) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        timestamp = utc_now()
        attempts = int(record.get("attempts") or 0) + 1
        record.update(
            {
                "status": "running",
                "started_at": timestamp,
                "heartbeat_at": timestamp,
                "attempts": attempts,
            }
        )
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        self.state["status"] = "running"
        self._write()

    def heartbeat(self, step_id: str, **fields: Any) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        record.update({"status": "running", "heartbeat_at": utc_now()})
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        self._write()

    def complete(
        self,
        step_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        *,
        artifact_snapshot: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        record.update({"status": "completed", "completed_at": utc_now()})
        if outputs is not None:
            record["outputs"] = outputs
        if artifact_snapshot is not None:
            record["artifact_snapshot"] = artifact_snapshot
        if all(step.get("status") == "completed" for step in self.state.get("steps", {}).values()):
            self.state["status"] = "completed"
        self._write()

    def fail(self, step_id: str, error: str) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        record.update({"status": "failed", "failed_at": utc_now(), "error": error})
        self.state["status"] = "failed"
        self._write()

    def cancel(self, step_id: str, reason: str) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        record.update({"status": "cancelled", "cancelled_at": utc_now(), "error": reason})
        self.state["status"] = "cancelled"
        self._write()

    def prepare_resume(self, step_id: str, *, from_status: str, reason: str) -> None:
        record = self.state["steps"].setdefault(step_id, {})
        retry_count = int(record.get("retry_count") or 0) + 1
        record.update(
            {
                "status": "pending",
                "resume_from_status": from_status,
                "resume_reason": reason,
                "resumed_at": utc_now(),
                "retry_count": retry_count,
            }
        )
        self.state["status"] = "running"
        self._write()
