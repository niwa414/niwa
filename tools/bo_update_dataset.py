#!/usr/bin/env python3
"""Update BO/active-learning dataset from orchestrator runs or bootstrap cases."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def resolve_dataset_paths(repo_root: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_root = repo_root / str(config.get("output_root") or "outputs/bo/default")
    return output_root / "dataset.jsonl", output_root / "dataset.csv", output_root / "dataset_summary.json"


def configured_fidelity_names(config: dict[str, Any]) -> list[str]:
    levels = ((config.get("fidelity") or {}).get("levels") or [])
    names = []
    for level in levels:
        name = str((level or {}).get("name") or "").strip()
        if name:
            names.append(name)
    return names


def detect_fidelity(
    config: dict[str, Any],
    case_meta: dict[str, Any],
    explicit_fidelity: str | None,
    role_hint: str | None,
) -> str:
    names = configured_fidelity_names(config)
    default_name = str((config.get("fidelity") or {}).get("default") or (names[0] if names else "unknown"))

    candidates = []
    if explicit_fidelity:
        candidates.append(str(explicit_fidelity))
    trade = (case_meta.get("metadata") or {}).get("trade_study") or {}
    bo = trade.get("bo") or {}
    if bo.get("fidelity") is not None:
        candidates.append(str(bo.get("fidelity")))
    if trade.get("fidelity") is not None:
        candidates.append(str(trade.get("fidelity")))
    if role_hint:
        candidates.append(str(role_hint))

    for cand in candidates:
        for name in names:
            if cand == name or name in cand:
                return name
    return default_name


def pick_load_metric(metrics: dict[str, Any], source: dict[str, Any], load_keys: list[str]) -> tuple[str | None, float | None]:
    for key in load_keys:
        value = to_float(metrics.get(key))
        if value is not None:
            return key, value
        value = to_float(source.get(key))
        if value is not None:
            return key, value
    return None, None


def make_entry(
    *,
    repo_root: Path,
    config: dict[str, Any],
    case_id: str,
    source_label: str,
    source_run_id: str | None,
    case_ref: str | None,
    explicit_fidelity: str | None,
    role_hint: str | None,
) -> dict[str, Any] | None:
    metrics_path = repo_root / "outputs" / case_id / "analysis" / "metrics.json"
    source_path = repo_root / "outputs" / case_id / "analysis" / "metrics_source.json"
    passfail_path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"

    metrics = load_json(metrics_path)
    if not metrics:
        return None
    source_metrics = load_json(source_path)
    passfail = load_json(passfail_path)

    case_meta: dict[str, Any] = {}
    if case_ref:
        p = Path(case_ref)
        if not p.is_absolute():
            p = repo_root / case_ref
        case_meta = load_json(p)

    fidelity = detect_fidelity(config, case_meta, explicit_fidelity, role_hint)

    knob_name = str(metrics.get("knob_name") or "shift")
    knob_value = to_float(metrics.get("knob_value"))
    if knob_value is None:
        trade = (case_meta.get("metadata") or {}).get("trade_study") or {}
        knob_value = to_float(trade.get("knob_value"))

    objective_cfg = config.get("objective") or {}
    constraints_cfg = config.get("constraints") or {}
    weights = objective_cfg.get("weights") or {}

    compression = to_float(metrics.get(objective_cfg.get("compression_key", "compression_ratio")))
    recapture = to_float(metrics.get(objective_cfg.get("recapture_key", "recapture_efficiency")))
    if recapture is None:
        recapture = to_float(source_metrics.get("eta_recaptured"))
    tilt = to_float(metrics.get(objective_cfg.get("tilt_key", "tilt_amp_max")))

    energy_key = str(constraints_cfg.get("energy_ok_key") or "energy_accounting_ok")
    energy_ok = bool(metrics.get(energy_key))

    load_keys = [str(k) for k in (constraints_cfg.get("load_metric_keys") or [])]
    load_key, load_value = pick_load_metric(metrics, source_metrics, load_keys)
    load_max = to_float(constraints_cfg.get("load_max"))

    load_penalty = 0.0
    load_ok = True
    if load_max is not None and load_value is not None:
        load_ok = load_value <= load_max
        load_penalty = max(0.0, (load_value - load_max) / max(abs(load_max), 1e-12))

    w_recapture = float(weights.get("recapture", 0.0))
    w_tilt = float(weights.get("tilt", 0.0))
    w_load = float(weights.get("load_penalty", 0.0))

    objective_score = None
    if compression is not None and recapture is not None and tilt is not None:
        objective_score = compression + w_recapture * recapture - w_tilt * tilt - w_load * load_penalty

    pass_status = str(passfail.get("status") or passfail.get("result") or "UNKNOWN").upper()
    pass_gate = pass_status == "PASS"

    feasible = bool(energy_ok and load_ok and pass_gate)

    return {
        "updated_at": now_iso(),
        "source": source_label,
        "source_run_id": source_run_id,
        "case_id": case_id,
        "case_ref": case_ref,
        "fidelity": fidelity,
        "knob_name": knob_name,
        "knob_value": knob_value,
        "compression_ratio": compression,
        "recapture_efficiency": recapture,
        "tilt_amp_max": tilt,
        "load_metric_key": load_key,
        "load_metric_value": load_value,
        "load_metric_max": load_max,
        "load_penalty": load_penalty,
        "energy_accounting_ok": energy_ok,
        "passfail_status": pass_status,
        "pass_gate": pass_gate,
        "feasible": feasible,
        "objective_score": objective_score,
        "metrics_path": str(metrics_path),
        "metrics_source_path": str(source_path),
        "passfail_path": str(passfail_path),
    }


def collect_from_run_ids(repo_root: Path, config: dict[str, Any], run_ids: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for run_id in run_ids:
        state_path = repo_root / "outputs" / "orchestrator" / run_id / "state.json"
        state = load_json(state_path)
        if not state:
            continue
        for job in state.get("jobs", []) or []:
            case_id = str(job.get("case_id") or "").strip()
            if not case_id:
                continue
            entry = make_entry(
                repo_root=repo_root,
                config=config,
                case_id=case_id,
                source_label="orchestrator",
                source_run_id=run_id,
                case_ref=(job.get("case_ref") if isinstance(job.get("case_ref"), str) else None),
                explicit_fidelity=None,
                role_hint=(job.get("role") if isinstance(job.get("role"), str) else None),
            )
            if entry:
                entries.append(entry)
    return entries


def collect_from_bootstrap(repo_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    bootstrap = ((config.get("bootstrap") or {}).get("cases") or [])
    for item in bootstrap:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "").strip()
        if not case_id:
            continue
        entry = make_entry(
            repo_root=repo_root,
            config=config,
            case_id=case_id,
            source_label="bootstrap",
            source_run_id=None,
            case_ref=None,
            explicit_fidelity=(item.get("fidelity") if isinstance(item.get("fidelity"), str) else None),
            role_hint=None,
        )
        if entry:
            entries.append(entry)
    return entries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "updated_at",
        "source",
        "source_run_id",
        "case_id",
        "fidelity",
        "knob_name",
        "knob_value",
        "compression_ratio",
        "recapture_efficiency",
        "tilt_amp_max",
        "load_metric_key",
        "load_metric_value",
        "load_metric_max",
        "load_penalty",
        "energy_accounting_ok",
        "passfail_status",
        "feasible",
        "objective_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fidelity: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None

    for row in rows:
        fid = str(row.get("fidelity") or "unknown")
        info = by_fidelity.setdefault(fid, {"count": 0, "feasible": 0})
        info["count"] += 1
        if bool(row.get("feasible")):
            info["feasible"] += 1

        score = to_float(row.get("objective_score"))
        if score is None:
            continue
        if best is None or score > to_float(best.get("objective_score") or -1e99):
            best = {
                "case_id": row.get("case_id"),
                "fidelity": fid,
                "objective_score": score,
                "knob_value": row.get("knob_value"),
            }

    return {
        "updated_at": now_iso(),
        "rows": len(rows),
        "by_fidelity": by_fidelity,
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update BO dataset from orchestrator runs and bootstrap cases.")
    parser.add_argument("--config", required=True, help="Path to bo-config.json")
    parser.add_argument("--run-id", action="append", default=[], help="Orchestrator run_id to import (repeatable)")
    parser.add_argument("--bootstrap", action="store_true", help="Import bootstrap cases defined in config")
    parser.add_argument("--dry-run", action="store_true", help="Do not write dataset files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    config = load_json(config_path)
    if not config:
        raise SystemExit(f"Invalid config: {config_path}")

    dataset_path, csv_path, summary_path = resolve_dataset_paths(repo_root, config)

    existing_rows = read_jsonl(dataset_path)
    by_case: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
            by_case[case_id] = row

    new_rows: list[dict[str, Any]] = []
    if args.bootstrap:
        new_rows.extend(collect_from_bootstrap(repo_root, config))
    if args.run_id:
        new_rows.extend(collect_from_run_ids(repo_root, config, args.run_id))

    for row in new_rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        by_case[case_id] = row

    final_rows = sorted(by_case.values(), key=lambda r: (str(r.get("fidelity")), to_float(r.get("knob_value") or 0.0)))
    summary = summarize(final_rows)

    print(f"[bo-dataset] config={config_path}")
    print(f"[bo-dataset] existing={len(existing_rows)} imported={len(new_rows)} final={len(final_rows)}")
    if summary.get("best"):
        print(f"[bo-dataset] best={summary['best']}")

    if args.dry_run:
        return

    write_jsonl(dataset_path, final_rows)
    write_csv(csv_path, final_rows)
    write_json(summary_path, summary)

    print(f"[bo-dataset] dataset_jsonl={dataset_path}")
    print(f"[bo-dataset] dataset_csv={csv_path}")
    print(f"[bo-dataset] summary={summary_path}")


if __name__ == "__main__":
    main()
