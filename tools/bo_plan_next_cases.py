#!/usr/bin/env python3
"""Plan next simulation cases using constrained Bayesian optimization and multi-fidelity quotas."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


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


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def sanitize_token(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            allowed.append(ch)
        else:
            allowed.append("-")
    cleaned = "".join(allowed).strip("-")
    return cleaned or "token"


def format_knob_token(value: float, scale: int, width: int) -> str:
    sign = "p" if value >= 0 else "n"
    mag = int(round(abs(value) * scale))
    return f"{sign}{mag:0{width}d}"


def deep_replace_strings(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [deep_replace_strings(v, old, new) for v in obj]
    if isinstance(obj, dict):
        return {k: deep_replace_strings(v, old, new) for k, v in obj.items()}
    return obj


def set_flag_value(cmd: list[Any], flag: str, value: str) -> None:
    for i in range(len(cmd) - 1):
        if str(cmd[i]) == flag:
            cmd[i + 1] = value


def set_nested_value(root: Any, path_tokens: list[Any], value: Any) -> bool:
    cur = root
    for idx, token in enumerate(path_tokens):
        last = idx == len(path_tokens) - 1
        if isinstance(token, int):
            if not isinstance(cur, list) or token < 0 or token >= len(cur):
                return False
            if last:
                cur[token] = value
                return True
            cur = cur[token]
            continue

        key = str(token)
        if not isinstance(cur, dict) or key not in cur:
            return False
        if last:
            cur[key] = value
            return True
        cur = cur[key]
    return False


def resolve_dataset_path(repo_root: Path, config: dict[str, Any]) -> Path:
    output_root = repo_root / str(config.get("output_root") or "outputs/bo/default")
    return output_root / "dataset.jsonl"


def parse_levels(config: dict[str, Any]) -> list[dict[str, Any]]:
    levels = ((config.get("fidelity") or {}).get("levels") or [])
    out: list[dict[str, Any]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        name = str(level.get("name") or "").strip()
        if not name:
            continue
        out.append(level)
    return out


def estimate_feasible_probability(grid: np.ndarray, rows: list[dict[str, Any]], length_scale: float) -> np.ndarray:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = to_float(row.get("knob_value"))
        if x is None:
            continue
        y = row.get("feasible")
        if y is None:
            continue
        xs.append(float(x))
        ys.append(1.0 if bool(y) else 0.0)

    if not xs:
        return np.full(grid.shape, 0.5, dtype=float)

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    if len(x_arr) == 1:
        return np.full(grid.shape, y_arr[0], dtype=float)

    ls = max(float(length_scale), 1e-9)
    diff = grid[:, None] - x_arr[None, :]
    weights = np.exp(-0.5 * (diff / ls) ** 2) + 1e-12
    probs = (weights * y_arr[None, :]).sum(axis=1) / weights.sum(axis=1)
    return np.clip(probs, 0.0, 1.0)


def fit_gp_predict(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if train_x.size < 2:
        mean = float(np.mean(train_y)) if train_y.size else 0.0
        std = float(np.std(train_y) + 0.25) if train_y.size else 1.0
        return np.full(pred_x.shape, mean, dtype=float), np.full(pred_x.shape, std, dtype=float), "fallback_insufficient"

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.01, length_scale_bounds=(1e-4, 1.0)) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2)
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )

    try:
        model.fit(train_x.reshape(-1, 1), train_y)
        mu, std = model.predict(pred_x.reshape(-1, 1), return_std=True)
        return mu.astype(float), std.astype(float), "gp"
    except Exception:
        mean = float(np.mean(train_y))
        std = float(np.std(train_y) + 0.25)
        return np.full(pred_x.shape, mean, dtype=float), np.full(pred_x.shape, std, dtype=float), "fallback_fit_error"


def mask_close(values: np.ndarray, existing: list[float], tol: float) -> np.ndarray:
    if not existing:
        return np.zeros(values.shape, dtype=bool)
    arr = np.array(existing, dtype=float)
    return np.any(np.abs(values[:, None] - arr[None, :]) <= tol, axis=1)


def build_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def patch_case_for_candidate(
    *,
    repo_root: Path,
    template_case_path: Path,
    output_case_dir: Path,
    config: dict[str, Any],
    level: dict[str, Any],
    batch_id: str,
    ordinal: int,
    knob_value: float,
    prediction: dict[str, Any],
) -> tuple[Path, str]:
    template_case = load_json(template_case_path)
    if not template_case:
        raise RuntimeError(f"Invalid template case: {template_case_path}")

    template_dir = template_case_path.parent
    if output_case_dir.exists():
        raise RuntimeError(f"Case directory already exists: {output_case_dir}")
    output_case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_dir, output_case_dir)

    case_path = output_case_dir / "case.json"
    case_data = load_json(case_path)
    if not case_data:
        raise RuntimeError(f"Generated case.json missing: {case_path}")

    old_id = str(case_data.get("id") or template_dir.name)
    campaign = sanitize_token(str(config.get("campaign_name") or "campaign"))
    fid = sanitize_token(str(level.get("name") or "fid"))
    scale = int((config.get("knob") or {}).get("token_scale", 10000))
    width = int((config.get("knob") or {}).get("token_width", 5))
    knob_token = format_knob_token(knob_value, scale, width)
    batch_tag = batch_id.replace("-", "")
    new_id = f"{campaign}-bo-{fid}-b{batch_tag}-{ordinal:02d}-{knob_token}"

    case_data = deep_replace_strings(case_data, old_id, new_id)
    case_data["id"] = new_id
    case_data["description"] = f"BO candidate {ordinal} ({level.get('name')}), knob shift={knob_value:.6f}."

    md = case_data.setdefault("metadata", {})
    trade = md.setdefault("trade_study", {})
    trade["knob_name"] = str((config.get("knob") or {}).get("name") or "shift")
    trade["knob_value"] = float(knob_value)
    trade["role"] = "bo_candidate"
    trade["bo"] = {
        "fidelity": str(level.get("name") or "unknown"),
        "batch_id": batch_id,
        "ordinal": ordinal,
        "predicted_objective": prediction.get("predicted_objective"),
        "predicted_std": prediction.get("predicted_std"),
        "predicted_feasible_prob": prediction.get("predicted_feasible_prob"),
        "acquisition": prediction.get("acquisition"),
        "planner_model": prediction.get("planner_model"),
    }

    knob_name = str((config.get("knob") or {}).get("name") or "shift")
    knob_str = f"{knob_value:.10f}".rstrip("0").rstrip(".")

    for stage in (case_data.get("run") or []):
        cmd = stage.get("cmd")
        if isinstance(cmd, list):
            set_flag_value(cmd, "--run-tag", new_id)
            set_flag_value(cmd, "--source-case-id", new_id)

    for stage in (case_data.get("analyze") or []):
        cmd = stage.get("cmd")
        if isinstance(cmd, list):
            set_flag_value(cmd, "--knob-name", knob_name)
            set_flag_value(cmd, "--knob-value", knob_str)
            set_flag_value(cmd, "--label", f"bo_{level.get('name')}_{ordinal:02d}")
            set_flag_value(cmd, "--source-case-id", new_id)

    case_path.write_text(json.dumps(case_data, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    patch_cfg = (level.get("case_patch") or {})
    knob_file = patch_cfg.get("knob_file_relpath")
    knob_path = patch_cfg.get("knob_json_path") or []
    if knob_file and knob_path:
        knob_file_path = output_case_dir / str(knob_file)
        knob_data = load_json(knob_file_path)
        if knob_data:
            base_abs = to_float(patch_cfg.get("absolute_base_value"))
            if base_abs is None:
                base_abs = to_float((config.get("knob") or {}).get("absolute_base_value"))
            target = float(knob_value) if base_abs is None else float(base_abs + knob_value)
            ok = set_nested_value(knob_data, list(knob_path), target)
            if not ok:
                raise RuntimeError(f"Failed to patch knob path {knob_path} in {knob_file_path}")
            knob_file_path.write_text(json.dumps(knob_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return case_path, new_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan next BO/active-learning cases and emit orchestrator plan.")
    parser.add_argument("--config", required=True, help="Path to bo-config.json")
    parser.add_argument("--batch-size", type=int, default=None, help="Override planner.batch_size")
    parser.add_argument("--batch-id", default=None, help="Optional batch id; default UTC timestamp")
    parser.add_argument("--output-plan", default=None, help="Optional output path for orchestrator plan JSON")
    parser.add_argument("--dry-run", action="store_true", help="Only compute suggestions; do not create cases/plan")
    parser.add_argument("--random-seed", type=int, default=None, help="Override planner.random_seed")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    config = load_json(config_path)
    if not config:
        raise SystemExit(f"Invalid config: {config_path}")

    levels = parse_levels(config)
    if not levels:
        raise SystemExit("No fidelity levels configured")

    planner_cfg = config.get("planner") or {}
    batch_size = int(args.batch_size if args.batch_size is not None else planner_cfg.get("batch_size", 3))
    beta = float(planner_cfg.get("beta", 2.0))
    min_sep = float((config.get("knob") or {}).get("min_separation", 0.001))
    low = float((config.get("knob") or {}).get("relative_min", -0.02))
    high = float((config.get("knob") or {}).get("relative_max", 0.02))
    grid_points = int((config.get("knob") or {}).get("grid_points", 101))
    feas_ls = float(planner_cfg.get("feasibility_length_scale", 0.008))
    warmup = int(planner_cfg.get("warmup_random_points", 3))
    seed = int(args.random_seed if args.random_seed is not None else planner_cfg.get("random_seed", 42))
    rng = random.Random(seed)

    dataset_path = resolve_dataset_path(repo_root, config)
    rows = read_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"Dataset not found or empty: {dataset_path}. Run bo_update_dataset.py first.")

    usable_rows = []
    for row in rows:
        x = to_float(row.get("knob_value"))
        y = to_float(row.get("objective_score"))
        if x is None or y is None:
            continue
        usable_rows.append(row)

    if not usable_rows:
        raise SystemExit("No usable rows with knob_value and objective_score")

    grid = np.linspace(low, high, grid_points)
    allow_cross = bool(((config.get("fidelity") or {}).get("allow_cross_fidelity_same_point")))

    quotas_cfg = ((config.get("fidelity") or {}).get("batch_quota") or {})
    quotas: dict[str, int] = {}
    for level in levels:
        name = str(level.get("name"))
        quotas[name] = int(quotas_cfg.get(name, 0))

    if sum(quotas.values()) <= 0:
        # fallback: equal split
        each = max(1, batch_size // max(1, len(levels)))
        quotas = {str(level.get("name")): each for level in levels}

    ranked_by_level: dict[str, list[dict[str, Any]]] = {}

    for level in levels:
        name = str(level.get("name"))
        cost = float(level.get("cost", 1.0))
        level_rows = [r for r in usable_rows if str(r.get("fidelity")) == name]

        x_train = np.array([float(r["knob_value"]) for r in level_rows], dtype=float)
        y_train = np.array([float(r["objective_score"]) for r in level_rows], dtype=float)

        taken_same = list(x_train)
        blocked = mask_close(grid, taken_same, min_sep)
        candidates = grid[~blocked]
        if candidates.size == 0:
            ranked_by_level[name] = []
            continue

        random_mode = len(level_rows) < warmup

        if random_mode:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            ranked: list[dict[str, Any]] = []
            for idx, x in enumerate(shuffled):
                ranked.append(
                    {
                        "fidelity": name,
                        "knob_value": float(x),
                        "acquisition": 1.0 - idx / max(1, len(shuffled)),
                        "predicted_objective": None,
                        "predicted_std": None,
                        "predicted_feasible_prob": None,
                        "planner_model": "random_warmup",
                        "cost": cost,
                    }
                )
            ranked_by_level[name] = ranked
            continue

        mu, std, model_name = fit_gp_predict(x_train, y_train, candidates, seed)
        p_feas = estimate_feasible_probability(candidates, level_rows if level_rows else usable_rows, feas_ls)
        acq = (mu + beta * std) * p_feas / max(cost, 1e-9)

        order = np.argsort(-acq)
        ranked = []
        for idx in order:
            ranked.append(
                {
                    "fidelity": name,
                    "knob_value": float(candidates[idx]),
                    "acquisition": float(acq[idx]),
                    "predicted_objective": float(mu[idx]),
                    "predicted_std": float(std[idx]),
                    "predicted_feasible_prob": float(p_feas[idx]),
                    "planner_model": model_name,
                    "cost": cost,
                }
            )
        ranked_by_level[name] = ranked

    selected: list[dict[str, Any]] = []
    selected_points: list[float] = []

    for level in levels:
        name = str(level.get("name"))
        quota = max(0, quotas.get(name, 0))
        if quota <= 0:
            continue
        pool = ranked_by_level.get(name, [])
        count = 0
        for cand in pool:
            if len(selected) >= batch_size:
                break
            x = float(cand["knob_value"])
            if (not allow_cross) and any(abs(x - y) <= min_sep for y in selected_points):
                continue
            selected.append(cand)
            selected_points.append(x)
            count += 1
            if count >= quota:
                break

    if len(selected) < batch_size:
        merged_pool: list[dict[str, Any]] = []
        for name, pool in ranked_by_level.items():
            merged_pool.extend(pool)
        merged_pool.sort(key=lambda r: float(r.get("acquisition") or -1e18), reverse=True)
        for cand in merged_pool:
            if len(selected) >= batch_size:
                break
            x = float(cand["knob_value"])
            if (not allow_cross) and any(abs(x - y) <= min_sep for y in selected_points):
                continue
            # avoid duplicate fidelity+knob entries
            dup = False
            for s in selected:
                if str(s.get("fidelity")) == str(cand.get("fidelity")) and abs(float(s.get("knob_value")) - x) <= min_sep:
                    dup = True
                    break
            if dup:
                continue
            selected.append(cand)
            selected_points.append(x)

    if not selected:
        raise SystemExit("Planner could not select any candidate")

    batch_id = args.batch_id or build_batch_id()
    output_root = repo_root / str(config.get("output_root") or "outputs/bo/default")
    plan_dir = output_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)

    suggestions = {
        "generated_at": now_iso(),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "batch_id": batch_id,
        "seed": seed,
        "grid": {
            "min": low,
            "max": high,
            "points": grid_points,
            "min_separation": min_sep,
        },
        "selected": selected,
        "ranked_by_level": {k: v[: min(10, len(v))] for k, v in ranked_by_level.items()},
    }

    suggestions_path = plan_dir / f"{batch_id}.suggestions.json"
    write_json(suggestions_path, suggestions)

    if args.dry_run:
        print(f"[bo-plan] dry-run selected={len(selected)}")
        print(f"[bo-plan] suggestions={suggestions_path}")
        return

    levels_by_name = {str(level.get("name")): level for level in levels}
    generated_cases_root = repo_root / str(config.get("generated_cases_root") or "cases/bo_generated")
    batch_case_root = generated_cases_root / batch_id
    batch_case_root.mkdir(parents=True, exist_ok=True)

    plan_cases: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []

    for idx, cand in enumerate(selected, start=1):
        fid = str(cand.get("fidelity"))
        level = levels_by_name[fid]
        template_case_path = repo_root / str(level.get("template_case"))

        scale = int((config.get("knob") or {}).get("token_scale", 10000))
        width = int((config.get("knob") or {}).get("token_width", 5))
        knob_token = format_knob_token(float(cand["knob_value"]), scale, width)

        case_dir_name = f"{sanitize_token(fid)}-k{knob_token}-{idx:02d}"
        output_case_dir = batch_case_root / case_dir_name

        case_path, case_id = patch_case_for_candidate(
            repo_root=repo_root,
            template_case_path=template_case_path,
            output_case_dir=output_case_dir,
            config=config,
            level=level,
            batch_id=batch_id,
            ordinal=idx,
            knob_value=float(cand["knob_value"]),
            prediction=cand,
        )

        stage = str(level.get("stage") or "all")
        retries = int(level.get("retries", 1))
        key = f"{sanitize_token(fid)}-{idx:02d}"

        plan_cases.append(
            {
                "key": key,
                "role": fid,
                "case": rel_path(case_path, repo_root),
                "stage": stage,
                "retries": retries,
                "notes": f"BO candidate knob={cand['knob_value']:.6f}, acq={cand.get('acquisition')}",
            }
        )

        materialized.append(
            {
                "case_id": case_id,
                "case_path": rel_path(case_path, repo_root),
                "fidelity": fid,
                "knob_value": cand.get("knob_value"),
                "acquisition": cand.get("acquisition"),
                "predicted_objective": cand.get("predicted_objective"),
                "predicted_std": cand.get("predicted_std"),
                "predicted_feasible_prob": cand.get("predicted_feasible_prob"),
            }
        )

    orch_cfg = config.get("orchestrator_plan") or {}
    plan_payload = {
        "name": f"{sanitize_token(str(orch_cfg.get('name_prefix') or 'bo-plan'))}-{batch_id}",
        "description": str(orch_cfg.get("description") or "BO planned cases"),
        "max_retries": int(orch_cfg.get("max_retries", 1)),
        "slurm": copy.deepcopy(orch_cfg.get("slurm") or {}),
        "cases": plan_cases,
    }

    if args.output_plan:
        output_plan = Path(args.output_plan)
        if not output_plan.is_absolute():
            output_plan = (repo_root / output_plan).resolve()
    else:
        output_plan = plan_dir / f"{batch_id}.plan.json"

    write_json(output_plan, plan_payload)

    materialized_path = plan_dir / f"{batch_id}.materialized.json"
    write_json(
        materialized_path,
        {
            "generated_at": now_iso(),
            "batch_id": batch_id,
            "plan_path": str(output_plan),
            "materialized": materialized,
        },
    )

    print(f"[bo-plan] config={config_path}")
    print(f"[bo-plan] dataset={dataset_path}")
    print(f"[bo-plan] selected={len(selected)}")
    print(f"[bo-plan] suggestions={suggestions_path}")
    print(f"[bo-plan] materialized={materialized_path}")
    print(f"[bo-plan] orchestrator_plan={output_plan}")
    print("[bo-plan] run next:")
    print(f"  python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start --plan {output_plan} --mode slurm --poll-interval-s 30")


if __name__ == "__main__":
    main()
