from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from .moire_errors import MoReWorkflowError

DEFAULT_MOIRE_KMC_SEED = 3401
DEFAULT_MOIRE_DIFFUSION_TEMPERATURES = (700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0)
KB_EV_PER_K = 8.617333262145e-5


def normalize_kmc_seeds(
    kmc_seed: Optional[int] = None,
    kmc_seeds: Optional[Sequence[int]] = None,
) -> list[int]:
    raw_values: list[int] = []
    if kmc_seed is not None:
        raw_values.append(int(kmc_seed))
    if kmc_seeds:
        raw_values.extend(int(item) for item in kmc_seeds)
    if not raw_values:
        raw_values = [DEFAULT_MOIRE_KMC_SEED]

    seeds: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        if value <= 0:
            raise MoReWorkflowError(f"KMC seed must be a positive integer, got: {value}")
        if value in seen:
            continue
        seeds.append(value)
        seen.add(value)
    return seeds


def apply_retry_seeds(seed_list: Sequence[int], retry_attempts: int) -> tuple[list[int], Dict[str, Any]]:
    retries = max(0, int(retry_attempts or 0))
    seeds = list(seed_list)
    added: list[int] = []
    if retries and len(seeds) == 1:
        candidate = seeds[0] + 1
        while len(added) < retries:
            if candidate not in seeds and candidate not in added:
                added.append(candidate)
            candidate += 1
    return seeds + added, {
        "requested_retry_attempts": retries,
        "added_retry_seeds": added,
        "enabled": bool(added),
    }


def series_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    return {
        "count": len(numeric),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(numeric),
        "max": max(numeric),
    }


def summarize_seed_runs(seed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    completed_runs = [item for item in seed_runs if item.get("status") == "completed"]
    failed_runs = [item for item in seed_runs if item.get("status") != "completed"]
    metrics: Dict[str, Any] = {}
    metric_extractors = {
        "accepted_events": lambda run: (run.get("parsed_run") or {}).get("accepted_events"),
        "rejected_events": lambda run: (run.get("parsed_run") or {}).get("rejected_events"),
        "final_time": lambda run: (run.get("parsed_run") or {}).get("final_time"),
        "final_energy": lambda run: (run.get("parsed_run") or {}).get("final_energy"),
        "loop_time_seconds": lambda run: (run.get("parsed_run") or {}).get("loop_time_seconds"),
        "jump_frequency_hz": lambda run: (run.get("derived_metrics") or {}).get("jump_frequency_hz"),
        "final_msd": lambda run: (run.get("diffusion_analysis") or {}).get("final_msd"),
        "final_diffusion_coefficient": lambda run: (run.get("diffusion_analysis") or {}).get("final_diffusion_coefficient"),
    }
    for field, extractor in metric_extractors.items():
        values = [float(value) for run in completed_runs if (value := extractor(run)) is not None]
        if values:
            metrics[field] = series_stats(values)
    representative = completed_runs[0] if completed_runs else (seed_runs[0] if seed_runs else None)
    return {
        "count": len(seed_runs),
        "completed_count": len(completed_runs),
        "failed_count": len(failed_runs),
        "seeds": [int(item["seed"]) for item in seed_runs],
        "completed_seeds": [int(item["seed"]) for item in completed_runs],
        "failed_seeds": [int(item["seed"]) for item in failed_runs],
        "representative_seed": representative.get("seed") if representative else None,
        "metrics": metrics,
    }


def normalize_temperature_list(temperatures_k: Optional[Sequence[float]] = None) -> list[float]:
    raw_values = list(temperatures_k or DEFAULT_MOIRE_DIFFUSION_TEMPERATURES)
    values: list[float] = []
    seen = set()
    for raw in raw_values:
        numeric = float(raw)
        if numeric <= 0:
            raise MoReWorkflowError(f"Temperature must be positive, got: {raw}")
        rounded = round(numeric, 6)
        if rounded in seen:
            continue
        values.append(numeric)
        seen.add(rounded)
    if not values:
        raise MoReWorkflowError("At least one temperature is required for the diffusion sweep.")
    return sorted(values)


def temperature_dir_label(temperature_k: float) -> str:
    if abs(temperature_k - round(temperature_k)) < 1.0e-9:
        return str(int(round(temperature_k)))
    return f"{temperature_k:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[Dict[str, float]]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return {"slope": slope, "intercept": intercept}
