#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RHO_STRONG_THRESHOLD = 0.05
RHO_FLOOR = 1.0e-8


@dataclass
class Sample:
    case_id: str
    drive_amp: float
    repeat: float
    off_step: float
    repeat_pre: float
    e_load: float
    u2_off: float
    rho_phase: float


@dataclass
class ModelSpec:
    e_coeffs: np.ndarray
    u_coeffs: np.ndarray
    rho_coeffs: np.ndarray
    x1_min: float
    x1_max: float
    x2_min: float
    x2_max: float


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def canonical_sha1(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_sample(repo_root: Path, case_id: str) -> Sample | None:
    pulse_path = repo_root / "outputs" / case_id / "analysis" / "metrics_pulse_kpi.json"
    formation_path = repo_root / "outputs" / case_id / "analysis" / "metrics_formation.json"

    pulse = read_json(pulse_path).get("pulse_kpi", {})
    formation = read_json(formation_path)
    phase = formation.get("formation_kpi_phase", {})

    try:
        drive_amp = float(pulse.get("driveAmp_scale"))
        repeat = float(pulse.get("inject_repeat_nsteps"))
        off_step = float(pulse.get("off_step") or 0.0)
        if off_step <= 0.0:
            off_step = repeat
        repeat_pre = min(repeat, off_step)
        e_load = float((pulse.get("energy_chain") or {}).get("e_load"))
        u2_off = float((pulse.get("tail_proxy") or {}).get("u2_p99_at_stepOff"))
        rho_phase = float(phase.get("rho_delta_rel_phase"))
    except Exception:
        return None

    if drive_amp <= 0.0 or repeat_pre <= 0.0 or e_load <= 0.0 or u2_off <= 0.0:
        return None

    return Sample(
        case_id=case_id,
        drive_amp=drive_amp,
        repeat=repeat,
        off_step=off_step,
        repeat_pre=repeat_pre,
        e_load=e_load,
        u2_off=u2_off,
        rho_phase=max(rho_phase, RHO_FLOOR),
    )


def design_e(samples: list[Sample]) -> np.ndarray:
    x1 = np.log([s.drive_amp for s in samples])
    return np.c_[np.ones(len(samples)), x1]


def design_u(samples: list[Sample]) -> np.ndarray:
    x1 = np.log([s.drive_amp for s in samples])
    x2 = np.log([s.repeat_pre for s in samples])
    return np.c_[np.ones(len(samples)), x2, x2 * x2, x1]


def fit_models(samples: list[Sample]) -> ModelSpec:
    a_e = design_e(samples)
    a_u = design_u(samples)

    y_e = np.log([s.e_load for s in samples])
    y_u = np.log([s.u2_off for s in samples])
    y_rho = np.log([max(s.rho_phase, RHO_FLOOR) for s in samples])

    e_coeffs = np.linalg.lstsq(a_e, y_e, rcond=None)[0]
    u_coeffs = np.linalg.lstsq(a_u, y_u, rcond=None)[0]
    rho_coeffs = np.linalg.lstsq(a_u, y_rho, rcond=None)[0]

    x1 = np.log([s.drive_amp for s in samples])
    x2 = np.log([s.repeat_pre for s in samples])

    return ModelSpec(
        e_coeffs=e_coeffs,
        u_coeffs=u_coeffs,
        rho_coeffs=rho_coeffs,
        x1_min=float(np.min(x1)),
        x1_max=float(np.max(x1)),
        x2_min=float(np.min(x2)),
        x2_max=float(np.max(x2)),
    )


def predict_raw(model: ModelSpec, sample: Sample) -> tuple[float, float, float]:
    x1 = math.log(sample.drive_amp)
    x2 = math.log(sample.repeat_pre)
    e = float(np.exp(np.array([1.0, x1]) @ model.e_coeffs))
    u2 = float(np.exp(np.array([1.0, x2, x2 * x2, x1]) @ model.u_coeffs))
    rho = float(np.exp(np.array([1.0, x2, x2 * x2, x1]) @ model.rho_coeffs))
    return e, u2, max(rho, RHO_FLOOR)


def nearest_neighbor_fallback(train: list[Sample], query: Sample) -> tuple[float, float, float]:
    qx1 = math.log(query.drive_amp)
    qx2 = math.log(query.repeat_pre)
    best = None
    best_dist = None
    for sample in train:
        x1 = math.log(sample.drive_amp)
        x2 = math.log(sample.repeat_pre)
        dist = (qx1 - x1) ** 2 + (qx2 - x2) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = sample
    assert best is not None
    return best.e_load, best.u2_off, best.rho_phase


def predict_with_fallback(model: ModelSpec, train: list[Sample], query: Sample) -> tuple[float, float, float, str]:
    x1 = math.log(query.drive_amp)
    x2 = math.log(query.repeat_pre)
    in_range = model.x1_min <= x1 <= model.x1_max and model.x2_min <= x2 <= model.x2_max
    if in_range:
        e, u2, rho = predict_raw(model, query)
        return e, u2, rho, "model"
    e, u2, rho = nearest_neighbor_fallback(train, query)
    return e, u2, rho, "nearest_neighbor"


def rel_err(pred: float, truth: float) -> float:
    return abs(pred - truth) / max(abs(truth), 1.0e-300)


def loocv_metrics(samples: list[Sample]) -> dict:
    e_err = []
    u_err = []
    rho_cls = []
    for idx in range(len(samples)):
        train = [samples[j] for j in range(len(samples)) if j != idx]
        test = samples[idx]
        model = fit_models(train)
        pe, pu, pr, _ = predict_with_fallback(model, train, test)
        e_err.append(rel_err(pe, test.e_load))
        u_err.append(rel_err(pu, test.u2_off))
        rho_cls.append((pr >= RHO_STRONG_THRESHOLD) == (test.rho_phase >= RHO_STRONG_THRESHOLD))

    return {
        "e_load_loocv_rel_err_max": float(max(e_err) if e_err else 0.0),
        "e_load_loocv_rel_err_mean": float(sum(e_err) / len(e_err) if e_err else 0.0),
        "u2_loocv_rel_err_max": float(max(u_err) if u_err else 0.0),
        "u2_loocv_rel_err_mean": float(sum(u_err) / len(u_err) if u_err else 0.0),
        "rho_cls_loocv_acc": float(sum(1.0 for ok in rho_cls if ok) / len(rho_cls) if rho_cls else 0.0),
    }


def oos_metrics(model: ModelSpec, train: list[Sample], samples: list[Sample]) -> tuple[dict, list[dict]]:
    rows = []
    fallback_count = 0
    for sample in samples:
        pe, pu, pr, mode = predict_with_fallback(model, train, sample)
        if mode != "model":
            fallback_count += 1
        rows.append(
            {
                "case_id": sample.case_id,
                "mode": mode,
                "drive_amp": sample.drive_amp,
                "repeat_pre": sample.repeat_pre,
                "e_load_true": sample.e_load,
                "e_load_pred": pe,
                "u2_true": sample.u2_off,
                "u2_pred": pu,
                "rho_true": sample.rho_phase,
                "rho_pred": pr,
                "u2_rel_err": rel_err(pu, sample.u2_off),
                "e_load_rel_err": rel_err(pe, sample.e_load),
                "rho_cls_true": sample.rho_phase >= RHO_STRONG_THRESHOLD,
                "rho_cls_pred": pr >= RHO_STRONG_THRESHOLD,
                "rho_cls_ok": (pr >= RHO_STRONG_THRESHOLD)
                == (sample.rho_phase >= RHO_STRONG_THRESHOLD),
            }
        )

    u2_err = [row["u2_rel_err"] for row in rows]
    e_err = [row["e_load_rel_err"] for row in rows]
    rho_cls = [row["rho_cls_ok"] for row in rows]
    metrics = {
        "oos_samples": len(rows),
        "oos_u2_rel_err_max": float(max(u2_err) if u2_err else 0.0),
        "oos_u2_rel_err_mean": float(sum(u2_err) / len(u2_err) if u2_err else 0.0),
        "oos_e_load_rel_err_max": float(max(e_err) if e_err else 0.0),
        "oos_rho_cls_acc": float(sum(1.0 for ok in rho_cls if ok) / len(rho_cls) if rho_cls else 0.0),
        "oos_fallback_count": int(fallback_count),
    }
    return metrics, rows


def write_prediction_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "mode",
        "drive_amp",
        "repeat_pre",
        "e_load_true",
        "e_load_pred",
        "e_load_rel_err",
        "u2_true",
        "u2_pred",
        "u2_rel_err",
        "rho_true",
        "rho_pred",
        "rho_cls_true",
        "rho_cls_pred",
        "rho_cls_ok",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# ML1 Transport Surrogate")
    lines.append("")
    lines.append(f"- surrogate_ready: `{metrics.get('surrogate_ready')}`")
    lines.append(f"- training_samples: `{metrics.get('training_samples')}`")
    lines.append(f"- u2_loocv_rel_err_max: `{metrics.get('u2_loocv_rel_err_max')}`")
    lines.append(f"- rho_cls_loocv_acc: `{metrics.get('rho_cls_loocv_acc')}`")
    lines.append(f"- oos_u2_rel_err_max: `{metrics.get('oos_u2_rel_err_max')}`")
    lines.append(f"- oos_rho_cls_acc: `{metrics.get('oos_rho_cls_acc')}`")
    lines.append(f"- fallback_policy_defined: `{metrics.get('fallback_policy_defined')}`")
    lines.append(f"- fallback_simulated_triggered: `{metrics.get('fallback_simulated_triggered')}`")
    lines.append("")
    lines.append("Artifacts include model coeffs, inference config, and prediction error tables.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_case_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML1 surrogate artifact for transport proxy metrics.")
    parser.add_argument("--train-cases", required=True, help="Comma-separated case IDs")
    parser.add_argument("--oos-cases", default="", help="Comma-separated OOS case IDs")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--inference-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_ids = parse_case_list(args.train_cases)
    oos_ids = parse_case_list(args.oos_cases)

    train_samples = []
    missing_train = []
    for case_id in train_ids:
        sample = load_sample(repo_root, case_id)
        if sample is None:
            missing_train.append(case_id)
            continue
        train_samples.append(sample)

    oos_samples = []
    missing_oos = []
    for case_id in oos_ids:
        sample = load_sample(repo_root, case_id)
        if sample is None:
            missing_oos.append(case_id)
            continue
        oos_samples.append(sample)

    if len(train_samples) < 5:
        metrics = {
            "surrogate_ready": False,
            "training_samples": len(train_samples),
            "missing_train_cases": missing_train,
            "error": "insufficient_training_samples",
        }
        out_metrics = Path(args.metrics)
        if not out_metrics.is_absolute():
            out_metrics = (repo_root / out_metrics).resolve()
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    model = fit_models(train_samples)
    loocv = loocv_metrics(train_samples)
    oos_stat, oos_rows = oos_metrics(model, train_samples, oos_samples)

    model_payload = {
        "kind": "ml1_transport_surrogate",
        "features": ["log_drive_amp", "log_repeat_pre", "log_repeat_pre_sq"],
        "targets": {
            "e_load": "log(e_load)",
            "u2_off": "log(u2_p99_at_stepOff)",
            "rho_phase": "log(max(rho_delta_rel_phase,1e-8))",
        },
        "coefficients": {
            "e_load": model.e_coeffs.tolist(),
            "u2_off": model.u_coeffs.tolist(),
            "rho_phase": model.rho_coeffs.tolist(),
        },
        "train_cases": train_ids,
        "train_cases_used": [s.case_id for s in train_samples],
        "missing_train_cases": missing_train,
    }
    model_payload["sha1"] = canonical_sha1(model_payload)

    inference_payload = {
        "kind": "ml1_inference_config",
        "feature_bounds": {
            "log_drive_amp": [model.x1_min, model.x1_max],
            "log_repeat_pre": [model.x2_min, model.x2_max],
        },
        "rho_strong_threshold": RHO_STRONG_THRESHOLD,
        "fallback": {
            "mode": "nearest_neighbor",
            "trigger": "feature_out_of_training_bounds",
            "distance": "euclidean(log_drive_amp,log_repeat_pre)",
        },
        "oos_cases": oos_ids,
        "oos_cases_used": [s.case_id for s in oos_samples],
        "missing_oos_cases": missing_oos,
    }
    inference_payload["sha1"] = canonical_sha1(inference_payload)

    synth_query = Sample(
        case_id="synthetic_fallback_probe",
        drive_amp=3.0,
        repeat=32.0,
        off_step=352.0,
        repeat_pre=32.0,
        e_load=0.0,
        u2_off=0.0,
        rho_phase=RHO_FLOOR,
    )
    _, _, _, synth_mode = predict_with_fallback(model, train_samples, synth_query)

    metrics = {
        "training_samples": len(train_samples),
        "oos_samples": len(oos_samples),
        "missing_train_case_count": len(missing_train),
        "missing_oos_case_count": len(missing_oos),
        "e_load_loocv_rel_err_max": loocv["e_load_loocv_rel_err_max"],
        "e_load_loocv_rel_err_mean": loocv["e_load_loocv_rel_err_mean"],
        "u2_loocv_rel_err_max": loocv["u2_loocv_rel_err_max"],
        "u2_loocv_rel_err_mean": loocv["u2_loocv_rel_err_mean"],
        "rho_cls_loocv_acc": loocv["rho_cls_loocv_acc"],
        "oos_u2_rel_err_max": oos_stat["oos_u2_rel_err_max"],
        "oos_u2_rel_err_mean": oos_stat["oos_u2_rel_err_mean"],
        "oos_e_load_rel_err_max": oos_stat["oos_e_load_rel_err_max"],
        "oos_rho_cls_acc": oos_stat["oos_rho_cls_acc"],
        "oos_fallback_count": oos_stat["oos_fallback_count"],
        "model_sha1": model_payload["sha1"],
        "inference_sha1": inference_payload["sha1"],
        "model_artifact_exists": True,
        "inference_artifact_exists": True,
        "fallback_policy_defined": True,
        "fallback_simulated_mode": synth_mode,
        "fallback_simulated_triggered": synth_mode == "nearest_neighbor",
    }

    metrics["surrogate_ready"] = bool(
        metrics["training_samples"] >= 8
        and metrics["u2_loocv_rel_err_max"] <= 0.1
        and metrics["e_load_loocv_rel_err_max"] <= 1.0e-3
        and metrics["rho_cls_loocv_acc"] >= 0.85
        and metrics["oos_u2_rel_err_max"] <= 0.1
        and metrics["oos_rho_cls_acc"] >= 0.85
        and metrics["fallback_policy_defined"]
        and metrics["fallback_simulated_triggered"]
    )

    out_metrics = Path(args.metrics)
    out_model = Path(args.model_out)
    out_infer = Path(args.inference_out)
    out_report = Path(args.report)
    out_pred = Path(args.predictions)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_model.is_absolute():
        out_model = (repo_root / out_model).resolve()
    if not out_infer.is_absolute():
        out_infer = (repo_root / out_infer).resolve()
    if not out_report.is_absolute():
        out_report = (repo_root / out_report).resolve()
    if not out_pred.is_absolute():
        out_pred = (repo_root / out_pred).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_infer.parent.mkdir(parents=True, exist_ok=True)

    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    out_model.write_text(json.dumps(model_payload, indent=2, sort_keys=True), encoding="utf-8")
    out_infer.write_text(json.dumps(inference_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_prediction_csv(out_pred, oos_rows)
    write_report(out_report, metrics)


if __name__ == "__main__":
    main()
