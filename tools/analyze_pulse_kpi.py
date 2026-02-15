#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_case(case_ref: str, root: Path) -> tuple[str, Path]:
    case_path = Path(case_ref)
    if case_path.is_file():
        case_dir = case_path.parent
        case_id = case_dir.name
    else:
        case_dir = root / "cases" / case_ref
        case_id = case_ref
    return case_id, case_dir


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_source(root: Path, case_id: str) -> Path:
    return root / "outputs" / case_id / "analysis"


def pick_metric(metrics: dict, keys: list[str]):
    for key in keys:
        if key in metrics:
            return metrics.get(key)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Join pulse KPIs from existing metrics.")
    parser.add_argument("--case", required=True, help="Case id or path to case.json")
    args = parser.parse_args()

    root = repo_root()
    case_id, case_dir = load_case(args.case, root)
    output_root = root / "outputs" / case_id / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    warpx_args_path = case_dir / "inputs" / "warpx_args.json"
    cfg = read_json(warpx_args_path) if warpx_args_path.exists() else {}

    sources_cfg = cfg.get("pulse_kpi_sources") or {}
    sources = {
        "formation": sources_cfg.get("formation") or case_id,
        "u2": sources_cfg.get("u2") or case_id,
        "u2hist": sources_cfg.get("u2hist") or case_id,
        "circuit": sources_cfg.get("circuit") or case_id,
        "runtime_guards": sources_cfg.get("runtime_guards") or sources_cfg.get("u2") or case_id,
    }

    join_audit = {
        "sources_present": [],
        "missing_sources": [],
        "source_paths": {},
    }

    # Formation metrics
    formation_metrics = {}
    formation_dir = resolve_source(root, sources["formation"])
    formation_path = formation_dir / "metrics_formation.json"
    if formation_path.exists():
        formation_metrics = read_json(formation_path)
        join_audit["sources_present"].append("formation")
        join_audit["source_paths"]["formation"] = str(formation_path)
    else:
        join_audit["missing_sources"].append("formation")

    # U2 stats metrics
    u2_metrics = {}
    u2_dir = resolve_source(root, sources["u2"])
    u2_path = u2_dir / "metrics_u2.json"
    if u2_path.exists():
        u2_metrics = read_json(u2_path)
        join_audit["sources_present"].append("u2")
        join_audit["source_paths"]["u2"] = str(u2_path)
    else:
        join_audit["missing_sources"].append("u2")

    # U2 histogram metrics
    u2hist_metrics = {}
    u2hist_dir = resolve_source(root, sources["u2hist"])
    u2hist_path = u2hist_dir / "metrics_u2hist.json"
    if u2hist_path.exists():
        u2hist_metrics = read_json(u2hist_path)
        join_audit["sources_present"].append("u2hist")
        join_audit["source_paths"]["u2hist"] = str(u2hist_path)
    else:
        join_audit["missing_sources"].append("u2hist")

    # Circuit metrics
    circuit_metrics = {}
    circuit_dir = resolve_source(root, sources["circuit"])
    circuit_path = circuit_dir / "metrics_circuit.json"
    if not circuit_path.exists():
        circuit_path = circuit_dir / "metrics.json"
    if circuit_path.exists():
        circuit_metrics = read_json(circuit_path)
        if "e_load_J" in circuit_metrics or "vind_peak_V" in circuit_metrics:
            join_audit["sources_present"].append("circuit")
            join_audit["source_paths"]["circuit"] = str(circuit_path)
        else:
            join_audit["missing_sources"].append("circuit")
            circuit_metrics = {}
    else:
        join_audit["missing_sources"].append("circuit")

    # Runtime guard metrics (inject controls)
    guard_metrics = {}
    guard_dir = resolve_source(root, sources["runtime_guards"])
    guard_path = guard_dir / "metrics_runtime_guards.json"
    if guard_path.exists():
        guard_metrics = read_json(guard_path)
        join_audit["sources_present"].append("runtime_guards")
        join_audit["source_paths"]["runtime_guards"] = str(guard_path)
    else:
        join_audit["missing_sources"].append("runtime_guards")

    drive_amp = cfg.get("driveAmp_scale")
    if drive_amp is None:
        drive_amp = cfg.get("opmd_b_scale")
    off_step = cfg.get("drive_envelope_off_step")

    formation_chain = formation_metrics.get("formation_chain") or {}
    Wb_end = formation_chain.get("Wb_end")
    if Wb_end is None:
        Wb_end = pick_metric(formation_metrics, ["Wb_last", "Wb_end", "Wb_peak"])

    step_B_energy_peak = pick_metric(formation_metrics, ["step_B_energy_peak"])

    inject_repeat_nsteps = guard_metrics.get("inject_repeat_nsteps_effective")
    if inject_repeat_nsteps is None:
        inject_repeat_nsteps = cfg.get("inject_repeat_nsteps")
    if inject_repeat_nsteps is None:
        inject_repeat_nsteps = cfg.get("m1_rho_cos_nsteps")

    inject_end_main_step = guard_metrics.get("inject_end_main_step_effective")
    if inject_end_main_step is None:
        inject_end_main_step = guard_metrics.get("inject_end_istep_effective")
    if inject_end_main_step is None:
        inject_end_main_step = cfg.get("inject_end_istep")

    inject_particles_total = guard_metrics.get("inject_particles_total")

    pulse_kpi = {
        "driveAmp_scale": drive_amp,
        "off_step": off_step,
        "inject_repeat_nsteps": inject_repeat_nsteps,
        "inject_end_main_step": inject_end_main_step,
        "inject_particles_total": inject_particles_total,
        "energy_chain": {
            "Wb_end": Wb_end,
            "vind_peak": circuit_metrics.get("vind_peak_V"),
            "e_load": circuit_metrics.get("e_load_J"),
        },
        "tail_proxy": {
            "u2_max_end": u2_metrics.get("u2_max_end"),
            "u2_p99_at_step0": u2hist_metrics.get("u2_p99_at_step0"),
            "u2_p99_at_stepOff": u2hist_metrics.get("u2_p99_at_stepOff"),
            "u2_p99_at_stepEnd": u2hist_metrics.get("u2_p99_at_stepEnd"),
            "u2_p99_ratio_off_to_end": u2hist_metrics.get("u2_p99_ratio_off_to_end"),
            "tail_gain": None,
        },
        "inject": {
            "repeat_nsteps": inject_repeat_nsteps,
            "end_main_step": inject_end_main_step,
            "particles_total": inject_particles_total,
        },
        "stage": {
            "step_B_energy_peak": step_B_energy_peak,
        },
        "join_audit": join_audit,
    }

    p99_step0 = pulse_kpi["tail_proxy"].get("u2_p99_at_step0")
    p99_stepoff = pulse_kpi["tail_proxy"].get("u2_p99_at_stepOff")
    if p99_step0 is not None and p99_stepoff is not None:
        try:
            p99_step0_val = float(p99_step0)
            if p99_step0_val != 0.0:
                pulse_kpi["tail_proxy"]["tail_gain"] = float(p99_stepoff) / p99_step0_val
        except Exception:
            pulse_kpi["tail_proxy"]["tail_gain"] = None

    out_path = output_root / "metrics_pulse_kpi.json"
    out_path.write_text(json.dumps({"pulse_kpi": pulse_kpi}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
