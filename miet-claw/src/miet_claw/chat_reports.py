from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from .run_inspection import (
    RUN_KIND_BRIDGE,
    RUN_KIND_MOIRE_CHAIN,
    get_log_excerpt,
    inspect_run,
    list_artifacts,
    _to_float,
)


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"


def stylize(text: str, *styles: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(styles) + text + RESET


def format_compare_report(report: Dict[str, Any]) -> str:
    left = report["left"]
    right = report["right"]
    left_temp = _to_float(left.get("temperature_k"))
    right_temp = _to_float(right.get("temperature_k"))
    left_diff = left.get("latest_diffusion") or {}
    right_diff = right.get("latest_diffusion") or {}
    hotter = None
    colder = None
    if left_temp is not None and right_temp is not None:
        hotter, colder = (left, right) if left_temp >= right_temp else (right, left)

    lines = [
        stylize("Run comparison", BOLD, GREEN),
        f"- newer run: {left['job_id']}",
        f"  - mode: {left.get('mode')}",
        f"  - material: {left.get('material_name')}",
        f"  - temperature: {left_temp if left_temp is not None else '—'} K",
        f"- older run: {right['job_id']}",
        f"  - mode: {right.get('mode')}",
        f"  - material: {right.get('material_name')}",
        f"  - temperature: {right_temp if right_temp is not None else '—'} K",
        "- LAMMPS barriers:",
    ]

    if report["barriers"]:
        identical = True
        for row in report["barriers"]:
            left_value = row["left"]
            right_value = row["right"]
            delta = row["delta"]
            if left_value is None or right_value is None:
                identical = False
                lines.append(f"  - {row['species']}: 数据不完整 ({left_value} vs {right_value})")
                continue
            if abs(delta) > 1.0e-9:
                identical = False
            lines.append(
                f"  - {row['species']}: {left_value:.6f} eV vs {right_value:.6f} eV "
                f"(Δ={delta:+.6f} eV)"
            )
        lines.append(f"- barrier conclusion: {'两次 barrier 完全一致' if identical else '两次 barrier 存在差异'}")
    else:
        lines.append("  - 没有读到 barrier 事件表。")

    def _fmt_metric(name: str, payload: Dict[str, Any]) -> str:
        value = payload.get(name)
        return "—" if value is None else f"{value:.6g}"

    lines.extend(
        [
            "- KMC latest metrics:",
            (
                f"  - newer: jumps={_fmt_metric('jumps', left_diff)}, "
                f"jump_frequency={_fmt_metric('jump_frequency', left_diff)}, "
                f"diffusion_coefficient={_fmt_metric('diffusion_coefficient', left_diff)}"
            ),
            (
                f"  - older: jumps={_fmt_metric('jumps', right_diff)}, "
                f"jump_frequency={_fmt_metric('jump_frequency', right_diff)}, "
                f"diffusion_coefficient={_fmt_metric('diffusion_coefficient', right_diff)}"
            ),
        ]
    )

    lines.append("- interpretation:")
    if left_temp is None or right_temp is None:
        lines.append("  - 温度信息缺失，无法判断是否符合温度升高后的直觉。")
    else:
        lines.append(
            f"  - 更高温的是 {hotter['job_id']} ({_to_float(hotter.get('temperature_k')):.1f} K)，"
            f"更低温的是 {colder['job_id']} ({_to_float(colder.get('temperature_k')):.1f} K)。"
        )

    left_jump = left_diff.get("jump_frequency")
    right_jump = right_diff.get("jump_frequency")
    left_dc = left_diff.get("diffusion_coefficient")
    right_dc = right_diff.get("diffusion_coefficient")
    if left_jump is None or right_jump is None or left_dc is None or right_dc is None:
        lines.append("  - KMC 的 jump frequency 或 diffusion coefficient 缺失，无法完成完整趋势判断。")
    elif hotter and colder:
        hotter_jump = (left_jump, left_dc) if hotter["job_id"] == left["job_id"] else (right_jump, right_dc)
        colder_jump = (right_jump, right_dc) if hotter["job_id"] == left["job_id"] else (left_jump, left_dc)
        jump_ok = hotter_jump[0] >= colder_jump[0]
        diff_ok = hotter_jump[1] >= colder_jump[1]
        lines.append(
            f"  - jump frequency: {'升温后更高，符合直觉' if jump_ok else '升温后没有更高，不符合常见直觉'}。"
        )
        lines.append(
            f"  - diffusion coefficient: {'升温后更高，符合直觉' if diff_ok else '升温后没有更高，不符合常见直觉'}。"
        )
    return "\n".join(lines)


def format_run_list(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "还没有可显示的 runs。"
    lines = [stylize("Recent runs", BOLD, CYAN)]
    for item in items:
        mode = item.get("mode") or "—"
        completed = item.get("completed_steps", 0)
        total = item.get("total_steps", 0)
        lines.append(
            f"- {stylize(item['job_id'], BOLD)} [{item['status']}] "
            f"{completed}/{total} · {mode} · {item['updated_at']}"
        )
        lines.append(f"  {item['material_name']}")
    return "\n".join(lines)


def format_draft_report(report: Dict[str, Any]) -> str:
    generated = report.get("generated_files", {})
    lines = [
        stylize("Draft ready", BOLD, GREEN),
        f"- job_id: {report['job_id']}",
        f"- mode: {report['mode']}",
        f"- material: {report.get('material_name')}",
        f"- provider: {report.get('provider_used')}",
        f"- template: {report.get('selected_template', {}).get('file_name')}",
        f"- workspace: {report.get('workspace_dir')}",
    ]
    if report.get("mode") == "md_to_kmc_chain":
        lines.append("- chain: LAMMPS CI-NEB barrier → repo misa-kmc continuation")
    if report.get("assumptions"):
        lines.append("- assumptions:")
        lines.extend([f"  - {item}" for item in report["assumptions"][:5]])
    if report.get("warnings"):
        lines.append("- warnings:")
        lines.extend([f"  - {item}" for item in report["warnings"][:5]])
    for label in ["job_spec", "md_script", "md_neb_campaign", "kmc_preview_input"]:
        if generated.get(label):
            lines.append(f"- {label}: {generated[label]}")
    return "\n".join(lines)


def format_run_report(report: Dict[str, Any]) -> str:
    execution = report.get("execution", {})
    final_run_dir = execution.get("final_run_dir") or execution.get("validation_run_dir")
    lines = [
        stylize("Run complete", BOLD, GREEN),
        f"- job_id: {report['job_id']}",
        f"- provider: {report.get('provider_used')}",
        f"- validation: {execution.get('validation_run_dir')}",
        f"- final run: {final_run_dir}",
    ]
    if execution.get("dry_run_only"):
        lines.append("- execution mode: dry-run / preview only")
    if execution.get("resume_existing"):
        lines.append("- recovery mode: resume existing run if previous progress is found")
    if execution.get("overwrite_existing"):
        lines.append("- recovery mode: overwrite existing run directory if needed")
    if final_run_dir:
        run_info = inspect_run(Path(final_run_dir))
        lines.append("- chain: LAMMPS barrier extraction → repo misa-kmc continuation")
        lines.append(f"- workflow: {run_info.get('workflow_kind')}")
        lines.append(f"- barrier source: {run_info.get('barrier_source_mode')}")
        if run_info.get("resume_summary"):
            resume_summary = run_info["resume_summary"]
            lines.append(
                "- resumed from existing progress: "
                f"completed={len(resume_summary.get('completed_steps') or [])}, "
                f"failed={len(resume_summary.get('failed_steps') or [])}, "
                f"cancelled={len(resume_summary.get('cancelled_steps') or [])}"
            )
        if run_info.get("recovery_plan"):
            recovery_steps = [
                item.get("step_id")
                for item in run_info["recovery_plan"].get("steps", [])
                if item.get("action") in {"restart_resumable_step", "rebuild_from_checkpoint", "rerun_step"}
            ]
            if recovery_steps:
                lines.append(f"- recovery plan reapplied steps: {', '.join(recovery_steps[:4])}")
            missing_outputs = [
                f"{item.get('step_id')}[{', '.join((item.get('missing_outputs') or [])[:2])}]"
                for item in run_info["recovery_plan"].get("steps", [])
                if item.get("missing_outputs")
            ]
            if missing_outputs:
                lines.append(f"- recovery validation found missing outputs: {', '.join(missing_outputs[:3])}")
            drifted_outputs = [
                f"{item.get('step_id')}[{', '.join((item.get('drifted_outputs') or [])[:2])}]"
                for item in run_info["recovery_plan"].get("steps", [])
                if item.get("drifted_outputs")
            ]
            if drifted_outputs:
                lines.append(f"- recovery validation found changed outputs: {', '.join(drifted_outputs[:3])}")
            cascaded = [
                f"{item.get('step_id')}←{item.get('invalidated_by')}"
                for item in run_info["recovery_plan"].get("steps", [])
                if item.get("invalidated_by")
            ]
            if cascaded:
                lines.append(f"- downstream steps invalidated by recovery: {', '.join(cascaded[:4])}")
        if run_info.get("checkpoint_count"):
            lines.append(f"- checkpoints recorded: {run_info['checkpoint_count']}")
        if run_info.get("neb_images") is not None:
            lines.append(f"- NEB images: {run_info['neb_images']}")
        for event in run_info.get("events", [])[:6]:
            lines.append(
                f"  - {event['species']}: {float(event['barrier_ev']):.6f} eV "
                f"({event.get('barrier_source', 'unknown')})"
            )
    return "\n".join(lines)


def format_inspect_report(info: Dict[str, Any]) -> str:
    if info.get("kind") == RUN_KIND_MOIRE_CHAIN:
        lines = [
            stylize("Run detail", BOLD, MAGENTA),
            f"- job_id: {info['job_id']}",
            f"- mode: {info.get('mode')}",
            f"- status: {info.get('status')}",
            f"- material: {info.get('material_name')}",
            f"- path: {info.get('path')}",
            f"- source case: {info.get('source_case_dir', '—')}",
            f"- working copy: {info.get('copied_case_dir', '—')}",
            f"- generated input: {info.get('generated_lammps_input', '—')}",
            f"- neb.txt: {info.get('neb_txt', '—')}",
            f"- barrier: {info.get('barrier_eV', '—')} eV",
            f"- runtime health: {(info.get('runtime_health') or {}).get('status', '—')}",
        ]
        if info.get("step_statuses"):
            lines.append("- steps:")
            lines.extend([f"  - {step}: {status}" for step, status in info["step_statuses"].items()])
        if info.get("events"):
            lines.append("- barriers:")
            for event in info["events"][:8]:
                lines.append(
                    f"  - {event['species']}: {float(event['barrier_ev']):.6f} eV "
                    f"({event.get('barrier_source', 'unknown')})"
                )
        lines.extend(
            [
                "- KMC:",
                f"  - accepted events: {info.get('accepted_events', '—')}",
                f"  - final time: {info.get('final_time', '—')}",
                f"  - md log: {info.get('md_log_path', '—')}",
                f"  - kmc log: {info.get('kmc_log_path', '—')}",
                f"  - summary: {info.get('summary_path', '—')}",
            ]
        )
        return "\n".join(lines)
    if info.get("kind") == RUN_KIND_BRIDGE:
        lines = [
            stylize("Run detail", BOLD, MAGENTA),
            f"- job_id: {info['job_id']}",
            f"- mode: {info.get('mode')}",
            f"- status: {info.get('status')}",
            f"- material: {info.get('material_name')}",
            f"- path: {info.get('path')}",
            f"- barrier: {info.get('barrier_eV', '—')} eV",
            f"- validation passed: {info.get('validation_passed', '—')}",
            f"- lookup hits: {info.get('lookup_hits', '—')}",
            f"- live ML misses: {info.get('live_ml_misses', '—')}",
            f"- runtime health: {(info.get('runtime_health') or {}).get('status', '—')}",
            f"- kmc log: {info.get('kmc_log_path', '—')}",
            f"- summary: {info.get('summary_path', '—')}",
        ]
        return "\n".join(lines)
    lines = [
        stylize("Run detail", BOLD, MAGENTA),
        f"- job_id: {info['job_id']}",
        f"- status: {info.get('status', '—')}",
        f"- mode: {info.get('mode')}",
        f"- material: {info.get('material_name')}",
        f"- temperature: {info.get('temperature_k', '—')} K",
        f"- path: {info.get('path')}",
        f"- workflow: {info.get('workflow_kind')}",
        f"- barrier source: {info.get('barrier_source_mode')}",
    ]
    provenance = info.get("execution_provenance") or {}
    if provenance:
        lines.append(f"- execution provenance: {provenance.get('label', '—')}")
        for stage_name, stage in (provenance.get("stages") or {}).items():
            detail = f"  - {stage_name}: mode={stage.get('mode', '—')}"
            if stage.get("diffusion_mode"):
                detail += f", diffusion={stage.get('diffusion_mode')}"
            if stage.get("reason"):
                detail += f", reason={stage.get('reason')}"
            lines.append(detail)
    if info.get("resume_summary"):
        resume_summary = info["resume_summary"]
        lines.append(
            "- resumed from existing progress: "
            f"completed={len(resume_summary.get('completed_steps') or [])}, "
            f"failed={len(resume_summary.get('failed_steps') or [])}, "
            f"cancelled={len(resume_summary.get('cancelled_steps') or [])}"
        )
    if info.get("recovery_plan"):
        recovery_steps = [
            item.get("step_id")
            for item in info["recovery_plan"].get("steps", [])
            if item.get("action") in {"restart_resumable_step", "rebuild_from_checkpoint", "rerun_step"}
        ]
        if recovery_steps:
            lines.append(f"- recovery plan reapplied steps: {', '.join(recovery_steps[:4])}")
        missing_outputs = [
            f"{item.get('step_id')}[{', '.join((item.get('missing_outputs') or [])[:2])}]"
            for item in info["recovery_plan"].get("steps", [])
            if item.get("missing_outputs")
        ]
        if missing_outputs:
            lines.append(f"- recovery validation found missing outputs: {', '.join(missing_outputs[:3])}")
        drifted_outputs = [
            f"{item.get('step_id')}[{', '.join((item.get('drifted_outputs') or [])[:2])}]"
            for item in info["recovery_plan"].get("steps", [])
            if item.get("drifted_outputs")
        ]
        if drifted_outputs:
            lines.append(f"- recovery validation found changed outputs: {', '.join(drifted_outputs[:3])}")
        cascaded = [
            f"{item.get('step_id')}←{item.get('invalidated_by')}"
            for item in info["recovery_plan"].get("steps", [])
            if item.get("invalidated_by")
        ]
        if cascaded:
            lines.append(f"- downstream steps invalidated by recovery: {', '.join(cascaded[:4])}")
    if info.get("checkpoint_count"):
        lines.append(f"- checkpoints recorded: {info.get('checkpoint_count')}")
    if info.get("neb_images") is not None:
        lines.append(f"- NEB images: {info['neb_images']}")
    if info.get("step_statuses"):
        lines.append("- steps:")
        lines.extend([f"  - {step}: {status}" for step, status in info["step_statuses"].items()])
    if info.get("events"):
        lines.append("- barriers:")
        for event in info["events"][:8]:
            lines.append(
                f"  - {event['species']}: {float(event['barrier_ev']):.6f} eV "
                f"({event.get('barrier_source', 'unknown')})"
            )
    latest_diffusion = info.get("latest_diffusion") or {}
    if latest_diffusion:
        lines.append("- latest KMC metrics:")
        lines.append(f"  - jumps: {latest_diffusion.get('jumps')}")
        lines.append(f"  - jump frequency: {latest_diffusion.get('jump_frequency')}")
        lines.append(f"  - diffusion coefficient: {latest_diffusion.get('diffusion_coefficient')}")
    if info.get("summary"):
        lines.append("- summary:")
        excerpt = "\n".join(info["summary"].splitlines()[:12])
        lines.append(textwrap.indent(excerpt, "  "))
    return "\n".join(lines)


def format_artifact_report(run_dir: Path) -> str:
    items = list_artifacts(run_dir)
    if not items:
        return "这个 run 还没有归档 artifacts。"
    lines = [stylize("Artifacts", BOLD, CYAN), f"- run: {run_dir}"]
    lines.extend([f"- {item}" for item in items])
    return "\n".join(lines)


def format_log_report(run_dir: Path, target: str = "auto") -> str:
    info = get_log_excerpt(run_dir, target=target)
    if not info.get("available"):
        return f"没有找到 {target} 日志。"
    header = f"Log excerpt ({info['target']})\n- path: {info['path']}\n"
    return header + "\n" + info["content"]


def format_bridge_report(summary: Dict[str, Any]) -> str:
    files = summary.get("files") or {}
    validation = summary.get("validation") or {}
    runtime_health = summary.get("runtime_health") or {}
    dispatch = summary.get("dispatch") or {}
    workdir = Path(files.get("barriers_tsv", "")).parent if files.get("barriers_tsv") else None
    lines = [
        stylize("KMC bridge complete", BOLD, GREEN),
        f"- barrier: {float(summary.get('barrier_eV', 0.0)):.6f} eV",
        f"- workdir: {workdir or '—'}",
        f"- lookup: {files.get('barriers_tsv', '—')}",
        f"- state: {files.get('state_values_sites', '—')}",
    ]
    if files.get("input_ml"):
        lines.append(f"- input: {files['input_ml']}")
    if files.get("run_out"):
        lines.append(f"- run.out: {files['run_out']}")
    if dispatch:
        lines.append(f"- dispatch: {dispatch.get('transport', '—')} ({dispatch.get('tool', '—')})")
    if summary.get("validation_passed") is not None:
        lines.append(f"- validation passed: {summary.get('validation_passed')}")
        if summary.get("safe_validation_passed") is not None:
            lines.append(f"- safe to continue: {summary.get('safe_validation_passed')}")
        lines.append(f"- lookup hits: {validation.get('lookup_hits')}")
        lines.append(f"- live ML misses: {validation.get('live_ml_misses')}")
    if runtime_health:
        lines.append(f"- runtime health: {runtime_health.get('status')}")
        for item in (runtime_health.get("warnings") or [])[:4]:
            lines.append(f"  - warning: {item}")
    return "\n".join(lines)


def format_moire_lammps_report(summary: Dict[str, Any]) -> str:
    dispatch = summary.get("dispatch") or {}
    visualization = summary.get("visualization") or {}
    model = summary.get("model") or {}
    lines = [
        stylize("MoRe LAMMPS complete", BOLD, GREEN),
        f"- status: {summary.get('status', '—')}",
        f"- source case: {summary.get('source_case_dir', '—')}",
        f"- working copy: {summary.get('copied_case_dir', '—')}",
        f"- generated input: {summary.get('generated_lammps_input', '—')}",
        f"- barrier script: {summary.get('generated_barrier_script', '—')}",
        f"- neb.txt: {summary.get('neb_txt', '—')}",
        f"- barrier: {float(summary.get('barrier_eV', 0.0)):.6f} eV",
        f"- LAMMPS status: {(summary.get('lammps') or {}).get('status', '—')}",
        f"- LAMMPS log: {(summary.get('lammps') or {}).get('log', '—')}",
        f"- postprocess log: {(summary.get('postprocess') or {}).get('log', '—')}",
    ]
    if model:
        lines.append(f"- model mode: {model.get('mode', '—')}")
        if model.get("event_json"):
            lines.append(f"- event: {model.get('event_json')}")
        if model.get("generated_data_final_lmp"):
            lines.append(f"- generated final data: {model.get('generated_data_final_lmp')}")
        if model.get("kmc_data_lmp_assist"):
            lines.append(f"- KMC lattice assist data.lmp: {model.get('kmc_data_lmp_assist')}")
    if visualization:
        lines.extend(
            [
                f"- LAMMPS OVITO: {visualization.get('status', '—')}",
                f"- initial structure: {visualization.get('initial_structure', '—')}",
                f"- final structure: {visualization.get('final_structure', '—')}",
            ]
        )
        if visualization.get("initial_snapshot"):
            lines.append(f"- initial snapshot: {visualization['initial_snapshot']}")
        if visualization.get("final_snapshot"):
            lines.append(f"- final snapshot: {visualization['final_snapshot']}")
    if dispatch:
        lines.append(f"- dispatch: {dispatch.get('transport', '—')} ({dispatch.get('tool', '—')})")
    if summary.get("summary_json"):
        lines.append(f"- summary: {summary['summary_json']}")
    return "\n".join(lines)


def format_moire_kmc_report(summary: Dict[str, Any]) -> str:
    dispatch = summary.get("dispatch") or {}
    parsed_run = summary.get("parsed_run") or {}
    state_transform = summary.get("state_generation") or summary.get("state_transform") or {}
    barrier_assignment = summary.get("barrier_assignment") or {}
    files = summary.get("files") or {}
    runtime_health = summary.get("runtime_health") or {}
    generated_event = summary.get("generated_event") or {}
    ensemble = summary.get("ensemble") or {}
    visualization = summary.get("visualization") or {}
    lines = [
        stylize("MoRe repo KMC complete", BOLD, GREEN),
        f"- status: {summary.get('status', '—')}",
        f"- barrier: {float(summary.get('barrier_eV', 0.0)):.6f} eV",
        f"- generated event: {generated_event.get('event_json', summary.get('event_json', '—'))}",
        f"- generated state: {files.get('state_values_sites', '—')}",
        f"- generated input: {files.get('input_kmc', '—')}",
        f"- run.out: {files.get('run_out', '—')}",
        f"- state source: {state_transform.get('source', '—')}",
        f"- event: {state_transform.get('event_json', summary.get('event_json', '—'))}",
        f"- data.lmp assist: {state_transform.get('data_lmp', '—')}",
        f"- lattice: {state_transform.get('lattice_style', '—')} {state_transform.get('lattice_constant', '—')}",
        f"- cells: {state_transform.get('cells', '—')}",
        f"- vacancy source: {(generated_event.get('vacancy_source') if generated_event else state_transform.get('vacancy_source', '—'))}",
        f"- barrier assignment: Mo={barrier_assignment.get('Mo', '—')} eV, Re={barrier_assignment.get('Re', '—')} eV",
        f"- pair sites remapped: {state_transform.get('converted_pair_markers', '—')}",
        f"- pair sites from data.lmp: {state_transform.get('pair_sites_from_data_lmp', '—')}",
        f"- pair sites fallback host type: {state_transform.get('defaulted_pair_sites', '—')}",
        f"- accepted events: {parsed_run.get('accepted_events', '—')}",
        f"- final time: {parsed_run.get('final_time', '—')}",
        f"- runtime health: {runtime_health.get('status', '—')}",
    ]
    if ensemble:
        accepted_stats = (ensemble.get("metrics") or {}).get("accepted_events") or {}
        final_time_stats = (ensemble.get("metrics") or {}).get("final_time") or {}
        lines.extend(
            [
                f"- seeds: {ensemble.get('seeds', [])}",
                f"- completed seeds: {ensemble.get('completed_count', 0)}/{ensemble.get('count', 0)}",
                f"- representative seed: {ensemble.get('representative_seed', summary.get('representative_seed', '—'))}",
            ]
        )
        if accepted_stats.get("count"):
            lines.append(
                f"- accepted events mean±std: {accepted_stats.get('mean'):.3f} ± {accepted_stats.get('std'):.3f}"
            )
        if final_time_stats.get("count"):
            lines.append(
                f"- final time mean±std: {final_time_stats.get('mean'):.6g} ± {final_time_stats.get('std'):.6g}"
            )
    if visualization.get("requested"):
        lines.append(f"- OVITO: {visualization.get('status', '—')}")
        completed = [item.get("output_png") for item in (visualization.get("per_seed") or []) if item.get("output_png")]
        if completed:
            lines.append(f"- OVITO sample: {completed[0]}")
    if visualization.get("comparison_chart_svg"):
        lines.append(f"- comparison chart: {visualization['comparison_chart_svg']}")
    if visualization.get("gif_status") and visualization.get("gif_status") != "disabled":
        lines.append(f"- GIF: {visualization.get('gif_status', '—')}")
        if visualization.get("animated_gif"):
            lines.append(f"- GIF file: {visualization['animated_gif']}")
    if dispatch:
        lines.append(f"- dispatch: {dispatch.get('transport', '—')} ({dispatch.get('tool', '—')})")
    if summary.get("summary_json"):
        lines.append(f"- summary: {summary['summary_json']}")
    return "\n".join(lines)


def format_moire_workflow_report(summary: Dict[str, Any]) -> str:
    kmc = summary.get("kmc") or summary.get("bridge") or {}
    lammps_visualization = summary.get("lammps_visualization") or {}
    lammps_event_binding = summary.get("lammps_event_binding") or {}
    lammps_model = summary.get("lammps_model") or {}
    files = kmc.get("files") or {}
    parsed_run = kmc.get("parsed_run") or {}
    state_transform = kmc.get("state_generation") or kmc.get("state_transform") or {}
    barrier_assignment = kmc.get("barrier_assignment") or {}
    runtime_health = summary.get("runtime_health") or kmc.get("runtime_health") or {}
    dispatch = summary.get("dispatch") or {}
    generated_event = summary.get("generated_event") or kmc.get("generated_event") or {}
    ensemble = kmc.get("ensemble") or {}
    visualization = kmc.get("visualization") or {}
    lines = [
        stylize("MoRe LAMMPS → KMC complete", BOLD, GREEN),
        f"- status: {summary.get('status', '—')}",
        f"- event: {summary.get('event_json')}",
        f"- generated event: {generated_event.get('event_json', '—')}",
        f"- source case: {summary.get('source_case_dir')}",
        f"- working copy: {summary.get('copied_case_dir')}",
        f"- generated LAMMPS input: {summary.get('generated_lammps_input', '—')}",
        f"- generated barrier script: {summary.get('generated_barrier_script', '—')}",
        f"- neb.txt: {summary.get('neb_txt')}",
        f"- LAMMPS status: {(summary.get('lammps') or {}).get('status', '—')}",
        f"- LAMMPS log: {(summary.get('lammps') or {}).get('log', '—')}",
        f"- postprocess log: {(summary.get('postprocess') or {}).get('log', '—')}",
        f"- barrier: {float(kmc.get('barrier_eV', 0.0)):.6f} eV",
        f"- KMC state: {files.get('state_values_sites', '—')}",
        f"- KMC input: {files.get('input_kmc', files.get('input_ml', '—'))}",
        f"- KMC run.out: {files.get('run_out', '—')}",
        f"- KMC accepted events: {parsed_run.get('accepted_events', '—')}",
        f"- KMC final time: {parsed_run.get('final_time', '—')}",
        f"- KMC state source: {state_transform.get('source', '—')}",
        f"- KMC event source: {state_transform.get('event_json', '—')}",
        f"- KMC data.lmp assist: {state_transform.get('data_lmp', '—')}",
        f"- KMC lattice: {state_transform.get('lattice_style', '—')} {state_transform.get('lattice_constant', '—')}",
        f"- KMC cells: {state_transform.get('cells', '—')}",
        f"- KMC vacancy source: {generated_event.get('vacancy_source', state_transform.get('vacancy_source', '—'))}",
        f"- pair-site remap: {state_transform.get('converted_pair_markers', '—')} -> type {state_transform.get('pair_marker_host_type', '—')}",
        f"- pair sites from data.lmp: {state_transform.get('pair_sites_from_data_lmp', '—')}",
        f"- pair sites fallback host type: {state_transform.get('defaulted_pair_sites', '—')}",
        f"- barrier assignment: Mo={barrier_assignment.get('Mo', '—')} eV, Re={barrier_assignment.get('Re', '—')} eV",
    ]
    if lammps_event_binding.get("expected_pair"):
        expected_pair = lammps_event_binding["expected_pair"]
        vacancy = expected_pair.get("vacancy") or {}
        jump = expected_pair.get("jump") or {}
        pair_label = (
            "LAMMPS requested jump pair"
            if lammps_event_binding.get("mode") == "event_json_requested_model"
            else "LAMMPS case jump pair"
        )
        lines.append(
            f"- {pair_label}: "
            f"vacancy site {vacancy.get('site_id', '—')} -> jump site {jump.get('site_id', '—')}"
        )
        lines.append(f"- LAMMPS case pair source: {lammps_event_binding.get('source', '—')}")
        if lammps_event_binding.get("matches_requested_event") is not None:
            lines.append(f"- LAMMPS event matched request: {lammps_event_binding.get('matches_requested_event')}")
    if lammps_model:
        lines.append(f"- LAMMPS model mode: {lammps_model.get('mode', '—')}")
        if lammps_model.get("generated_data_final_lmp"):
            lines.append(f"- LAMMPS generated final data: {lammps_model.get('generated_data_final_lmp')}")
        if lammps_model.get("kmc_data_lmp_assist"):
            lines.append(f"- KMC lattice assist data.lmp: {lammps_model.get('kmc_data_lmp_assist')}")
    if ensemble:
        accepted_stats = (ensemble.get("metrics") or {}).get("accepted_events") or {}
        final_time_stats = (ensemble.get("metrics") or {}).get("final_time") or {}
        lines.extend(
            [
                f"- KMC seeds: {ensemble.get('seeds', [])}",
                f"- KMC completed seeds: {ensemble.get('completed_count', 0)}/{ensemble.get('count', 0)}",
                f"- KMC representative seed: {ensemble.get('representative_seed', kmc.get('representative_seed', '—'))}",
            ]
        )
        if accepted_stats.get("count"):
            lines.append(
                f"- KMC accepted mean±std: {accepted_stats.get('mean'):.3f} ± {accepted_stats.get('std'):.3f}"
            )
        if final_time_stats.get("count"):
            lines.append(
                f"- KMC final time mean±std: {final_time_stats.get('mean'):.6g} ± {final_time_stats.get('std'):.6g}"
            )
    if visualization.get("requested"):
        lines.append(f"- OVITO: {visualization.get('status', '—')}")
        completed = [item.get("output_png") for item in (visualization.get("per_seed") or []) if item.get("output_png")]
        if completed:
            lines.append(f"- OVITO sample: {completed[0]}")
    if visualization.get("comparison_chart_svg"):
        lines.append(f"- comparison chart: {visualization['comparison_chart_svg']}")
    if visualization.get("gif_status") and visualization.get("gif_status") != "disabled":
        lines.append(f"- GIF: {visualization.get('gif_status', '—')}")
        if visualization.get("animated_gif"):
            lines.append(f"- GIF file: {visualization['animated_gif']}")
    if lammps_visualization:
        lines.append(f"- LAMMPS OVITO: {lammps_visualization.get('status', '—')}")
        lines.append(f"- LAMMPS initial structure: {lammps_visualization.get('initial_structure', '—')}")
        lines.append(f"- LAMMPS final structure: {lammps_visualization.get('final_structure', '—')}")
        if lammps_visualization.get("initial_snapshot"):
            lines.append(f"- LAMMPS initial snapshot: {lammps_visualization['initial_snapshot']}")
        if lammps_visualization.get("final_snapshot"):
            lines.append(f"- LAMMPS final snapshot: {lammps_visualization['final_snapshot']}")
    if dispatch.get("lammps"):
        item = dispatch["lammps"]
        lines.append(f"- LAMMPS dispatch: {item.get('transport', '—')} ({item.get('tool', '—')})")
    if dispatch.get("kmc"):
        item = dispatch["kmc"]
        lines.append(f"- KMC dispatch: {item.get('transport', '—')} ({item.get('tool', '—')})")
    if runtime_health:
        lines.append(f"- runtime health: {runtime_health.get('status')}")
        for item in (runtime_health.get("warnings") or [])[:4]:
            lines.append(f"  - warning: {item}")
    if summary.get("summary_json"):
        lines.append(f"- summary: {summary['summary_json']}")
    return "\n".join(lines)


def format_moire_compare_report(summary: Dict[str, Any]) -> str:
    event_runs = summary.get("event_runs") or []
    barrier_ranking = summary.get("barrier_ranking") or []
    barrier_stats = summary.get("barrier_stats") or {}
    kmc_metrics = summary.get("kmc_metrics") or {}
    warnings = summary.get("warnings") or []
    lines = [
        stylize("MoRe event compare complete", BOLD, GREEN),
        f"- status: {summary.get('status', '—')}",
        f"- source case: {summary.get('case_dir', '—')}",
        f"- workdir: {summary.get('workdir', '—')}",
        f"- events: {summary.get('completed_count', 0)}/{summary.get('event_count', len(event_runs))} completed",
        f"- KMC enabled: {summary.get('run_kmc')}",
        f"- OVITO requested: {summary.get('ovito_requested')}",
    ]
    if barrier_stats.get("count"):
        lines.append(f"- barrier range: {barrier_stats.get('min'):.6f} → {barrier_stats.get('max'):.6f} eV")
    if summary.get("barrier_span_eV") is not None:
        lines.append(f"- barrier span: {float(summary.get('barrier_span_eV')):.6f} eV")
    if summary.get("run_kmc") and summary.get("kmc_seeds"):
        lines.append(f"- KMC seeds: {summary.get('kmc_seeds')}")
    if barrier_ranking:
        lines.append("- barrier ranking:")
        for item in barrier_ranking[:8]:
            delta = item.get("delta_vs_lowest_eV")
            delta_text = "" if delta is None else f" (Δlowest={float(delta):+.6f} eV)"
            lines.append(
                f"  - #{item.get('rank')}: {item.get('label')} = {float(item.get('barrier_eV', 0.0)):.6f} eV{delta_text}"
            )
    if kmc_metrics.get("accepted_events", {}).get("count"):
        accepted = kmc_metrics["accepted_events"]
        lines.append(f"- KMC accepted mean±std: {accepted.get('mean'):.3f} ± {accepted.get('std'):.3f}")
    if kmc_metrics.get("final_time", {}).get("count"):
        final_time = kmc_metrics["final_time"]
        lines.append(f"- KMC final time mean±std: {final_time.get('mean'):.6g} ± {final_time.get('std'):.6g}")
    lines.append("- per event:")
    for item in event_runs[:8]:
        barrier = item.get("barrier_eV")
        barrier_text = "—" if barrier is None else f"{float(barrier):.6f} eV"
        extras = []
        if item.get("accepted_events") is not None:
            extras.append(f"accepted={item.get('accepted_events')}")
        if item.get("final_time") is not None:
            extras.append(f"final_time={item.get('final_time')}")
        if item.get("kmc_visualization_sample"):
            extras.append(f"kmc_ovito={item.get('kmc_visualization_sample')}")
        if item.get("lammps_final_snapshot"):
            extras.append(f"lammps_ovito={item.get('lammps_final_snapshot')}")
        details = f" · {' · '.join(extras)}" if extras else ""
        lines.append(f"  - {item.get('label')}: [{item.get('status', '—')}] barrier={barrier_text}{details}")
        if item.get("summary_json"):
            lines.append(f"    summary: {item.get('summary_json')}")
        if item.get("error"):
            lines.append(f"    error: {item.get('error')}")
    if warnings:
        lines.append("- warnings:")
        for item in warnings[:6]:
            lines.append(f"  - {item}")
    if summary.get("comparison_json"):
        lines.append(f"- comparison: {summary['comparison_json']}")
    if summary.get("summary_json"):
        lines.append(f"- summary: {summary['summary_json']}")
    return "\n".join(lines)


def format_moire_diffusion_sweep_report(summary: Dict[str, Any]) -> str:
    temperature_runs = summary.get("temperature_runs") or []
    arrhenius_fit = summary.get("arrhenius_fit") or {}
    temperature_trend = summary.get("temperature_trend") or {}
    diffusion_stats = summary.get("diffusion_stats") or {}
    lammps_visualization = summary.get("lammps_visualization") or {}
    warnings = summary.get("warnings") or []
    lines = [
        stylize("MoRe diffusion-vs-temperature sweep complete", BOLD, GREEN),
        f"- status: {summary.get('status', '—')}",
        f"- source case: {summary.get('case_dir', '—')}",
        f"- event.json: {summary.get('event_json', '—')}",
        f"- workdir: {summary.get('workdir', '—')}",
        f"- barrier: {float(summary.get('barrier_eV', 0.0)):.6f} eV",
        f"- temperatures (K): {summary.get('temperatures_k')}",
        f"- KMC seeds: {summary.get('kmc_seeds')}",
        f"- run_time: {summary.get('run_time', '—')}",
        f"- stats_step: {summary.get('stats_step', '—')}",
        f"- completed temperatures: {summary.get('completed_count', 0)}/{len(temperature_runs)}",
        f"- OVITO requested: {summary.get('ovito_requested')}",
    ]
    if lammps_visualization:
        lines.append(f"- LAMMPS OVITO: {lammps_visualization.get('status', '—')}")
        if lammps_visualization.get("final_snapshot"):
            lines.append(f"- LAMMPS final snapshot: {lammps_visualization['final_snapshot']}")
    if diffusion_stats.get("count"):
        lines.append(
            f"- diffusion mean±std: {float(diffusion_stats.get('mean', 0.0)):.6g} ± {float(diffusion_stats.get('std', 0.0)):.6g}"
        )
    if temperature_trend.get("monotonic_increasing") is not None:
        lines.append(f"- diffusion increases with temperature: {temperature_trend.get('monotonic_increasing')}")
    if arrhenius_fit.get("activation_energy_eV") is not None:
        lines.append(f"- Arrhenius fitted activation energy: {float(arrhenius_fit['activation_energy_eV']):.6f} eV")
    lines.append("- per temperature:")
    for item in temperature_runs[:12]:
        coeff = item.get("diffusion_coefficient")
        coeff_text = "—" if coeff is None else f"{float(coeff):.6g}"
        details: List[str] = []
        if item.get("diffusion_std") is not None and coeff is not None:
            details.append(f"std={float(item.get('diffusion_std') or 0.0):.3g}")
        if item.get("jump_frequency_hz") is not None:
            details.append(f"jump_freq={float(item['jump_frequency_hz']):.6g} Hz")
        if item.get("accepted_events") is not None:
            details.append(f"accepted={item.get('accepted_events')}")
        if item.get("ovito_sample"):
            details.append(f"ovito={item.get('ovito_sample')}")
        suffix = f" · {' · '.join(details)}" if details else ""
        lines.append(f"  - {item.get('label')}: [{item.get('status', '—')}] D={coeff_text}{suffix}")
        if item.get("error"):
            lines.append(f"    error: {item.get('error')}")
        if item.get("kmc_summary_json"):
            lines.append(f"    summary: {item.get('kmc_summary_json')}")
    files = summary.get("files") or {}
    if files.get("diffusion_vs_temperature_csv"):
        lines.append(f"- diffusion csv: {files['diffusion_vs_temperature_csv']}")
    if files.get("diffusion_vs_temperature_svg"):
        lines.append(f"- diffusion chart: {files['diffusion_vs_temperature_svg']}")
    if files.get("arrhenius_csv"):
        lines.append(f"- arrhenius csv: {files['arrhenius_csv']}")
    if files.get("arrhenius_svg"):
        lines.append(f"- arrhenius chart: {files['arrhenius_svg']}")
    if summary.get("summary_json"):
        lines.append(f"- summary: {summary['summary_json']}")
    if warnings:
        lines.append("- warnings:")
        for item in warnings[:6]:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def format_progress_event(stage: str, payload: Dict[str, Any]) -> str:
    labels = {
        "autonomy.run.start": "开始准备 autonomy 运行",
        "autonomy.workspace.start": "开始生成工作区",
        "autonomy.provider.resolved": "provider 已解析",
        "autonomy.template.selected": "模板已选定",
        "autonomy.files.materialized": "工作区文件已生成",
        "autonomy.workspace.ready": "草稿工作区已准备完成",
        "autonomy.plan.ready": "执行计划已生成",
        "autonomy.validation.start": "开始 dry-run 验证",
        "autonomy.validation.complete": "dry-run 验证完成",
        "autonomy.final_run.start": "开始真实运行",
        "autonomy.final_run.complete": "真实运行完成",
        "autonomy.run.complete": "autonomy 任务完成",
        "executor.job.initialized": "执行器已初始化",
        "executor.job.resume": "准备从已有执行进度继续",
        "executor.job.recovery_plan": "已生成恢复执行计划",
        "executor.job.start": "开始按计划执行作业",
        "executor.job.cancelled": "作业执行已取消",
        "executor.step.start": "开始执行单个步骤",
        "executor.step.resume": "准备恢复单个步骤",
        "executor.step.checkpoint_ready": "步骤检查点已就绪",
        "executor.step.complete": "单个步骤执行完成",
        "executor.step.failed": "单个步骤执行失败",
        "executor.step.cancelled": "单个步骤已取消",
        "executor.step.skipped": "跳过已完成步骤",
        "executor.step.heartbeat": "步骤仍在运行",
        "executor.job.complete": "作业执行完成",
        "bridge.start": "开始桥接 LAMMPS 能垒到 KMC",
        "bridge.command.complete": "桥接脚本执行完成",
        "bridge.complete": "KMC bridge 完成",
        "mcp.tool.start": "开始通过 MCP 调用工具",
        "mcp.tool.complete": "MCP 工具调用完成",
        "moire.run.start": "开始 MoRe 全链路运行",
        "moire.case.copied": "MoRe 算例已复制到工作目录",
        "moire.lammps.start": "开始运行 LAMMPS/NEB",
        "moire.lammps.complete": "LAMMPS/NEB 完成",
        "moire.lammps.failed": "LAMMPS/NEB 失败",
        "moire.postprocess.start": "开始整理 NEB 输出",
        "moire.postprocess.complete": "NEB 输出整理完成",
        "moire.postprocess.failed": "NEB 输出整理失败",
        "moire.barrier.ready": "已从 LAMMPS/NEB 提取迁移能垒",
        "moire.kmc.start": "开始写入 repo KMC 输入并运行 KMC",
        "moire.kmc.complete": "repo KMC 运行完成",
        "moire.kmc.failed": "repo KMC 运行失败",
        "moire.run.complete": "MoRe 全链路运行完成",
        "moire.compare.start": "开始比较多个 MoRe 事件",
        "moire.compare.event.start": "开始运行单个比较事件",
        "moire.compare.event.complete": "单个比较事件完成",
        "moire.compare.event.failed": "单个比较事件失败",
        "moire.compare.complete": "MoRe 事件比较完成",
        "moire.diffusion.start": "开始构建扩散系数-温度关系算例",
        "moire.diffusion.temperature.start": "开始运行单个温度点的 KMC",
        "moire.diffusion.temperature.completed": "单个温度点的 KMC 已完成",
        "moire.diffusion.temperature.failed": "单个温度点的 KMC 失败",
        "moire.diffusion.temperature.warning": "单个温度点的 KMC 部分完成",
        "moire.diffusion.complete": "扩散系数-温度关系算例完成",
    }
    base = labels.get(stage, stage)
    extras = []
    if payload.get("template"):
        extras.append(f"template={payload['template']}")
    if payload.get("mode"):
        extras.append(f"mode={payload['mode']}")
    if payload.get("provider_used"):
        extras.append(f"provider={payload['provider_used']}")
    if payload.get("workspace_dir"):
        extras.append(f"workspace={payload['workspace_dir']}")
    if payload.get("run_dir"):
        extras.append(f"run={payload['run_dir']}")
    if payload.get("output_dir"):
        extras.append(f"output={payload['output_dir']}")
    if payload.get("tool"):
        extras.append(f"tool={payload['tool']}")
    if payload.get("purpose"):
        extras.append(str(payload["purpose"]))
    if payload.get("barrier_eV") is not None:
        extras.append(f"barrier={float(payload['barrier_eV']):.6f} eV")
    if payload.get("steps") is not None:
        extras.append(f"steps={payload['steps']}")
    if payload.get("recovery_steps") is not None:
        extras.append(f"recovery_steps={payload['recovery_steps']}")
    if payload.get("step_id"):
        extras.append(f"step={payload['step_id']}")
    if payload.get("final_run_dir"):
        extras.append(f"final={payload['final_run_dir']}")
    if payload.get("workdir"):
        extras.append(f"workdir={payload['workdir']}")
    if payload.get("bridge_workdir"):
        extras.append(f"bridge={payload['bridge_workdir']}")
    if payload.get("copied_case_dir"):
        extras.append(f"case={payload['copied_case_dir']}")
    if payload.get("log"):
        extras.append(f"log={payload['log']}")
    if payload.get("health"):
        extras.append(f"health={payload['health']}")
    if payload.get("status"):
        extras.append(f"status={payload['status']}")
    if payload.get("previous_status"):
        extras.append(f"previous={payload['previous_status']}")
    if payload.get("resume_existing") is not None:
        extras.append(f"resume={payload['resume_existing']}")
    if payload.get("overwrite_existing") is not None:
        extras.append(f"overwrite={payload['overwrite_existing']}")
    if payload.get("checkpoint_kind"):
        extras.append(f"checkpoint={payload['checkpoint_kind']}")
    if payload.get("recovery_action"):
        extras.append(f"recovery={payload['recovery_action']}")
    if payload.get("decision_reason"):
        extras.append(f"reason={payload['decision_reason']}")
    if payload.get("missing_outputs"):
        missing = payload["missing_outputs"]
        if isinstance(missing, list) and missing:
            extras.append(f"missing={','.join(missing[:2])}")
    if payload.get("drifted_outputs"):
        drifted = payload["drifted_outputs"]
        if isinstance(drifted, list) and drifted:
            extras.append(f"drifted={','.join(drifted[:2])}")
    if payload.get("invalidated_by"):
        extras.append(f"invalidated_by={payload['invalidated_by']}")
    if payload.get("safe_validation_passed") is not None:
        extras.append(f"safe={payload['safe_validation_passed']}")
    if payload.get("validation_passed") is not None:
        extras.append(f"validated={payload['validation_passed']}")
    return f"[progress] {base}" + (f" · {' | '.join(extras)}" if extras else "")
