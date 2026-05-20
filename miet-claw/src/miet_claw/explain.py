import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_event_table(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_diffusion_row(path: Path) -> Dict:
    rows = _read_event_table(path)
    return rows[0] if rows else {}


def _execution_mode_lines(stage: str, payload: Dict) -> List[str]:
    if not payload:
        return []
    pieces = [f"- {stage}: mode={payload.get('mode', 'unknown')}"]
    if payload.get("diffusion_mode"):
        pieces[0] += f", diffusion={payload.get('diffusion_mode')}"
    if payload.get("reason"):
        pieces[0] += f", reason={payload.get('reason')}"
    return pieces


def write_summary(run_dir: Path, spec: Dict, state: Dict) -> Path:
    explain_dir = run_dir / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)
    summary_path = explain_dir / "summary.md"

    event_rows = _read_event_table(run_dir / "artifacts" / "chain" / "event_table.csv")
    if not event_rows:
        event_rows = _read_event_table(run_dir / "artifacts" / "kmc" / "event_table.csv")
    md_barriers = _read_json(run_dir / "artifacts" / "md" / "barriers.json")
    md_execution = _read_json(run_dir / "artifacts" / "md" / "md_execution.json")
    kmc_execution = _read_json(run_dir / "artifacts" / "kmc" / "kmc_execution.json")
    diffusion_row = _read_first_diffusion_row(run_dir / "artifacts" / "kmc" / "diffusion.csv")

    lines = [
        f"# Job Summary: {spec['job_id']}",
        "",
        f"- 模式：`{spec['mode']}`",
        f"- 材料体系：{spec.get('material_system', {}).get('name', '未命名')}",
        "",
        "## 执行状态",
    ]

    for step_id, record in state.get("steps", {}).items():
        lines.append(f"- `{step_id}`: {record.get('status', 'unknown')}")

    execution_lines = _execution_mode_lines("MD", md_execution) + _execution_mode_lines("KMC", kmc_execution)
    if execution_lines:
        lines.extend(["", "## 执行来源 / 真实性", ""])
        lines.extend(execution_lines)
        has_simulated = any(
            payload.get("mode") in {"dry-run", "simulated"} or payload.get("diffusion_mode") == "simulated"
            for payload in [md_execution, kmc_execution]
            if payload
        )
        if has_simulated:
            lines.append("- 注意：这个 run 包含 dry-run 或模拟输出，不能当作完整真实计算结果。")

    if event_rows:
        lines.extend(["", "## 关键 barrier / rate", ""])
        for row in event_rows:
            lines.append(
                f"- {row['event_id']}: {row['species']} barrier={float(row['barrier_ev']):.4f} eV, rate={float(row['rate_hz']):.4e} Hz @ {row['temperature_k']} K"
            )

    reference_energy = md_barriers.get("metadata", {}).get("reference_energy_ev")
    workflow_kind = md_barriers.get("metadata", {}).get("workflow_kind")
    neb_images = md_barriers.get("metadata", {}).get("neb_images")
    barrier_source_mode = md_barriers.get("metadata", {}).get("barrier_source_mode")
    parsed_species_count = md_barriers.get("metadata", {}).get("parsed_species_count")
    species_count = md_barriers.get("metadata", {}).get("species_count")
    if reference_energy is not None:
        lines.extend(
            [
                "",
                "## MD 参考信息",
                "",
                f"- LAMMPS reference total energy: {float(reference_energy):.6f} eV",
            ]
        )
    elif workflow_kind:
        lines.extend(
            [
                "",
                "## MD 工作流信息",
                "",
                f"- workflow: {workflow_kind}",
            ]
        )

    if workflow_kind and neb_images is not None:
        lines.append(f"- NEB images: {neb_images}")
    if barrier_source_mode:
        label = barrier_source_mode
        if parsed_species_count is not None and species_count is not None:
            label = f"{label} ({parsed_species_count}/{species_count} species parsed)"
        lines.append(f"- barrier source: {label}")

    if diffusion_row:
        lines.extend(
            [
                "",
                "## KMC 结果摘要",
                "",
                f"- jumps={diffusion_row.get('jumps')}, simulation_time={diffusion_row.get('simulation_time')}, diffusion_coefficient={diffusion_row.get('diffusion coefficient')}",
            ]
        )

    lines.extend(
        [
            "",
            "## 产物位置",
            "",
            f"- 状态文件：`{run_dir / 'state.json'}`",
            f"- 归档清单：`{run_dir / 'archive' / 'manifest.json'}`",
        ]
    )

    if spec["mode"] == "md_to_kmc_chain":
        lines.append(f"- KMC 输入：`{run_dir / 'artifacts' / 'kmc' / 'generated_kmc.in'}`")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "这个输出强调的是链路透明化：先拿到 barrier，再生成事件表与速率，再喂给 KMC。这样后续换引擎时，上层任务定义不用重做。",
        ]
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
