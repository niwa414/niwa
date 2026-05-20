from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class Step:
    id: str
    stage: str
    description: str
    mutating: bool = True
    resumable: bool = False
    checkpoint_kind: Optional[str] = None


def build_plan(spec: Dict) -> List[Step]:
    mode = spec["mode"]
    plan: List[Step] = []

    if mode in {"md_only", "md_to_kmc_chain"}:
        plan.append(
            Step(
                "md.run",
                "md",
                "运行或接入 MD 阶段，产出 barrier 数据",
                resumable=True,
                checkpoint_kind="state_store",
            )
        )

    if mode == "md_to_kmc_chain":
        plan.append(
            Step(
                "chain.compile_event_table",
                "chain",
                "把 MD barrier 编译成事件表、速率和 KMC barrier 映射",
                checkpoint_kind="artifacts",
            )
        )

    if mode in {"kmc_only", "md_to_kmc_chain"}:
        plan.append(
            Step(
                "kmc.prepare_input",
                "kmc",
                "准备 KMC 输入文件与配套资产",
                checkpoint_kind="artifacts",
            )
        )
        plan.append(
            Step(
                "kmc.run",
                "kmc",
                "运行或模拟运行 KMC 长时演化",
                resumable=True,
                checkpoint_kind="state_store",
            )
        )

    plan.append(
        Step(
            "explain.summary",
            "explain",
            "生成面向科研人员的解释性总结",
            checkpoint_kind="report",
        )
    )
    plan.append(
        Step(
            "archive.results",
            "archive",
            "登记产物，生成归档清单",
            checkpoint_kind="archive",
        )
    )
    return plan


def build_plan_payload(spec: Dict) -> List[Dict[str, str]]:
    return [asdict(step) for step in build_plan(spec)]
