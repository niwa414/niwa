import json
from pathlib import Path
from typing import Any, Dict

SUPPORTED_MODES = {"md_only", "kmc_only", "md_to_kmc_chain"}


class SpecError(ValueError):
    pass


def _resolve_maybe_path(value: str, base_dir: Path) -> str:
    if not isinstance(value, str):
        return value
    if value.startswith(".") or "/" in value:
        return str((base_dir / value).resolve())
    return value


def validate_job_spec(spec: Dict[str, Any]) -> None:
    mode = spec.get("mode")
    if mode not in SUPPORTED_MODES:
        raise SpecError(f"Unsupported mode: {mode}")
    if not spec.get("job_id"):
        raise SpecError("job_id is required")

    if mode in {"md_only", "md_to_kmc_chain"} and "md" not in spec:
        raise SpecError(f"mode={mode} requires md section")
    if mode in {"kmc_only", "md_to_kmc_chain"} and "kmc" not in spec:
        raise SpecError(f"mode={mode} requires kmc section")

    if "kmc" in spec:
        kmc = spec["kmc"]
        if "temperature_k" not in kmc:
            raise SpecError("kmc.temperature_k is required")
        if "template" not in kmc:
            raise SpecError("kmc.template is required")

    if mode == "kmc_only":
        precomputed = spec["kmc"]["template"].get("precomputed_barriers")
        if not precomputed:
            raise SpecError("kmc_only requires kmc.template.precomputed_barriers")


def load_job_spec(path: str) -> Dict[str, Any]:
    spec_path = Path(path).resolve()
    base_dir = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_job_spec(spec)

    md = spec.get("md")
    if md:
        if "barriers_source" in md:
            md["barriers_source"] = _resolve_maybe_path(md["barriers_source"], base_dir)
        if "command" in md:
            md["command"] = [_resolve_maybe_path(part, base_dir) for part in md["command"]]
        if "working_dir" in md:
            md["working_dir"] = _resolve_maybe_path(md["working_dir"], base_dir)

    kmc = spec.get("kmc")
    if kmc:
        if "command" in kmc:
            kmc["command"] = [_resolve_maybe_path(part, base_dir) for part in kmc["command"]]
        template = kmc.get("template", {})
        if "cluster_xyz" in template:
            template["cluster_xyz"] = _resolve_maybe_path(template["cluster_xyz"], base_dir)
        assets = template.get("potential_assets", [])
        template["potential_assets"] = [_resolve_maybe_path(asset, base_dir) for asset in assets]

    spec["_meta"] = {
        "job_spec_path": str(spec_path),
        "job_spec_dir": str(base_dir),
    }
    return spec
