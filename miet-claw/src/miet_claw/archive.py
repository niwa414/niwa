import hashlib
import json
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(run_dir: Path, spec: Dict, state: Dict) -> Dict:
    files: List[Dict] = []
    for root_name in ["artifacts", "explain"]:
        root = run_dir / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(run_dir)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "job_id": spec["job_id"],
        "mode": spec["mode"],
        "job_spec": spec["_meta"]["job_spec_path"],
        "state_file": str((run_dir / "state.json").relative_to(run_dir)),
        "files": files,
        "step_status": {key: value.get("status") for key, value in state.get("steps", {}).items()},
    }


def write_manifest(run_dir: Path, spec: Dict, state: Dict) -> Path:
    manifest = build_manifest(run_dir, spec, state)
    archive_dir = run_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
