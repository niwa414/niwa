#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SECTIONS = [
    ("gpu_runtime_proof", "source_path", "source_sha256"),
    ("private_shot_dataset_binding", "dataset_path", "dataset_sha256"),
    ("private_shot_dataset_binding", "calibration_report_path", "calibration_report_sha256"),
    ("private_hardware_model_binding", "model_path", "model_sha256"),
    ("private_hardware_model_binding", "validation_report_path", "validation_report_sha256"),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_path(repo_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill SHA-256 fields for existing internal parity manifest artifacts.")
    parser.add_argument(
        "--manifest",
        default="evidence/internal/helion_internal_parity_manifest.json",
        help="Manifest JSON path.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even when hash field already has a value.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = resolve_path(repo_root, args.manifest)
    if manifest_path is None:
        raise SystemExit("Manifest path is empty.")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Manifest is invalid JSON: {manifest_path}")

    updates = 0
    checked = 0
    for section_key, path_key, hash_key in SECTIONS:
        section = manifest.get(section_key)
        if not isinstance(section, dict):
            continue
        raw_path = section.get(path_key)
        path = resolve_path(repo_root, raw_path)
        if path is None or not path.exists():
            continue
        checked += 1
        if (not args.force) and str(section.get(hash_key) or "").strip():
            continue
        section[hash_key] = sha256_file(path)
        updates += 1

    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"manifest={manifest_path}")
    print(f"checked_artifacts={checked}")
    print(f"updated_hashes={updates}")


if __name__ == "__main__":
    main()
