#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def git_sha(root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def format_value(value: float) -> str:
    return f"{int(round(value * 1000)):04d}"


def update_inputs(case_dir: Path, case: dict, knob: str, value: float) -> list[str]:
    updated = []
    for rel in case.get("inputs", []):
        path = case_dir / rel
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        changed = False
        if isinstance(data, dict):
            if knob == "drift" and "opmd_double_seed_drift" in data:
                drift = data.get("opmd_double_seed_drift")
                if isinstance(drift, list) and drift:
                    drift[0] = float(value)
                    data["opmd_double_seed_drift"] = drift
                    changed = True
            if knob == "shift" and "opmd_double_seed_shift" in data:
                shift = data.get("opmd_double_seed_shift")
                if isinstance(shift, list) and shift:
                    shift[0] = float(value)
                    data["opmd_double_seed_shift"] = shift
                    changed = True
        if changed:
            path.write_text(json.dumps(data, indent=2) + "\n")
            updated.append(rel)
    return updated


def read_shift0(case_dir: Path, case: dict) -> float | None:
    for rel in case.get("inputs", []):
        path = case_dir / rel
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "opmd_double_seed_shift" in data:
            shift = data.get("opmd_double_seed_shift")
            if isinstance(shift, list) and shift:
                return float(shift[0])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Create 1D sweep cases by cloning a base case.")
    parser.add_argument("--base-case", required=True, help="Path to base case.json")
    parser.add_argument("--out-dir", required=True, help="Output cases directory")
    parser.add_argument("--knob", required=True, help="Knob name (e.g., drift)")
    parser.add_argument("--values", nargs="+", type=float, help="Sweep values")
    parser.add_argument("--deltas", nargs="+", type=float, help="Sweep deltas (for shift)")
    args = parser.parse_args()

    base_case_path = Path(args.base_case)
    if not base_case_path.exists():
        raise SystemExit(f"base_case_not_found: {base_case_path}")
    base_case_dir = base_case_path.parent
    base = json.loads(base_case_path.read_text())
    base_id = base.get("id") or base_case_dir.name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    sha = git_sha(repo_root)

    values = args.values
    shift0 = None
    if args.deltas:
        shift0 = read_shift0(base_case_dir, base)
        if shift0 is None:
            raise SystemExit("shift0_not_found_in_inputs")
        values = [shift0 + delta for delta in args.deltas]
    if not values:
        raise SystemExit("no_values_provided")

    for idx, value in enumerate(values):
        suffix = format_value(value)
        if args.deltas:
            delta = args.deltas[idx]
            delta_suffix = f"{'p' if delta >= 0 else 'n'}{format_value(abs(delta) / 1.0)}"
            new_id = f"{base_id}-{args.knob}{delta_suffix}"
        else:
            new_id = f"{base_id}-{args.knob}{suffix}"
        new_dir = out_dir / new_id
        if new_dir.exists():
            raise SystemExit(f"case_dir_exists: {new_dir}")
        shutil.copytree(base_case_dir, new_dir)

        case = json.loads((new_dir / "case.json").read_text())
        case["id"] = new_id
        case["description"] = f"{case.get('description','').rstrip()} sweep {args.knob}={value}".strip()
        meta = case.get("metadata", {})
        meta.setdefault("sweep", {})
        meta["sweep"].update(
            {
                "base_case": base_id,
                "knob": args.knob,
                "value": float(value),
                "git_sha": sha,
            }
        )
        if shift0 is not None and args.deltas:
            meta["sweep"].update(
                {
                    "shift0": float(shift0),
                    "delta": float(args.deltas[idx]),
                }
            )
        case["metadata"] = meta

        updated = update_inputs(new_dir, case, args.knob, float(value))
        if not updated:
            print(f"[warn] no_drift_field_updated in {new_id}")

        (new_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        print(f"{new_id} drift={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
