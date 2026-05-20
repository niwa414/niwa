import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
POTENTIAL_PATH = REPO_ROOT / "crystalkmc-fix-diffusion-coef/TEST/1Compatibility/case2/FeCuNi.eam.alloy"
BARRIER_TEMPLATE_PATH = REPO_ROOT / "examples/sample-data/md/barriers.fe-cu-ni.json"


def build_lammps_input(potential_path: Path) -> str:
    return f"""# Minimal LAMMPS reference-energy demo for Miet Claw
units           metal
dimension       3
boundary        p p p
atom_style      atomic

lattice         bcc 2.86
region          box block 0 4 0 4 0 4
create_box      3 box
create_atoms    1 box

mass            1 55.845
mass            2 63.546
mass            3 58.6934

pair_style      eam/alloy
pair_coeff      * * {potential_path} Fe Cu Ni

neighbor        2.0 bin
compute         peratom all pe/atom
thermo          0
run             0

print           "Total energy: $(etotal) eV"
print           "Potential energy per atom: $(pe/atoms) eV"
"""


def parse_total_energy(stdout: str) -> float:
    match = re.search(r"Total energy:\s*([-+0-9.eE]+)\s*eV", stdout)
    if not match:
        raise RuntimeError("Could not find 'Total energy' in LAMMPS output")
    return float(match.group(1))


def main() -> int:
    if not POTENTIAL_PATH.exists():
        raise FileNotFoundError(f"Potential file not found: {POTENTIAL_PATH}")

    workdir = Path.cwd()
    input_path = workdir / "in.reference.lmp"
    stdout_path = workdir / "lammps_demo.out"
    barrier_output_path = workdir / "barriers.generated.json"

    input_path.write_text(build_lammps_input(POTENTIAL_PATH), encoding="utf-8")
    completed = subprocess.run(
        ["lmp", "-in", input_path.name],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"LAMMPS demo command failed with return code {completed.returncode}")

    total_energy = parse_total_energy(completed.stdout)
    payload = json.loads(BARRIER_TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["source"] = "lammps-reference-energy-demo-adapter"
    payload["metadata"] = {
        "reference_energy_ev": total_energy,
        "lammps_stdout": stdout_path.name,
        "lammps_input": input_path.name,
        "potential_path": str(POTENTIAL_PATH),
    }
    barrier_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {barrier_output_path.name}")
    print(f"Reference total energy: {total_energy:.6f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
