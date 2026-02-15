import sys
import os
import glob
import argparse
from pathlib import Path
import numpy as np


def add_athena_vis_path():
    env_path = os.environ.get("ATHENA_VIS_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "athena-public-version-21.0" / "vis" / "python",
            repo_root / "athena-24.0" / "vis" / "python",
        ]
    )
    for path in candidates:
        if path.exists():
            sys.path.append(str(path))
            return True
    return False


if not add_athena_vis_path():
    raise SystemExit(
        "athena_read not found. Set ATHENA_VIS_PATH or keep athena-public-version-21.0/vis/python available."
    )

import athena_read

def inspect_vtk(filepath):
    print(f"\nInspecting: {os.path.basename(filepath)}")
    try:
        x_faces, y_faces, z_faces, data = athena_read.vtk(filepath)
        
        print(f"Dimensions: ({len(z_faces)-1}, {len(y_faces)-1}, {len(x_faces)-1})")
        print(f"Variables: {list(data.keys())}")
        
        for key in data:
            val = data[key]
            if val.ndim == 4: # Vector (1, ny, nx, 3)
                for i, comp in enumerate(['x', 'y', 'z']):
                    v = val[0, :, :, i]
                    print(f"  {key}_{comp}: min={v.min():.4e}, max={v.max():.4e}, mean={v.mean():.4e}")
            else: # Scalar (1, ny, nx)
                v = val[0]
        print(f"  {key}: min={v.min():.4e}, max={v.max():.4e}, mean={v.mean():.4e}")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Quickly inspect Athena++ VTK files.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=str(repo_root / "outputs" / "mhd"),
        help="Directory containing VTK files (default: outputs/mhd under repo root).",
    )
    args = parser.parse_args()

    vtk_files = sorted(glob.glob(os.path.join(args.data_dir, "*.vtk")))
    if not vtk_files:
        print(f"No VTK files found under {args.data_dir}.")
        return

    inspect_vtk(vtk_files[0])
    if len(vtk_files) > 1:
        inspect_vtk(vtk_files[-1])


if __name__ == "__main__":
    main()
