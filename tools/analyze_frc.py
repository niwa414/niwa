import sys
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def add_athena_vis_path():
    """Resolve athena_read location from env or local checkout."""
    env_path = os.environ.get("ATHENA_VIS_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "athena-24.0" / "vis" / "python",
            repo_root / "athena-public-version-21.0" / "vis" / "python",
        ]
    )
    for path in candidates:
        if path.exists():
            sys.path.append(str(path))
            return True
    return False


if not add_athena_vis_path():
    raise SystemExit(
        "athena_read not found. Set ATHENA_VIS_PATH or keep athena-24.0/vis/python available."
    )

import athena_read

def analyze_hst(data_dir):
    data_dir = Path(data_dir)
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "plots" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    hst_file = data_dir / "frc_merge.hst"
    if not hst_file.exists():
        print(f"Error: History file {hst_file} not found.")
        return

    print(f"Analyzing history file: {hst_file}")
    try:
        hst_data = athena_read.hst(hst_file)
        
        time = hst_data['time']
        dt = hst_data['dt']
        mass = hst_data['mass']
        tot_E = hst_data['tot-E']
        
        # 1. Check Conservation
        mass_err = (mass - mass[0]) / mass[0]
        energy_err = (tot_E - tot_E[0]) / tot_E[0]
        
        print("\n--- Conservation Check ---")
        print(f"Mass relative error range: [{mass_err.min():.2e}, {mass_err.max():.2e}]")
        print(f"Total Energy relative error range: [{energy_err.min():.2e}, {energy_err.max():.2e}]")
        
        if np.abs(mass_err).max() < 1e-10:
            print("  Mass is strictly conserved.")
        else:
            print("  Mass shows small variations (acceptable if small).")
            
        if np.abs(energy_err).max() < 1e-3:
            print("  Energy is well conserved.")
        else:
            print("  Energy shows some drift.")

        # 2. Check Time Step
        print("\n--- Time Step Check ---")
        print(f"dt range: [{dt.min():.2e}, {dt.max():.2e}]")
        print(f"Mean dt: {dt.mean():.2e}")
        
        # Check for sharp drops (e.g., drop by factor of 2)
        dt_ratio = dt[1:] / dt[:-1]
        min_ratio = dt_ratio.min()
        if min_ratio < 0.5:
             print(f"  WARNING: Sharp dt drop detected (min ratio {min_ratio:.2f}). Possible instability.")
        else:
             print("  Time step is stable.")

        # Plot Conservation and dt
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        axes[0].plot(time, mass_err)
        axes[0].set_ylabel('Mass Rel. Err')
        axes[0].set_title('Mass Conservation')
        axes[0].grid(True)
        
        axes[1].plot(time, energy_err)
        axes[1].set_ylabel('Energy Rel. Err')
        axes[1].set_title('Energy Conservation')
        axes[1].grid(True)
        
        axes[2].plot(time, dt)
        axes[2].set_ylabel('dt')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Time Step Evolution')
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_path = output_dir / "frc_conservation.png"
        plt.savefig(plot_path)
        print(f"Saved conservation plot to {plot_path}")

    except Exception as e:
        print(f"Error reading history file: {e}")

def get_vtk_time(filename):
    try:
        with open(filename, 'r', errors='replace') as f:
            f.readline() # Skip first line
            line = f.readline() # Read second line
            if 'time=' in line:
                t_str = line.split('time=')[1].split()[0]
                return float(t_str)
    except Exception:
        pass
    return None

def analyze_vtk(data_dir):
    data_dir = Path(data_dir)
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "plots" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_files = sorted(glob.glob(str(data_dir / "frc_merge.block0.out1.*.vtk")))
    if not vtk_files:
        print("No VTK files found.")
        return

    print(f"\nFound {len(vtk_files)} VTK files.")
    
    # Analyze first and last file
    files_to_analyze = [vtk_files[0], vtk_files[-1]]
    labels = ['Initial', 'Final']
    
    for vtk_file, label in zip(files_to_analyze, labels):
        print(f"\nAnalyzing {label} state: {os.path.basename(vtk_file)}")
        try:
            # Read time manually
            time_val = get_vtk_time(vtk_file)
            time_str = f"{time_val:.2f}" if time_val is not None else "?"

            # Read data
            # athena_read.vtk returns (x, y, z, data)
            # data is a dictionary
            x, y, z, data = athena_read.vtk(vtk_file)
            
            # Assuming 2D simulation (z is dummy)
            # Grid structure: x (x1), y (x2), z (x3)
            # data arrays shape: (nz, ny, nx) -> (1, ny, nx)
            
            # Extract fields (check keys if needed, assuming standard names)
            rho = data['rho'][0] # Density
            press = data['press'][0] # Pressure
            vel = data['vel'][0] # Velocity vector (3 components)
            bcc = data['Bcc'][0] # Magnetic field (3 components)
            
            vx = vel[:,:,0]
            vy = vel[:,:,1]
            bx = bcc[:,:,0]
            by = bcc[:,:,1]
            
            # Coordinates for plotting
            # x and y are cell faces, need cell centers
            xc = 0.5 * (x[:-1] + x[1:])
            yc = 0.5 * (y[:-1] + y[1:])
            X, Y = np.meshgrid(xc, yc)
            
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Density + B-field lines
            im0 = axes[0,0].pcolormesh(X, Y, rho, cmap='viridis', shading='auto')
            axes[0,0].streamplot(X, Y, bx, by, color='white', linewidth=0.5, density=1.5)
            axes[0,0].set_title(f'{label} Density & B-lines (t={time_str})')
            fig.colorbar(im0, ax=axes[0,0], label='rho')
            axes[0,0].set_aspect('equal')

            # 2. Pressure
            im1 = axes[0,1].pcolormesh(X, Y, press, cmap='inferno', shading='auto')
            axes[0,1].set_title(f'{label} Pressure')
            fig.colorbar(im1, ax=axes[0,1], label='press')
            axes[0,1].set_aspect('equal')
            
            # 3. B-field Magnitude
            b_mag = np.sqrt(bx**2 + by**2)
            im2 = axes[1,0].pcolormesh(X, Y, b_mag, cmap='plasma', shading='auto')
            axes[1,0].set_title(f'{label} |B|')
            fig.colorbar(im2, ax=axes[1,0], label='|B|')
            axes[1,0].set_aspect('equal')
            
            # 4. Velocity field (divergence/inflow check)
            # Use quiver for velocity, subsample for clarity
            skip = 4
            axes[1,1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                             vx[::skip, ::skip], vy[::skip, ::skip], scale=20, width=0.002)
            axes[1,1].set_title(f'{label} Velocity Field')
            axes[1,1].set_aspect('equal')
            
            plt.tight_layout()
            outfile = output_dir / f"frc_{label.lower()}.png"
            plt.savefig(outfile)
            print(f"Saved plot to {outfile}")
            
            # Quantitative checks
            print(f"  Max Density: {rho.max():.4e}")
            print(f"  Max Pressure: {press.max():.4e}")
            print(f"  Max |B|: {b_mag.max():.4e}")
            print(f"  Max |v|: {np.sqrt(vx**2 + vy**2).max():.4e}")

        except Exception as e:
            print(f"Error analyzing VTK file: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = str(Path(__file__).resolve().parents[1] / "outputs" / "mhd" / "frc_smoke")

    analyze_hst(data_dir)
    analyze_vtk(data_dir)
