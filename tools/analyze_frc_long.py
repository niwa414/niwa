import sys
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def add_athena_vis_path():
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
        
        # 2. Check Time Step
        print("\n--- Time Step Check ---")
        print(f"dt range: [{dt.min():.2e}, {dt.max():.2e}]")
        print(f"Final dt: {dt[-1]:.2e}")
        
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
        plot_path = output_dir / "frc_long_conservation.png"
        plt.savefig(plot_path)
        print(f"Saved conservation plot to {plot_path}")

    except Exception as e:
        print(f"Error reading history file: {e}")

def analyze_vtk_series(data_dir):
    data_dir = Path(data_dir)
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "plots" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_files = sorted(glob.glob(str(data_dir / "frc_merge.block0.out1.*.vtk")))
    if not vtk_files:
        print("No VTK files found.")
        return

    print(f"\nFound {len(vtk_files)} VTK files. Analyzing full time series...")
    
    times = []
    max_rho = []
    max_press = []
    max_B = []
    max_v = []
    
    # Indices for slice plotting (Initial, Middle, Final)
    idx_plot = [0, len(vtk_files)//2, len(vtk_files)-1]
    
    for i, vtk_file in enumerate(vtk_files):
        try:
            # Read time
            time_val = get_vtk_time(vtk_file)
            if time_val is None: continue
            
            # Read data
            x, y, z, data = athena_read.vtk(vtk_file)
            
            rho = data['rho'][0]
            press = data['press'][0]
            vel = data['vel'][0]
            bcc = data['Bcc'][0]
            
            vx, vy = vel[:,:,0], vel[:,:,1]
            bx, by = bcc[:,:,0], bcc[:,:,1]
            b_mag = np.sqrt(bx**2 + by**2)
            v_mag = np.sqrt(vx**2 + vy**2)
            
            # Store max values
            times.append(time_val)
            max_rho.append(rho.max())
            max_press.append(press.max())
            max_B.append(b_mag.max())
            max_v.append(v_mag.max())
            
            # Plot slices for selected frames
            if i in idx_plot:
                label = f"t={time_val:.1f}"
                print(f"  Plotting slice for {label}...")
                
                xc = 0.5 * (x[:-1] + x[1:])
                yc = 0.5 * (y[:-1] + y[1:])
                X, Y = np.meshgrid(xc, yc)
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Density + B-lines
                im0 = axes[0,0].pcolormesh(X, Y, rho, cmap='viridis', shading='auto')
                axes[0,0].streamplot(X, Y, bx, by, color='white', linewidth=0.5, density=1.5)
                axes[0,0].set_title(f'{label} Density & B-lines')
                fig.colorbar(im0, ax=axes[0,0], label='rho')
                axes[0,0].set_aspect('equal')

                # Pressure
                im1 = axes[0,1].pcolormesh(X, Y, press, cmap='inferno', shading='auto')
                axes[0,1].set_title(f'{label} Pressure')
                fig.colorbar(im1, ax=axes[0,1], label='press')
                axes[0,1].set_aspect('equal')
                
                # |B|
                im2 = axes[1,0].pcolormesh(X, Y, b_mag, cmap='plasma', shading='auto')
                axes[1,0].set_title(f'{label} |B|')
                fig.colorbar(im2, ax=axes[1,0], label='|B|')
                axes[1,0].set_aspect('equal')
                
                # Velocity
                skip = 4
                axes[1,1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                                 vx[::skip, ::skip], vy[::skip, ::skip], scale=20, width=0.002)
                axes[1,1].set_title(f'{label} Velocity Field')
                axes[1,1].set_aspect('equal')
                
                plt.tight_layout()
                outfile = output_dir / f"frc_slice_t{int(time_val):02d}.png"
                plt.savefig(outfile)
        
        except Exception as e:
            print(f"Error processing {vtk_file}: {e}")
            
    # Plot Time Series of Peak Values
    print("\nGenerating time series plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    axes[0,0].plot(times, max_rho, 'b-o')
    axes[0,0].set_ylabel('Max Density')
    axes[0,0].set_title('Peak Density Evolution')
    axes[0,0].grid(True)
    
    axes[0,1].plot(times, max_press, 'r-o')
    axes[0,1].set_ylabel('Max Pressure')
    axes[0,1].set_title('Peak Pressure Evolution (Heating)')
    axes[0,1].grid(True)
    
    axes[1,0].plot(times, max_B, 'g-o')
    axes[1,0].set_ylabel('Max |B|')
    axes[1,0].set_title('Peak Magnetic Field')
    axes[1,0].grid(True)
    
    axes[1,1].plot(times, max_v, 'k-o')
    axes[1,1].set_ylabel('Max |v|')
    axes[1,1].set_title('Peak Velocity (Decay)')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / "frc_long_peaks.png"
    plt.savefig(plot_path)
    print(f"Saved time series plot to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = str(Path(__file__).resolve().parents[1] / "outputs" / "mhd" / "frc_long_full")

    analyze_hst(data_dir)
    analyze_vtk_series(data_dir)
