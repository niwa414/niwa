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

def main():
    repo_root = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path(__file__).resolve().parents[1] / "outputs" / "mhd" / "open"

    print(f"Analyzing results in: {data_dir}")
    output_dir = repo_root / "outputs" / "plots" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    hst_file = data_dir / "hall_gem.hst"
    try:
        hst_data = athena_read.hst(str(hst_file))
        print("History Data Tail (last 5 steps):")
        keys_to_show = ['time', 'tot-E', '1-ME', '2-ME', '3-ME', '1-KE', '2-KE', '3-KE'] # Mapping similar to user request
        # Athena++ hst keys might differ slightly from user's generic list, using standard ones
        # Check available keys
        available_keys = [k for k in keys_to_show if k in hst_data]
        
        # Check for NaNs in key fields
        print("\nChecking for NaNs in History Data:")
        nan_found = False
        for k in available_keys:
            if np.isnan(hst_data[k]).any():
                print(f"  WARNING: NaN found in {k}!")
                nan_found = True
        if not nan_found:
            print("  No NaNs found in tracked history variables.")

        # Print header
        print(f"\n{'Step':<5} " + " ".join([f"{k:>12}" for k in available_keys]))
        
        # Print last 5 rows
        num_rows = len(hst_data['time'])
        start_idx = max(0, num_rows - 5)
        for i in range(start_idx, num_rows):
            row_str = f"{i:<5} " + " ".join([f"{hst_data[k][i]:12.4e}" for k in available_keys])
            print(row_str)
            
    except Exception as e:
        print(f"Error reading history file: {e}")

    # 2. Calculate Reconnection Rate from VTK files
    print("\nCalculating Reconnection Rate from VTK files...")
    # Try to find specific problem files first, otherwise default to all vtk
    vtk_pattern = str(data_dir / "hall_gem*.vtk")
    vtk_files = sorted(glob.glob(vtk_pattern))
    if not vtk_files:
        vtk_files = sorted(glob.glob(str(data_dir / "*.vtk")))
    
    rates = []
    
    print(f"Found {len(vtk_files)} VTK files.")
    
    for vtk_file in vtk_files:
        try:
            # Parse time from filename or file content
            # Filename format: hall_gem.block0.out1.00000.vtk
            # We can rely on athena_read to get the time if needed, but let's try to get it from data
            x_faces, y_faces, z_faces, data = athena_read.vtk(vtk_file)
            
            # Coordinates (cell centers)
            x = 0.5 * (x_faces[:-1] + x_faces[1:])
            y = 0.5 * (y_faces[:-1] + y_faces[1:])
            
            # Debug coordinates on first file
            if vtk_file == vtk_files[0]:
                 print(f"  Debug: y range [{y.min():.4f}, {y.max():.4f}], x range [{x.min():.4f}, {x.max():.4f}]")

            # Athena++ 2D: (nz, ny, nx, 3) -> (1, ny, nx, 3)
            # Variables
            if 'Bcc' in data:
                B = data['Bcc'][0] # (ny, nx, 3)
                V = data['vel'][0] # (ny, nx, 3)
            elif 'b' in data:
                B = data['b'][0]
                V = data['v'][0]
            else:
                print(f"Skipping {vtk_file}: missing fields")
                continue
                
            Bx = B[:, :, 0]
            By = B[:, :, 1]
            Bz = B[:, :, 2]
            
            # Check B statistics
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            print(f"  File: {os.path.basename(vtk_file)}")
            print(f"    B_max: {B_mag.max():.4e}, Bz_range: [{Bz.min():.4e}, {Bz.max():.4e}]")

            Vx = V[:, :, 0]
            Vy = V[:, :, 1]
            Vz = V[:, :, 2]
            
            # Calculate Reconnection Electric Field Ez = Vx*By - Vy*Bx
            # (Note: In 2D x-y GEM, reconnection E-field is out-of-plane, i.e., Ez)
            # User's script had Ey = -(V1*B3 - V3*B1), which is -((Vx*Bz) - (Vz*Bx)) = Vz*Bx - Vx*Bz = -Ey
            # But for x-y plane, Ez is the relevant one. We will calculate Ez.
            Ez = Vx * By - Vy * Bx
            
            # Masks
            # Central region: |y| < 0.05
            # Upstream region: y > 0.3 * y_max (assuming y_max is domain bound)
            # y is 1D array of centers
            
            # Broadcast y to 2D grid for masking if needed, or just use indices since it's structured
            # But numpy broadcasting works: Ez is (ny, nx), y is (ny,)
            # We need to reshape y to (ny, 1) to broadcast against (ny, nx)
            y_2d = y[:, np.newaxis]
            
            mask_c = np.abs(y_2d) < 0.05
            # Fallback if mask is empty (grid might be too coarse or offset)
            if not np.any(mask_c):
                idx_center = np.argmin(np.abs(y))
                mask_c = np.zeros_like(y_2d, dtype=bool)
                mask_c[idx_center] = True
                if vtk_file == vtk_files[0]:
                    print(f"  Note: |y|<0.05 empty. Using center index {idx_center} at y={y[idx_center]:.4f}")

            mask_up = y_2d > (0.3 * y.max())
            if not np.any(mask_up):
                # Fallback to top row
                mask_up = np.zeros_like(y_2d, dtype=bool)
                mask_up[-1] = True
                if vtk_file == vtk_files[0]:
                     print(f"  Note: Upstream mask empty. Using top row at y={y[-1]:.4f}")
            
            # Calculate means
            # Note: mask_c is boolean array of shape (ny, 1). Broadcasting to (ny, nx) works for indexing?
            # No, we need full 2D masks
            mask_c_2d = np.repeat(mask_c, len(x), axis=1)
            mask_up_2d = np.repeat(mask_up, len(x), axis=1)
            
            Ez_c = Ez[mask_c_2d].mean()
            Bx_up = Bx[mask_up_2d].mean()
            
            rate = Ez_c / Bx_up if Bx_up != 0 else 0.0
            
            # Get time
            # We can try to read it from header, or use file index * dt (if we knew dt)
            # Let's read header manually as athena_read.vtk doesn't return time directly in the tuple
            time = 0.0
            with open(vtk_file, 'rb') as f:
                header = f.readlines()[:3]
                for line in header:
                    line_str = line.decode('utf-8', errors='ignore')
                    if 'time=' in line_str:
                        time = float(line_str.split('time=')[1].split()[0])
                        break
            
            rates.append((time, rate))
            
        except Exception as e:
            print(f"Error processing {vtk_file}: {e}")
            continue

    # Sort by time
    rates.sort(key=lambda x: x[0])
    
    print("Reconnection Rate (Time, Rate) - First 5:")
    for r in rates[:5]:
        print(f"  ({r[0]:.4f}, {r[1]:.4e})")
        
    print("Reconnection Rate (Time, Rate) - Last 5:")
    for r in rates[-5:]:
        print(f"  ({r[0]:.4f}, {r[1]:.4e})")

    # 3. Plot Reconnection Rate
    times = [r[0] for r in rates]
    vals = [r[1] for r in rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, vals, '-o')
    plt.xlabel('Time')
    plt.ylabel('Reconnection Rate (Normalized)')
    plt.title('Reconnection Rate vs Time')
    plt.grid(True)
    plot_path = output_dir / "reconnection_rate.png"
    plt.savefig(plot_path)
    print(f"\nSaved {plot_path}")

    # 3.5 Plot Magnetic and Kinetic Energy History (Diagnostic)
    if 'time' in hst_data:
        plt.figure(figsize=(10, 6))
        
        # Magnetic Energy
        if '1-ME' in hst_data:
            me_tot = hst_data['1-ME'] + hst_data['2-ME'] + hst_data['3-ME']
            plt.plot(hst_data['time'], me_tot, '-r', label='Total Magnetic Energy')
        
        # Kinetic Energy
        if '1-KE' in hst_data:
            ke_tot = hst_data['1-KE'] + hst_data['2-KE'] + hst_data['3-KE']
            plt.plot(hst_data['time'], ke_tot, '-b', label='Total Kinetic Energy')

        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy History')
        plt.legend()
        plt.grid(True)
        plot_path = output_dir / "energy_history.png"
        plt.savefig(plot_path)
        print(f"Saved {plot_path} (Diagnostic)")
    
    # 4. Plot Bz Quadrupole
    if vtk_files:
        # Plot Initial State (t=0)
        first_file = vtk_files[0]
        print(f"\nGenerating Bz slice from {first_file} (Initial)")
        x_faces, y_faces, z_faces, data = athena_read.vtk(first_file)
        x = 0.5 * (x_faces[:-1] + x_faces[1:])
        y = 0.5 * (y_faces[:-1] + y_faces[1:])
        X, Y = np.meshgrid(x, y)
        if 'Bcc' in data: Bz0 = data['Bcc'][0, :, :, 2]
        elif 'b' in data: Bz0 = data['b'][0, :, :, 2]
        else: Bz0 = np.zeros_like(X)
        
        plt.figure(figsize=(10, 5))
        limit0 = max(abs(Bz0.min()), abs(Bz0.max()))
        if limit0 == 0: limit0 = 0.1
        plt.pcolormesh(X, Y, Bz0, cmap='RdBu_r', vmin=-limit0, vmax=limit0, shading='auto')
        plt.colorbar(label='Bz')
        plt.title('Bz Structure (Initial t=0)')
        plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
        plt.tight_layout()
        plot_path = output_dir / "bz_initial.png"
        plt.savefig(plot_path)
        print(f"Saved {plot_path}")

        # Plot Final State
        last_file = vtk_files[-1]
        print(f"Generating Bz slice from {last_file} (Final)")
        x_faces, y_faces, z_faces, data = athena_read.vtk(last_file)
        if 'Bcc' in data: Bz = data['Bcc'][0, :, :, 2]
        elif 'b' in data: Bz = data['b'][0, :, :, 2]
        else: Bz = np.zeros_like(X)
            
        plt.figure(figsize=(10, 5))
        limit = max(abs(Bz.min()), abs(Bz.max()))
        if limit == 0: limit = 0.1 # Handle zero field case
        
        plt.pcolormesh(X, Y, Bz, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
        plt.colorbar(label='Bz')
        plt.title(f'Bz Quadrupole Structure (t={times[-1]:.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plot_path = output_dir / "bz_slice.png"
        plt.savefig(plot_path)
        print(f"Saved {plot_path}")

if __name__ == "__main__":
    main()
