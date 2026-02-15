import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


if add_athena_vis_path():
    try:
        import athena_read
    except ImportError:
        athena_read = None
else:
    athena_read = None
if athena_read is None:
    print("Warning: athena_read not found. VTK processing will be unavailable.")

def get_vtk_time(filename):
    """Extract time from VTK header."""
    try:
        with open(filename, 'r', errors='replace') as f:
            for _ in range(5):
                line = f.readline()
                if 'time=' in line:
                    return float(line.split('time=')[1].split()[0])
    except Exception:
        pass
    return None

def extract_R_series(vtk_pattern):
    """Extract R(t) from a series of VTK files."""
    if athena_read is None:
        return None, None
        
    files = sorted(glob.glob(vtk_pattern))
    if not files:
        print(f"Warning: No VTK files found matching {vtk_pattern}")
        return None, None
    
    times = []
    Rs = []
    
    print(f"Extracting R(t) from {len(files)} VTK files...")
    
    for fname in files:
        try:
            t_val = get_vtk_time(fname)
            if t_val is None:
                continue
                
            x, y, z, data = athena_read.vtk(fname)
            
            # Assume 2D (z=0)
            # Data shape: (nz, ny, nx) -> we take [0]
            rho = data['rho'][0]
            
            # Calculate R based on density threshold (20% of peak)
            max_rho = rho.max()
            threshold = 0.2 * max_rho
            
            # Get Y coordinates (cell centers)
            yc = 0.5 * (y[:-1] + y[1:])
            
            # Find y indices where any x has rho > threshold
            # (Assuming plasma is centered in Y)
            mask_y = np.any(rho > threshold, axis=1)
            valid_ys = yc[mask_y]
            
            if len(valid_ys) > 0:
                # Diameter = max_y - min_y
                # Radius = Diameter / 2
                diameter = valid_ys.max() - valid_ys.min()
                R = diameter / 2.0
            else:
                R = 0.0
            
            times.append(t_val)
            Rs.append(R)
            
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            
    if not times:
        return None, None
        
    # Sort by time just in case
    times = np.array(times)
    Rs = np.array(Rs)
    idx = np.argsort(times)
    return times[idx], Rs[idx]

def lcr_circuit_with_compression(args):
    # Parameters
    V0 = args.V0
    C = args.C
    R_line = args.R_line
    L0 = args.L0
    L_alpha = args.L_alpha
    R_plasma0 = args.R_plasma0
    R_min = args.R_min
    v_ramp = args.v_ramp
    R_coil = args.R_coil
    turns = args.turns
    dt = args.dt
    tmax = args.tmax
    
    # Derived params
    kB = args.kB if args.kB else (4e-7 * np.pi * turns / R_coil)
    
    # Time array
    steps = int(tmax / dt)
    time = np.linspace(0, tmax, steps)
    
    # Load VTK R(t) if requested
    vtk_times, vtk_Rs = None, None
    if args.vtk_path:
        vtk_times, vtk_Rs = extract_R_series(args.vtk_path)
        if vtk_times is not None:
            print(f"Using VTK-derived R(t) from {vtk_times[0]:.2e}s to {vtk_times[-1]:.2e}s")
            # Interpolate R_plasma0 to match start of VTK data if needed, 
            # or just let it jump? Better to smooth.
            # For now, raw interpolation.
        else:
            print("Falling back to linear compression model.")
            
    # State variables
    I = 0.0
    Q = C * V0
    
    # Results storage
    results = []
    
    # Simulation loop
    for t in time:
        # 1. Calculate Plasma Radius R(t)
        if vtk_times is not None:
            # Interpolate from VTK data
            if t <= vtk_times[0]:
                R_plasma = vtk_Rs[0]
            elif t >= vtk_times[-1]:
                R_plasma = vtk_Rs[-1]
            else:
                R_plasma = np.interp(t, vtk_times, vtk_Rs)
        else:
            # Linear Compression Model
            t_compress_end = (R_plasma0 - R_min) / v_ramp if v_ramp > 0 else tmax
            if t < t_compress_end:
                R_plasma = R_plasma0 + (R_min - R_plasma0) * (t / t_compress_end)
            else:
                R_plasma = R_min
            
        # 2. Calculate L(t) and dL/dt
        L = L0 + L_alpha * R_plasma
        
        # dR/dt calculation
        if vtk_times is not None:
            # Finite difference derivative from interpolation
            # To be smooth, we look ahead slightly
            delta = 1e-9
            if t + delta <= vtk_times[-1] and t + delta >= vtk_times[0]:
                R_next = np.interp(t + delta, vtk_times, vtk_Rs)
                dR_dt = (R_next - R_plasma) / delta
            else:
                dR_dt = 0.0
        else:
            if t < t_compress_end:
                dR_dt = (R_min - R_plasma0) / t_compress_end
            else:
                dR_dt = 0.0
            
        dL_dt = L_alpha * dR_dt
        
        # 3. Circuit Equation
        V_cap = Q / C
        
        # dI/dt = (V_cap - I*(R_line + dL/dt)) / L
        dI_dt = (V_cap - I * (R_line + dL_dt)) / L
        
        # Update state (Forward Euler)
        I_new = I + dI_dt * dt
        Q_new = Q - I * dt
        
        # Estimate B
        B_est = kB * I
        
        # Energy
        E_cap = 0.5 * Q**2 / C
        E_ind = 0.5 * L * I**2
        
        results.append([t, I, V_cap, L, dL_dt, R_plasma, B_est, E_cap, E_ind])
        
        I = I_new
        Q = Q_new

    # Save to CSV
    header = "t,I,V_cap,L,dL_dt,R_plasma,B_est,E_cap,E_ind"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, results, delimiter=",", header=header, comments="")
    print(f"LCR waveform saved to {out_path}")
    
    if getattr(args, "no_plot", False):
        return

    # Plot
    data = np.array(results)
    t_data = data[:,0] * 1e6 # us
    I_data = data[:,1] / 1e3 # kA
    B_data = data[:,6]
    R_data = data[:,5]
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # Plot I and B
    color = 'tab:red'
    ax1.set_ylabel('Current (kA)', color=color)
    ax1.plot(t_data, I_data, color=color, label='Current')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('B_est (T)', color=color)
    ax2.plot(t_data, B_data, color=color, linestyle='--', label='B_est')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title('LCR Circuit Results')
    ax1.grid(True)
    
    # Plot R(t)
    color = 'tab:green'
    ax3.set_xlabel('Time (us)')
    ax3.set_ylabel('Plasma Radius (m)', color=color)
    ax3.plot(t_data, R_data, color=color, label='R_plasma')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_title('Plasma Compression Trajectory')
    ax3.grid(True)
    
    plt.tight_layout()
    repo_root = Path(__file__).resolve().parents[1]
    analysis_dir = repo_root / "outputs" / "analysis"
    plots_dir = repo_root / "outputs" / "plots"
    plot_dir = getattr(args, "plot_dir", None)
    if plot_dir:
        plot_path = Path(plot_dir) / f"{out_path.stem}.png"
    else:
        plot_path = out_path.with_suffix(".png")
        try:
            if analysis_dir in out_path.resolve().parents:
                plot_path = plots_dir / f"{out_path.stem}.png"
        except RuntimeError:
            pass
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="LCR Circuit Solver with Plasma Compression")
    parser.add_argument("--V0", type=float, default=20e3)
    parser.add_argument("--C", type=float, default=10e-6)
    parser.add_argument("--R_line", type=float, default=0.05)
    parser.add_argument("--L0", type=float, default=1e-6)
    parser.add_argument("--L_alpha", type=float, default=2e-6)
    parser.add_argument("--R_plasma0", type=float, default=0.30)
    parser.add_argument("--R_min", type=float, default=0.10)
    parser.add_argument("--v_ramp", type=float, default=5.0)
    parser.add_argument("--R_coil", type=float, default=0.35)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-7)
    parser.add_argument("--tmax", type=float, default=50e-6)
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root / "outputs" / "analysis" / "lcr_waveform.csv"),
    )
    parser.add_argument("--kB", type=float, help="Optional manual B coefficient")
    parser.add_argument("--vtk_path", type=str, help="Glob pattern for VTK files to extract R(t)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot output.")
    parser.add_argument("--plot-dir", type=str, help="Optional override for plot output directory.")
    
    args = parser.parse_args()
    lcr_circuit_with_compression(args)
