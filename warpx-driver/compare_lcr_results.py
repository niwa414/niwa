import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from types import SimpleNamespace
from pathlib import Path

# Add tools directory to sys.path to import lcr_coupling
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "tools"))
import lcr_coupling

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_lcr_results.py <json_metadata_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    args_dict = metadata.get('args', {})
    
    # Check if LCR was used
    if not args_dict.get('use_lcr', False):
        print("Error: This run did not use LCR.")
        sys.exit(1)

    lcr_out_csv = args_dict.get('lcr_out')
    if not lcr_out_csv:
        print("Error: No LCR output file specified in metadata.")
        sys.exit(1)
        
    # Handle relative paths
    if not os.path.isabs(lcr_out_csv):
        # Try relative to CWD first
        if not os.path.exists(lcr_out_csv):
            # Try relative to json file directory
            json_dir = os.path.dirname(json_file)
            alt_path = os.path.join(json_dir, os.path.basename(lcr_out_csv))
            if os.path.exists(alt_path):
                lcr_out_csv = alt_path
            else:
                 # Check if it's relative to project root (cwd of script execution usually)
                 pass

    if not os.path.exists(lcr_out_csv):
        print(f"Error: LCR output file {lcr_out_csv} not found.")
        sys.exit(1)

    print(f"Loading WarpX LCR history from {lcr_out_csv}...")
    warpx_df = pd.read_csv(lcr_out_csv)
    
    # Map JSON args to lcr_coupling args
    lcr_args = SimpleNamespace()
    lcr_args.V0 = args_dict.get('lcr_V0', 20e3)
    lcr_args.C = args_dict.get('lcr_C', 10e-6)
    lcr_args.R_line = args_dict.get('lcr_R_line', 0.05)
    lcr_args.L0 = args_dict.get('lcr_L0', 1e-6)
    lcr_args.L_alpha = args_dict.get('lcr_L_alpha', 2e-6)
    lcr_args.R_plasma0 = args_dict.get('lcr_R_plasma0', 0.30)
    lcr_args.R_min = args_dict.get('lcr_R_min', 0.10)
    lcr_args.v_ramp = args_dict.get('lcr_v_ramp', 5.0)
    lcr_args.R_coil = args_dict.get('lcr_R_coil', 0.35)
    lcr_args.turns = args_dict.get('lcr_turns', 1)
    lcr_args.dt = args_dict.get('dt', 1e-7) # WarpX dt
    
    # tmax should match the WarpX run duration
    max_steps = args_dict.get('max_steps', 100)
    lcr_args.tmax = max_steps * lcr_args.dt
    
    lcr_args.kB = args_dict.get('lcr_kB')
    lcr_args.out = "lcr_coupling_reference.csv"
    lcr_args.vtk_path = None # Assuming we want to compare with the linear model used in WarpX

    print(f"Running reference LCR solver (dt={lcr_args.dt}, tmax={lcr_args.tmax})...")
    lcr_coupling.lcr_circuit_with_compression(lcr_args)
    
    print(f"Loading reference LCR output from {lcr_args.out}...")
    ref_df = pd.read_csv(lcr_args.out)
    
    # Comparison
    # Interpolate ref to warpx times
    t_warpx = warpx_df['t'].values
    t_ref = ref_df['t'].values
    
    # If ref has fewer points than warpx (unlikely if dt matches), or different spacing
    I_ref_interp = np.interp(t_warpx, t_ref, ref_df['I'].values)
    
    E_cap_warpx = warpx_df['E_cap'].values
    E_ind_warpx = warpx_df['E_ind'].values
    E_total_warpx = E_cap_warpx + E_ind_warpx
    
    I_warpx = warpx_df['I'].values
    error_I = I_warpx - I_ref_interp
    rmse_I = np.sqrt(np.mean(error_I**2))
    max_error_I = np.max(np.abs(error_I))
    
    print("\n--- Comparison Results ---")
    print(f"RMSE Current: {rmse_I:.6e} A")
    print(f"Max Error Current: {max_error_I:.6e} A")
    
    # Energy Analysis
    delta_E_total = E_total_warpx - E_total_warpx[0]
    max_energy_deviation = np.max(np.abs(delta_E_total))
    print(f"Max Energy Deviation (WarpX): {max_energy_deviation:.6e} J")
    print(f"Initial Energy: {E_total_warpx[0]:.6e} J")
    print(f"Relative Energy Error: {max_energy_deviation/E_total_warpx[0]:.6e}")

    # Plotting
    plt.figure(figsize=(10, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_warpx, I_warpx, 'r-', label='WarpX', linewidth=2, alpha=0.7)
    plt.plot(t_ref, ref_df['I'].values, 'b--', label='Reference') 
    plt.ylabel('Current (A)')
    plt.legend()
    plt.title('Current Comparison')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t_warpx, error_I, 'k-')
    plt.ylabel('Error (A)')
    plt.title('Current Error (WarpX - Ref)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t_warpx, E_total_warpx, 'g-', label='Total Energy (WarpX)')
    plt.plot(t_warpx, E_cap_warpx, 'r--', label='E_cap')
    plt.plot(t_warpx, E_ind_warpx, 'b--', label='E_ind')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.title('Energy Conservation (WarpX)')
    plt.grid(True)
    
    plt.tight_layout()
    output_plot = os.path.splitext(lcr_out_csv)[0] + "_comparison.png"
    plt.savefig(output_plot)
    print(f"Comparison plot saved to {output_plot}")
    
    # Clean up temp file
    if os.path.exists(lcr_args.out):
        os.remove(lcr_args.out)

if __name__ == "__main__":
    main()
