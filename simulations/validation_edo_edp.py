import numpy as np
import pandas as pd
import argparse
import sys
import os
from scipy.interpolate import interp1d
from utils import (
    load_edp_fields,
    compute_velocity_norm,
    compute_radial_profile,
    save_plot,
    export_dataframe
)


def validate_edo_edp(edo_csv="edo_results.csv", pdf_output="validation_edo_edp_comparison.pdf", 
                    csv_output="validation_profiles.csv", show_plot=True, 
                    edp_dir=".", output_dir="results"):
    """
    Validate the PDE model by comparing it to the analytical ODE profile.

    This function loads ODE results from a CSV file and PDE simulation results,
    computes radial profiles, and creates comparison plots and data.

    Args:
        edo_csv (str, optional): Path to ODE results CSV file. Defaults to "edo_results.csv".
        pdf_output (str, optional): Output PDF filename for comparison plot. Defaults to "validation_edo_edp_comparison.pdf".
        csv_output (str, optional): Output CSV filename for validation data. Defaults to "validation_profiles.csv".
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        edp_dir (str, optional): Directory containing PDE field files. Defaults to current directory.
        output_dir (str, optional): Directory to save output files. Defaults to "results".

    Returns:
        bool: True if validation was successful, False otherwise
    """
    try:
        print("\nValidating PDE model against ODE profile...")

        # 1. Load ODE results
        try:
            edo_data = pd.read_csv(edo_csv)
            print(f"Loaded ODE data from {edo_csv}")

            xi_edo_km = edo_data["xi_km"].values
            a = edo_data["a_xi"].values
            b = edo_data["b_xi"].values
            v_norm_edo = np.sqrt(a**2 + b**2)

        except Exception as e:
            print(f"Error loading ODE data: {e}", file=sys.stderr)
            print(f"Make sure the file {edo_csv} exists and contains the required columns.")
            return False

        # 2. Load PDE fields
        try:
            u_sim, v_sim, x, y = load_edp_fields(directory=edp_dir)
            print(f"Loaded PDE fields from {edp_dir}")

            # Create coordinate grid and compute distance from center
            X, Y = np.meshgrid(x, y)
            R = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2)
            v_norm_sim_grid = compute_velocity_norm(u_sim, v_sim)

        except Exception as e:
            print(f"Error loading PDE fields: {e}", file=sys.stderr)
            return False

        # 3. Compute radial profile from PDE simulation
        r_centers, v_norm_sim_profile = compute_radial_profile(R, v_norm_sim_grid)
        r_centers_km = r_centers / 1000  # Convert to kilometers

        # 4. Interpolate ODE profile to match PDE radial points
        try:
            interp_edo = interp1d(xi_edo_km, v_norm_edo, bounds_error=False, fill_value="extrapolate")
            v_norm_edo_interp = interp_edo(r_centers_km)

        except Exception as e:
            print(f"Error in interpolation: {e}", file=sys.stderr)
            return False

        # 5. Create and save comparison plot
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, pdf_output)

        save_plot(
            x=r_centers_km,
            y_list=[v_norm_sim_profile, v_norm_edo_interp],
            labels=["PDE Simulation", "ODE Theory"],
            title="Cross-validation: Radial profile |v(ξ)| – ODE vs PDE",
            xlabel="Distance from center ξ (km)",
            ylabel="Velocity magnitude |v(ξ)| (m/s)",
            filename=pdf_output,
            styles=["--", "-"],
            show=show_plot,
            results_dir=output_dir
        )

        # 6. Export profiles to CSV
        csv_path = os.path.join(output_dir, csv_output)

        # Calculate relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(v_norm_sim_profile - v_norm_edo_interp) / v_norm_edo_interp
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, np.nan)

        df = pd.DataFrame({
            "r_km": r_centers_km,
            "v_norm_pde": v_norm_sim_profile,
            "v_norm_ode": v_norm_edo_interp,
            "rel_diff": rel_diff
        })

        export_dataframe(df, csv_path)

        # Calculate validation metrics
        valid_indices = ~np.isnan(rel_diff)
        if np.any(valid_indices):
            mean_rel_diff = np.mean(rel_diff[valid_indices]) * 100  # as percentage
            max_rel_diff = np.max(rel_diff[valid_indices]) * 100    # as percentage

            print("\nValidation metrics:")
            print(f"  - Mean relative difference: {mean_rel_diff:.2f}%")
            print(f"  - Maximum relative difference: {max_rel_diff:.2f}%")

        print("\nODE-PDE validation completed successfully.")
        print(f"  - Comparison plot: {pdf_path}")
        print(f"  - Validation data: {csv_path}")

        return True

    except Exception as e:
        print(f"Error in ODE-PDE validation: {e}", file=sys.stderr)
        return False


def main():
    """
    Main function for the ODE-PDE validation command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Validate PDE model against analytical ODE profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--ode-csv", type=str, default="edo_results.csv",
                       help="Path to ODE results CSV file")
    parser.add_argument("--pdf-output", type=str, default="validation_edo_edp_comparison.pdf",
                       help="Output PDF filename for comparison plot")
    parser.add_argument("--csv-output", type=str, default="validation_profiles.csv",
                       help="Output CSV filename for validation data")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display the plot")
    parser.add_argument("--edp-dir", type=str, default=".",
                       help="Directory containing PDE field files")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save output files")

    args = parser.parse_args()

    # Run validation
    success = validate_edo_edp(
        edo_csv=args.ode_csv,
        pdf_output=args.pdf_output,
        csv_output=args.csv_output,
        show_plot=not args.no_show,
        edp_dir=args.edp_dir,
        output_dir=args.output_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
