import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import time
from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward,
    simulate_edo_reduced_theta
)
from utils import save_plot, export_dataframe


def run_all_methods(show_progress=True, **kwargs):
    """
    Run all ODE solution methods with the same parameters.

    Args:
        show_progress (bool, optional): Whether to show progress updates. Defaults to True.
        **kwargs: Additional parameters to pass to all methods.

    Returns:
        dict: Dictionary containing results from all methods
    """
    methods_to_run = {
        "ivp": simulate_edo_ivp,
        "bdf": simulate_edo_ivp_bdf,
        "euler": simulate_edo_euler_backward,
        "theta": simulate_edo_reduced_theta,
    }

    results = {}

    try:
        for name, method in methods_to_run.items():
            if show_progress:
                print(f"Running method: {name}...")
                start_time = time.time()

            results[name] = method(**kwargs)

            if show_progress:
                elapsed = time.time() - start_time
                print(f"  Completed in {elapsed:.2f} seconds")

        return results

    except Exception as e:
        print(f"Error running ODE methods: {e}", file=sys.stderr)
        return {}


def create_comparison_plots(methods_results, output_dir="results"):
    """
    Create comparison plots for all solution methods.

    Args:
        methods_results (dict): Dictionary containing results from all methods
        output_dir (str, optional): Directory to save plots. Defaults to "results".
    """
    if not methods_results:
        print("No results to plot")
        return

    # Use the first method's grid as reference
    ref_method = next(iter(methods_results))
    xi = methods_results[ref_method]["xi_km"]

    try:
        # Radial velocity comparison
        save_plot(
            xi,
            [methods_results[m]["a_xi"] for m in methods_results],
            labels=[f"a(ξ) - {m}" for m in methods_results],
            title="Comparison of a(ξ) across methods",
            xlabel="ξ (km)", ylabel="a(ξ)",
            filename="compare_a_xi.pdf",
            results_dir=output_dir
        )

        # Tangential velocity comparison
        save_plot(
            xi,
            [methods_results[m]["b_xi"] for m in methods_results],
            labels=[f"b(ξ) - {m}" for m in methods_results],
            title="Comparison of b(ξ) across methods",
            xlabel="ξ (km)", ylabel="b(ξ)",
            filename="compare_b_xi.pdf",
            results_dir=output_dir
        )

        # Geostrophic factor comparison
        save_plot(
            xi,
            [methods_results[m]["theta_xi"] for m in methods_results],
            labels=[f"θ(ξ) - {m}" for m in methods_results],
            title="Comparison of θ(ξ) across methods",
            xlabel="ξ (km)", ylabel="θ(ξ)",
            filename="compare_theta.pdf",
            results_dir=output_dir
        )

        # Kinetic energy comparison
        save_plot(
            xi,
            [methods_results[m]["E_xi"] for m in methods_results],
            labels=[f"E(ξ) - {m}" for m in methods_results],
            title="Comparison of kinetic energy E(ξ)",
            xlabel="ξ (km)", ylabel="E(ξ)",
            filename="compare_energy.pdf",
            results_dir=output_dir
        )

        print(f"Comparison plots saved to {output_dir}/")

    except Exception as e:
        print(f"Error creating comparison plots: {e}", file=sys.stderr)


def calculate_differences(methods_results, reference="ivp"):
    """
    Calculate differences between methods and a reference method.

    Args:
        methods_results (dict): Dictionary containing results from all methods
        reference (str, optional): Reference method name. Defaults to "ivp".

    Returns:
        tuple: (DataFrame with all data, dictionary of differences)
    """
    if not methods_results or reference not in methods_results:
        print(f"Reference method '{reference}' not found in results")
        return None, {}

    try:
        # Get reference grid
        xi = methods_results[reference]["xi_km"]

        # Create DataFrame with all results
        df_data = {
            "xi_km": xi,
        }

        # Add all method results
        for m in methods_results:
            df_data[f"a_{m}"] = methods_results[m]["a_xi"]
            df_data[f"b_{m}"] = methods_results[m]["b_xi"]
            df_data[f"theta_{m}"] = methods_results[m]["theta_xi"]
            df_data[f"E_{m}"] = methods_results[m]["E_xi"]

        df = pd.DataFrame(df_data)

        # Calculate differences
        deltas = {}
        for m in methods_results:
            if m != reference:
                df[f"delta_a_{m}"] = df[f"a_{m}"] - df[f"a_{reference}"]
                df[f"delta_b_{m}"] = df[f"b_{m}"] - df[f"b_{reference}"]
                df[f"delta_theta_{m}"] = df[f"theta_{m}"] - df[f"theta_{reference}"]
                df[f"delta_E_{m}"] = df[f"E_{m}"] - df[f"E_{reference}"]

                deltas[m] = {
                    "a": df[f"delta_a_{m}"],
                    "b": df[f"delta_b_{m}"],
                    "theta": df[f"delta_theta_{m}"],
                    "E": df[f"delta_E_{m}"]
                }

        return df, deltas

    except Exception as e:
        print(f"Error calculating differences: {e}", file=sys.stderr)
        return None, {}


def create_difference_plots(xi, deltas, reference="ivp", output_dir="results"):
    """
    Create plots showing differences between methods and a reference method.

    Args:
        xi (array-like): x-axis data (radial distance)
        deltas (dict): Dictionary of differences
        reference (str, optional): Reference method name. Defaults to "ivp".
        output_dir (str, optional): Directory to save plots. Defaults to "results".
    """
    if not deltas:
        print("No differences to plot")
        return

    try:
        # Create difference plots for each method
        for m in deltas:
            # Radial velocity difference
            save_plot(
                xi,
                [deltas[m]["a"]],
                labels=[f"Δa(ξ) = {m} - {reference}"],
                title=f"Difference in a(ξ): {m} - {reference}",
                xlabel="ξ (km)", ylabel="Δa(ξ)",
                filename=f"delta_a_{m}.pdf",
                results_dir=output_dir
            )

            # Tangential velocity difference
            save_plot(
                xi,
                [deltas[m]["b"]],
                labels=[f"Δb(ξ) = {m} - {reference}"],
                title=f"Difference in b(ξ): {m} - {reference}",
                xlabel="ξ (km)", ylabel="Δb(ξ)",
                filename=f"delta_b_{m}.pdf",
                results_dir=output_dir
            )

            # Geostrophic factor difference
            save_plot(
                xi,
                [deltas[m]["theta"]],
                labels=[f"Δθ(ξ) = {m} - {reference}"],
                title=f"Difference in θ(ξ): {m} - {reference}",
                xlabel="ξ (km)", ylabel="Δθ(ξ)",
                filename=f"delta_theta_{m}.pdf",
                results_dir=output_dir
            )

            # Kinetic energy difference
            save_plot(
                xi,
                [deltas[m]["E"]],
                labels=[f"ΔE(ξ) = {m} - {reference}"],
                title=f"Difference in E(ξ): {m} - {reference}",
                xlabel="ξ (km)", ylabel="ΔE(ξ)",
                filename=f"delta_E_{m}.pdf",
                results_dir=output_dir
            )

        print(f"Difference plots saved to {output_dir}/")

    except Exception as e:
        print(f"Error creating difference plots: {e}", file=sys.stderr)


def cross_validate_methods(output_csv="validation_cross_table.csv", 
                          output_dir="results", 
                          reference="ivp",
                          show_progress=True,
                          **kwargs):
    """
    Perform cross-validation of all ODE solution methods.

    Args:
        output_csv (str, optional): Output CSV filename. Defaults to "validation_cross_table.csv".
        output_dir (str, optional): Directory to save results. Defaults to "results".
        reference (str, optional): Reference method for comparisons. Defaults to "ivp".
        show_progress (bool, optional): Whether to show progress updates. Defaults to True.
        **kwargs: Additional parameters to pass to all methods.

    Returns:
        bool: True if validation was successful, False otherwise
    """
    try:
        print("\nCross-validation of ODE solution methods")

        # Run all methods
        methods_results = run_all_methods(show_progress=show_progress, **kwargs)
        if not methods_results:
            return False

        # Create comparison plots
        create_comparison_plots(methods_results, output_dir=output_dir)

        # Calculate differences
        df, deltas = calculate_differences(methods_results, reference=reference)
        if df is None:
            return False

        # Create difference plots
        create_difference_plots(df["xi_km"], deltas, reference=reference, output_dir=output_dir)

        # Export data
        export_dataframe(df, output_csv)
        print(f"Comparison table exported to {output_csv}")

        print("\nCross-validation completed successfully:")
        print(f"  - Comparison plots: {output_dir}/compare_*.pdf")
        print(f"  - Difference plots: {output_dir}/delta_*.pdf")
        print(f"  - Data table: {output_csv}")

        return True

    except Exception as e:
        print(f"Error in cross-validation: {e}", file=sys.stderr)
        return False


def main():
    """
    Main function for the ODE cross-validation command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Cross-validation of cyclone ODE solution methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--output-csv", type=str, default="validation_cross_table.csv",
                       help="Output CSV filename")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--reference", type=str, default="ivp", choices=["ivp", "bdf", "euler", "theta"],
                       help="Reference method for comparisons")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress updates")
    parser.add_argument("--params", type=str,
                       help="Additional parameters in format 'key1=value1,key2=value2,...'")

    args = parser.parse_args()

    # Parse additional parameters if provided
    extra_params = {}
    if args.params:
        for param in args.params.split(','):
            key, value = param.split('=')
            # Try to convert to appropriate type
            try:
                # Try as int
                extra_params[key] = int(value)
            except ValueError:
                try:
                    # Try as float
                    extra_params[key] = float(value)
                except ValueError:
                    # Keep as string
                    extra_params[key] = value

    # Run cross-validation
    success = cross_validate_methods(
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        reference=args.reference,
        show_progress=not args.no_progress,
        **extra_params
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
