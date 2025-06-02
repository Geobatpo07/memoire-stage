import argparse
import sys
from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward,
    simulate_edo_reduced_theta
)
from utils import save_edo_result, save_plot

# Available ODE solution methods
METHODS = {
    "ivp": simulate_edo_ivp,
    "bdf": simulate_edo_ivp_bdf,
    "euler": simulate_edo_euler_backward,
    "theta": simulate_edo_reduced_theta,
}


def main():
    """
    Main function for the ODE simulation command-line interface.

    This function parses command-line arguments, runs the selected ODE solution
    method, saves the results, and optionally generates plots.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Cyclone ODE system simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command-line arguments
    parser.add_argument(
        "--method", 
        choices=METHODS.keys(), 
        default="ivp",
        help="Solution method to use"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="edo_results.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Display figures"
    )
    parser.add_argument(
        "--params", 
        type=str, 
        help="Additional parameters in format 'key1=value1,key2=value2,...'"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
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

        # Run the selected method
        print(f"\nRunning ODE simulation with method: {args.method}")
        result = METHODS[args.method](**extra_params)

        # Save results
        save_edo_result(result, args.output)
        print(f"Results saved to {args.output}")

        # Generate plots if requested
        if args.show:
            # Velocity components plot
            save_plot(
                result["xi_km"], 
                [result["a_xi"], result["b_xi"]],
                labels=["a(ξ) (radial)", "b(ξ) (tangential)"],
                title="Velocity Components", 
                xlabel="ξ (km)",
                ylabel="Velocity (m/s)", 
                filename="edo_ab_components.pdf"
            )

            # Geostrophic factor plot
            save_plot(
                result["xi_km"], 
                [result["theta_xi"]], 
                ["θ(ξ)"],
                title="Geostrophic Factor", 
                xlabel="ξ (km)",
                ylabel="θ(ξ)", 
                filename="edo_theta.pdf"
            )

            # Kinetic energy plot
            save_plot(
                result["xi_km"], 
                [result["E_xi"]], 
                ["Kinetic Energy"],
                title="Kinetic Energy", 
                xlabel="ξ (km)",
                ylabel="E(ξ)", 
                filename="edo_energy.pdf"
            )

            print("Plots generated successfully")

    except Exception as e:
        print(f"Error in ODE simulation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
