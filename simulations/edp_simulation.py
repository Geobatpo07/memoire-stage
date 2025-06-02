import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time
from utils import (
    shift,
    lax_friedrichs_2d,
    compute_velocity_norm,
    save_fields_edp,
    plot_quiver_field
)


def simulate_edp(L=300e3, N=200, Tmax=3600, u_star=5.0, v_star=0.0, 
                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5, 
                plot_every=300, show_progress=True, save_dir="results"):
    """
    Simulate the PDE system on a 2D grid using an explicit method.

    Args:
        L (float): Domain size (m)
        N (int): Grid resolution
        Tmax (float): Final simulation time (s)
        u_star (float): Background x-velocity
        v_star (float): Background y-velocity
        w0 (float): Earth's rotation rate (rad/s)
        phi_deg (float): Latitude in degrees
        k (float): Friction coefficient
        xi_star (float): Absorption coefficient
        plot_every (int): Interval for generating plots (in time steps)
        show_progress (bool): Whether to show progress updates
        save_dir (str): Directory to save results

    Returns:
        tuple: (u, v, x, y) arrays containing the final velocity fields and coordinate grids
    """
    try:
        # Calculate grid spacing and time step (with CFL condition)
        dx = L / N
        dt = 0.8 * dx / 10  # CFL factor for stability
        steps = int(Tmax / dt)
        phi = np.radians(phi_deg)

        if show_progress:
            print(f"Setting up PDE simulation:")
            print(f"  - Grid: {N}x{N} points ({dx/1000:.1f} km spacing)")
            print(f"  - Time step: {dt:.2f} s ({steps} steps)")
            print(f"  - Physical parameters: phi={phi_deg}°, k={k}, xi_star={xi_star}")

        # Create coordinate grid
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X, Y = np.meshgrid(x, y)

        # Initialize velocity field with Gaussian perturbation
        center_perturbation = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (1e9))
        u = u_star + 2.0 * center_perturbation
        v = v_star + 0.5 * center_perturbation

        # Main time integration loop
        start_time = time.time()
        last_update = start_time

        for n in range(steps):
            # Calculate advection terms using Lax-Friedrichs scheme
            u_adv = lax_friedrichs_2d(u, u, v, dx, dx)
            v_adv = lax_friedrichs_2d(v, u, v, dx, dx)

            # Calculate source terms (Coriolis force, friction, absorption)
            rhs_u = (xi_star - k)*(u - u_star) + w0 * np.sin(phi)*(v - v_star)
            rhs_v = (xi_star - k)*(v - v_star) - w0 * np.sin(phi)*(u - u_star)

            # Update velocity field
            u -= dt * u_adv - dt * rhs_u
            v -= dt * v_adv - dt * rhs_v

            # Generate plots at specified intervals
            if n % plot_every == 0:
                norm = compute_velocity_norm(u, v)
                max_vel = np.max(norm)
                plot_quiver_field(X, Y, u, v, 
                                 title=f"Wind field at t = {int(n*dt)} s (max: {max_vel:.2f} m/s)")

            # Show progress updates
            if show_progress and (time.time() - last_update > 5.0 or n == steps-1):
                elapsed = time.time() - start_time
                progress = (n + 1) / steps
                remaining = elapsed / progress - elapsed if progress > 0 else 0
                print(f"  Progress: {progress*100:.1f}% - Step {n+1}/{steps} - " 
                      f"Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
                last_update = time.time()

        # Save final fields
        save_fields_edp(u, v, x, y, results_dir=save_dir)
        print(f"PDE simulation completed and results saved to {save_dir}/")

        return u, v, x, y

    except Exception as e:
        print(f"Error in PDE simulation: {e}", file=sys.stderr)
        # Return empty arrays in case of error
        return np.zeros((N, N)), np.zeros((N, N)), np.linspace(0, L, N), np.linspace(0, L, N)


def main():
    """
    Main function for the PDE simulation command-line interface.

    This function parses command-line arguments and runs the PDE simulation.
    """
    parser = argparse.ArgumentParser(
        description="Cyclone PDE system simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command-line arguments
    parser.add_argument("--size", type=float, default=300e3, 
                       help="Domain size in meters")
    parser.add_argument("--resolution", type=int, default=200, 
                       help="Grid resolution (N×N points)")
    parser.add_argument("--time", type=float, default=3600, 
                       help="Simulation time in seconds")
    parser.add_argument("--latitude", type=float, default=20, 
                       help="Latitude in degrees")
    parser.add_argument("--friction", type=float, default=1e-5, 
                       help="Friction coefficient k")
    parser.add_argument("--absorption", type=float, default=5e-5, 
                       help="Absorption coefficient xi_star")
    parser.add_argument("--plot-every", type=int, default=300, 
                       help="Plot interval (in time steps)")
    parser.add_argument("--no-progress", action="store_true", 
                       help="Disable progress updates")
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Directory to save results")

    args = parser.parse_args()

    # Run simulation with command-line arguments
    simulate_edp(
        L=args.size,
        N=args.resolution,
        Tmax=args.time,
        phi_deg=args.latitude,
        k=args.friction,
        xi_star=args.absorption,
        plot_every=args.plot_every,
        show_progress=not args.no_progress,
        save_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
