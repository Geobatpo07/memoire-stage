import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import sys
import os
from utils import compute_velocity_norm, find_cyclone_center, save_fields_edp


def simulate_cyclone_on_globe(lon_cyclone=-72.42, lat_cyclone=19.0, extent=10, N=40, 
                             sigma=5e5, output_file="cyclone_simulation.png", 
                             save_fields=True, show_plot=True, output_dir="results"):
    """
    Simulate a cyclonic wind field centered on a geographic location and
    project it on a globe map using Cartopy.

    Args:
        lon_cyclone (float, optional): Longitude of cyclone center. Defaults to -72.42 (Port-au-Prince, Haiti).
        lat_cyclone (float, optional): Latitude of cyclone center. Defaults to 19.0 (Port-au-Prince, Haiti).
        extent (float, optional): Map extent in degrees around center. Defaults to 10.
        N (int, optional): Grid resolution. Defaults to 40.
        sigma (float, optional): Characteristic width of cyclone in meters. Defaults to 5e5.
        output_file (str, optional): Output image filename. Defaults to "cyclone_simulation.png".
        save_fields (bool, optional): Whether to save velocity fields. Defaults to True.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        output_dir (str, optional): Directory to save results. Defaults to "results".

    Returns:
        tuple: (U, V, LON, LAT, center_x, center_y) containing velocity fields, coordinate grids, and cyclone center
    """
    try:
        # Create geographic grid
        lons = np.linspace(lon_cyclone - extent, lon_cyclone + extent, N)
        lats = np.linspace(lat_cyclone - extent, lat_cyclone + extent, N)
        LON, LAT = np.meshgrid(lons, lats)

        # Convert to Cartesian coordinates (meters)
        R_earth = 6378135  # Earth radius in meters
        dx = np.radians(LON - lon_cyclone) * R_earth * np.cos(np.radians(lat_cyclone))
        dy = np.radians(LAT - lat_cyclone) * R_earth

        # Create cyclonic velocity field (circular Gaussian profile)
        # The factor 1e5 scales the velocity to realistic values
        distance_squared = dx**2 + dy**2
        gaussian = np.exp(-distance_squared / (2 * sigma**2)) / 1e5

        # Cyclonic rotation (counterclockwise in Northern Hemisphere)
        U = -dy * gaussian  # x-component (zonal)
        V = dx * gaussian   # y-component (meridional)

        # Analyze the field
        velocity_norm = compute_velocity_norm(U, V)
        center_x, center_y = find_cyclone_center(velocity_norm, LON, LAT)
        max_velocity = np.max(velocity_norm)

        print(f"Cyclone simulation:")
        print(f"  - Center: ({center_x:.2f}°E, {center_y:.2f}°N)")
        print(f"  - Maximum wind speed: {max_velocity:.2f} m/s")

        # Save velocity fields if requested
        if save_fields:
            os.makedirs(output_dir, exist_ok=True)
            save_fields_edp(U, V, LON, LAT, results_dir=output_dir)
            print(f"Velocity fields saved to {output_dir}/")

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot velocity field (quiver)
        skip = 2  # Skip factor to avoid overcrowding arrows
        ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                 U[::skip, ::skip], V[::skip, ::skip],
                 scale=50, color='blue', transform=ccrs.PlateCarree())

        # Mark cyclone center
        ax.plot(center_x, center_y, 'ro', markersize=10, 
               transform=ccrs.PlateCarree(), label='Cyclone center')

        # Set plot labels and extent
        ax.set_title("Tropical Cyclone Simulation - Wind Field on Globe")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_extent([lon_cyclone - extent, lon_cyclone + extent,
                      lat_cyclone - extent, lat_cyclone + extent],
                     crs=ccrs.PlateCarree())

        # Add grid and legend
        ax.gridlines(draw_labels=True)
        ax.legend()
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Figure saved: {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

        return U, V, LON, LAT, center_x, center_y

    except Exception as e:
        print(f"Error in cyclone simulation: {e}", file=sys.stderr)
        return None, None, None, None, None, None


def main():
    """
    Main function for the cyclone globe simulation command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a cyclonic wind field on a geographic map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command-line arguments
    parser.add_argument("--longitude", type=float, default=-72.42,
                       help="Longitude of cyclone center")
    parser.add_argument("--latitude", type=float, default=19.0,
                       help="Latitude of cyclone center")
    parser.add_argument("--extent", type=float, default=10,
                       help="Map extent in degrees around center")
    parser.add_argument("--resolution", type=int, default=40,
                       help="Grid resolution (N×N points)")
    parser.add_argument("--sigma", type=float, default=5e5,
                       help="Characteristic width of cyclone in meters")
    parser.add_argument("--output", type=str, default="cyclone_simulation.png",
                       help="Output image filename")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--no-save-fields", action="store_true",
                       help="Don't save velocity fields")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display the plot")

    args = parser.parse_args()

    # Run simulation with command-line arguments
    simulate_cyclone_on_globe(
        lon_cyclone=args.longitude,
        lat_cyclone=args.latitude,
        extent=args.extent,
        N=args.resolution,
        sigma=args.sigma,
        output_file=args.output,
        save_fields=not args.no_save_fields,
        show_plot=not args.no_show,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
