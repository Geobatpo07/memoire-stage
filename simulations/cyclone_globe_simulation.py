import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from utils import compute_velocity_norm, find_cyclone_center, save_fields_edp, load_edp_fields


def ensure_grids_are_2d(X, Y):
    """
    Convertit X et Y en grilles 2D si elles sont en 1D.
    """
    if X.ndim == 1 and Y.ndim == 1:
        print("Conversion de grilles 1D en 2D via np.meshgrid.")
        return np.meshgrid(X, Y)
    elif X.ndim == 2 and Y.ndim == 2:
        return X, Y
    else:
        raise ValueError("Les grilles X et Y doivent être toutes deux 1D ou 2D.")


def simulate_cyclone_on_globe(lon_cyclone=-72.42, lat_cyclone=19.0, extent=10,
                              output_file="cyclone_simulation.png",
                              save_fields=True, show_plot=True, output_dir="results"):
    try:
        # Chargement des champs simulés
        U, V, X, Y = load_edp_fields(directory=output_dir)

        # Vérification et conversion éventuelle des grilles
        X, Y = ensure_grids_are_2d(X, Y)

        # Conversion des grilles cartésiennes en géographiques
        R_earth = 6378135  # Rayon moyen de la Terre en mètres
        LAT = lat_cyclone + (Y / R_earth) * (180 / np.pi)
        LON = lon_cyclone + (X / (R_earth * np.cos(np.radians(lat_cyclone)))) * (180 / np.pi)

        # Analyse du champ de vitesse
        velocity_norm = compute_velocity_norm(U, V)
        center_lon, center_lat = find_cyclone_center(velocity_norm, LON, LAT)
        max_velocity = np.max(velocity_norm)

        print("Simulation basée sur les vitesses EDP :")
        print(f"  - Centre détecté : ({center_lon:.2f}°E, {center_lat:.2f}°N)")
        print(f"  - Vitesse maximale : {max_velocity:.2f} m/s")

        if save_fields:
            os.makedirs(output_dir, exist_ok=True)
            save_fields_edp(U, V, LON, LAT, results_dir=output_dir)
            print(f"Champs de vitesse sauvegardés dans : {output_dir}/")

        # Visualisation du champ de vent sur la carte
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        skip = 2
        ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                  U[::skip, ::skip], V[::skip, ::skip],
                  scale=50, color='blue', transform=ccrs.PlateCarree())

        ax.plot(center_lon, center_lat, 'ro', markersize=10,
                transform=ccrs.PlateCarree(), label='Centre du cyclone')

        ax.set_title("Simulation de cyclone (EDP) – Champ de vent sur le globe")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_extent([LON.min(), LON.max(), LAT.min(), LAT.max()], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True)
        ax.legend()
        plt.tight_layout()

        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path)
        print(f"Figure enregistrée : {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return U, V, LON, LAT, center_lon, center_lat

    except Exception as e:
        print(f"Erreur dans la simulation du cyclone : {e}", file=sys.stderr)
        return None, None, None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Visualisation des vitesses EDP projetées sur le globe")
    parser.add_argument("--longitude", type=float, default=-72.42,
                        help="Longitude du centre du cyclone (origine X)")
    parser.add_argument("--latitude", type=float, default=19.0,
                        help="Latitude du centre du cyclone (origine Y)")
    parser.add_argument("--output", type=str, default="cyclone_simulation.png",
                        help="Nom du fichier image de sortie")
    parser.add_argument("--output-dir", type=str, default="../results",
                        help="Dossier de chargement et de sortie")
    parser.add_argument("--no-save-fields", action="store_true",
                        help="Ne pas enregistrer les nouveaux champs de vitesse")
    parser.add_argument("--no-show", action="store_true",
                        help="Ne pas afficher la figure")

    args = parser.parse_args()

    simulate_cyclone_on_globe(
        lon_cyclone=args.longitude,
        lat_cyclone=args.latitude,
        output_file=args.output,
        save_fields=not args.no_save_fields,
        show_plot=not args.no_show,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
