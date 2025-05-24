import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import compute_velocity_norm, find_cyclone_center, save_fields_edp

def simulate_cyclone_on_globe(lon_cyclone=-72.42, lat_cyclone=19.0, extent=10, N=40, sigma=5e5):
    """
    Simule un champ de vent cyclonique centré sur une localisation géographique
    et le projette sur une carte du globe à l'aide de Cartopy. (par défaut, Port-au-Prince, Haïti)

    Args:
        lon_cyclone (float): Longitude du centre du cyclone.
        lat_cyclone (float): Latitude du centre du cyclone.
        extent (float): Étendue (en degrés) autour du centre.
        N (int): Résolution de la grille.
        sigma (float): Largeur caractéristique du cyclone.
    """
    # Grille géographique
    lons = np.linspace(lon_cyclone - extent, lon_cyclone + extent, N)
    lats = np.linspace(lat_cyclone - extent, lat_cyclone + extent, N)
    LON, LAT = np.meshgrid(lons, lats)

    # Conversion en coordonnées (m)
    R_earth = 6378135  # m
    dx = np.radians(LON - lon_cyclone) * R_earth * np.cos(np.radians(lat_cyclone))
    dy = np.radians(LAT - lat_cyclone) * R_earth

    # Champ de vitesse cyclonique (profil gaussien circulaire)
    U = -dy * np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) / 1e5
    V = dx * np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) / 1e5

    # Analyse
    velocity_norm = compute_velocity_norm(U, V)
    center_x, center_y = find_cyclone_center(velocity_norm, LON, LAT)

    # Sauvegarde des champs
    save_fields_edp(U, V, LON, LAT)

    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    skip = 2
    ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
              U[::skip, ::skip], V[::skip, ::skip],
              scale=50, color='blue', transform=ccrs.PlateCarree())

    ax.plot(center_x, center_y, 'ro', markersize=10, transform=ccrs.PlateCarree(), label='Centre du cyclone')

    ax.set_title("Simulation d'un cyclone tropical – Champ de vent sur le globe")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_extent([lon_cyclone - extent, lon_cyclone + extent,
                   lat_cyclone - extent, lat_cyclone + extent],
                  crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("cyclone_simulation.png")
    print("Figure sauvegardée : cyclone_simulation.png")
    plt.show()

# Exécution directe
if __name__ == "__main__":
    simulate_cyclone_on_globe()
