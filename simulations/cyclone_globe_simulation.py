import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import compute_velocity_norm, find_cyclone_center, save_fields

# --- Paramètres géographiques (Haïti) ---
lon_cyclone = -72.42  # centre du cyclone (longitude)
lat_cyclone = 19.0    # centre du cyclone (latitude)
extent = 10        # extension en degrés autour du centre

N = 40  # résolution de la grille
lons = np.linspace(lon_cyclone - extent, lon_cyclone + extent, N)
lats = np.linspace(lat_cyclone - extent, lat_cyclone + extent, N)
LON, LAT = np.meshgrid(lons, lats)

# --- Conversion en radians pour calculs ---
lon_rad = np.radians(LON)
lat_rad = np.radians(LAT)

# --- Coordonnées relatives (km approx. avec sphère de rayon 6378135 m) ---
R_earth = 6378135  # rayon terrestre en m
dx = np.radians(LON - lon_cyclone) * R_earth * np.cos(np.radians(lat_cyclone))
dy = np.radians(LAT - lat_cyclone) * R_earth

# --- Champ de vitesse (cyclonique simple, champ circulaire) ---
sigma = 5e5  # largeur (m)
U = -dy * np.exp(-(dx**2 + dy**2) / (2*sigma**2)) / 1e5
V = dx * np.exp(-(dx**2 + dy**2) / (2*sigma**2)) / 1e5

# Calculer la norme de la vitesse
velocity_norm = compute_velocity_norm(U, V)

# Trouver le centre du cyclone
center_x, center_y = find_cyclone_center(velocity_norm, LON, LAT)

# Sauvegarder les champs
save_fields(U, V, LON, LAT)

# --- Tracer la carte ---
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Afficher le champ de vent
skip = 2
plt.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
           U[::skip, ::skip], V[::skip, ::skip],
           scale=50, color='blue', transform=ccrs.PlateCarree())

# Marquer le centre du cyclone
plt.plot(center_x, center_y, 'ro', markersize=10, transform=ccrs.PlateCarree(),
         label='Centre du cyclone')

# Personnalisation
plt.title("Simulation d'un cyclone tropical – Champ de vent sur le globe")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_extent([lon_cyclone - extent, lon_cyclone + extent,
               lat_cyclone - extent, lat_cyclone + extent],
              crs=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
plt.legend()
plt.tight_layout()

# Sauvegarder et afficher
plt.savefig('cyclone_simulation.png')
print("Figure sauvegardée : cyclone_simulation.png")
plt.show()