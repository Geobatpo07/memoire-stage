import numpy as np
import os
from utils import load_edp_fields, save_fields_edp


def reshape_grids_if_needed(LON, LAT):
    """
    Vérifie si LON et LAT sont 1D. Si oui, les convertit en 2D avec meshgrid.
    """
    print("\nVérification des grilles LON et LAT...")
    print(f"  LON shape: {LON.shape}")
    print(f"  LAT shape: {LAT.shape}")

    if LON.ndim == 1 and LAT.ndim == 1:
        print("Conversion 1D → 2D via meshgrid.")
        LON_2D, LAT_2D = np.meshgrid(LON, LAT)
        return LON_2D, LAT_2D
    elif LON.ndim == 2 and LAT.ndim == 2:
        print("Déjà en 2D. Aucune action nécessaire.")
        return LON, LAT
    else:
        raise ValueError("Les grilles LON et LAT doivent être toutes deux 1D ou 2D.")


def main():
    input_dir = "../results"
    print(f"\nChargement des champs EDP depuis : {input_dir}")

    try:
        u_field, v_field, x_grid, y_grid = load_edp_fields(directory=input_dir)

        # Vérification/Conversion des grilles
        LON, LAT = reshape_grids_if_needed(x_grid, y_grid)

        # Sauvegarde des champs corrigés
        print("\nSauvegarde des champs EDP avec grilles corrigées...")
        save_fields_edp(u_field, v_field, LON, LAT, results_dir=input_dir)
        print("Fichiers mis à jour avec succès.")

    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
