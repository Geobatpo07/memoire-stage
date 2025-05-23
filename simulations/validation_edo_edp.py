import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils import (
    load_edp_fields,
    compute_velocity_norm,
    compute_radial_profile,
    save_plot,
    export_dataframe
)

def validate_edo_edp(edo_csv="edo_results.csv", pdf_output="validation_edo_edp_comparaison.pdf", csv_output="validation_profiles.csv", show_plot=True):
    """
    Valide le modèle EDP par comparaison au profil analytique EDO.

    Args:
        edo_csv (str): chemin du fichier EDO (CSV)
        pdf_output (str): nom du fichier graphique PDF à exporter
        csv_output (str): nom du fichier CSV de validation à exporter
        show_plot (bool): True pour afficher le graphique
    """

    # --- 1. Charger les résultats EDO ---
    edo_data = pd.read_csv(edo_csv)
    xi_edo_km = edo_data["xi_km"].values
    a = edo_data["a_xi"].values
    b = edo_data["b_xi"].values
    v_norm_edo = np.sqrt(a**2 + b**2)

    # --- 2. Charger les champs EDP ---
    u_sim, v_sim, x, y = load_edp_fields()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2)
    v_norm_sim_grid = compute_velocity_norm(u_sim, v_sim)

    # --- 3. Construire le profil radial simulé ---
    r_centers, v_norm_sim_profile = compute_radial_profile(R, v_norm_sim_grid)
    r_centers_km = r_centers / 1000

    # --- 4. Interpoler le profil EDO sur les rayons simulés ---
    interp_edo = interp1d(xi_edo_km, v_norm_edo, bounds_error=False, fill_value="extrapolate")
    v_norm_edo_interp = interp_edo(r_centers_km)

    # --- 5. Tracer et sauvegarder la comparaison ---
    save_plot(
        x=r_centers_km,
        y_list=[v_norm_sim_profile, v_norm_edo_interp],
        labels=["Simulation EDP", "Théorie EDO"],
        title="Validation croisée : Profil radial |v(ξ)| – EDO vs EDP",
        xlabel="Distance au centre ξ (km)",
        ylabel="Norme de la vitesse |v(ξ)| (m/s)",
        filename=pdf_output,
        styles=["--", "-"],
        show=show_plot
    )

    # --- 6. Export CSV des profils ---
    df = pd.DataFrame({
        "r_km": r_centers_km,
        "v_norm_edp": v_norm_sim_profile,
        "v_norm_edo": v_norm_edo_interp
    })
    export_dataframe(df, csv_output)

    print("Validation EDO-EDP terminée.")
    print(f"   Graphique : {pdf_output}")
    print(f"   Données   : {csv_output}")


# Point d'entrée direct
if __name__ == "__main__":
    validate_edo_edp()
