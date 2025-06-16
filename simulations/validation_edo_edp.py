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
    Valider le modèle EDP en le comparant au profil analytique EDO.

    Cette fonction charge les résultats EDO à partir d’un fichier CSV ainsi que les résultats
    de la simulation EDP, calcule les profils radiaux et génère des graphiques et données
    de comparaison.

    Args:
        edo_csv (str, optional): Chemin du fichier CSV contenant les résultats EDO. Par défaut "edo_results.csv".
        pdf_output (str, optional): Nom du fichier PDF de sortie pour le graphique comparatif. Par défaut "validation_edo_edp_comparison.pdf".
        csv_output (str, optional): Nom du fichier CSV de sortie pour les données de validation. Par défaut "validation_profiles.csv".
        show_plot (bool, optional): Afficher ou non le graphique. Par défaut True.
        edp_dir (str, optional): Répertoire contenant les fichiers de champs EDP. Par défaut le répertoire courant.
        output_dir (str, optional): Répertoire pour enregistrer les fichiers de sortie. Par défaut "results".

    Returns:
        bool: True si la validation est réussie, False sinon.
    """
    try:
        print("\nValidation du modèle EDP par rapport au profil EDO...")

        # 1. Charger les résultats EDO
        try:
            edo_data = pd.read_csv(edo_csv)
            print(f"Données EDO chargées depuis {edo_csv}")

            xi_edo_km = edo_data["xi_km"].values
            a = edo_data["a_xi"].values
            b = edo_data["b_xi"].values
            v_norm_edo = np.sqrt(a**2 + b**2)

        except Exception as e:
            print(f"Erreur lors du chargement des données EDO : {e}", file=sys.stderr)
            print(f"Assurez-vous que le fichier {edo_csv} existe et contient les colonnes requises.")
            return False

        # 2. Charger les champs EDP
        try:
            u_sim, v_sim, x, y = load_edp_fields(directory=edp_dir)
            print(f"Champs EDP chargés depuis {edp_dir}")

            # Créer une grille de coordonnées et calculer la distance au centre
            X, Y = np.meshgrid(x, y)
            R = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2)
            v_norm_sim_grid = compute_velocity_norm(u_sim, v_sim)

        except Exception as e:
            print(f"Erreur lors du chargement des champs EDP : {e}", file=sys.stderr)
            return False

        # 3. Calculer le profil radial à partir de la simulation EDP
        r_centers, v_norm_sim_profile = compute_radial_profile(R, v_norm_sim_grid)
        r_centers_km = r_centers / 1000  # Conversion en kilomètres

        # 4. Interpoler le profil EDO pour correspondre aux points radiaux EDP
        try:
            interp_edo = interp1d(xi_edo_km, v_norm_edo, bounds_error=False, fill_value="extrapolate")
            v_norm_edo_interp = interp_edo(r_centers_km)

        except Exception as e:
            print(f"Erreur lors de l'interpolation : {e}", file=sys.stderr)
            return False

        # 5. Créer et enregistrer le graphique de comparaison
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, pdf_output)

        save_plot(
            x=r_centers_km,
            y_list=[v_norm_sim_profile, v_norm_edo_interp],
            labels=["Simulation EDP", "Théorie EDO"],
            title="Validation croisée : Profil radial |v(ξ)| – EDO vs EDP",
            xlabel="Distance au centre ξ (km)",
            ylabel="Norme de la vitesse |v(ξ)| (m/s)",
            filename=pdf_output,
            styles=["--", "-"],
            show=show_plot,
            results_dir=output_dir
        )

        # 6. Exporter les profils vers un fichier CSV
        csv_path = os.path.join(output_dir, csv_output)

        # Calculer la différence relative
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

        # Calculer les métriques de validation
        valid_indices = ~np.isnan(rel_diff)
        if np.any(valid_indices):
            mean_rel_diff = np.mean(rel_diff[valid_indices]) * 100  # en pourcentage
            max_rel_diff = np.max(rel_diff[valid_indices]) * 100    # en pourcentage

            print("\nMétriques de validation :")
            print(f"  - Différence relative moyenne : {mean_rel_diff:.2f}%")
            print(f"  - Différence relative maximale : {max_rel_diff:.2f}%")

        print("\nValidation EDO-EDP terminée avec succès.")
        print(f"  - Graphique de comparaison : {pdf_path}")
        print(f"  - Données de validation : {csv_path}")

        return True

    except Exception as e:
        print(f"Erreur dans la validation EDO-EDP : {e}", file=sys.stderr)
        return False


def main():
    """
    Fonction principale pour l’interface en ligne de commande de validation EDO-EDP.
    """
    parser = argparse.ArgumentParser(
        description="Valider le modèle EDP par rapport au profil analytique EDO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--ode-csv", type=str, default="edo_results.csv",
                       help="Chemin du fichier CSV contenant les résultats EDO")
    parser.add_argument("--pdf-output", type=str, default="validation_edo_edp_comparison.pdf",
                       help="Nom du fichier PDF de sortie pour le graphique")
    parser.add_argument("--csv-output", type=str, default="validation_profiles.csv",
                       help="Nom du fichier CSV de sortie pour les données de validation")
    parser.add_argument("--no-show", action="store_true",
                       help="Ne pas afficher le graphique")
    parser.add_argument("--edp-dir", type=str, default=".",
                       help="Répertoire contenant les fichiers de champs EDP")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Répertoire où enregistrer les fichiers de sortie")

    args = parser.parse_args()

    # Exécuter la validation
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
