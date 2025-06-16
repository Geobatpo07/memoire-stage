import argparse
import sys
import os

from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward,
    simulate_edo_reduced_theta
)
from utils import (
    save_edo_result,
    save_plot,
    generate_velocity_fields_from_edo,
    get_versioned_filename
)

# Méthodes disponibles
METHODES = {
    "ivp": simulate_edo_ivp,
    "bdf": simulate_edo_ivp_bdf,
    "euler": simulate_edo_euler_backward,
    "theta": simulate_edo_reduced_theta,
}


def main():
    parser = argparse.ArgumentParser(
        description="Simulation du système d’EDO pour le cyclone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--method", choices=METHODES.keys(), default="ivp",
                        help="Méthode de résolution à utiliser")
    parser.add_argument("--output", type=str, default="../results",
                        help="Nom du dossier de sortie")
    parser.add_argument("--show", action="store_true",
                        help="Afficher les figures")
    parser.add_argument("--params", type=str,
                        help="Paramètres supplémentaires au format 'clé1=valeur1,clé2=valeur2,...'")

    args = parser.parse_args()

    try:
        print(f"\nMéthode EDO sélectionnée : {args.method}")

        # Paramètres supplémentaires
        extra_params = {}
        if args.params:
            for param in args.params.split(','):
                key, value = param.split('=')
                try:
                    extra_params[key] = int(value)
                except ValueError:
                    try:
                        extra_params[key] = float(value)
                    except ValueError:
                        extra_params[key] = value

        # Exécution de la simulation
        result = METHODES[args.method](**extra_params)

        if not isinstance(result, dict) or "xi_km" not in result or len(result["xi_km"]) == 0:
            print("Aucune donnée — a(ξ) est probablement devenu négatif prématurément.")
            return

        # Déterminer le fichier CSV versionné
        results_dir = os.path.dirname(args.output) or "."
        base_csv_name = f"edo_results_{args.method}.csv"
        versioned_csv = get_versioned_filename(results_dir, base_csv_name)

        # Sauvegarder les résultats CSV
        save_edo_result(result, versioned_csv)
        print(f"Résultats enregistrés : {versioned_csv}")

        # Génération des champs vectoriels et sauvegarde des .npy
        generate_velocity_fields_from_edo(result, output_dir=results_dir)

        # Graphiques si demandé
        if args.show:
            save_plot(
                result["xi_km"],
                [result["a_xi"], result["b_xi"]],
                labels=["a(ξ) (radiale)", "b(ξ) (tangentielle)"],
                title="Composantes de vitesse",
                xlabel="ξ (km)",
                ylabel="Vitesse (m/s)",
                filename=get_versioned_filename(args.output, f"edo_ab_components_{args.method}.png")
            )

            save_plot(
                result["xi_km"],
                [result["theta_xi"]],
                labels=["θ(ξ)"],
                title="Facteur géostrophique",
                xlabel="ξ (km)",
                ylabel="θ(ξ)",
                filename=get_versioned_filename(args.output, f"edo_theta_{args.method}.png")
            )

            save_plot(
                result["xi_km"],
                [result["E_xi"]],
                labels=["Énergie cinétique"],
                title="Énergie cinétique du vent",
                xlabel="ξ (km)",
                ylabel="E(ξ)",
                filename=get_versioned_filename(args.output, f"edo_energy_{args.method}.png")
            )

            print("Graphiques générés avec succès")

    except Exception as e:
        print(f"Erreur lors de la simulation EDO : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
