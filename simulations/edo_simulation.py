import argparse
from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward,
    simulate_edo_reduced_theta
)
from utils import save_edo_result, save_plot

# === Mapping des méthodes disponibles ===
METHODS = {
    "ivp": simulate_edo_ivp,
    "bdf": simulate_edo_ivp_bdf,
    "euler": simulate_edo_euler_backward,
    "theta": simulate_edo_reduced_theta,
}

def main():
    parser = argparse.ArgumentParser(description="Simulation du système EDO pour modélisation cyclonique")
    parser.add_argument("--method", choices=METHODS.keys(), default="ivp",
                        help="Méthode de résolution à utiliser")
    parser.add_argument("--output", type=str, default="edo_results.csv",
                        help="Nom du fichier CSV de sortie")
    parser.add_argument("--show", action="store_true", help="Afficher les figures")
    args = parser.parse_args()

    # Exécution de la méthode choisie
    print(f"\n Simulation EDO avec la méthode : {args.method}")
    result = METHODS[args.method]()
    save_edo_result(result, args.output)

    # Affichage des figures si demandé
    if args.show:
        save_plot(result["xi_km"], [result["a_xi"], result["b_xi"]],
                  labels=["a(ξ) (radial)", "b(ξ) (tangentiel)"],
                  title="Composantes de la vitesse", xlabel="ξ (km)",
                  ylabel="vitesse (m/s)", filename="edo_ab_components.pdf")

        save_plot(result["xi_km"], [result["theta_xi"]], ["θ(ξ)"],
                  title="Facteur géostrophique", xlabel="ξ (km)",
                  ylabel="θ(ξ)", filename="edo_theta.pdf")

        save_plot(result["xi_km"], [result["E_xi"]], ["Énergie cinétique"],
                  title="Énergie cinétique", xlabel="ξ (km)",
                  ylabel="E(ξ)", filename="edo_energy.pdf")

if __name__ == "__main__":
    main()
