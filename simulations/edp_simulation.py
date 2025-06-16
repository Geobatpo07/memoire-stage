import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time

from cyclone_globe_simulation import ensure_grids_are_2d
from utils import (
    shift,
    lax_friedrichs_2d,
    compute_velocity_norm,
    save_fields_edp,
    plot_quiver_field, load_edp_fields
)

def simulate_edp(L=300e3, N=200, Tmax=3600, u_star=5.0, v_star=0.0,
                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5,
                plot_every=300, show_progress=True, save_dir="../results"):
    """
    Simule le système d’EDP sur une grille 2D à l’aide d’une méthode explicite.

    Paramètres :
        L (float) : Taille du domaine (m)
        N (int) : Résolution de la grille
        Tmax (float) : Temps final de la simulation (s)
        u_star (float) : Vitesse de fond en x
        v_star (float) : Vitesse de fond en y
        w0 (float) : Taux de rotation de la Terre (rad/s)
        phi_deg (float) : Latitude en degrés
        k (float) : Coefficient de frottement
        xi_star (float) : Coefficient d’absorption
        plot_every (int) : Intervalle de génération de graphiques (en pas de temps)
        show_progress (bool) : Affiche ou non la progression
        save_dir (str) : Répertoire de sauvegarde des résultats

    Retourne :
        tuple : (u, v, x, y) champs de vitesse finaux et grilles de coordonnées
    """
    try:
        # Calcul du pas d’espace et du pas de temps (condition CFL)
        dx = L / N
        dt = 0.8 * dx / 10  # facteur CFL pour la stabilité
        steps = int(Tmax / dt)
        phi = np.radians(phi_deg)

        if show_progress:
            print("Initialisation de la simulation EDP :")
            print(f"  - Grille : {N}x{N} points (espacement {dx/1000:.1f} km)")
            print(f"  - Pas de temps : {dt:.2f} s ({steps} pas)")
            print(f"  - Paramètres physiques : phi={phi_deg}°, k={k}, xi_star={xi_star}")

            # Chargement des champs simulés
            U, V, X, Y = load_edp_fields(directory="../results")

            # Vérification et conversion éventuelle des grilles
            X, Y = ensure_grids_are_2d(X, Y)

        # Initialisation du champ de vitesse avec perturbation gaussienne
        center_perturbation = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (1e9))
        u = u_star + 2.0 * center_perturbation
        v = v_star + 0.5 * center_perturbation

        # Boucle principale d’intégration temporelle
        start_time = time.time()
        last_update = start_time

        for n in range(steps):
            # Calcul des termes d’advection par Lax-Friedrichs
            u_adv = lax_friedrichs_2d(u, u, v, dx, dx)
            v_adv = lax_friedrichs_2d(v, u, v, dx, dx)

            # Calcul des sources (Coriolis, frottement, absorption)
            rhs_u = (xi_star - k)*(u - u_star) + w0 * np.sin(phi)*(v - v_star)
            rhs_v = (xi_star - k)*(v - v_star) - w0 * np.sin(phi)*(u - u_star)

            # Mise à jour du champ de vitesse
            u -= dt * u_adv - dt * rhs_u
            v -= dt * v_adv - dt * rhs_v

            # Tracé à intervalles réguliers
            if n % plot_every == 0:
                norm = compute_velocity_norm(u, v)
                max_vel = np.max(norm)
                plot_quiver_field(X, Y, u, v,
                                 title=f"Champ de vent à t = {int(n*dt)} s (max : {max_vel:.2f} m/s)")

            # Affichage de la progression
            if show_progress and (time.time() - last_update > 5.0 or n == steps-1):
                elapsed = time.time() - start_time
                progress = (n + 1) / steps
                remaining = elapsed / progress - elapsed if progress > 0 else 0
                print(f"  Progression : {progress*100:.1f}% - Étape {n+1}/{steps} - " 
                      f"Écoulé : {elapsed:.1f}s - Restant : {remaining:.1f}s")
                last_update = time.time()

        # Sauvegarde finale
        save_fields_edp(u, v, x, y, results_dir=save_dir)
        print(f"Simulation EDP terminée, résultats enregistrés dans {save_dir}/")

        return u, v, x, y

    except Exception as e:
        print(f"Erreur dans la simulation EDP : {e}", file=sys.stderr)
        # Retourne des tableaux vides en cas d’erreur
        return np.zeros((N, N)), np.zeros((N, N)), np.linspace(0, L, N), np.linspace(0, L, N)


def main():
    """
    Fonction principale pour l’interface en ligne de commande de simulation EDP.

    Cette fonction analyse les arguments passés et lance la simulation.
    """
    parser = argparse.ArgumentParser(
        description="Simulation du système EDP pour cyclone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Définition des arguments
    parser.add_argument("--size", type=float, default=300e3,
                       help="Taille du domaine en mètres")
    parser.add_argument("--resolution", type=int, default=200,
                       help="Résolution de la grille (N×N)")
    parser.add_argument("--time", type=float, default=3600,
                       help="Temps total de la simulation en secondes")
    parser.add_argument("--latitude", type=float, default=20,
                       help="Latitude en degrés")
    parser.add_argument("--friction", type=float, default=1e-5,
                       help="Coefficient de frottement k")
    parser.add_argument("--absorption", type=float, default=5e-5,
                       help="Coefficient d’absorption xi_star")
    parser.add_argument("--plot-every", type=int, default=300,
                       help="Intervalle de tracé (en pas de temps)")
    parser.add_argument("--no-progress", action="store_true",
                       help="Désactiver l’affichage de la progression")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Répertoire de sortie des résultats")

    args = parser.parse_args()

    # Lancement de la simulation avec les arguments
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
