import numpy as np
import matplotlib.pyplot as plt
from utils import (
    shift,
    lax_friedrichs_2d,
    compute_velocity_norm,
    save_fields,
    plot_quiver_field
)

def simulate_edp(L=300e3, N=200, Tmax=3600, u_star=5.0, v_star=0.0, w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5, plot_every=300):
    """
    Simule le système EDP sur une grille 2D avec méthode explicite.

    Args:
        L (float): taille du domaine (m)
        N (int): résolution de la grille
        Tmax (float): temps final (s)
        u_star, v_star (float): vitesses de fond
        w0 (float): vitesse de rotation terrestre (rad/s)
        phi_deg (float): latitude (en degrés)
        k, xi_star (float): friction et absorption
        plot_every (int): intervalle d'affichage (pas de temps)
    """
    # Validations
    assert N > 0, "La résolution de la grille N doit être positive."
    assert L > 0, "La taille du domaine L doit être positive."
    assert Tmax > 0, "Le temps final Tmax doit être positif."

    dx = L / N
    dy = dx
    dt = 0.8 * dx / 10
    steps = int(Tmax / dt)
    phi = np.radians(phi_deg)

    # Grille
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    # Conditions initiales (perturbation gaussienne)
    u = u_star + 2.0 * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (1e9))
    v = v_star + 0.5 * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (1e9))

    # Simulation temporelle
    for n in range(steps):
        # Validations
        assert np.all(np.isfinite(u)), "Valeurs infinies détectées dans u."
        assert np.all(np.isfinite(v)), "Valeurs infinies détectées dans v."

        u_adv = lax_friedrichs_2d(u, u, v, dx, dy)
        v_adv = lax_friedrichs_2d(v, u, v, dx, dy)

        rhs_u = (xi_star - k)*(u - u_star) + w0 * np.sin(phi)*(v - v_star)
        rhs_v = (xi_star - k)*(v - v_star) - w0 * np.sin(phi)*(u - u_star)

        u = u - dt * u_adv + dt * rhs_u
        v = v - dt * v_adv + dt * rhs_v

        if n % plot_every == 0:
            norm = compute_velocity_norm(u, v)
            plot_quiver_field(X, Y, u, v, title=f"Champ de vent à t = {int(n*dt)} s")

    # Sauvegarde
    save_fields(u, v, x, y)
    print("Simulation EDP terminée et résultats sauvegardés.")

# Point d'entrée facultatif
if __name__ == "__main__":
    simulate_edp()
