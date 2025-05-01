import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from utils import save_plot, export_dataframe

def simulate_edo_ivp(Rext=300000, xi0=10000, N=1000, w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5, theta0=2e-4, show_plots=True, filename="edo_results.csv"):
    """
    Résout le système EDO du modèle cyclonique par méthode solve_ivp.

    Args:
        Rext (float): rayon externe (m)
        xi0 (float): rayon initial (m)
        N (int): nombre de points
        w0 (float): rotation terrestre
        phi_deg (float): latitude en degrés
        k, xi_star (float): friction, absorption
        theta0 (float): facteur géostrophique en Rext
        show_plots (bool): affichage des figures
        filename (str): nom du fichier CSV de sortie
    """

    phi_star = np.radians(phi_deg)
    xi_eval = np.linspace(xi0, Rext, N)

    a_R = 1e-6
    b_R = -theta0 * a_R

    def system(xi, y):
        a, b = y
        if np.abs(a) < 1e-10:
            return [0, 0]
        da = ((xi_star - k)*a + w0*np.sin(phi_star)*b - a**2 + b**2) / (xi * a)
        db = ((xi_star - k)*b - w0*np.sin(phi_star)*a - 2*a*b) / (xi * a)
        return [da, db]

    # Résolution
    sol = solve_ivp(system, [Rext, xi0], [a_R, b_R], t_eval=xi_eval[::-1], method='RK45', rtol=1e-6)

    xi_sol = sol.t[::-1] / 1000
    a_sol = sol.y[0][::-1]
    b_sol = sol.y[1][::-1]
    theta_sol = -b_sol / a_sol
    E_sol = 0.5 * (a_sol**2 + b_sol**2)

    # Sauvegarde CSV
    df = pd.DataFrame({
        "xi_km": xi_sol,
        "a_xi": a_sol,
        "b_xi": b_sol,
        "theta_xi": theta_sol,
        "E_xi": E_sol
    })
    export_dataframe(df, filename)

    # Graphiques
    if show_plots:
        save_plot(xi_sol, [a_sol, b_sol],
                  labels=["a(ξ) (radial)", "b(ξ) (tangentiel)"],
                  title="Composantes radiale et tangente de la vitesse",
                  xlabel="ξ (km)", ylabel="vitesse (m/s)",
                  filename="edo_ab_components.pdf")

        save_plot(xi_sol, [theta_sol], ["θ(ξ)"],
                  title="Facteur géostrophique",
                  xlabel="ξ (km)", ylabel="θ(ξ)",
                  filename="edo_theta.pdf")

        save_plot(xi_sol, [E_sol], ["Énergie cinétique"],
                  title="Énergie cinétique du champ de vitesse",
                  xlabel="ξ (km)", ylabel="E(ξ)",
                  filename="edo_energy.pdf")

    print(f"Simulation EDO terminée. Résultats exportés dans : {filename}")


# Point d'entrée
if __name__ == "__main__":
    simulate_edo_ivp()
