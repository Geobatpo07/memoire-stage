import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from utils import export_dataframe, format_edo_solution, save_edo_result


def simulate_edo_ivp(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4, w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5, method="RK45"):
    """
    Simule le système d’EDO du cyclone à l’aide de solve_ivp de SciPy.

    Retour :
        dict : Dictionnaire contenant la solution et les grandeurs dérivées.
    """
    phi_star = np.radians(phi_deg)
    xi_eval = np.linspace(xi0, Rext, N)

    a_R = -1e-6  # Négatif : flux radial entrant
    b_R = theta0 * abs(a_R)  # Positif : rotation cyclonique

    def system(xi, y):
        a, b = y
        if np.abs(a) < 1e-10:
            return [0, 0]
        da = ((xi_star - k)*a + w0*np.sin(phi_star)*b - a**2 + b**2) / (xi * a)
        db = ((xi_star - k)*b - w0*np.sin(phi_star)*a - 2*a*b) / (xi * a)
        return [da, db]

    def stop_when_a_positive(xi, y):
        return y[0]
    stop_when_a_positive.terminal = True
    stop_when_a_positive.direction = 1

    try:
        sol = solve_ivp(
            system,
            [Rext, xi0],
            [a_R, b_R],
            t_eval=xi_eval[::-1],
            method=method,
            events=stop_when_a_positive
        )

        if sol.status == 1:
            print(f"Intégration stoppée automatiquement : a(ξ) est devenu négatif à ξ = {sol.t[-1]:.2f} m")

        return format_edo_solution(sol.t[::-1], sol.y[0][::-1], sol.y[1][::-1])
    except Exception as e:
        print(f"Erreur lors de l'intégration de l'EDO ({method}) : {e}")
        empty = np.zeros(N)
        return format_edo_solution(xi_eval, empty, empty)


def simulate_edo_ivp_bdf(**kwargs):
    """
    Simule le système d’EDO du cyclone avec la méthode BDF (implicite), en arrêtant
    la simulation si a devient positif.

    Arguments :
        **kwargs : Arguments à transmettre à simulate_edo_ivp.

    Retour :
        dict : Résultats de la simulation.
    """
    return simulate_edo_ivp(method="BDF", **kwargs)


def simulate_edo_euler_backward(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4, w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    """
    Simule le système d’EDO du cyclone avec une méthode d’Euler explicite inversée.
    Interrompt la simulation si a devient positif.

    Retour :
        dict : Résultats de la simulation.
    """
    try:
        phi_star = np.radians(phi_deg)
        h = (Rext - xi0) / N
        xi = np.linspace(Rext, xi0, N)

        a = np.zeros(N)
        b = np.zeros(N)
        theta = np.zeros(N)

        a[0] = -1e-6
        b[0] = theta0 * abs(a[0])
        theta[0] = theta0

        for j in range(1, N):
            xi_j = xi[j]
            a_prev, b_prev = a[j-1], b[j-1]
            theta_prev = theta[j-1]

            b[j] = (xi[j-1]*b_prev - w0*np.sin(phi_star)*h + b_prev*h +
                   (xi_star - k)*theta_prev*h) / xi_j

            a[j] = (xi[j-1]*a_prev - theta_prev**2 * a_prev*h -
                   (xi_star - k)*h + w0*np.sin(phi_star)*theta_prev*h) / xi_j

            if a[j] > 0:
                print(f"Arrêt de la simulation : a({xi_j/1000:.2f} km) > 0")
                return format_edo_solution(xi[:j+1][::-1], a[:j+1][::-1], b[:j+1][::-1])

            theta[j] = -b[j]/a[j] if abs(a[j]) > 1e-10 else 0.0

        return format_edo_solution(xi[::-1], a[::-1], b[::-1])

    except Exception as e:
        print(f"Erreur dans l'intégration par Euler inversé : {e}")
        empty = np.zeros(N)
        return format_edo_solution(np.linspace(xi0, Rext, N), empty, empty)


def simulate_edo_reduced_theta(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    """
    Simule le système d’EDO dans une version réduite avec a(xi) et θ(xi).
    Interrompt la simulation si a devient positif ou trop proche de zéro.
    """
    try:
        phi_star = np.radians(phi_deg)
        xi_eval = np.linspace(xi0, Rext, N)

        a_R = -1e-6
        theta_R = theta0

        def reduced_system(xi, y):
            a, theta = y
            # Tolérance légèrement plus souple
            if np.abs(a) < 1e-8 or abs(theta**2 - 1) < 1e-8:
                return [0, 0]
            da = ((xi_star - k) + w0*np.sin(phi_star)*theta) / (xi * (1 - theta**2)) - 2*a/xi
            dtheta = (w0*np.sin(phi_star)/a - theta) * (1 + theta**2)
            return [da * a, dtheta]

        def stop_if_a_positive(xi, y):
            return -y[0]  # s’arrête si a > 0
        stop_if_a_positive.terminal = True
        stop_if_a_positive.direction = -1

        sol = solve_ivp(
            reduced_system,
            [Rext, xi0],
            [a_R, theta_R],
            t_eval=xi_eval[::-1],
            events=stop_if_a_positive,
            method="RK45"
        )

        if sol.status == 1:
            print(f"Intégration stoppée automatiquement : a(ξ) est devenu positif à ξ = {sol.t[-1]:.2f} m")

        if sol.t.size == 0 or sol.y.shape[1] == 0:
            print("Aucune donnée enregistrée — la simulation a échoué dès le début.")
            return format_edo_solution(xi_eval, np.zeros(N), np.zeros(N))

        a_sol = sol.y[0][::-1]
        theta_sol = sol.y[1][::-1]
        b_sol = -theta_sol * a_sol

        return format_edo_solution(sol.t[::-1], a_sol, b_sol)

    except Exception as e:
        print(f"Erreur dans l'intégration avec theta réduit : {e}")
        empty = np.zeros(N)
        return format_edo_solution(xi_eval, empty, empty)
