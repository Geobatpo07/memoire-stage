import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from utils import export_dataframe, format_edo_solution, save_edo_result

# === 1. solve_ivp RK45 classique ===
def simulate_edo_ivp(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                      w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
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

    sol = solve_ivp(system, [Rext, xi0], [a_R, b_R], t_eval=xi_eval[::-1], method="RK45")
    return format_edo_solution(sol.t[::-1], sol.y[0][::-1], sol.y[1][::-1])


# === 2. solve_ivp BDF implicite ===
def simulate_edo_ivp_bdf(**kwargs):
    return simulate_edo_ivp(method="BDF", **kwargs)


# === 3. Euler retrograde explicite (marche arriere sur grille) ===
def simulate_edo_euler_backward(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    phi_star = np.radians(phi_deg)
    h = (Rext - xi0) / N
    xi = np.linspace(Rext, xi0, N)

    a = np.zeros(N)
    b = np.zeros(N)
    theta = np.zeros(N)

    a[0] = 1e-6
    b[0] = -theta0 * a[0]
    theta[0] = theta0

    for j in range(1, N):
        xi_j = xi[j]
        a_prev, b_prev = a[j-1], b[j-1]
        theta_prev = theta[j-1]

        b[j] = (xi[j-1]*b_prev - w0*np.sin(phi_star)*h + b_prev*h + (xi_star - k)*theta_prev*h) / xi_j
        a[j] = (xi[j-1]*a_prev - theta_prev**2 * a_prev*h - (xi_star - k)*h + w0*np.sin(phi_star)*theta_prev*h) / xi_j
        theta[j] = -b[j]/a[j] if a[j] != 0 else 0.0

    return format_edo_solution(xi[::-1], a[::-1], b[::-1])


# === 4. Forme réduite avec a(xi) et theta(xi) ===
def simulate_edo_reduced_theta(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    phi_star = np.radians(phi_deg)
    xi_eval = np.linspace(xi0, Rext, N)
    a_R = 1e-6
    theta_R = theta0

    def reduced_system(xi, y):
        a, theta = y
        if np.abs(a) < 1e-10:
            return [0, 0]
        da = ((xi_star - k) + w0*np.sin(phi_star)*theta) / (xi * (1 - theta**2)) - 2*a/xi
        dtheta = (w0*np.sin(phi_star)/a - theta) * (1 + theta**2)
        return [da * a, dtheta]

    sol = solve_ivp(reduced_system, [Rext, xi0], [a_R, theta_R], t_eval=xi_eval[::-1], method="RK45")
    a_sol = sol.y[0][::-1]
    theta_sol = sol.y[1][::-1]
    b_sol = -theta_sol * a_sol
    return format_edo_solution(sol.t[::-1], a_sol, b_sol)
