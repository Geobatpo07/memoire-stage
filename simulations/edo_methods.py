import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from utils import export_dataframe, format_edo_solution, save_edo_result


def simulate_edo_ivp(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                     w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5, method="RK45"):
    """
    Simulate the cyclone ODE system using SciPy's solve_ivp.

    Args:
        N (int, optional): Number of evaluation points. Defaults to 1000.
        xi0 (float, optional): Inner radius (m). Defaults to 1e4.
        Rext (float, optional): Outer radius (m). Defaults to 3e5.
        theta0 (float, optional): Initial geostrophic factor. Defaults to 2e-4.
        w0 (float, optional): Earth's rotation rate (rad/s). Defaults to 7.2921e-5.
        phi_deg (float, optional): Latitude in degrees. Defaults to 20.
        k (float, optional): Friction coefficient. Defaults to 1e-5.
        xi_star (float, optional): Absorption coefficient. Defaults to 5e-5.
        method (str, optional): Integration method. Defaults to "RK45".

    Returns:
        dict: Dictionary containing the solution and derived quantities
    """
    # Convert latitude to radians
    phi_star = np.radians(phi_deg)

    # Create evaluation points (from inner to outer radius)
    xi_eval = np.linspace(xi0, Rext, N)

    # Initial conditions at the outer radius
    a_R = 1e-6  # Small initial radial velocity
    b_R = -theta0 * a_R  # Initial tangential velocity

    def system(xi, y):
        """
        Define the ODE system.

        Args:
            xi (float): Radial coordinate (independent variable)
            y (list): State vector [a, b]

        Returns:
            list: Derivatives [da/dxi, db/dxi]
        """
        a, b = y

        # Avoid division by zero
        if np.abs(a) < 1e-10:
            return [0, 0]

        # System equations
        da = ((xi_star - k)*a + w0*np.sin(phi_star)*b - a**2 + b**2) / (xi * a)
        db = ((xi_star - k)*b - w0*np.sin(phi_star)*a - 2*a*b) / (xi * a)

        return [da, db]

    try:
        # Solve the system (note: integration is from Rext to xi0, so we reverse the arrays)
        sol = solve_ivp(
            system, 
            [Rext, xi0],  # Integration bounds (from outer to inner)
            [a_R, b_R],   # Initial conditions at outer radius
            t_eval=xi_eval[::-1],  # Evaluation points (reversed)
            method=method
        )

        # Format the solution (reversing arrays to go from inner to outer)
        return format_edo_solution(sol.t[::-1], sol.y[0][::-1], sol.y[1][::-1])

    except Exception as e:
        print(f"Error in ODE integration ({method}): {e}")
        # Return empty arrays of correct size in case of error
        empty = np.zeros(N)
        return format_edo_solution(xi_eval, empty, empty)


def simulate_edo_ivp_bdf(**kwargs):
    """
    Simulate the cyclone ODE system using the BDF (implicit) method.

    Args:
        **kwargs: Arguments to pass to simulate_edo_ivp

    Returns:
        dict: Dictionary containing the solution and derived quantities
    """
    return simulate_edo_ivp(method="BDF", **kwargs)


def simulate_edo_euler_backward(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                                w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    """
    Simulate the cyclone ODE system using an explicit backward Euler method.

    This method integrates from the outer radius (Rext) to the inner radius (xi0)
    using a custom explicit scheme.

    Args:
        N (int, optional): Number of grid points. Defaults to 1000.
        xi0 (float, optional): Inner radius (m). Defaults to 1e4.
        Rext (float, optional): Outer radius (m). Defaults to 3e5.
        theta0 (float, optional): Initial geostrophic factor. Defaults to 2e-4.
        w0 (float, optional): Earth's rotation rate (rad/s). Defaults to 7.2921e-5.
        phi_deg (float, optional): Latitude in degrees. Defaults to 20.
        k (float, optional): Friction coefficient. Defaults to 1e-5.
        xi_star (float, optional): Absorption coefficient. Defaults to 5e-5.

    Returns:
        dict: Dictionary containing the solution and derived quantities
    """
    try:
        # Convert latitude to radians
        phi_star = np.radians(phi_deg)

        # Grid spacing
        h = (Rext - xi0) / N

        # Create grid points (from outer to inner radius)
        xi = np.linspace(Rext, xi0, N)

        # Initialize solution arrays
        a = np.zeros(N)
        b = np.zeros(N)
        theta = np.zeros(N)

        # Initial conditions at the outer radius
        a[0] = 1e-6  # Small initial radial velocity
        b[0] = -theta0 * a[0]  # Initial tangential velocity
        theta[0] = theta0  # Initial geostrophic factor

        # Backward Euler integration
        for j in range(1, N):
            xi_j = xi[j]
            a_prev, b_prev = a[j-1], b[j-1]
            theta_prev = theta[j-1]

            # Update tangential velocity component
            b[j] = (xi[j-1]*b_prev - w0*np.sin(phi_star)*h + b_prev*h + 
                   (xi_star - k)*theta_prev*h) / xi_j

            # Update radial velocity component
            a[j] = (xi[j-1]*a_prev - theta_prev**2 * a_prev*h - 
                   (xi_star - k)*h + w0*np.sin(phi_star)*theta_prev*h) / xi_j

            # Update geostrophic factor with division by zero check
            theta[j] = -b[j]/a[j] if abs(a[j]) > 1e-10 else 0.0

        # Format the solution (reversing arrays to go from inner to outer)
        return format_edo_solution(xi[::-1], a[::-1], b[::-1])

    except Exception as e:
        print(f"Error in backward Euler integration: {e}")
        # Return empty arrays of correct size in case of error
        empty = np.zeros(N)
        return format_edo_solution(np.linspace(xi0, Rext, N), empty, empty)


def simulate_edo_reduced_theta(N=1000, xi0=1e4, Rext=3e5, theta0=2e-4,
                              w0=7.2921e-5, phi_deg=20, k=1e-5, xi_star=5e-5):
    """
    Simulate the cyclone ODE system using a reduced form with a(xi) and theta(xi).

    This method uses a different formulation of the equations in terms of
    the radial velocity a and the geostrophic factor theta.

    Args:
        N (int, optional): Number of evaluation points. Defaults to 1000.
        xi0 (float, optional): Inner radius (m). Defaults to 1e4.
        Rext (float, optional): Outer radius (m). Defaults to 3e5.
        theta0 (float, optional): Initial geostrophic factor. Defaults to 2e-4.
        w0 (float, optional): Earth's rotation rate (rad/s). Defaults to 7.2921e-5.
        phi_deg (float, optional): Latitude in degrees. Defaults to 20.
        k (float, optional): Friction coefficient. Defaults to 1e-5.
        xi_star (float, optional): Absorption coefficient. Defaults to 5e-5.

    Returns:
        dict: Dictionary containing the solution and derived quantities
    """
    try:
        # Convert latitude to radians
        phi_star = np.radians(phi_deg)

        # Create evaluation points (from inner to outer radius)
        xi_eval = np.linspace(xi0, Rext, N)

        # Initial conditions at the outer radius
        a_R = 1e-6  # Small initial radial velocity
        theta_R = theta0  # Initial geostrophic factor

        def reduced_system(xi, y):
            """
            Define the reduced ODE system in terms of a and theta.

            Args:
                xi (float): Radial coordinate (independent variable)
                y (list): State vector [a, theta]

            Returns:
                list: Derivatives [d(a)/dxi, d(theta)/dxi]
            """
            a, theta = y

            # Avoid division by zero and numerical instabilities
            if np.abs(a) < 1e-10 or abs(theta**2 - 1) < 1e-10:
                return [0, 0]

            # System equations
            da = ((xi_star - k) + w0*np.sin(phi_star)*theta) / (xi * (1 - theta**2)) - 2*a/xi
            dtheta = (w0*np.sin(phi_star)/a - theta) * (1 + theta**2)

            return [da * a, dtheta]

        # Solve the system (note: integration is from Rext to xi0, so we reverse the arrays)
        sol = solve_ivp(
            reduced_system, 
            [Rext, xi0],  # Integration bounds (from outer to inner)
            [a_R, theta_R],  # Initial conditions at outer radius
            t_eval=xi_eval[::-1],  # Evaluation points (reversed)
            method="RK45"
        )

        # Extract and reverse solution components
        a_sol = sol.y[0][::-1]
        theta_sol = sol.y[1][::-1]

        # Calculate tangential velocity from geostrophic factor
        b_sol = -theta_sol * a_sol

        # Format the solution
        return format_edo_solution(sol.t[::-1], a_sol, b_sol)

    except Exception as e:
        print(f"Error in reduced theta integration: {e}")
        # Return empty arrays of correct size in case of error
        empty = np.zeros(N)
        return format_edo_solution(xi_eval, empty, empty)
