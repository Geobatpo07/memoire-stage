import sympy as sp
import sys
import argparse
from utils import display_series_info, extract_order_equations


def define_symbols():
    """
    Define symbolic variables for the eye wall asymptotic analysis.

    Returns:
        tuple: Symbolic variables (xi, xim, delta, epsilon, alpha, beta, w0, phi, k, xi_star)
    """
    # Spatial variables
    xi, xim = sp.symbols('xi xim', real=True)
    delta = xi - xim  # Expansion variable
    epsilon = sp.Symbol('\varepsilon', positive=True)  # Small parameter

    # Taylor series coefficients
    alpha = sp.symbols('alpha1 alpha2 alpha3', real=True)  # For a(xi)
    beta = sp.symbols('beta0 beta1 beta2', real=True)      # For b(xi)

    # Physical parameters
    w0, phi = sp.symbols('w0 phi', real=True, positive=True)  # Earth rotation and latitude
    k, xi_star = sp.symbols('k xi_star', real=True, positive=True)  # Friction and absorption

    return xi, xim, delta, epsilon, alpha, beta, w0, phi, k, xi_star


def define_series(delta, alpha, beta):
    """
    Define Taylor series expansions for velocity components near the eye wall.

    Args:
        delta (sympy.Symbol): Expansion variable (xi - xim)
        alpha (list): Coefficients for radial velocity expansion
        beta (list): Coefficients for tangential velocity expansion

    Returns:
        tuple: (a, b) Taylor series for radial and tangential velocities
    """
    # Radial velocity a(xi) = alpha1*delta + alpha2*delta^2 + alpha3*delta^3
    a = alpha[0]*delta + alpha[1]*delta**2 + alpha[2]*delta**3

    # Tangential velocity b(xi) = beta0 + beta1*delta + beta2*delta^2
    b = beta[0] + beta[1]*delta + beta[2]*delta**2

    return a, b


def edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star):
    """
    Define the ODE system equations in symbolic form for eye wall analysis.

    Args:
        xi (sympy.Symbol): Radial coordinate
        a, b (sympy.Expr): Radial and tangential velocity expressions
        a_prime, b_prime (sympy.Expr): Derivatives of a and b
        w0, phi, k, xi_star (sympy.Symbol): Physical parameters

    Returns:
        tuple: (eq1, eq2) Expanded equations
    """
    # First equation (radial momentum)
    eq1 = a * a_prime * xi + a**2 - b**2 - ((xi_star - k)*a + w0*sp.sin(phi)*b)

    # Second equation (tangential momentum)
    eq2 = a * b_prime * xi + 2*a*b - ((xi_star - k)*b - w0*sp.sin(phi)*a)

    # Expand the equations to collect terms
    return eq1.expand(), eq2.expand()


def asymptotic_expansion(n=4, save_to_file=None):
    """
    Perform asymptotic analysis of the cyclone ODE system near the eye wall.

    This function expands the ODE system in a Taylor series around xi = xim
    (the eye wall radius) and extracts equations at different orders.

    Args:
        n (int, optional): Maximum order of expansion. Defaults to 4.
        save_to_file (str, optional): File to save results. Defaults to None.

    Returns:
        list: Equations at orders 0 to 2 for both momentum equations
    """
    try:
        # Redirect output to file if specified
        original_stdout = sys.stdout
        if save_to_file:
            sys.stdout = open(save_to_file, 'w')

        print("\n Asymptotic analysis near the eye wall (ξ = ξ_m)\n")

        # Define symbols and series
        xi, xim, delta, epsilon, alpha, beta, w0, phi, k, xi_star = define_symbols()
        a, b = define_series(delta, alpha, beta)

        # Calculate derivatives
        a_prime = sp.diff(a, xi)
        b_prime = sp.diff(b, xi)

        # Define and expand equations
        eq1, eq2 = edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star)

        # Series expansion in delta
        eq1_series = eq1.series(delta, n=n).removeO()
        eq2_series = eq2.series(delta, n=n).removeO()

        # Display results
        print("\nEquation 1 expanded:")
        display_series_info(eq1_series, delta)

        print("\nEquation 2 expanded:")
        display_series_info(eq2_series, delta)

        print("\n🧪 Extraction of dominant terms:")
        orders = extract_order_equations([eq1_series, eq2_series], delta, orders=[0, 1, 2])
        for i, eq in enumerate(orders):
            print(f"\n— Equation {i//3 + 1}, order {i%3} —")
            sp.pprint(sp.Eq(eq, 0))

        # Restore stdout if needed
        if save_to_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Eye wall analysis results saved to {save_to_file}")

        return orders

    except Exception as e:
        if save_to_file and sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"Error in eye wall asymptotic analysis: {e}", file=sys.stderr)
        return []


def main():
    """
    Main function for the eye wall asymptotic analysis command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Asymptotic analysis of cyclone ODE system near the eye wall",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--order", type=int, default=4,
                       help="Maximum order of expansion")
    parser.add_argument("--output", type=str, default=None,
                       help="File to save results")

    args = parser.parse_args()

    asymptotic_expansion(n=args.order, save_to_file=args.output)


if __name__ == "__main__":
    main()
