import sympy as sp
from utils import display_series_info, extract_order_equations

def define_symbols():
    xi, xim = sp.symbols('xi xim', real=True)
    delta = xi - xim

    # Coefficients de Taylor
    alpha = sp.symbols('alpha1 alpha2 alpha3', real=True)
    beta = sp.symbols('beta0 beta1 beta2', real=True)

    # Paramètres physiques
    w0, phi, k, xi_star = sp.symbols('w0 phi k xi_star', real=True, positive=True)

    return xi, xim, delta, alpha, beta, w0, phi, k, xi_star

def define_taylor_series(delta, alpha, beta):
    a = alpha[0]*delta + alpha[1]*delta**2 + alpha[2]*delta**3
    b = beta[0] + beta[1]*delta + beta[2]*delta**2
    return a, b

def define_edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star):
    eq1 = a * a_prime * xi + a**2 - b**2 - ((xi_star - k)*a + w0*sp.sin(phi)*b)
    eq2 = a * b_prime * xi + 2*a*b - ((xi_star - k)*b - w0*sp.sin(phi)*a)
    return eq1.expand(), eq2.expand()

def asymptotic_analysis(n=4):
    print("\n Analyse asymptotique au voisinage de ξ = ξ_m\n")

    # Déclarations
    xi, xim, delta, alpha, beta, w0, phi, k, xi_star = define_symbols()
    a, b = define_taylor_series(delta, alpha, beta)

    a_prime = sp.diff(a, xi)
    b_prime = sp.diff(b, xi)

    eq1, eq2 = define_edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star)

    # Développement en série en delta
    eq1_series = eq1.series(delta, n=n).removeO()
    eq2_series = eq2.series(delta, n=n).removeO()

    # Affichage et extraction
    print(" Équation 1 développée :")
    display_series_info(eq1_series, delta)

    print("\n Équation 2 développée :")
    display_series_info(eq2_series, delta)

    print("\n Équations aux ordres 0 à 2 :")
    eqs_0_2 = extract_order_equations([eq1_series, eq2_series], delta, orders=[0, 1, 2])
    for i, eq in enumerate(eqs_0_2):
        print(f"\n— Équation {i//3 + 1}, ordre {i%3} —")
        sp.pprint(sp.Eq(eq, 0))

    return eqs_0_2

if __name__ == "__main__":
    asymptotic_analysis()
