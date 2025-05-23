import sympy as sp
from utils import display_series_info, extract_order_equations

# === Définition des symboles ===
def define_symbols():
    xi, xim = sp.symbols('xi xim', real=True)
    delta = xi - xim
    epsilon = sp.Symbol('\varepsilon', positive=True)

    alpha = sp.symbols('alpha1 alpha2 alpha3', real=True)
    beta = sp.symbols('beta0 beta1 beta2', real=True)
    w0, phi, k, xi_star = sp.symbols('w0 phi k xi_star', real=True, positive=True)

    return xi, xim, delta, epsilon, alpha, beta, w0, phi, k, xi_star

# === Définir les séries de Taylor pour a et b ===
def define_series(delta, alpha, beta):
    a = alpha[0]*delta + alpha[1]*delta**2 + alpha[2]*delta**3
    b = beta[0] + beta[1]*delta + beta[2]*delta**2
    return a, b

# === Équations asymptotiques EDO ===
def edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star):
    eq1 = a * a_prime * xi + a**2 - b**2 - ((xi_star - k)*a + w0*sp.sin(phi)*b)
    eq2 = a * b_prime * xi + 2*a*b - ((xi_star - k)*b - w0*sp.sin(phi)*a)
    return eq1.expand(), eq2.expand()

# === Analyse complète ===
def asymptotic_expansion(n=4):
    print("\n Analyse asymptotique au voisinage du mur de l'œil\n")

    xi, xim, delta, epsilon, alpha, beta, w0, phi, k, xi_star = define_symbols()
    a, b = define_series(delta, alpha, beta)

    a_prime = sp.diff(a, xi)
    b_prime = sp.diff(b, xi)
    eq1, eq2 = edo_equations(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star)

    eq1_series = eq1.series(delta, n=n).removeO()
    eq2_series = eq2.series(delta, n=n).removeO()

    print("\nÉquation 1 développée :")
    display_series_info(eq1_series, delta)

    print("\nÉquation 2 développée :")
    display_series_info(eq2_series, delta)

    print("\n\U0001f9ea Extraction des termes dominants :")
    orders = extract_order_equations([eq1_series, eq2_series], delta, orders=[0, 1, 2])
    for i, eq in enumerate(orders):
        print(f"\n— Équation {i//3 + 1}, ordre {i%3} —")
        sp.pprint(sp.Eq(eq, 0))

    return orders

if __name__ == "__main__":
    asymptotic_expansion()
