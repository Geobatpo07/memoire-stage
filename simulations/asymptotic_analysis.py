import sympy as sp
import sys
import argparse
from utils import display_series_info, extract_order_equations


def definir_symboles():
    """
    Définir les variables symboliques pour l’analyse asymptotique.

    Retourne :
        tuple : Variables symboliques (xi, xim, delta, alpha, beta, w0, phi, k, xi_star)
    """
    # Variables spatiales
    xi, xim = sp.symboles('xi xim', réel=True)
    delta = xi - xim  # Variable de développement

    # Coefficients de la série de Taylor
    alpha = sp.symboles('alpha1 alpha2 alpha3', réel=True)  # Pour a(xi)
    beta = sp.symboles('beta0 beta1 beta2', réel=True)      # Pour b(xi)

    # Paramètres physiques
    w0, phi = sp.symboles('w0 phi', réel=True, positif=True)  # Rotation terrestre et latitude
    k, xi_star = sp.symboles('k xi_star', réel=True, positif=True)  # Friction et absorption

    return xi, xim, delta, alpha, beta, w0, phi, k, xi_star


def definir_series_taylor(delta, alpha, beta):
    """
    Définir les développements de Taylor pour les composantes de vitesse.

    Arguments :
        delta (sympy.Symbol) : Variable de développement (xi - xim)
        alpha (liste) : Coefficients pour a(xi)
        beta (liste) : Coefficients pour b(xi)

    Retourne :
        tuple : (a, b) séries de Taylor pour les vitesses radiale et tangentielle
    """
    a = alpha[0]*delta + alpha[1]*delta**2 + alpha[2]*delta**3
    b = beta[0] + beta[1]*delta + beta[2]*delta**2
    return a, b


def definir_equations_edo(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star):
    """
    Définir les équations du système EDO sous forme symbolique.

    Arguments :
        xi (sympy.Symbol) : Coordonnée radiale
        a, b (sympy.Expr) : Expressions des vitesses radiale et tangentielle
        a_prime, b_prime (sympy.Expr) : Dérivées de a et b
        w0, phi, k, xi_star (sympy.Symbol) : Paramètres physiques

    Retourne :
        tuple : (eq1, eq2) équations développées
    """
    eq1 = a * a_prime * xi + a**2 - b**2 - ((xi_star - k)*a + w0*sp.sin(phi)*b)
    eq2 = a * b_prime * xi + 2*a*b - ((xi_star - k)*b - w0*sp.sin(phi)*a)

    return eq1.expand(), eq2.expand()


def analyse_asymptotique(n=4, save_to_file=None):
    """
    Effectue l’analyse asymptotique du système EDO de cyclone.

    Cette fonction développe le système en série de Taylor autour de xi = xim
    et extrait les équations aux ordres successifs.

    Arguments :
        n (int, optionnel) : Ordre maximal du développement. Par défaut 4.
        save_to_file (str, optionnel) : Nom de fichier pour sauvegarder. Par défaut None.

    Retourne :
        list : Équations aux ordres 0 à 2 pour les deux équations de quantité de mouvement.
    """
    try:
        original_stdout = sys.stdout
        if save_to_file:
            sys.stdout = open(save_to_file, 'w')

        print("\n Analyse asymptotique au voisinage de ξ = ξ_m\n")

        xi, xim, delta, alpha, beta, w0, phi, k, xi_star = definir_symboles()
        a, b = definir_series_taylor(delta, alpha, beta)

        a_prime = sp.diff(a, xi)
        b_prime = sp.diff(b, xi)

        eq1, eq2 = definir_equations_edo(xi, a, b, a_prime, b_prime, w0, phi, k, xi_star)

        eq1_series = eq1.series(delta, n=n).removeO()
        eq2_series = eq2.series(delta, n=n).removeO()

        print(" Équation 1 développée :")
        display_series_info(eq1_series, delta)

        print("\n Équation 2 développée :")
        display_series_info(eq2_series, delta)

        print("\n Équations aux ordres 0 à 2 :")
        eqs_0_2 = extract_order_equations([eq1_series, eq2_series], delta, orders=[0, 1, 2])
        for i, eq in enumerate(eqs_0_2):
            print(f"\n— Équation {i//3 + 1}, ordre {i%3} —")
            sp.pprint(sp.Eq(eq, 0))

        if save_to_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Résultats enregistrés dans {save_to_file}")

        return eqs_0_2

    except Exception as e:
        if save_to_file and sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"Erreur lors de l’analyse asymptotique : {e}", file=sys.stderr)
        return []


def main():
    """
    Fonction principale pour exécuter l’analyse depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(
        description="Analyse asymptotique du système EDO de cyclone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--order", type=int, default=4,
                       help="Ordre maximal du développement")
    parser.add_argument("--output", type=str, default=None,
                       help="Fichier de sortie pour les résultats")

    args = parser.parse_args()

    analyse_asymptotique(n=args.order, save_to_file=args.output)


if __name__ == "__main__":
    main()
