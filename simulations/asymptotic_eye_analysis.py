import sympy as sp
import sys
import argparse
import os

def define_symbols():
    xi, xim = sp.symbols('xi xim', real=True)
    delta = sp.Symbol('delta', real=True)

    A, alpha = sp.symbols('A alpha', positive=True, real=True)
    b0, C = sp.symbols('b0 C', real=True)

    w0, phi, k, xi_star = sp.symbols('w0 phi k xi_star', real=True, positive=True)
    return xi, xim, delta, A, alpha, b0, C, w0, phi, k, xi_star

def define_asymptotic_profiles(delta, A, alpha, b0, C):
    a = -A * delta**alpha
    a_prime = sp.diff(a, delta)

    b = b0 * sp.exp(-C * delta**(1 - alpha))
    b_prime = sp.diff(b, delta)

    return a, a_prime, b, b_prime

def edo_equations(xi, delta, a, a_prime, b, b_prime, w0, phi, k, xi_star):
    eq1 = a * a_prime * xi + a**2 - b**2 - ((xi_star - k)*a + w0*sp.sin(phi)*b)
    eq2 = a * b_prime * xi + 2*a*b - ((xi_star - k)*b - w0*sp.sin(phi)*a)
    return eq1.simplify(), eq2.simplify()

def save_to_latex(a, b, eq1, eq2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tex_file = os.path.join(output_dir, "asymptotic_equations.tex")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("% Fichier généré automatiquement : Profils asymptotiques\n")
        f.write("\\section*{Analyse asymptotique au voisinage de l’œil (approche CSIM)}\n")
        f.write("Les profils asymptotiques considérés sont :\n")
        f.write("\\begin{align*}\n")
        f.write("a(\\xi) &= %s \\\\\n" % sp.latex(a))
        f.write("b(\\xi) &= %s\n" % sp.latex(b))
        f.write("\\end{align*}\n\n")
        f.write("Les équations différentielles approchées associées à ces profils sont :\n")
        f.write("\\begin{align*}\n")
        f.write("0 &= %s \\\\ \n" % sp.latex(eq1))
        f.write("0 &= %s \n" % sp.latex(eq2))
        f.write("\\end{align*}\n")

    print(f"\nÉquations enregistrées dans {tex_file}")

def analyze(save_to_file=None, latex_output_dir=None):
    xi, xim, delta, A, alpha, b0, C, w0, phi, k, xi_star = define_symbols()
    a, a_prime, b, b_prime = define_asymptotic_profiles(delta, A, alpha, b0, C)
    eq1, eq2 = edo_equations(xi, delta, a, a_prime, b, b_prime, w0, phi, k, xi_star)

    original_stdout = sys.stdout
    if save_to_file:
        sys.stdout = open(save_to_file, 'w', encoding="utf-8")

    print("\n--- Analyse asymptotique (modèle Hasler/CSIM 2019) ---\n")
    print("Profil asymptotique radial : a(xi)")
    sp.pprint(a)
    print("\nProfil asymptotique tangentielle : b(xi)")
    sp.pprint(b)

    print("\nÉquation différentielle radiale :")
    sp.pprint(sp.Eq(eq1, 0))

    print("\nÉquation différentielle tangentielle :")
    sp.pprint(sp.Eq(eq2, 0))

    if save_to_file:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"\nRésultats enregistrés dans {save_to_file}")

    save_to_latex(a, b, eq1, eq2, latex_output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Analyse asymptotique au voisinage de l’œil (approche CSIM 2019)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=str, default=None, help="Fichier de sortie console")
    parser.add_argument("--latex-dir", type=str, default="../sections", help="Répertoire pour le fichier LaTeX")
    args = parser.parse_args()

    analyze(save_to_file=args.output, latex_output_dir=args.latex_dir)

if __name__ == "__main__":
    main()
