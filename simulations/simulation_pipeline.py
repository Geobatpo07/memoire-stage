import argparse
import time
from functools import wraps
from tqdm import tqdm

from edp_simulation import simulate_edp
from edo_simulation import main as run_edo_cli
from validation_edo_edp import validate_edo_edp
from edo_methods import simulate_edo_ivp, simulate_edo_ivp_bdf, simulate_edo_euler_backward, simulate_edo_reduced_theta

# Define methods dictionary similar to what was in edo_validation_cross
edo_methods = {
    "ivp": simulate_edo_ivp,
    "bdf": simulate_edo_ivp_bdf,
    "euler": simulate_edo_euler_backward,
    "theta": simulate_edo_reduced_theta,
}

# --- Décorateur pour mesurer la durée des étapes ---
def timed_step(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{name} — DÉBUT")
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"{name} — TERMINÉ en {duration:.2f} sec")
            return result
        return wrapper
    return decorator

@timed_step("Simulation EDP")
def run_edp():
    simulate_edp()

@timed_step("Simulation EDO (par défaut : solve_ivp)")
def run_edo():
    from edo_methods import simulate_edo_ivp
    simulate_edo_ivp()

@timed_step("Validation croisée EDO vs EDP")
def run_validation():
    validate_edo_edp(show_plot=True, edp_dir="results")

@timed_step("Validation croisée entre méthodes EDO")
def run_cross_validation():
    import edo_validation_cross

# --- Fonction principale orchestrant le pipeline ---
def main(args):
    selected = [args.edp, args.edo, args.validate, args.cross]
    if not any(selected):
        print("\n Pipeline complet lancé via tqdm...\n")
        for task, label in tqdm([
            (run_edp, "EDP"),
            (run_edo, "EDO"),
            (run_validation, "Validation EDO/EDP"),
            (run_cross_validation, "Cross-validation EDO")
        ], total=4, desc="Étapes du pipeline", ncols=80):
            task()
    else:
        if args.edp:
            run_edp()
        if args.edo:
            run_edo()
        if args.validate:
            run_validation()
        if args.cross:
            run_cross_validation()

# --- Interface ligne de commande ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de simulation de cyclone : EDP + EDO + validation.")
    parser.add_argument("--edp", action="store_true", help="Exécute uniquement la simulation EDP")
    parser.add_argument("--edo", action="store_true", help="Exécute uniquement la simulation EDO")
    parser.add_argument("--validate", action="store_true", help="Exécute uniquement la validation croisée EDO/EDP")
    parser.add_argument("--cross", action="store_true", help="Exécute la comparaison entre les méthodes EDO")
    args = parser.parse_args()
    main(args)
