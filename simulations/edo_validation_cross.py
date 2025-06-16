import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import argparse
import sys
import time
import os
from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward
)
from utils import save_plot, export_dataframe


def run_all_methods(show_progress=True, **kwargs):
    methods_to_run = {
        "ivp": simulate_edo_ivp,
        "bdf": simulate_edo_ivp_bdf,
        "euler": simulate_edo_euler_backward
    }

    results = {}
    for name, method in methods_to_run.items():
        if show_progress:
            print(f"Exécution de la méthode : {name}...")
            start_time = time.time()
        results[name] = method(**kwargs)
        if show_progress:
            print(f"  Terminé en {time.time() - start_time:.2f} secondes")
    return results


def interpolate_methods(methods_results_raw, xi_common):
    interpolated = {}
    for name, res in methods_results_raw.items():
        try:
            xi_raw = res["xi_km"]
            interpolated[name] = {
                "xi_km": xi_common,
                "a_xi": interp1d(xi_raw, res["a_xi"], bounds_error=False, fill_value="extrapolate")(xi_common),
                "b_xi": interp1d(xi_raw, res["b_xi"], bounds_error=False, fill_value="extrapolate")(xi_common),
                "theta_xi": interp1d(xi_raw, res["theta_xi"], bounds_error=False, fill_value="extrapolate")(xi_common),
                "E_xi": interp1d(xi_raw, res["E_xi"], bounds_error=False, fill_value="extrapolate")(xi_common),
            }
        except Exception as e:
            print(f"Erreur d'interpolation pour {name} : {e}")
    return interpolated


def cross_validate_methods(output_csv="validation_cross_table_interp.csv",
                          output_dir="results",
                          reference="ivp",
                          show_progress=True,
                          **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    print("\nValidation croisée des méthodes de résolution des EDO")
    methods_results_raw = run_all_methods(show_progress=show_progress, **kwargs)
    xi_common = np.linspace(10, 300, 720)
    methods_results = interpolate_methods(methods_results_raw, xi_common)

    # Export CSV
    df_data = {"xi_km": xi_common}
    for m in methods_results:
        df_data[f"a_{m}"] = methods_results[m]["a_xi"]
        df_data[f"b_{m}"] = methods_results[m]["b_xi"]
        df_data[f"theta_{m}"] = methods_results[m]["theta_xi"]
        df_data[f"E_{m}"] = methods_results[m]["E_xi"]
    df = pd.DataFrame(df_data)
    export_dataframe(df, os.path.join(output_dir, output_csv))

    # Graphiques comparatifs
    for metric, label, ylabel in [
        ("a_xi", "a(ξ)", "a(ξ)"),
        ("b_xi", "b(ξ)", "b(ξ)"),
        ("theta_xi", "θ(ξ)", "θ(ξ)"),
        ("E_xi", "E(ξ)", "E(ξ)")
    ]:
        save_plot(
            xi_common,
            [methods_results[m][metric] for m in methods_results],
            labels=[f"{label} - {m}" for m in methods_results],
            title=f"Comparaison de {label} selon les méthodes",
            xlabel="ξ (km)", ylabel=ylabel,
            filename=f"compare_{metric}.png",
            results_dir=output_dir
        )

    print(f"\nValidation croisée complétée avec succès.")
    print(f"  - CSV exporté : {os.path.join(output_dir, output_csv)}")
    print(f"  - Graphiques dans {output_dir}/compare_*.png")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validation croisée des méthodes de résolution d'EDO pour les cyclones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--output-csv", type=str, default="validation_cross_table_interp.csv")
    parser.add_argument("--output-dir", type=str, default="../results")
    parser.add_argument("--reference", type=str, default="ivp", choices=["ivp", "bdf", "euler"])
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--params", type=str)

    args = parser.parse_args()
    extra_params = {}

    if args.params:
        for param in args.params.split(','):
            key, value = param.split('=')
            try:
                extra_params[key] = int(value)
            except ValueError:
                try:
                    extra_params[key] = float(value)
                except ValueError:
                    extra_params[key] = value

    success = cross_validate_methods(
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        reference=args.reference,
        show_progress=not args.no_progress,
        **extra_params
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
