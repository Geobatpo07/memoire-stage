import matplotlib.pyplot as plt
import pandas as pd
from edo_methods import (
    simulate_edo_ivp,
    simulate_edo_ivp_bdf,
    simulate_edo_euler_backward,
    simulate_edo_reduced_theta
)
from utils import save_plot, export_dataframe

# === Chargement des résultats pour toutes les méthodes ===
print("\n Validation croisée des méthodes de résolution EDO")

methods = {
    "ivp": simulate_edo_ivp(),
    "bdf": simulate_edo_ivp_bdf(),
    "euler": simulate_edo_euler_backward(),
    "theta": simulate_edo_reduced_theta(),
}

# === Tracé superposé des composantes ===
xi = methods["ivp"]["xi_km"]  # Grille commune

save_plot(
    xi,
    [methods[m]["a_xi"] for m in methods],
    labels=[f"a(ξ) - {m}" for m in methods],
    title="Comparaison a(ξ) selon la méthode",
    xlabel="ξ (km)", ylabel="a(ξ)",
    filename="compare_a_xi.pdf"
)

save_plot(
    xi,
    [methods[m]["b_xi"] for m in methods],
    labels=[f"b(ξ) - {m}" for m in methods],
    title="Comparaison b(ξ) selon la méthode",
    xlabel="ξ (km)", ylabel="b(ξ)",
    filename="compare_b_xi.pdf"
)

save_plot(
    xi,
    [methods[m]["theta_xi"] for m in methods],
    labels=[f"θ(ξ) - {m}" for m in methods],
    title="Comparaison θ(ξ) selon la méthode",
    xlabel="ξ (km)", ylabel="θ(ξ)",
    filename="compare_theta.pdf"
)

save_plot(
    xi,
    [methods[m]["E_xi"] for m in methods],
    labels=[f"E(ξ) - {m}" for m in methods],
    title="Comparaison de l'énergie cinétique E(ξ)",
    xlabel="ξ (km)", ylabel="E(ξ)",
    filename="compare_energy.pdf"
)

# === Export d'un tableau de comparaison ===
print("\n Génération du tableau comparatif...")
df = pd.DataFrame({
    "xi_km": xi,
    **{f"a_{m}": methods[m]["a_xi"] for m in methods},
    **{f"b_{m}": methods[m]["b_xi"] for m in methods},
    **{f"theta_{m}": methods[m]["theta_xi"] for m in methods},
    **{f"E_{m}": methods[m]["E_xi"] for m in methods},
})

# === Calcul des écarts relatifs par rapport à la méthode ivp ===
ref = "ivp"
deltas = {}
for m in methods:
    if m != ref:
        df[f"delta_a_{m}"] = df[f"a_{m}"] - df[f"a_{ref}"]
        df[f"delta_b_{m}"] = df[f"b_{m}"] - df[f"b_{ref}"]
        df[f"delta_theta_{m}"] = df[f"theta_{m}"] - df[f"theta_{ref}"]
        df[f"delta_E_{m}"] = df[f"E_{m}"] - df[f"E_{ref}"]
        deltas[m] = {
            "a": df[f"delta_a_{m}"],
            "b": df[f"delta_b_{m}"],
            "theta": df[f"delta_theta_{m}"],
            "E": df[f"delta_E_{m}"]
        }

# === Tracer les écarts ===
for m in deltas:
    save_plot(
        xi,
        [deltas[m]["a"]],
        labels=[f"Δa(ξ) = {m} - {ref}"],
        title=f"Écart sur a(ξ) : {m} - {ref}",
        xlabel="ξ (km)", ylabel="Δa(ξ)",
        filename=f"delta_a_{m}.pdf"
    )
    save_plot(
        xi,
        [deltas[m]["b"]],
        labels=[f"Δb(ξ) = {m} - {ref}"],
        title=f"Écart sur b(ξ) : {m} - {ref}",
        xlabel="ξ (km)", ylabel="Δb(ξ)",
        filename=f"delta_b_{m}.pdf"
    )
    save_plot(
        xi,
        [deltas[m]["theta"]],
        labels=[f"Δθ(ξ) = {m} - {ref}"],
        title=f"Écart sur θ(ξ) : {m} - {ref}",
        xlabel="ξ (km)", ylabel="Δθ(ξ)",
        filename=f"delta_theta_{m}.pdf"
    )
    save_plot(
        xi,
        [deltas[m]["E"]],
        labels=[f"ΔE(ξ) = {m} - {ref}"],
        title=f"Écart sur E(ξ) : {m} - {ref}",
        xlabel="ξ (km)", ylabel="ΔE(ξ)",
        filename=f"delta_E_{m}.pdf"
    )

export_dataframe(df, "validation_cross_table.csv")

print("\n Comparaison enregistrée : compare_*.pdf")
print(" Écarts enregistrés : delta_*.pdf")
print(" Tableau exporté : validation_cross_table.csv")
