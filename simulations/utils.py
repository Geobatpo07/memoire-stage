import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Décalage périodique ---
def shift(f, axis, direction):
    return np.roll(f, shift=direction, axis=axis)

# --- Schéma de convection Lax-Friedrichs 2D ---
def lax_friedrichs_2d(f, u, v, dx, dy):
    fx = 0.5 * (shift(f, 1, -1) - shift(f, 1, 1)) / (2 * dx)
    fy = 0.5 * (shift(f, 0, -1) - shift(f, 0, 1)) / (2 * dy)
    return u * fx + v * fy

# --- Norme de la vitesse ---
def compute_velocity_norm(u, v):
    return np.sqrt(u**2 + v**2)

# --- Centre du cyclone ---
def find_cyclone_center(norm_v, x, y):
    max_idx = np.unravel_index(np.argmax(norm_v), norm_v.shape)
    return float(x[max_idx]), float(y[max_idx])

# --- Sauvegarde des champs ---
def get_versioned_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    version = 0.0
    versioned_filename = os.path.join(directory, f"{base}_v{version:.1f}{ext}")
    while os.path.exists(versioned_filename):
        version += 0.1
        versioned_filename = os.path.join(directory, f"{base}_v{version:.1f}{ext}")
    return versioned_filename

def save_fields_edp(u, v, x, y, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    for name, data in [("u_field", u), ("v_field", v), ("x_grid", x), ("y_grid", y)]:
        path = get_versioned_filename(results_dir, f"{name}.npy")
        np.save(path, data)
        print(f"Champ sauvegardé : {path}")

# --- Chargement ---
def load_edp_fields():
    return (
        np.load("u_field.npy"),
        np.load("v_field.npy"),
        np.load("x_grid.npy"),
        np.load("y_grid.npy")
    )

# --- Profils radiaux ---
def compute_radial_profile(R, V, num_bins=100):
    r_bins = np.linspace(0, R.max(), num_bins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    profile = [np.mean(V[(R >= r_bins[i]) & (R < r_bins[i+1])]) if np.any((R >= r_bins[i]) & (R < r_bins[i+1])) else np.nan for i in range(len(r_bins) - 1)]
    return r_centers, np.array(profile)

# --- Tracés ---
def save_plot(x, y_list, labels, title, xlabel, ylabel, filename, styles=None, show=True, legend_loc="best", results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i, y in enumerate(y_list):
        style = styles[i] if styles and i < len(styles) else "-"
        plt.plot(x, y, style, label=labels[i], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.grid(True)
    plt.tight_layout()
    path = get_versioned_filename(results_dir, filename)
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Graphique sauvegardé : {path}")

def plot_quiver_field(X, Y, u, v, scale=50, skip=5, title="Champ de vent", savepath="results"):
    plt.figure(figsize=(8, 6))
    plt.quiver(X[::skip, ::skip]/1000, Y[::skip, ::skip]/1000, u[::skip, ::skip], v[::skip, ::skip], scale=scale)
    plt.title(title)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.grid(True)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Champ de vent sauvegardé : {savepath}")
        plt.close()
    else:
        plt.show()

# --- Séries et asymptotiques ---
def display_series_info(series_expr, delta):
    for i in range(4):
        coeff = series_expr.coeff(delta, i)
        if coeff != 0:
            print(f"  • Ordre {i} :", coeff)

def extract_order_equations(series_list, delta, orders=[0, 1, 2]):
    return [series.coeff(delta, n) for series in series_list for n in orders]

# --- Exports ---
def export_dataframe(df, filename):
    df.to_csv(filename, index=False)
    print(f"Données exportées : {filename}")

def export_numpy_dict(data_dict, prefix=""):
    for key, arr in data_dict.items():
        path = f"{prefix}{key}.npy"
        np.save(path, arr)
        print(f"Fichier sauvegardé : {path}")

# --- Formatage EDO ---
def format_edo_solution(xi_m, a_sol, b_sol):
    xi_km = xi_m / 1000
    theta = -b_sol / a_sol
    E = 0.5 * (a_sol**2 + b_sol**2)
    return {
        "xi_km": xi_km,
        "a_xi": a_sol,
        "b_xi": b_sol,
        "theta_xi": theta,
        "E_xi": E
    }

def save_edo_result(result_dict, filename):
    df = pd.DataFrame(result_dict)
    export_dataframe(df, filename)
