import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd


# --- Décalage périodique d’un tableau (utile pour schéma de convection) ---
def shift(f, axis, direction):
    return np.roll(f, shift=direction, axis=axis)

# --- Schéma de convection Lax-Friedrichs 2D ---
def lax_friedrichs_2d(f, u, v, dx, dy):
    """
    Implement Lax-Friedrichs 2D scheme with input validation and debugging.
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [f, u, v]):
        raise TypeError("f, u, and v must be numpy arrays")
    
    if not (f.shape == u.shape == v.shape):
        raise ValueError(f"Shape mismatch: f:{f.shape}, u:{u.shape}, v:{v.shape}")
        
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid grid spacing: dx={dx}, dy={dy}")
    
    try:
        fx = 0.5 * (shift(f, 1, -1) - shift(f, 1, 1)) / (2 * dx)
        fy = 0.5 * (shift(f, 0, -1) - shift(f, 0, 1)) / (2 * dy)
        result = u * fx + v * fy
        
        # Check for numerical instabilities
        if not np.all(np.isfinite(result)):
            print("Warning: Non-finite values detected in Lax-Friedrichs scheme")
            
        return result
    except Exception as e:
        print(f"Error in Lax-Friedrichs calculation: {str(e)}")
        raise

def compute_velocity_norm(u, v):
    """
    Compute velocity norm with validation.
    """
    if not (isinstance(u, np.ndarray) and isinstance(v, np.ndarray)):
        raise TypeError("u and v must be numpy arrays")
        
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u:{u.shape}, v:{v.shape}")
        
    try:
        norm = np.sqrt(u**2 + v**2)
        
        # Check for NaN or inf values
        if not np.all(np.isfinite(norm)):
            print("Warning: Non-finite values detected in velocity norm")
            
        return norm
    except Exception as e:
        print(f"Error computing velocity norm: {str(e)}")
        raise

def find_cyclone_center(norm_v, x, y):
    """
    Find cyclone center with validation and debugging.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [norm_v, x, y]):
        raise TypeError("All inputs must be numpy arrays")
    
    try:
        max_idx = np.unravel_index(np.argmax(norm_v), norm_v.shape)
        # For meshgrid coordinates, we need to index both dimensions
        center_x = float(x[max_idx[0], max_idx[1]])
        center_y = float(y[max_idx[0], max_idx[1]])
        
        print(f"Found cyclone center at: ({center_x:.2f}, {center_y:.2f})")
        print(f"Maximum velocity norm: {float(norm_v[max_idx]):.2f}")
        
        return center_x, center_y
    except Exception as e:
        print(f"Error finding cyclone center: {str(e)}")
        raise

# --- Sauvegarde des résultats EDP (u, v, x, y) ---
# --- Save fields with versioning ---
def save_fields(u, v, x, y, results_dir="results"):
    """
    Save fields with validation, error handling, and versioning.
    """
    try:
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Input validation
        if not all(isinstance(arr, np.ndarray) for arr in [u, v, x, y]):
            raise TypeError("All inputs must be numpy arrays")

        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch between u {u.shape} and v {v.shape}")

        # Save with versioning
        for name, data in [("u_field", u), ("v_field", v),
                           ("x_grid", x), ("y_grid", y)]:
            try:
                filename = get_versioned_filename(results_dir, f"{name}.npy")
                np.save(filename, data)
                print(f"Successfully saved {filename}")
            except Exception as e:
                raise ValueError(f"Error saving {name}.npy: {str(e)}")

    except Exception as e:
        raise ValueError(f"Error in save_fields: {str(e)}")

# --- Sauvegarde des graphiques ---
def save_plot(x, y_list, labels, title, xlabel, ylabel, filename, styles=None, show=True, legend_loc="best", results_dir="results"):
    """
    Génère et sauvegarde un graphique simple à plusieurs courbes.

    Args:
        x (array): abscisses communes
        y_list (list of arrays): liste des courbes y
        labels (list of str): noms des courbes
        title (str): titre du graphique
        xlabel (str): nom de l’axe x
        ylabel (str): nom de l’axe y
        filename (str): nom du fichier à sauvegarder (avec extension)
        styles (list of str): styles matplotlib optionnels pour chaque courbe
        show (bool): True = affiche ; False = ne fait que sauvegarder
        legend_loc (str): position de la légende (default = "best")
        results_dir (str): répertoire de sauvegarde
    """
    try:
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Generate the plot
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

        # Save with versioning
        versioned_filename = get_versioned_filename(results_dir, filename)
        plt.savefig(versioned_filename)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Graph saved: {versioned_filename}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        raise

def load_edp_fields():
    u = np.load("u_field.npy")
    v = np.load("v_field.npy")
    x = np.load("x_grid.npy")
    y = np.load("y_grid.npy")
    return u, v, x, y

# --- Construction d’un profil radial moyen à partir d’un champ 2D ---
def compute_radial_profile(R, V, num_bins=100):
    r_bins = np.linspace(0, R.max(), num_bins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    v_profile = []

    for i in range(len(r_bins) - 1):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        values = V[mask]
        v_profile.append(np.mean(values) if values.size > 0 else np.nan)

    return r_centers, np.array(v_profile)

def export_dataframe(df, filename):
    df.to_csv(filename, index=False)
    print(f"Données exportées : {filename}")

def export_numpy_dict(data_dict, prefix=""):
    for key, arr in data_dict.items():
        path = f"{prefix}{key}.npy"
        np.save(path, arr)
        print(f"Fichier sauvegardé : {path}")

def plot_quiver_field(X, Y, u, v, scale=50, skip=5, title="Champ de vent", savepath="results"):
    plt.figure(figsize=(8, 6))
    plt.quiver(X[::skip, ::skip]/1000, Y[::skip, ::skip]/1000,
               u[::skip, ::skip], v[::skip, ::skip], scale=scale)
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

def display_series_info(series_expr, delta):
    """
    Affiche les coefficients ordonnés d'une série développée en delta.
    """
    for i in range(4):
        coeff = series_expr.coeff(delta, i)
        if coeff != 0:
            print(f"  • Ordre {i} : ", coeff)

def extract_order_equations(series_list, delta, orders=[0, 1, 2]):
    """
    Extrait les équations aux ordres spécifiés pour un ensemble de séries.
    """
    equations = []
    for series in series_list:
        for n in orders:
            equations.append(series.coeff(delta, n))
    return equations

# --- Helper function for versioning ---
def get_versioned_filename(directory, filename):
    """
    Generate a versioned filename if the file already exists.
    """
    base, ext = os.path.splitext(filename)
    version = 0.0
    versioned_filename = os.path.join(directory, f"{base}_v{version}{ext}")
    while os.path.exists(versioned_filename):
        version += 0.1
        versioned_filename = os.path.join(directory, f"{base}_v{version}{ext}")
    return versioned_filename

# --- Save fields with versioning ---
def save_fields_edp(u, v, x, y, results_dir="results"):
    """
    Save fields with validation, error handling, and versioning.
    """
    try:
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Input validation
        if not all(isinstance(arr, np.ndarray) for arr in [u, v, x, y]):
            raise TypeError("All inputs must be numpy arrays")

        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch between u {u.shape} and v {v.shape}")

        # Save with versioning
        for name, data in [("u_field", u), ("v_field", v),
                           ("x_grid", x), ("y_grid", y)]:
            try:
                filename = get_versioned_filename(results_dir, f"{name}.npy")
                np.save(filename, data)
                print(f"Successfully saved {filename}")
            except Exception as e:
                raise ValueError(f"Error saving {name}.npy: {str(e)}")

    except Exception as e:
        raise ValueError(f"Error in save_fields_edp: {str(e)}")

# --- Save plot with versioning ---
def save_plot(x, y_list, labels, title, xlabel, ylabel, filename, styles=None, show=True, legend_loc="best", results_dir="results"):
    """
    Generate and save a plot with versioning.
    """
    try:
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Generate the plot
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

        # Save with versioning
        versioned_filename = get_versioned_filename(results_dir, filename)
        plt.savefig(versioned_filename)
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Graph saved: {versioned_filename}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        raise

# === Formatage commun des résultats ===
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


# === Export CSV helper ===
def save_edo_result(result_dict, filename):
    df = pd.DataFrame(result_dict)
    export_dataframe(df, filename)