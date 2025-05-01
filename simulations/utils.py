import numpy as np
import matplotlib.pyplot as plt

# --- Décalage périodique d’un tableau (utile pour schéma de convection) ---
def shift(f, axis, direction):
    return np.roll(f, shift=direction, axis=axis)

# --- Schéma de convection Lax-Friedrichs 2D ---
def lax_friedrichs_2d(f, u, v, dx, dy):
    fx = 0.5 * (shift(f, 1, -1) - shift(f, 1, 1)) / (2 * dx)
    fy = 0.5 * (shift(f, 0, -1) - shift(f, 0, 1)) / (2 * dy)
    return u * fx + v * fy

def compute_velocity_norm(u, v):
    return np.sqrt(u**2 + v**2)

def find_cyclone_center(norm_v, x, y):
    max_idx = np.unravel_index(np.argmax(norm_v), norm_v.shape)
    return x[max_idx[1]], y[max_idx[0]]

# --- Sauvegarde des résultats EDP (u, v, x, y) ---
def save_fields_edp(u, v, x, y):
    np.save("u_field.npy", u)
    np.save("v_field.npy", v)
    np.save("x_grid.npy", x)
    np.save("y_grid.npy", y)
    print("Champs EDP sauvegardés : u_field.npy, v_field.npy, x_grid.npy, y_grid.npy")

# --- Sauvegarde des graphiques ---
import matplotlib.pyplot as plt

def save_plot(x, y_list, labels, title, xlabel, ylabel, filename,
              styles=None, show=True, legend_loc="best"):
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
    """
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
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Graphique sauvegardé : {filename}")

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
    print(f"🧾 Données exportées : {filename}")

def export_numpy_dict(data_dict, prefix=""):
    for key, arr in data_dict.items():
        path = f"{prefix}{key}.npy"
        np.save(path, arr)
        print(f"💾 Fichier sauvegardé : {path}")

def plot_quiver_field(X, Y, u, v, scale=50, skip=5, title="Champ de vent", savepath=None):
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

