import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def shift(f, axis, direction):
    """
    Applique un décalage périodique à un tableau le long d'un axe spécifié.

    Args:
        f (numpy.ndarray): Le tableau à décaler
        axis (int): L'axe le long duquel effectuer le décalage
        direction (int): Le nombre de positions à décaler (positif ou négatif)

    Returns:
        numpy.ndarray: Le tableau décalé
    """
    return np.roll(f, shift=direction, axis=axis)


def lax_friedrichs_2d(f, u, v, dx, dy):
    """
    Applique le schéma de convection Lax-Friedrichs 2D.

    Args:
        f (numpy.ndarray): Le champ à advecter
        u (numpy.ndarray): La composante x du champ de vitesse
        v (numpy.ndarray): La composante y du champ de vitesse
        dx (float): L'espacement de la grille dans la direction x
        dy (float): L'espacement de la grille dans la direction y

    Returns:
        numpy.ndarray: Le terme d'advection
    """
    fx = 0.5 * (shift(f, 1, -1) - shift(f, 1, 1)) / (2 * dx)
    fy = 0.5 * (shift(f, 0, -1) - shift(f, 0, 1)) / (2 * dy)
    return u * fx + v * fy


def compute_velocity_norm(u, v):
    """
    Calcule la norme d'un champ de vitesse.

    Args:
        u (numpy.ndarray): La composante x du champ de vitesse
        v (numpy.ndarray): La composante y du champ de vitesse

    Returns:
        numpy.ndarray: La norme de la vitesse à chaque point
    """
    return np.sqrt(u**2 + v**2)


def find_cyclone_center(norm_v, x, y):
    """
    Find the center of a cyclone based on the maximum of the velocity norm.

    Args:
        norm_v (numpy.ndarray): The velocity norm field
        x (numpy.ndarray): The x-coordinates grid
        y (numpy.ndarray): The y-coordinates grid

    Returns:
        tuple: The (x, y) coordinates of the cyclone center
    """
    max_idx = np.unravel_index(np.argmax(norm_v), norm_v.shape)
    return float(x[max_idx]), float(y[max_idx])

def get_versioned_filename(directory, filename):
    """
    Generate a versioned filename to avoid overwriting existing files.

    Args:
        directory (str): The directory where the file will be saved
        filename (str): The base filename

    Returns:
        str: A unique filename with version number
    """
    base, ext = os.path.splitext(filename)
    version = 0.0
    versioned_filename = os.path.join(directory, f"{base}_v{version:.1f}{ext}")
    while os.path.exists(versioned_filename):
        version += 0.1
        versioned_filename = os.path.join(directory, f"{base}_v{version:.1f}{ext}")
    return versioned_filename


def save_fields_edp(u, v, x, y, results_dir="results"):
    """
    Save velocity fields and coordinate grids to numpy files.

    Args:
        u (numpy.ndarray): The x-component of the velocity field
        v (numpy.ndarray): The y-component of the velocity field
        x (numpy.ndarray): The x-coordinates grid
        y (numpy.ndarray): The y-coordinates grid
        results_dir (str, optional): Directory to save results. Defaults to "results".
    """
    os.makedirs(results_dir, exist_ok=True)
    for name, data in [("u_field", u), ("v_field", v), ("x_grid", x), ("y_grid", y)]:
        path = get_versioned_filename(results_dir, f"{name}.npy")
        np.save(path, data)
        print(f"Field saved: {path}")


def load_edp_fields(directory="."):
    """
    Load velocity fields and coordinate grids from numpy files.
    Loads the latest version of each file based on version number.

    Args:
        directory (str, optional): Directory containing the field files. Defaults to current directory.

    Returns:
        tuple: (u, v, x, y) arrays containing the velocity fields and coordinate grids

    Raises:
        FileNotFoundError: If any of the required files are not found
    """
    try:
        # Define the base filenames
        base_filenames = ["u_field", "v_field", "x_grid", "y_grid"]
        loaded_data = []

        # Find the latest version for each file
        for base_name in base_filenames:
            # Get all files matching the pattern base_name_v*.npy
            matching_files = [f for f in os.listdir(directory) 
                             if f.startswith(f"{base_name}_v") and f.endswith(".npy")]

            # If no versioned files found, try the non-versioned file
            if not matching_files:
                file_path = os.path.join(directory, f"{base_name}.npy")
                if os.path.exists(file_path):
                    loaded_data.append(np.load(file_path))
                    continue
                else:
                    raise FileNotFoundError(f"No {base_name} files found in {directory}")

            # Extract versions from filenames and find the latest
            versions = []
            for filename in matching_files:
                # Extract version number from filename (format: base_name_vX.Y.npy)
                version_str = filename.replace(f"{base_name}_v", "").replace(".npy", "")
                try:
                    version = float(version_str)
                    versions.append((version, filename))
                except ValueError:
                    # Skip files with invalid version format
                    continue

            if not versions:
                raise FileNotFoundError(f"No valid versioned {base_name} files found in {directory}")

            # Sort by version and get the latest
            versions.sort(reverse=True)
            latest_version, latest_filename = versions[0]

            # Load the latest version
            file_path = os.path.join(directory, latest_filename)
            loaded_data.append(np.load(file_path))
            print(f"Loaded {base_name} version {latest_version:.1f} from {latest_filename}")

        # Return the loaded data as a tuple (u, v, x, y)
        return tuple(loaded_data)
    except FileNotFoundError as e:
        print(f"Error loading field files: {e}")
        print("Make sure to run the EDP simulation first or specify the correct directory.")
        raise

def compute_radial_profile(R, V, num_bins=100):
    """
    Compute a radial profile of a field by binning and averaging.

    Args:
        R (numpy.ndarray): Radial distance field
        V (numpy.ndarray): Value field to be averaged
        num_bins (int, optional): Number of radial bins. Defaults to 100.

    Returns:
        tuple: (r_centers, profile) where r_centers are the bin centers and
               profile contains the averaged values
    """
    r_bins = np.linspace(0, R.max(), num_bins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # More readable implementation of the profile calculation
    profile = []
    for i in range(len(r_bins) - 1):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.any(mask):
            profile.append(np.mean(V[mask]))
        else:
            profile.append(np.nan)

    return r_centers, np.array(profile)


def save_plot(x, y_list, labels, title, xlabel, ylabel, filename, 
              styles=None, show=True, legend_loc="best", results_dir="results"):
    """
    Create and save a line plot with multiple data series.

    Args:
        x (array-like): x-axis data
        y_list (list): List of y-axis data series
        labels (list): List of labels for each data series
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        filename (str): Output filename
        styles (list, optional): Line styles for each series. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        legend_loc (str, optional): Legend location. Defaults to "best".
        results_dir (str, optional): Output directory. Defaults to "results".
    """
    try:
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

        print(f"Plot saved: {path}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_quiver_field(X, Y, u, v, scale=50, skip=5, title="Wind Field", savepath=None):
    """
    Create and save a quiver plot of a vector field.

    Args:
        X (numpy.ndarray): x-coordinates grid
        Y (numpy.ndarray): y-coordinates grid
        u (numpy.ndarray): x-component of the vector field
        v (numpy.ndarray): y-component of the vector field
        scale (int, optional): Quiver scale factor. Defaults to 50.
        skip (int, optional): Sampling interval for quiver arrows. Defaults to 5.
        title (str, optional): Plot title. Defaults to "Wind Field".
        savepath (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    try:
        plt.figure(figsize=(12, 12))
        plt.quiver(X[::skip, ::skip]/1000, Y[::skip, ::skip]/1000, 
                  u[::skip, ::skip], v[::skip, ::skip], scale=scale)
        plt.title(title)
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.grid(True)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
            print(f"Vector field plot saved: {savepath}")
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error creating quiver plot: {e}")

def display_series_info(series_expr, delta):
    """
    Display the coefficients of a series expansion.

    Args:
        series_expr (sympy.Expr): The series expression
        delta (sympy.Symbol): The variable of the series expansion
    """
    for i in range(4):
        coeff = series_expr.coeff(delta, i)
        if coeff != 0:
            print(f"  • Order {i}: {coeff}")


def extract_order_equations(series_list, delta, orders=[0, 1, 2]):
    """
    Extract coefficients of specific orders from a list of series.

    Args:
        series_list (list): List of sympy series expressions
        delta (sympy.Symbol): The variable of the series expansion
        orders (list, optional): Orders to extract. Defaults to [0, 1, 2].

    Returns:
        list: Coefficients of the specified orders for each series
    """
    return [series.coeff(delta, n) for series in series_list for n in orders]


def export_dataframe(df, filename):
    """
    Export a pandas DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to export
        filename (str): Output filename
    """
    try:
        df.to_csv(filename, index=False)
        print(f"Data exported: {filename}")
    except Exception as e:
        print(f"Error exporting DataFrame: {e}")


def export_numpy_dict(data_dict, prefix=""):
    """
    Export a dictionary of numpy arrays to files.

    Args:
        data_dict (dict): Dictionary of numpy arrays
        prefix (str, optional): Prefix for filenames. Defaults to "".
    """
    try:
        for key, arr in data_dict.items():
            path = f"{prefix}{key}.npy"
            np.save(path, arr)
            print(f"File saved: {path}")
    except Exception as e:
        print(f"Error exporting numpy arrays: {e}")


def format_edo_solution(xi_m, a_sol, b_sol):
    """
    Format ODE solution arrays into a dictionary with derived quantities.

    Args:
        xi_m (numpy.ndarray): Radial distance array (in meters)
        a_sol (numpy.ndarray): Radial velocity component
        b_sol (numpy.ndarray): Tangential velocity component

    Returns:
        dict: Dictionary with xi_km, a_xi, b_xi, theta_xi, and E_xi
    """
    # Convert to kilometers for display
    xi_km = xi_m / 1000

    # Calculate derived quantities
    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.where(np.abs(a_sol) > 1e-10, -b_sol / a_sol, 0)

    # Energy
    E = 0.5 * (a_sol**2 + b_sol**2)

    return {
        "xi_km": xi_km,
        "a_xi": a_sol,
        "b_xi": b_sol,
        "theta_xi": theta,
        "E_xi": E
    }


def save_edo_result(result_dict, output_csv):
    """
    Enregistre les résultats EDO dans un fichier CSV.
    """
    df = pd.DataFrame({
        "xi_km": result_dict["xi_km"],
        "a_xi": result_dict["a_xi"],
        "b_xi": result_dict["b_xi"],
        "theta_xi": result_dict["theta_xi"],
        "E_xi": result_dict["E_xi"]
    })
    df.to_csv(output_csv, index=False)
    print(f"Data exported: {output_csv}")


def generate_velocity_fields_from_edo(result, output_dir="results"):
    """
    Génére les champs U(x,y), V(x,y), X, Y à partir des composantes radiales et tangentielles
    a(ξ) et b(ξ) et les enregistre dans des fichiers versionnés.

    Paramètres :
        result (dict) : Résultat retourné par simulate_edo_...
        output_dir (str) : Dossier de sortie pour enregistrer les .npy
    """
    a_xi = np.array(result["a_xi"])
    b_xi = np.array(result["b_xi"])
    xi = np.array(result["xi_km"])

    if len(xi) == 0 or len(a_xi) == 0 or len(b_xi) == 0:
        print("Impossible de générer les champs vectoriels : données incomplètes.")
        return

    N = len(xi)
    r = xi  # ξ représente la distance radiale
    theta = np.linspace(0, 2 * np.pi, N)

    R, T = np.meshgrid(r, theta, indexing='ij')  # (N, N)

    # Composantes dans le repère polaire
    A = np.tile(a_xi[:, np.newaxis], (1, N))  # a(ξ)
    B = np.tile(b_xi[:, np.newaxis], (1, N))  # b(ξ)

    # Coordonnées cartésiennes (X, Y) à partir de (R, T)
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Calcul des composantes dans le repère cartésien
    U = A * X / R - B * Y / R
    V = A * Y / R + B * X / R

    # Gestion des divisions par zéro
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)

    # Versionnement et sauvegarde
    for name, field in zip(['u_field', 'v_field', 'x_grid', 'y_grid'], [U, V, X, Y]):
        versioned_name = get_versioned_filename(output_dir, f"{name}.npy")
        np.save(versioned_name, field)
        print(f"Field saved: {versioned_name}")

    print(f"Champs vectoriels 2D sauvegardés dans {output_dir}/")
