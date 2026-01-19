import numpy as np
import pandas as pd
import tqdm
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

FPS = 30
PIXEL_PER_CM = 36.39
DIST_THRESH = PIXEL_PER_CM*2.5

def load_dlc_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".h5", ".hdf5"]:
        return pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
def euklidean_distance(x1, y1, x2, y2):
        """
        This func returns the euklidean distance between two points.
        (x1, y1) and (x2, y2) are the cartesian coordinates of the points.
        """
        if np.isnan(x1):
            distance = np.nan
        elif np.isnan(x2):
            distance = np.nan
        else:
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        return distance

def interpolate_with_max_gap(df, max_gap=30, method="linear"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    #print(num_cols)

    # 1) Nur „echte“ Interpolation zwischen gültigen Punkten
    out[num_cols] = out[num_cols].interpolate(method=method,
                                              limit_direction="both",
                                              limit_area="inside")
    
    # 2) NaN-Runs > max_gap identifizieren und wieder auf NaN setzen
    for col in num_cols:
        s = df[col]  # Original mit NaNs
        # Gruppen-IDs zwischen Nicht-NaNs erstellen
        grp = s.notna().cumsum()
        # Länge jedes NaN-Runs
        run_len = s.isna().groupby(grp).transform("sum")
        # Maske: Positionen in zu langen NaN-Runs
        too_long = s.isna() & (run_len > max_gap)
        # Zurücksetzen
        out.loc[too_long, col] = np.nan
    
    return out

def plot_distance_histogram(
    distance_values,
    bins=50,
    title="Distance per frame",
    xlabel="Distance (pixel)",
    save_as=None
):
    """
    Plots a histogram of per-frame distances.
    NaN values are ignored automatically.
    """

    # NaNs entfernen
    distances = np.asarray(distance_values)
    distances = distances[np.isfinite(distances)]

    if distances.size == 0:
        print("[plot_distance_histogram] No valid distance values to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Frames")
    plt.title(title)

    if save_as is not None:
        plt.savefig(save_as, format="svg")

    plt.show()

all_mice = ["109", "121", "122", "125"]
mouse = "109"

"""
Daten einlesen und in Stimulus und Kontrolle sortieren
"""

exp_path = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\raw"

day1_files = sorted(glob.glob(os.path.join(exp_path + "/Day1/" + mouse, '*.h5')))

m1_d1_files = [file for file in day1_files if "top1" in file]

for file in m1_d1_files:
    df = load_dlc_df(file)
    scorer = df.columns.levels[0][0]

    filter_value = 0.8
    stim_nose_likelihood = df.loc[:, (scorer, ["nose"], ["likelihood"])].to_numpy().ravel()
    stim_likelihood_mask = stim_nose_likelihood >= filter_value
    #stim_nose_filtered = m_stim_df.loc[stim_likelihood_mask, (scorer, ["nose"], ["x", "y", "likelihood"])].copy()
    stim_nose_filtered = df.loc[:, (scorer, ["nose"], ["x", "y", "likelihood"])].copy()
    stim_nose_filtered.loc[~stim_likelihood_mask, :] = np.nan
    stim_nose_filtered = interpolate_with_max_gap(stim_nose_filtered)