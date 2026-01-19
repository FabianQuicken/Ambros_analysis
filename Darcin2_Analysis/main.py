"""
Analyseziele:
- Aufenthaltsdauer im Stimulusmodul vs Kontrollmodul Gesamt
- Aufenthaltsdauer im Stimulusmodul vs Kontrollmodul pro Visit
- Interaktionsdauer Stimulus vs Kontrolle
- Strecke/Zeit in Stimulusmodul vs Kontrollmodul

Plots: 
- Eventplots für jede einzelne Maus

Output: 
- Metriken werden in eine Excel Datei geschrieben

Generelles:
- Jeder Run analysiert eine Maus
- Die Daten aller Mäuse sind im Ordner "raw" über 3 Subordner "Day1", "Day2", "Day3" verteilt
- Kameraöffnen markiert Experimentstart
- dann enstehen über 20min Videoschnipsel, die zeitlich eingeordnet werden müssen
- "none_none" im Namen markiert dishes ohne stimuli (Day1 und Day3)
- Beispiel um die Benennung am Stimulustag 2 zu verstehen --> "153darcin_152_hepes": 
    - Wenn darcin vor hepes modul1 = Stimulus, sonst modul2 = Stimulus
    - Modul1 Stimulus = Urin Maus 153 1:1 Darcin(in HEPES) 
    - Modul2 Kontrolle = Urin Maus 152 1:1 HEPES
- likelihood Filterung muss stattfinden, da die Mäuse nicht immer im Modul sind 
"""


import numpy as np
import pandas as pd
import tqdm
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm



"""
Funktionen
"""
def plot_distance_histogram(
    distance_values,
    bins=50,
    title="Distance per frame",
    xlabel="Distance (pixel)",
    save_as=None,
    show_plot=False
):
    """
    Plots a histogram of per-frame distances.
    NaN values are ignored automatically.
    """

    # NaNs entfernen
    distances = np.asarray(distance_values)
    distances = distances[np.isfinite(distances)]
    distances = np.asarray([i for i in distances if i <= 60])
    

    if distances.size == 0:
        print("[plot_distance_histogram] No valid distance values to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=bins)
    plt.xlim(0,60)
    plt.ylim(0,2500)
    plt.xlabel(xlabel)
    plt.ylabel("Frames")
    plt.title(title)

    if save_as is not None:
        plt.savefig(save_as, format="svg")
    if show_plot:
        plt.show()

def heatmap_plot(x_values = np.array, y_values = np.array, plotname = str, save_as = str, num_bins = 35, cmap = 'hot', plot_time_frame_hours = (None, None)):
    """
    This function plots a heatmap of x and y coordinates, e.g. of the snout. Pass the coordinates of a complete experiment, and they get filtered for all
    values that are not "0". Plotname and savepath need to be provided. Binsize is 50 per default. Colormap is "hot" per default.
    """

    if plot_time_frame_hours[1]:
        plot_time_frame_frames = (round(plot_time_frame_hours[0]*108000), round(plot_time_frame_hours[1]*108000))
        x_values = x_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]
        y_values = y_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]

    # Filter: finite Werte und ungleich 0
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values != 0) & (y_values != 0)
    heatmap_x = x_values[mask]
    heatmap_y = y_values[mask]

    if heatmap_x.size == 0:
        print(f"[heatmap_plot] No valid data after filtering for: {plotname}")
        return


    # get max x value and max y value to scale the heatmap
    x_max = np.nanmax(heatmap_x)

    # tested, ob die y-Werte schon invertiert wurden
    if np.nanmax(heatmap_y) > 0:
        y_max = np.nanmax(heatmap_y)
    else:
        y_max = np.nanmin(heatmap_y)

        y_max = round(y_max *-1)

    # calculate number of y-bins based on ratio between x and y axis
    y_bins = round((y_max / x_max) * num_bins)
    bins = (num_bins, y_bins)


    # create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(heatmap_x, heatmap_y, bins=bins)

    plt.figure(figsize=(8,6))
    #sns.heatmap(heatmap.T, cmap=cmap, square=True, cbar=True, xticklabels=True, yticklabels=True)
    
    plt.imshow(heatmap.T, origin='lower', cmap=cmap,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect=1)  # Set aspect ratio 
    
    plt.colorbar(label='Frames')
    plt.title(plotname)
    plt.savefig(save_as, format='svg')
    plt.show()

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

def time_to_seconds(time_str: str) -> int:
    hours, minutes, seconds = map(int, time_str.split("_"))
    return hours * 3600 + minutes * 60 + seconds

def seconds_since_first(first_file: str, this_file: str) -> int:
    first_name = os.path.splitext(os.path.basename(first_file))[0]
    this_name  = os.path.splitext(os.path.basename(this_file))[0]

    first_time = first_name[11:19]  # HH_MM_SS
    this_time  = this_name[11:19]

    return time_to_seconds(this_time) - time_to_seconds(first_time)

def load_dlc_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".h5", ".hdf5"]:
        return pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def build_master_dlc_dataframe(
    files_in_order: list[str],
    fps: int,
    total_frames: int | None = None,
    fill_value=np.nan,
    allow_overlap: bool = False
) -> pd.DataFrame:
    """
    files_in_order: Liste der Fragment-Dateien in zeitlicher Reihenfolge.
    fps: Frames per second des Experiments.
    total_frames: Optional. Wenn None, wird aus erstem/letztem File geschätzt:
                 (end-start)*fps + len(last_df)
    allow_overlap: Wenn False, wird bei Überschneidungen ein Fehler geworfen.
    """

    if not files_in_order:
        raise ValueError("files_in_order is empty")

    first_file = files_in_order[0]
    last_file = files_in_order[-1]

    # Lade erstes und letztes Fragment einmal, um Spalten + Default total_frames zu bestimmen
    df_first = load_dlc_df(first_file)
    df_last  = load_dlc_df(last_file)

    if total_frames is None:
        exp_seconds = seconds_since_first(first_file, last_file)
        total_frames = exp_seconds * fps + len(df_last)

    # Master-DF (NaNs) in Zielgröße
    master = pd.DataFrame(
        data=fill_value,
        index=pd.RangeIndex(total_frames),
        columns=df_first.columns
    )

    # Optional: Tracking, ob Zeilen schon befüllt wurden (für Overlap-Check)
    filled_mask = np.zeros(total_frames, dtype=bool)

    for f in files_in_order:
        df_part = load_dlc_df(f)

        # Spalten konsistent machen (falls einzelne Fragmente Spaltenreihenfolge abweicht)
        # -> fehlende Spalten werden NaN, zusätzliche Spalten werden verworfen
        df_part = df_part.reindex(columns=master.columns)

        offset_s = seconds_since_first(first_file, f)
        if offset_s < 0:
            raise ValueError(f"File {f} starts before first_file (negative offset).")
        offset_frames = int(round(offset_s * fps))

        start = offset_frames
        end = offset_frames + len(df_part)

        if end > total_frames:
            raise ValueError(
                f"Fragment {f} would exceed master length: end={end}, total_frames={total_frames}. "
                f"Increase total_frames or check timestamps/FPS."
            )

        if not allow_overlap:
            if filled_mask[start:end].any():
                raise ValueError(
                    f"Overlap detected when inserting {f} into range [{start}:{end}]. "
                    f"Set allow_overlap=True or resolve timestamp issues."
                )

        # Einfügen (schnell, spaltenweise kompatibel)
        master.iloc[start:end] = df_part.to_numpy(copy=False)

        filled_mask[start:end] = True

    return master

def filter_and_interpolate_all_bodyparts(
    df: pd.DataFrame,
    scorer: str,
    bodyparts: list[str] | None = None,
    filter_value: float = 0.8,
    max_gap: int = 30,
    method: str = "linear",
    keep_likelihood: bool = True,
):
    """
    Für jeden Bodypart:
    - erstellt Maske: likelihood >= filter_value
    - setzt x/y bei schlechter likelihood auf NaN
    - interpoliert nur x/y mit max_gap-Regel
    - lässt likelihood unverändert (keine Interpolation), optional aber mit zurückgeben

    Returns:
      df_out: DataFrame mit gefilterten+interpolierten x/y (und optional likelihood)
      masks: dict[bodypart] -> boolean mask (True = valid)
    """

    df_out = df.copy()
    masks = {}

    # Wenn bodyparts nicht angegeben: aus Spalten ableiten
    if bodyparts is None:
        bodyparts = list(df_out.loc[:, (scorer, slice(None), slice(None))].columns.levels[1])

    for bp in bodyparts:
        # Likelihood-Serien (Index: Frames)
        lh = df_out.loc[:, (scorer, bp, "likelihood")]
        mask = lh >= filter_value
        masks[bp] = mask.to_numpy()

        # x/y extrahieren
        xy = df_out.loc[:, (scorer, bp, ["x", "y"])].copy()

        # schlechte Frames -> NaN (nur x/y)
        xy.loc[~mask, :] = np.nan

        # interpolieren (nur x/y)
        xy_interp = interpolate_with_max_gap(xy, max_gap=max_gap, method=method)

        # zurückschreiben
        df_out.loc[:, (scorer, bp, ["x", "y"])] = xy_interp

        # optional: likelihood in Ausgabe behalten oder komplett auf NaN setzen bei invalid
        if not keep_likelihood:
            df_out.loc[:, (scorer, bp, "likelihood")] = np.nan

    return df_out, masks

FPS = 30
PIXEL_PER_CM = 36.39
DIST_THRESH = PIXEL_PER_CM*2.5

all_mice = ["109", "121", "122", "125"]
mouse = "109"

"""
Daten einlesen und in Stimulus und Kontrolle sortieren
"""

exp_path = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\raw"

day1_files = sorted(glob.glob(os.path.join(exp_path + "/Day1/" + mouse, '*.h5')))
day2_files = sorted(glob.glob(os.path.join(exp_path + "/Day2/" + mouse, '*.h5')))
day3_files = sorted(glob.glob(os.path.join(exp_path + "/Day3/" + mouse, '*.h5')))

m1_d1_files = [file for file in day1_files if "top1" in file]
m2_d1_files = [file for file in day1_files if "top2" in file]

m1_d2_files = [file for file in day2_files if "top1" in file]
m2_d2_files = [file for file in day2_files if "top2" in file]

m1_d3_files = [file for file in day3_files if "top1" in file]
m2_d3_files = [file for file in day3_files if "top2" in file]


stim_modul = None
if "darcin" in os.path.basename(m1_d2_files[0])[43:52]:
    stim_modul = 1
elif "hepes" in os.path.basename(m1_d2_files[0])[43:52]: 
    stim_modul = 2
else:
    raise NameError(f"Filename seems to be incorrect.\nFilename: {m1_d2_files[0]}")


if stim_modul == 1:
    stim_data = [m1_d1_files, m1_d2_files, m1_d3_files]
    con_data = [m2_d1_files, m2_d2_files, m2_d3_files]
else:
    con_data = [m1_d1_files, m1_d2_files, m1_d3_files]
    stim_data = [m2_d1_files, m2_d2_files, m2_d3_files]

dish_inv = {
        "day1": {
            "stim_dish": None,
            "con_dish": None
        },
        "day2": {
            "stim_dish": None,
            "con_dish": None
        },
        "day3": {
            "stim_dish": None,
            "con_dish": None
        }
    }

time_present = {
        "day1": {
            "stim_modul": None,
            "con_modul": None
        },
        "day2": {
            "stim_modul": None,
            "con_modul": None
        },
        "day3": {
            "stim_modul": None,
            "con_modul": None
        }
    }



for i in tqdm(range(3)): # über jeden Experimenttag iterieren, hier später 3  einfügen

    d_stim_data = stim_data[i]
    d_con_data = con_data[i]

    # Master_df erstellen: packt alle einzelnen h5 dateien in eine zeitlich sortierte Datei
    m_stim_df = build_master_dlc_dataframe(files_in_order=d_stim_data, fps=FPS, allow_overlap=True)
    m_con_df = build_master_dlc_dataframe(files_in_order=d_con_data, fps=FPS, allow_overlap=True)

    scorer = m_stim_df.columns.levels[0][0]

    
    
    def get_dish_coords(df, scorer, dish_name):

        dish_likelihood = df.loc[:, (scorer, [dish_name], ["likelihood"])].to_numpy().ravel()

        best_likelihood_frame = 0
        max_likelihood = 0

        for index, likelihood in enumerate(dish_likelihood):
            if np.isfinite(likelihood) and likelihood > max_likelihood:
                max_likelihood = likelihood
                best_likelihood_frame = index
                    

        dish_x = df.loc[best_likelihood_frame, (scorer, [dish_name], ["x"])].to_numpy().item()
        dish_y = df.loc[best_likelihood_frame, (scorer, [dish_name], ["y"])].to_numpy().item()


        return dish_x, dish_y


    s_dish_x, s_dish_y = get_dish_coords(df=load_dlc_df(d_stim_data[0]), scorer=scorer, dish_name="dish")
    c_dish_x, c_dish_y = get_dish_coords(df=load_dlc_df(d_con_data[0]), scorer=scorer, dish_name="dish")

    s_dish_inv = np.zeros(len(m_stim_df))
    c_dish_inv = np.zeros(len(m_con_df))

    # nose coordinaten likelihood filtern über maske
    filter_value = 0.99

    stim_nose_likelihood = m_stim_df.loc[:, (scorer, ["nose"], ["likelihood"])].to_numpy().ravel()
    stim_likelihood_mask = stim_nose_likelihood >= filter_value
    #stim_nose_filtered = m_stim_df.loc[stim_likelihood_mask, (scorer, ["nose"], ["x", "y", "likelihood"])].copy()
    stim_nose_filtered = m_stim_df.loc[:, (scorer, ["nose"], ["x", "y", "likelihood"])].copy()
    stim_nose_filtered.loc[~stim_likelihood_mask, :] = np.nan
    stim_nose_filtered = interpolate_with_max_gap(stim_nose_filtered)

    # nose coordinaten likelihood filtern über maske
    con_nose_likelihood = m_con_df.loc[:, (scorer, ["nose"], ["likelihood"])].to_numpy().ravel()
    con_likelihood_mask = con_nose_likelihood >= filter_value
    #con_nose_filtered = m_con_df.loc[con_likelihood_mask, (scorer, ["nose"], ["x", "y"])]
    con_nose_filtered = m_con_df.loc[:, (scorer, ["nose"], ["x", "y", "likelihood"])].copy()
    con_nose_filtered.loc[~con_likelihood_mask, :] = np.nan
    con_nose_filtered = interpolate_with_max_gap(con_nose_filtered)

    # zeit im modul based on hoher nose likelihood
    time_present_s = (stim_nose_filtered.loc[:, (scorer, "nose", "likelihood")] > filter_value).sum()
    time_present_c = (con_nose_filtered.loc[:, (scorer, "nose", "likelihood")] > filter_value).sum()

    #time_present[f"day{str(i+1)}"]["stim_modul"] = round(counter/len(stim_likelihood_mask), 3)
    time_present[f"day{str(i+1)}"]["stim_modul"] = round(time_present_s/len(stim_likelihood_mask), 3)
    time_present[f"day{str(i+1)}"]["con_modul"] = round(time_present_c/len(con_likelihood_mask), 3)

    # nose coordinaten extrahieren s= stimulus, c=control
    s_nose_x = stim_nose_filtered.loc[:, (scorer, ["nose"], ["x"])].to_numpy().ravel()
    s_nose_y = stim_nose_filtered.loc[:, (scorer, ["nose"], ["y"])].to_numpy().ravel()

    #heatmap_plot(x_values=s_nose_x, y_values=s_nose_y, plotname=mouse + " stim modul", save_as=exp_path + f"/{mouse}_day{i+1}_stim.svg")

    c_nose_x = con_nose_filtered.loc[:, (scorer, ["nose"], ["x"])].to_numpy().ravel()
    c_nose_y = con_nose_filtered.loc[:, (scorer, ["nose"], ["y"])].to_numpy().ravel()

    #heatmap_plot(x_values=c_nose_x, y_values=c_nose_y, plotname=mouse + " con modul", save_as=exp_path + f"/{mouse}_day{i+1}_con.svg")

    # über Abstand die investigation time checken
    for j, (x,y) in enumerate(zip(s_nose_x, s_nose_y)):
        d_dist = euklidean_distance(x1=s_dish_x, y1=s_dish_y, x2=x, y2=y)
        if d_dist <= DIST_THRESH:
            s_dish_inv[j] = 1

    for j, (x,y) in enumerate(zip(c_nose_x, c_nose_y)):
        d_dist = euklidean_distance(x1=c_dish_x, y1=c_dish_y, x2=x, y2=y)
        if d_dist <= DIST_THRESH:
            c_dish_inv[j] = 1

    dish_inv[f"day{str(i+1)}"]["stim_dish"] = int(np.nansum(s_dish_inv))
    dish_inv[f"day{str(i+1)}"]["con_dish"] = int(np.nansum(c_dish_inv))

    def calculate_dist(df):

        df, masks = filter_and_interpolate_all_bodyparts(df=df, scorer=scorer, bodyparts=["nose", "left_ear", "right_ear"], filter_value=0.8)
        # zurückgelegte Strecke (Pixel pro Frame)
        x = df.loc[:, (scorer, ["nose", "left_ear", "right_ear"], "x")]
        y = df.loc[:, (scorer, ["nose", "left_ear", "right_ear"], "y")]

        # pro Frame den Mittelwert über die 3 Punkte (NaNs werden ignoriert)
        mean_x = np.nanmean(x.to_numpy(), axis=1)  # shape: (n_frames,)
        mean_y = np.nanmean(y.to_numpy(), axis=1)  # shape: (n_frames,)

        distance_values = np.full(len(mean_x), np.nan)

        for k in range(len(mean_x) - 1):
            distance_values[k] = euklidean_distance(
                x1=mean_x[k], y1=mean_y[k],
                x2=mean_x[k+1], y2=mean_y[k+1]
        )
        return distance_values
    
    dist_stim = calculate_dist(m_stim_df)
    dist_con = calculate_dist(m_con_df)

    plot_distance_histogram(distance_values=dist_stim,
                            title=f"Distance per Frame, mouse{mouse}, day {i+1}, stim",
                            save_as=exp_path+f"/disthist_day{i+1}_{mouse}_stim.svg")
    plot_distance_histogram(distance_values=dist_con,
                            title=f"Distance per Frame, mouse{mouse}, day {i+1}, con",
                            save_as=exp_path+f"/disthist_day{i+1}_{mouse}_con.svg")
    
    dist_sum_stim = np.nansum(dist_stim)
    dist_sum_con = np.nansum(dist_con)
    dist_cm_s = dist_sum_stim/PIXEL_PER_CM
    dist_cm_c = dist_sum_con/PIXEL_PER_CM

    # geschwindigkeit
    avg_speed_s = dist_cm_s / (time_present_s/FPS)
    avg_speed_c = dist_cm_c / (time_present_c/FPS)

    """
    print("\nDistance and speed Stim:\n")
    print(dist_cm_s)
    print(avg_speed_s)
    print("\nDistance and speed Con:\n")
    print(dist_cm_c)
    print(avg_speed_c)
    """


#print(dish_inv)
#print(time_present)



