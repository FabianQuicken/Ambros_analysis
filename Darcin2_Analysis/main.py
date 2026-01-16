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

"""
Funktionen
"""
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

FPS = 30
PIXEL_PER_CM = 36.39
DIST_THRESH = PIXEL_PER_CM*2.5

all_mice = ["109", "121", "122", "125"]
mouse = "125"

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


for i in range(3): # über jeden Experimenttag iterieren, hier später 3  einfügen

    d_stim_data = stim_data[i]
    d_con_data = con_data[i]

    print(dish_inv[f"day{str(i+1)}"])

    # Master_df erstellen
    
    m_stim_df = build_master_dlc_dataframe(files_in_order=d_stim_data, fps=FPS, allow_overlap=True)
    m_con_df = build_master_dlc_dataframe(files_in_order=d_con_data, fps=FPS, allow_overlap=True)

    scorer = m_stim_df.columns.levels[0][0]

    
    
    def get_dish_coords(df, scorer, dish_name):

        dish_likelihood = df.loc[:, (scorer, [dish_name], ["likelihood"])].to_numpy().ravel()

        best_likelihood_frame = 0
        max_likelihood = 0

        for index, likelihood in enumerate(dish_likelihood):
            if likelihood > max_likelihood:
                    best_likelihood_frame = index
                    

        dish_x = df.loc[best_likelihood_frame, (scorer, [dish_name], ["x"])].to_numpy().item()
        dish_y = df.loc[best_likelihood_frame, (scorer, [dish_name], ["y"])].to_numpy().item()


        return dish_x, dish_y


    s_dish_x, s_dish_y = get_dish_coords(df=load_dlc_df(d_stim_data[0]), scorer=scorer, dish_name="dish")
    c_dish_x, c_dish_y = get_dish_coords(df=load_dlc_df(d_con_data[0]), scorer=scorer, dish_name="dish")

    s_dish_inv = np.zeros(len(m_stim_df))
    c_dish_inv = np.zeros(len(m_con_df))

    s_nose_x = m_stim_df.loc[:, (scorer, ["nose"], ["x"])].to_numpy().ravel()
    s_nose_y = m_stim_df.loc[:, (scorer, ["nose"], ["y"])].to_numpy().ravel()

    c_nose_x = m_con_df.loc[:, (scorer, ["nose"], ["x"])].to_numpy().ravel()
    c_nose_y = m_con_df.loc[:, (scorer, ["nose"], ["y"])].to_numpy().ravel()

    # likelihood filtern über maske
    stim_nose_likelihood = m_stim_df.loc[:, (scorer, ["nose"], ["likelihood"])].to_numpy().ravel()
    likelihood_mask = stim_nose_likelihood >= 0.6
    m_stim_df.loc[~likelihood_mask, (scorer, ["nose"], ["x", "y"])]

    


    for j, (x,y) in enumerate(zip(s_nose_x, s_nose_y)):
        d_dist = euklidean_distance(x1=s_dish_x, y1=s_dish_y, x2=x, y2=y)
        if d_dist <= DIST_THRESH:
            s_dish_inv[j] = 1

    for j, (x,y) in enumerate(zip(c_nose_x, c_nose_y)):
        d_dist = euklidean_distance(x1=c_dish_x, y1=c_dish_y, x2=x, y2=y)
        if d_dist <= DIST_THRESH:
            c_dish_inv[j] = 1

    







