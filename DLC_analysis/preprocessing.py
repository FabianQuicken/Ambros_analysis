"""

Here, the preprocessing of DLC data is performed. The preprocessing includes the following steps:
1. Loading the DLC data from h5 or CSV files in a destination folder.
2. Metadata extraction and saving.
3. Output DF and Working DF initialization.
4. Likelihood Filtering & Interpolation. 
5. Inserting DLC Data in the Working DF, last files overwrite if overlapping. 
6. Cropping the Working DF to the time range of interest (experiment length).
7. Returning Working DF, Output DF and Metadata (optionally returning original DLC data).

"""

import glob as glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# # # # # # # _________________________________________________________________________________________________________________

# Variablen 

# # # # # # # _________________________________________________________________________________________________________________

path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_30_45_46\hab"
FPS = 30


# # # # # # # _________________________________________________________________________________________________________________

# Funktionen  

# # # # # # # _________________________________________________________________________________________________________________


def file_discovery(path):
    
    h5_files = sorted(glob.glob(os.path.join(path, "*.h5")))
    csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))

    if len(h5_files) > 0:
        print(f"Found {len(h5_files)} h5 files in the specified path.")
        return h5_files, ".h5"
    elif len(csv_files) > 0:
        print(f"Found {len(csv_files)} csv files in the specified path.")
        return csv_files, ".csv"
    else:
        raise FileNotFoundError("No DLC data files found in the specified path. Please check the path and ensure that the files are in the correct format (.h5 or .csv).")

def calculate_experiment_length(first_file, last_file):
      
      name_first_file = os.path.splitext(os.path.basename(first_file))[0]
      name_last_file = os.path.splitext(os.path.basename(last_file))[0]

      # Datum noch mit auslesen
      date = name_first_file[0:10]


      #Zeit immer an selber stelle
      startzeit = name_first_file[11:19] 
      endzeit = name_last_file[11:19]

      start_in_s = time_to_seconds(startzeit)
      ende_in_s = time_to_seconds(endzeit)

      experiment_dauer_in_s = ende_in_s - start_in_s 
      basename, filetype = os.path.splitext(last_file)
      # gesamte experimentdauer in frames
      if 'csv' in filetype:
        df_last_file = _load_dlc_csv(last_file)[0]
      elif 'h5' in filetype:
        df_last_file = pd.read_hdf(rf'{last_file}')
      exp_duration_frames = experiment_dauer_in_s * FPS + len(df_last_file)

      return exp_duration_frames, startzeit, endzeit, date

def time_to_seconds(time_str):
        
        hours, minutes, seconds = map(int, time_str.split("_"))
        return hours * 3600 + minutes * 60 + seconds

def get_metadata(file_list, filetype, fps):
    # get recording length
    exp_duration_frames, startzeit, endzeit, date = calculate_experiment_length(file_list[0], file_list[-1])
    filename = os.path.basename(file_list[0])
    name_parts = filename.split("_")


    # get rid of parts related to the DLC network
    dlc_part = None
    for i, part in enumerate(name_parts):
        if "DLC" in part:
            dlc_part = i
    name_parts = name_parts[6:dlc_part] if dlc_part is not None else name_parts[6:]

    camera = name_parts[-1]

    animal_info = name_parts[0] + "_" + name_parts[1]

    experiment_info = "_".join(name_parts[2:-1])  # Join all parts except the last one (camera)

    return {
        "fps": fps,
        "start": startzeit,
        "end": endzeit,
        "date": date,
        "experiment_length_frames": exp_duration_frames,
        "animal_info": animal_info,
        "experiment_info": experiment_info,
        "camera": camera
    }

def _load_dlc_csv(file):
    # Nur die ersten 4 Zeilen anschauen
    preview = pd.read_csv(file, header=None, nrows=4)

    header_labels = preview.iloc[:, 0].astype(str).str.lower().tolist()
    ma = False
    if "individuals" in header_labels and "coords" in header_labels:
        print("hello")
        header = [0, 1, 2, 3]
        ma = True
    elif "coords" in header_labels:
        header = [0, 1, 2]
    else:
        raise ValueError(f"Unexpected header format in CSV file: {file}. Expected 'coords' or 'individuals' in the first column.")

    df = pd.read_csv(
        file,
        header=header,
        index_col=0
    )

    return df, ma

def load_dlc_df(first_file, filetype):

    if "h5" in filetype:
        df = pd.read_hdf(first_file)
        cols = df.columns.nlevels
        ma = False
        if "individuals" in df.columns.names:
            ma = True 
        if not "coords" in df.columns.names:
            raise ValueError(f"Unexpected column names in HDF5 file: {first_file}. Expected 'coords' in the column names.")  

    elif "csv" in filetype:
        df, ma = _load_dlc_csv(first_file)


    return df, ma

def add_individual_level(df, individual_name):
    df = df.copy()

    df.columns = pd.MultiIndex.from_tuples(
        [
            (scorer, individual_name, bodypart, coord)
            for scorer, bodypart, coord in df.columns
        ],
        names=["scorer", "individuals", "bodyparts", "coords"]
    )

    return df

def create_working_df(files, filetype, metadata):

    first_df, ma = load_dlc_df(files[0], filetype)

    if not ma:
        first_df = add_individual_level(first_df, metadata["animal_info"])
    # Create a new DataFrame with the same columns as the original DataFrame
    working_df = pd.DataFrame(index=range(metadata["experiment_length_frames"]), columns=first_df.columns)

    return working_df


def likelihood_filtering(df, filter_value = 0.3):

    df = df.copy()
    individuals = df.columns.get_level_values("individuals").unique()
    bodyparts = df.columns.get_level_values("bodyparts").unique()
    for ind in individuals:
        for bp in bodyparts:
            # likelihood als 1D-Array
            lh = df.loc[:, (slice(None), ind, bp, "likelihood")].squeeze() # aus lh (df) eine series machen

            # Maske: True wo likelihood < threshold
            mask = lh < filter_value

            # x und y auf NaN setzen
            df.loc[mask, (slice(None), ind, bp, ["x", "y"])] = np.nan

    return df


def interpolate_with_max_gap(df, max_gap=30, method="linear"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns

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

def find_relative_startframe(first_file, filename):
    if not filename:
            return 0
    
    # Nur den Dateinamen ohne Path nehmen
    first_file = os.path.splitext(os.path.basename(first_file))[0]
    this_file = os.path.splitext(os.path.basename(filename))[0]
    
    # Extrahiere Zeit aus den Dateinamen
    first_time = first_file[11:19]  # Erwarte Format HH_MM_SS
    this_time = this_file[11:19]
    
    start_seconds = time_to_seconds(first_time)
    current_seconds = time_to_seconds(this_time)
    
    # Berechne die Differenz in Frames
    return (current_seconds - start_seconds) * FPS
    

def insert_into_working_df(working_df, files, filetype, metadata):

    for file in tqdm(files):
        df, ma = load_dlc_df(file, filetype)

        if not ma:
            df = add_individual_level(df, metadata["animal_info"])

        start_idx = find_relative_startframe(files[0], file)

        df = likelihood_filtering(df, filter_value=0.3)
        df = interpolate_with_max_gap(df, max_gap=30, method="linear")

        # Insert the data into the working DataFrame
        working_df.iloc[start_idx:start_idx + len(df), :] = df.values

    return working_df



def find_overlap():
    pass

def crop_working_df():
    pass

def create_metric_df():
    pass

def save_metadata():
    pass



# # # # # # # _________________________________________________________________________________________________________________

# Preprocessing  

# # # # # # # _________________________________________________________________________________________________________________


path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"

# h5 bzw. csv files finden
files, filetype = file_discovery(path)

# Experiment Metadata aus Namen extrahieren
metadata = get_metadata(files, filetype, FPS)

# ein leeres DF in Experimentlänge erstellen
dlc_df = create_working_df(files, filetype, metadata)

dlc_df = insert_into_working_df(dlc_df, files, filetype, metadata)


