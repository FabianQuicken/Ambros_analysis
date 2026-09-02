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

# # # # # # # _________________________________________________________________________________________________________________

# Variablen 

# # # # # # # _________________________________________________________________________________________________________________

path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
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
        df_last_file = pd.read_csv(rf'{last_file}')
      elif 'h5' in filetype:
        df_last_file = pd.read_hdf(rf'{last_file}')
      exp_duration_frames = np.zeros(experiment_dauer_in_s * FPS + len(df_last_file))

      return exp_duration_frames, startzeit, endzeit, date

def time_to_seconds(time_str):
        
        hours, minutes, seconds = map(int, time_str.split("_"))
        return hours * 3600 + minutes * 60 + seconds

def get_metadata(file_list, filetype, fps):
    # get recording length
    exp_duration_frames, startzeit, endzeit, date = calculate_experiment_length(file_list[0], file_list[-1])
    print(os.path.splitext(os.path.basename(file_list[0]))[1])
    filename = os.path.basename(file_list[0])
    name_parts = filename.split("_")
    print(name_parts)

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
        "experiment_length_frames": len(exp_duration_frames),
        "animal_info": animal_info,
        "experiment_info": experiment_info,
        "camera": camera
    }

def define_experiment_length(startzeit, endzeit, last_file):

      start_in_s = time_to_seconds(startzeit)
      ende_in_s = time_to_seconds(endzeit)

      experiment_dauer_in_s = ende_in_s - start_in_s 
      basename, filetype = os.path.splitext(last_file)
      # gesamte experimentdauer in frames
      if 'csv' in filetype:
        df_last_file = pd.read_csv(rf'{last_file}')
      elif 'h5' in filetype:
        df_last_file = pd.read_hdf(rf'{last_file}')
      exp_duration_frames = np.zeros(experiment_dauer_in_s * FPS + len(df_last_file))

      return exp_duration_frames

def create_working_df():
    pass

def likelihood_filtering():
    pass

def interpolation():
    pass

def find_relative_startframe():
    pass

def insert_into_working_df():
    pass

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

files, filetype = file_discovery(path)

metadata = get_metadata(files, filetype, FPS)

exp_len = define_experiment_length(metadata["start"], metadata["end"], files[-1])