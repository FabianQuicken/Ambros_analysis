import numpy as np
import pandas as pd
import os
from config import FPS


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



def fill_missing_values(array):
        """
        Replacing np.nans with a linear interpolation. Takes and returns an array.
        """
        nan_indices = np.isnan(array)
        array[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), array[~nan_indices])
        return array



# Wandle Zeit in Sekunden seit Mitternacht um
def time_to_seconds(time_str):
        hours, minutes, seconds = map(int, time_str.split("_"))
        return hours * 3600 + minutes * 60 + seconds



def convert_videostart_to_experiment_length(first_file, filename):
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

        # Berechne die Differenz in Sekunden
        return current_seconds - start_seconds

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

      # gesamte experimentdauer in frames
      df_last_file = pd.read_csv(last_file)
      exp_duration_frames = np.zeros(experiment_dauer_in_s * FPS + len(df_last_file))

      return exp_duration_frames, startzeit, endzeit, date