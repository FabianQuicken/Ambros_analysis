import numpy as np
import pandas as pd
import os
from shapely.geometry import Point, Polygon
from config import FPS
import warnings

def create_polygon(polygon_coords=list):
    return Polygon(polygon_coords)

def create_point(x, y):
    return Point(x, y)

def is_point_in_polygon(polygon, point):
    return polygon.contains(point)

def shrink_rectangle(coords, scale):
      coords = np.array(coords)
      center = coords.mean(axis=0)
      shrunk_coords = center + (coords - center) * scale
      return shrunk_coords.tolist()

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
      basename, filetype = os.path.splitext(last_file)
      # gesamte experimentdauer in frames
      if 'csv' in filetype:
        df_last_file = pd.read_csv(rf'{last_file}')
      elif 'h5' in filetype:
        df_last_file = pd.read_hdf(rf'{last_file}')
      exp_duration_frames = np.zeros(experiment_dauer_in_s * FPS + len(df_last_file))

      return exp_duration_frames, startzeit, endzeit, date

def mouse_center(df, scorer, individuals, bodyparts, min_bp):
     n_frames = len(df)
     n_ind = len(individuals)

     all_center_x = np.empty((n_ind, n_frames), dtype=float) 
     all_center_y = np.empty((n_ind, n_frames), dtype=float)

     for i, ind in enumerate(individuals):
        # x arrays für alle bodyparts eines individuums 
        arr_x = df.loc[:, (scorer, ind, bodyparts, ["x", "y"])].values[:,::2]
        # y arrays für alle bodyparts eines individuums
        arr_y = df.loc[:, (scorer, ind, bodyparts, ["x", "y"])].values[:,1::2] * -1

        valid = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
        valid_counts = valid.sum(axis=1)   

        # center der maus als mean aller punkte bestimmen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centroid_x = np.nanmean(arr_x, axis=1)
            centroid_y = np.nanmean(arr_y, axis=1)

        # Wenn kein Minimum an Bodyparts getracfkt wird, wird der center wert auf nan gesetzt
        too_few = valid_counts < min_bp
        centroid_x[too_few] = np.nan
        centroid_y[too_few] = np.nan

        all_center_x[i] = centroid_x
        all_center_y[i] = centroid_y

     return all_center_x, all_center_y


