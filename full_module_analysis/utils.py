import numpy as np
import pandas as pd
import os
from shapely.geometry import Point, Polygon
from config import FPS
import math
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

def mouse_center(df, scorer, individuals, bodyparts, min_bodyparts=None):
    """
    Compute the geometric center (centroid) of each mouse across frames
    using the x/y coordinates of DeepLabCut bodyparts.

    Parameters
    ----------
    df : pandas.DataFrame
        DLC multi-animal prediction dataframe with columns:
        (scorer, individual, bodypart, [x, y, likelihood]).
    scorer : str
        Name of the DLC scorer (df.columns.levels[0]).
    individuals : list of str
        List of individual mouse identifiers.
    bodyparts : list of str
        List of bodyparts belonging to each mouse.
    min_bodyparts : int, optional
        Minimum number of valid bodyparts required to compute a center.
        If None, defaults to ceil(n_bodyparts / 2).

    Returns
    -------
    centers : dict
        Dictionary:
        {
            individual_name: (center_x, center_y)
        }
        where center_x and center_y are 1D arrays of length n_frames,
        containing NaNs for frames with insufficient valid points.
    """

    centers = {}

    n_ind = len(individuals)
    n_frames = len(df)

    all_center_x = np.full((n_ind, n_frames), np.nan, dtype=float)
    all_center_y = np.full((n_ind, n_frames), np.nan, dtype=float)

    for i, ind in enumerate(individuals):
        # Extrahiere alle x- und y-Koordinaten dieses Individuums
        data = df.loc[:, (scorer, ind, bodyparts, ["x", "y"])].to_numpy()

        # x-Werte sind in Spalten [0, 2, 4, ...]
        arr_x = data[:, ::2]
        # y-Werte sind in Spalten [1, 3, 5, ...]
        arr_y = data[:, 1::2]

        # Y invertieren für "echte" geometrische Orientierung
        arr_y = -arr_y

        n_frames, n_bp = arr_x.shape

        # Defaults: mindestens die Hälfte der Bodyparts müssen valid sein
        if min_bodyparts is None:
            min_bodyparts = math.ceil(n_bp / 2)

        # Valid Masks (x und y müssen beide gültig sein)
        valid = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
        valid_counts = valid.sum(axis=1)

        # Warnungen über "Mean of empty slice" unterdrücken
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center_x = np.nanmean(arr_x, axis=1)
            center_y = np.nanmean(arr_y, axis=1)

        # Frames mit zu wenigen validen Punkten → hart auf NaN setzen
        too_few = valid_counts < min_bodyparts
        center_x[too_few] = np.nan
        center_y[too_few] = np.nan

        centers[ind] = (center_x, center_y)

        all_center_x[i] = center_x
        all_center_y[i] = center_y

    return all_center_x, all_center_y
     