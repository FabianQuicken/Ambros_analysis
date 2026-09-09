


# # # # # # # _________________________________________________________________________________________________________________

# Imports

# # # # # # # _________________________________________________________________________________________________________________

import glob as glob
import os

import numpy as np
import pandas as pd
import math

from tqdm import tqdm
import warnings

from variables import FPS, PIXEL_PER_CM

# # # # # # # _________________________________________________________________________________________________________________

# Funktionen

# # # # # # # _________________________________________________________________________________________________________________

def load_experiment_data(folder_path):

    expected_working_df_headings = ["scorer", "individuals", "bodyparts", "coords"]

    expected_metrics_df_headings = ["name", "individuals", "metrics"]

    # read in correct file
    possible_files = glob.glob(os.path.join(folder_path, "*.h5"))
    experiment_file = None
    for file in possible_files:
        if "experiment" in os.path.basename(file):
            experiment_file = file
        else:
            raise ValueError(f"No experiment file found in {folder_path}")

    # read in dlc data and check if valid
    working_df = pd.read_hdf(experiment_file, key="pose")
    for heading in working_df.columns.names:
        if heading not in expected_working_df_headings:
            raise ValueError(f"Unexpected column heading found in dlc dataframe of {experiment_file}: {heading}")

    working_df = working_df.droplevel("scorer", axis=1)

    # read in empty metric df and check if valid
    metrics_df = pd.read_hdf(experiment_file, key="metrics")

    for heading in metrics_df.columns.names:
        if heading not in expected_metrics_df_headings:
            raise ValueError(f"Unexpected column heading found in metrics dataframe of {experiment_file}: {heading}")
        
    return working_df, metrics_df

def drop_dlc_columns(df, individuals=None, bodyparts=None):
    if individuals is not None:
        df = df.drop(columns=individuals, level="individuals")

    if bodyparts is not None:
        df = df.drop(columns=bodyparts, level="bodyparts")

    df.columns = df.columns.remove_unused_levels()

    return df

def add_metric_to_metric_df(metrics_df, name, individual, metric_name, metric_data):

    metrics_df = metrics_df.copy()

    if len(metric_data) != len(metrics_df):
        if len(metric_data) > len(metrics_df):
            raise ValueError(f"Unexpected metric length {len(metric_data)} does not fit into metrics df length of {len(metrics_df)}")
        else:
            array = np.full(len(metrics_df), np.nan)
            array[0:len(metric_data)] = metric_data
            metric_data = array


    metrics_df[(name, individual, metric_name)] = metric_data

    return metrics_df

def calculate_mean_likelihood(working_df, metrics_df, ind, bodyparts):

    likelihoods = working_df.loc[:, (ind, bodyparts, "likelihood")].to_numpy()

    # mean pro frame
    mean_likelihood = np.nanmean(likelihoods, axis=1)
    
    # speichern
    mean_likelihood = np.round(mean_likelihood, decimals=2)

    return mean_likelihood

def mouse_center(df, individual, bodyparts, min_bodyparts=None):
    """
    Compute per-frame mouse centers from DeepLabCut bodypart coordinates.

    For each individual, the center is calculated as the mean x/y position of
    the available bodyparts in each frame. Frames with fewer valid bodyparts
    than ``min_bodyparts`` are set to NaN. Y coordinates are inverted when the
    input values appear to be in image coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DeepLabCut multi-animal prediction dataframe with MultiIndex columns in
        the form ``(scorer, individual, bodypart, coord)``. Only ``"x"`` and
        ``"y"`` coordinates are used.
    scorer : str
        Name of the DLC scorer level to read from ``df``.
    individuals : sequence of str
        Individual mouse identifiers. The output rows follow this order.
    bodyparts : sequence of str
        Bodyparts used to estimate each mouse center.
    min_bodyparts : int, optional
        Minimum number of valid bodyparts required for a frame to receive a
        center value. If None, defaults to ``ceil(n_bodyparts / 2)``.

    Returns
    -------
    all_center_x : numpy.ndarray
        Array of shape ``(n_individuals, n_frames)`` containing center x
        coordinates.
    all_center_y : numpy.ndarray
        Array of shape ``(n_individuals, n_frames)`` containing center y
        coordinates.
    """

    # Extrahiere alle x- und y-Koordinaten dieses Individuums
    data = df.loc[:, (individual, bodyparts, ["x", "y"])].to_numpy()

    # x-Werte sind in Spalten [0, 2, 4, ...]
    arr_x = data[:, ::2]
    # y-Werte sind in Spalten [1, 3, 5, ...]
    arr_y = data[:, 1::2]

    # Y invertieren für "echte" geometrische Orientierung, falls noch nicht geschehen
    if np.max(arr_y) > 0:
        print("Inverting y")
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



    return center_x, center_y

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

def distance_travelled_arraybased(x_arr,y_arr):
    """
    Compute frame-to-frame Euclidean distances based on center coordinates.

    The function calculates the distance travelled between consecutive frames
    using paired x and y coordinate arrays. The output array has length
    len(x_arr) - 1, where each entry corresponds to the distance between
    frame i and i+1.

    Parameters
    ----------
    x_arr : array-like
        1D array of x-coordinates (e.g. center positions per frame).
    y_arr : array-like
        1D array of y-coordinates (same length as x_arr).

    Returns
    -------
    distance_values : np.ndarray
        1D array of Euclidean distances between consecutive frames.
    """
    distance_values = np.zeros((len(x_arr))-1)
    prev_coords = None
    for i, (x, y) in enumerate(zip(x_arr,y_arr)):
        if prev_coords:
            
            dist = euklidean_distance(x1=x,
                                    y1=y,
                                    x2=prev_coords[0],
                                    y2=prev_coords[1]
                                    )
            # falls dist values einen unlogischen Schwellenwert überschreiten, wird der letzte gültige Wert genommen
            # falls es noch keine vorherigen Werte gibt, wird 0 eingesetzt
            if i > 2 and dist > 200:
                dist = distance_values[i-2]
            elif dist > 200:
                dist = 0
            
            distance_values[i-1] = dist
            
                
        prev_coords = (x,y)    

    return distance_values

def remove_distance_jitter(dist_values, thrsh = 4):
    # dist values kuratieren um jitter rauszurechnen
    curated_dist = np.zeros(len(dist_values))
    for i, val in enumerate(dist_values):
        if val > thrsh:
            curated_dist[i] = val - thrsh
        elif np.isnan(val):
             curated_dist[i] = np.nan
    return curated_dist

def moving_average(data, window=15):
    """
    Centered moving average smoothing.

    Parameters
    ----------
    data : array-like
    window : int
        Window size in samples

    Returns
    -------
    smoothed : ndarray
        Same length as input
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')

def acceleration(dist_values, fps=FPS, px_per_cm=PIXEL_PER_CM):

    v = dist_values # speed px pro frame
    a = np.diff(v) # acceleration pro frame

    a_cms2 = a / PIXEL_PER_CM * FPS # umrechnung in cm/s2
    return a, a_cms2

def acceleration_events(a, acc_thr = 5):
    count = 0
    acc_events = np.where(a >= acc_thr, 1, 0)
    acc_events[0] = 0
    diffs = np.diff(acc_events)
    acc_eventstarts = np.where(diffs == 1)[0]
    for value in diffs:
        if  value == 1:
            count += 1

    return count, list(acc_eventstarts)

def get_all_traj(x, y, len_thr=FPS):



        # x und y arrays müssen gleich lang sein
        if len(x) != len(y):
            raise ValueError("x and y arrays have different length.")
        
        # testen ob daten da sind für das jeweilige individum, sonst nächstes Ind
        valid = np.isfinite(x) & np.isfinite(y)

        # um Randfälle (Maus wird schon im ersten Frame getrackt bzw noch im letzten) zu berechnen:
        if valid[0]:
            valid[0] = False
        if valid[-1]:
            valid[-1] = False

        # finden wo coordinaten neu getrackt werden
        diff = np.diff(valid.astype(int))
        appearances = np.where(diff == 1)[0] + 1
        disappearances = np.where(diff == -1)[0] + 1

        # wenn alles klappt, müsste es für jeden entry einen exit geben
        if len(appearances) != len(disappearances):
            print(f"\nEntry number ({len(appearances)}) and exit number dont match ({len(disappearances)})")
        appearances.sort()
        disappearances.sort()
        # robustes Pairing: für jeden entry den nächsten exit danach
        traj_slices = []
        dis_ptr = 0
        for a in appearances:
            while dis_ptr < len(disappearances) and disappearances[dis_ptr] <= a:
                dis_ptr += 1
            if dis_ptr >= len(disappearances):
                raise ValueError(f"Entry at {a} has no subsequent exit.")
            d = disappearances[dis_ptr]-1
            dis_ptr += 1
            # trajectories unter 1s sind vermutlich fake
            if (d-a + 1) < len_thr:
                continue
            traj_slices.append((a, d))


        # slice indices speichern
        all_traj = []
        len_traj = []
        start_traj = []
        for a, d in traj_slices:
            t = (x[a:d+1], y[a:d+1])
            t_len = len(t[0])
            all_traj.append(t)
            len_traj.append(t_len)
            start_traj.append(a)
            #plot_trajectory_segment(x=x, y=y, e=a, ex=d)

        return all_traj, traj_slices, len_traj, start_traj

# # # # # # # _________________________________________________________________________________________________________________

# Analysis Main 

# # # # # # # _________________________________________________________________________________________________________________

folder_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2\metric_analysis"
folder_path = r"C:\Users\Fabian\Desktop\Transfer\analysis_testing\metric_analysis"

individuals_to_remove = None
bodyparts_to_remove = ["food1"]

working_df, metrics_df = load_experiment_data(folder_path)

working_df = drop_dlc_columns(working_df, individuals_to_remove, bodyparts_to_remove)

name = metrics_df.columns.get_level_values("name").unique().item()

individuals = working_df.columns.get_level_values("individuals").unique()

bodyparts = working_df.columns.get_level_values("bodyparts").unique()






# einzene Metrics werden pro Individual erstellt und ins metric dataframe eingefügt
for individual in individuals:

    working_df = working_df.copy()
    #working_df.loc[:, (individual, bodyparts, ["y"])] *= -1

    # mean likelihood als Maß für die Tracking Qualität
    mean_lh = calculate_mean_likelihood(working_df, metrics_df, individual, bodyparts)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "mean_likelihood", mean_lh)

    # mouse center als mean aller koordinaten
    center_x, center_y = mouse_center(working_df, individual, bodyparts, min_bodyparts=len(bodyparts)/3)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "center_x", center_x)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "center_y", center_y)

    # nose für investigation metrics rausholen
    nose_x = working_df.loc[:, (individual, "nose", "x")].to_numpy()
    nose_y = working_df.loc[:, (individual, "nose", "y")].to_numpy()

    # time visible
    visible = (~np.isnan(center_x)).astype(int)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "visible", visible)

    # speed, inklusive moving average und smoothing um jitter entgegen zu wirken
    speed = distance_travelled_arraybased(center_x, center_y)
    speed = moving_average(speed, window=int(FPS/2))
    # speed wird unter threshold auf 0 gesetzt (Maus ist immobile, Bewegung ist getrieben von Keypoint Jitter)
    speed = remove_distance_jitter(speed, thrsh=4)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "speed", speed)

    # immobile
    immobile = np.where(speed == 0, 1, 0)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "immobile", immobile)

    # acceleration
    acc, acc_cm_s = acceleration(speed, FPS, PIXEL_PER_CM)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "acceleration", acc)

    # speed events
    count_speed_events, speed_event_frame_idx = acceleration_events(acc)

    # all trajectories (not regarding if a trajectory starts in the "entry area" of a module)
    all_traj, traj_slices, len_traj, start_traj = get_all_traj(center_x, center_y)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "trajectory_start", start_traj)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "trajectory_length", len_traj)


  







