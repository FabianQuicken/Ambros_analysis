


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

def posture_compactness(df, ind, bodyparts, center_x, center_y):
    """
    Calculate per-frame posture compactness from bodypart distances.

    Compactness is defined as the mean Euclidean distance between the mouse
    center and the selected bodyparts in each frame. Lower values indicate that
    the selected bodyparts are closer to the center, while higher values
    indicate a more extended posture.

    Parameters
    ----------
    df : pandas.DataFrame
        DeepLabCut multi-animal prediction dataframe with MultiIndex columns in
        the form ``(scorer, individual, bodypart, coord)``.
    scorer : str
        Name of the DLC scorer level to read from ``df``.
    ind : str
        Individual mouse identifier to analyze.
    bodyparts : sequence of str
        Bodyparts included in the compactness calculation.
    center_x, center_y : array-like
        Per-frame x and y coordinates of the mouse center. These arrays must be
        aligned to the rows of ``df``.

    Returns
    -------
    mean_dists : numpy.ndarray
        One-dimensional array with one compactness value per frame. Missing
        center or bodypart coordinates are ignored with ``nanmean``; frames
        where all bodypart distances are invalid remain NaN.
    """

    center_x = np.asarray(center_x, dtype=float)
    center_y = np.asarray(center_y, dtype=float)

    # Spaltenamen für x oder y Koordinaten bauen
    x_cols = [(ind, bp, "x") for bp in bodyparts]
    y_cols = [(ind, bp, "y") for bp in bodyparts]

    # holt alle x und y Koordinaten aus dem Dataframe (array mit n_frames, n_bodyparts)
    bp_x = df.loc[:, x_cols].to_numpy(dtype=float)
    bp_y = df.loc[:, y_cols].to_numpy(dtype=float)

    # vektorisierte from der euklidean distance berechnung
    dists = np.sqrt(
        (bp_x - center_x[:, None]) ** 2 +
        (bp_y - center_y[:, None]) ** 2
    )

    mean_dists = np.nanmean(dists, axis=1)

    return mean_dists

def calculate_bodylength(working_df, individual, bodyparts):

    """
    Calculates bodylength over coordinates of the nose -> tailbase axis
    """
    axis_bp = ["nose", "head_center", "dorsal1", "dorsal2", "dorsal3", "dorsal4", "tailbase"]

    bl = np.zeros(len(working_df))


    for bp in axis_bp:
        if bp not in bodyparts:
            print("Bodylength could not be calculated due to keypoints missing in the label scheme. Check 'axis_bp' in the function.")
            return bl

        

    for i in range(bl):

        sum_lengths = 0
        flag = False

        for i, bp in enumerate(axis_bp):

            if bp == axis_bp[-1]:
                        continue
            
            x1 = working_df.loc[i, (individual, bp, "x")].ravel().to_numpy()
            y1 = working_df.loc[i, (individual, bp, "y")].ravel().to_numpy()

            x2 = working_df.loc[i, (individual, axis_bp[i+1], "x")].ravel().to_numpy()
            y2 = working_df.loc[i, (individual, axis_bp[i+1], "y")].ravel().to_numpy()

            # versuchen den nächsten bp zu nehmen, falls einer in der Reihe fehlt
            if np.isnan(x2) and i+2 <= len(axis_bp):
                x2 = working_df.loc[i, (individual, axis_bp[i+2], "x")].ravel().to_numpy()
                y2 = working_df.loc[i, (individual, axis_bp[i+2], "y")].ravel().to_numpy()


            dist = euklidean_distance(x1, y1, x2, y2)

            if np.isnan(dist):
                flag = True
            else:
                sum_lengths += dist

        if flag:
            continue
        else:
            bl[i] = sum_lengths

    return bl

def get_orientation(front_x, front_y, rear_x, rear_y):
    """
    Calculate the per-frame orientation angle of a vector from rear to front.

    The orientation is computed with ``atan2(dy, dx)``, where ``dx`` and ``dy``
    are the coordinate differences between the front and rear points. This gives
    the direction of the vector ``rear -> front`` in the coordinate system of
    the input data.

    Parameters
    ----------
    front_x, front_y : array-like
        X and Y coordinates of the front point for each frame.
    rear_x, rear_y : array-like
        X and Y coordinates of the rear point for each frame.

    Returns
    -------
    orientations : numpy.ndarray
        One-dimensional array of orientation angles in degrees, with one value
        per frame. Values are in the range ``[-180, 180]``. Missing coordinates
        propagate to NaN.

    Notes
    -----
    The angle convention follows the input coordinate system: 0 degrees points
    along positive x, and the sign of positive rotation depends on whether y
    increases upward (Cartesian coordinates) or downward (image coordinates).
    """
    orientations = np.full(shape=(len(front_x)), fill_value=np.nan, dtype=float)

    for i in range(len(orientations)):
        dx = front_x[i] - rear_x[i]
        dy = front_y[i] - rear_y[i]

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        orientations[i] = angle_deg
    
    return orientations

def get_all_traj(x, y, len_thr=FPS):

    def plot_trajectory_segment(x, y, e, ex, close_after=2.0):
        """
        Plots a trajectory segment x[e:ex+1], y[e:ex+1] and closes automatically.

        Parameters
        ----------
        x, y : array-like
            Coordinate arrays (same length).
        e : int
            Entry index (start).
        ex : int
            Exit index (end, inclusive).
        close_after : float
            Seconds after which the plot closes automatically.
        """

        x_seg = np.asarray(x[e:ex+1])
        y_seg = np.asarray(y[e:ex+1])

        if x_seg.size == 0:
            print("[plot_trajectory_segment] Empty segment, nothing to plot.")
            return

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(x_seg, y_seg, "-o", markersize=3)
        ax.scatter(x_seg[0], y_seg[0], c="green", label="start", zorder=3)
        ax.scatter(x_seg[-1], y_seg[-1], c="red", label="end", zorder=3)

        ax.set_aspect("equal")
        ax.set_xlim(0,2000)
        ax.set_ylim(0,-1100)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Trajectory {e}:{ex}")
        ax.legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(close_after)
        plt.close(fig)

    all_traj = []
    traj_slices = {}
    start_len_traj = []



    x = np.round(x, decimals=0)
    y = np.round(y, decimals=0)

    # x und y arrays müssen gleich lang sein
    if len(x) != len(y):
        raise ValueError("x and y arrays have different length.")
        
    # testen ob daten da sind 
    valid = np.isfinite(x) & np.isfinite(y)
    # mind 1 sekunde insgesamt getrackt?
    if valid.sum() < FPS:
        return None

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
    paired = []
    dis_ptr = 0
    traj_start = []
    traj_end = []
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
        paired.append((a, d))
        traj_start.append(a)
        traj_end.append(d)


    # slice indices speichern
    traj_slices = paired
    for a, d in paired:
        t = (x[a:d+1], y[a:d+1])
        t_len = len(t[0])
        all_traj.append(t)
        start_len_traj.append((a, t_len))
    #plot_trajectory_segment


    return traj_start, traj_end

def get_theta(a, b1, b2, signed_angle=True):
    """
    Computes the angle between two direction vectors defined by a common
    intersection point.

    The function calculates the angle between the vectors a→b1 and a→b2,
    which can be interpreted as a change in movement or body orientation
    between two time points.

    Depending on the parameter `signed_angle`, the function returns either:
    - the unsigned (inner) angle in the range [0°, 180°], or
    - the signed angle in the range (-180°, 180°], encoding left/right turns.

    Parameters
    ----------
    a : array-like of shape (2,)
        Intersection point (x, y), e.g. the tail base of a mouse.
        This point defines the origin of both direction vectors.
    b1 : array-like of shape (2,)
        Endpoint of the first direction vector (a→b1),
        e.g. the front of the animal at time point t.
    b2 : array-like of shape (2,)
        Endpoint of the second direction vector (a→b2),
        e.g. the front of the animal at time point t+1.
    signed_angle : bool, default=True
        If True, returns a signed angle using atan2(cross, dot):
            - positive values indicate a left (counter-clockwise) turn
            - negative values indicate a right (clockwise) turn
        If False, returns the unsigned inner angle using arccos(dot),
        always in the range [0°, 180°].

        Note: The interpretation of left/right assumes a standard
        Cartesian coordinate system (x right, y up). For image-based
        coordinate systems (x right, y down), the sign needs to be inverted.

    Returns
    -------
    angle : float
        Angle between vectors a→b1 and a→b2 in degrees.
        Range depends on `signed_angle`:
        - signed_angle=True  → (-180°, 180°]
        - signed_angle=False → [0°, 180°]

    Notes
    -----
    - If either vector has zero length (a == b1 or a == b2), the angle
      is undefined and a warning is raised.
    - The signed angle is numerically more stable and should be preferred
      for trajectory analysis, turning-angle statistics, and left/right
      bias estimation.
    - The unsigned angle is useful when only the magnitude of direction
      change is of interest.
    """
    # hilfsfunktion für Vektorbetrag 2D
    def vektorbetrag(v):
        return math.sqrt(v[0]**2 + v[1]**2)
    
    # punkte als array für einfachere rechnungen
    a = np.asarray(a, float)
    b1 = np.asarray(b1, float)
    b2 = np.asarray(b2, float)

    # vektoren berechnen ("spitze minus fuss")
    v1 = b1 - a
    v2 = b2 - a

    # wir wollen den inneren Winkel θ Theta berechnen, der von beiden Vektoren eingeschlossen wird
    # cos θ = Skalarprodukt v1 * v2 dividied by |v1| * |v2|
    # daraus folgt θ = cos^-1 ((Skalarprodukt v1 * v2) / (|v1| * |v2|))

    # betrag v1 und v2:
    betrag1 = vektorbetrag(v1)
    betrag2 = vektorbetrag(v2)

    # skalarprodukt v1 * v2:
    sp = v1[0] * v2[0] + v1[1] * v2[1]

    # Nullvektoren info
    if betrag1 == 0 or betrag2 == 0:
        warnings.warn(
            "Zero-length direction vector encountered (a == b1 or a == b2). "
            "Angle is undefined; returning NaN.",
            RuntimeWarning
            )
        return np.nan
        
    
    if signed_angle:
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        radians = np.arctan2(cross, sp)  # radians in (-pi, pi]
        return np.round(np.degrees(radians), decimals=0)


    # tetha berechnen
    cos_theta = sp / (betrag1 * betrag2)
    # float rundungsfehler könnten 1.00000000000001 ergeben, was zu NaN Werten führen könnte
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    radians = np.arccos(cos_theta)
    return np.round(np.degrees(radians), decimals=0)


def theta_analysis(front_x, front_y, rear_x, rear_y, start_idxs, end_idxs, stepsize = 10):
    """
    """
    thetas = []


    discarded_frames = 0
    length_all_traj = 0
        
    for e, ex in zip(start_idxs, end_idxs):
            
        step = 0
            
        while (e + step + stepsize) <= ex:
                
            current_frame = e + step
            comparison_frame = e + step + stepsize
            step += stepsize
            length_all_traj += stepsize

            rx = rear_x[current_frame]
            ry = rear_y[current_frame]
            fx1 = front_x[current_frame]
            fy1 = front_y[current_frame]
            fx2 = front_x[comparison_frame]
            fy2 = front_y[comparison_frame]

            if not np.isfinite(rx) or not np.isfinite(fx1) or not np.isfinite(fx2):
                discarded_frames += stepsize
                continue
                
            t = get_theta(a = [rx, ry], b1 = [fx1, fy1], b2 = [fx2, fy2])
            thetas.append(t)
            
    if discarded_frames > 0:
        if int(discarded_frames/len(start_idxs)) > 100:
            print(f"\nWarning: For Theta analysis, {discarded_frames} out of a total of {length_all_traj} frames have been discarded due to missing tracking during trajectory for {ind}."
                f"\nAn average of {int(discarded_frames/len(start_idxs))} frames was discarded per visit."
                "\nDepending on Trajectory length, thus might be an unusual high amount, \nchecking the tracking of the video is recommended.")
            

    return thetas

def arc_chord_ratio(traj_start, traj_end, x_coords, y_coords, fragmentsize_divisor=3, speed_thr=2.0):
    """

    """
    if fragmentsize_divisor <= 0:
        raise ValueError("fragmentsize_divisor must be positive.")
    if speed_thr < 0:
        raise ValueError("speed_thr must be non-negative.")

    for start, end in zip(traj_start, traj_end):



        x = x_coords[start:end]
        y = y_coords[start:end]

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        tortuosity = []
    
        t_len = len(x)

        fragment_size = int(FPS/fragmentsize_divisor)
        if fragment_size < 2:
            raise ValueError("fragmentsize_divisor creates fragments shorter than 2 frames.")


        # int um abzurunden
        n_fragments = int(t_len / fragment_size)

        fragments = []

        counter = 0
        f_window = 0
        while counter < n_fragments:
            x_y = (x[f_window:f_window+fragment_size], y[f_window:f_window+fragment_size])
            fragments.append(x_y)
            f_window += fragment_size
            counter += 1
        
        

        for f in fragments:
            valid = np.isfinite(f[0]) & np.isfinite(f[1])
            if valid.sum() < 2:
                #print("fragment_skipped")
                continue

            f_x = f[0][valid]
            f_y = f[1][valid]

            start_end_dist = euklidean_distance(x1=f_x[0], y1=f_y[0], x2 = f_x[-1], y2 = f_y[-1])
            if not np.isfinite(start_end_dist) or start_end_dist <= 0:
                #print("fragment_skipped du to start _end dist")
                continue

            # hier summe der dist values berechnen
            distance_values = np.zeros((len(f_x)-1))
            for i in range(len(f_x)-1):
                distance_values[i] = euklidean_distance(x1=f_x[i],
                                                        y1=f_y[i],
                                                        x2=f_x[i+1],
                                                        y2=f_y[i+1])
            curve_length = np.sum(distance_values)
            if not np.isfinite(curve_length):
                #print("fragment_skipped du to curve length")
                continue

            # fragmente unter speed threshold (= Maus ist immobile) fallen raus
            duration_s = (len(f_x) - 1) / FPS
            speed = (curve_length / PIXEL_PER_CM) / duration_s
            if speed < speed_thr:
                #print("fragment_skipped du to speed")
                continue

            tortuosity.append(curve_length / start_end_dist)
    
    #print(f"\nT = {np.mean(tortuosity)}")
    if len(tortuosity) == 0:
        return np.nan, [np.nan]

    return np.mean(tortuosity), tortuosity

def social_inv(working_df, nose_x, nose_y, individual, individuals, bodyparts, face_bp, anogenital_bp, body_bp):

    # mögliche individuals die investigated werden
    individuals.pop(individual)

    face_inv = np.full(shape=len(nose_x), fill_value=np.nan)
    body_inv = face_inv.copy()
    anogenital_inv = face_inv.copy()

    for ind in individuals:
        
        for i in range(len(nose_x)):

            for bp in face_bp:
                
                dist = euklidean_distance(
                    x1 = nose_x[i],
                    y1 = nose_y[i],
                    x2 = working_df.loc[i, (ind, bp, "x")].to_numpy(),
                    y2 = working_df.loc[i, (ind, bp, "y")].to_numpy()
                    )
                if dist >= 20:
                    flag = True

            


    

    return face_inv, body_inv, anogenital_inv




def save_metrics(metrics_df, folder_path):

    # Experiment-Datei suchen
    experiment_files = [
        file for file in glob.glob(os.path.join(folder_path, "*.h5"))
        if "experiment" in os.path.basename(file).lower()
    ]

    if len(experiment_files) == 0:
        raise FileNotFoundError(
            f"No experiment file found in {folder_path}"
        )

    if len(experiment_files) > 1:
        raise ValueError(
            f"Multiple experiment files found in {folder_path}: "
            f"{experiment_files}"
        )

    experiment_file = experiment_files[0]

    # Aktualisiertes DataFrame unter demselben Key speichern
    metrics_df.to_hdf(
        experiment_file,
        key="metrics",
        mode="a",
        format="fixed"
    )

    print(f"Metrics saved to: {experiment_file}")

# # # # # # # _________________________________________________________________________________________________________________

# Analysis Main 

# # # # # # # _________________________________________________________________________________________________________________


folder_path = r"C:\Users\Fabian\Desktop\Transfer\analysis_testing\metric_analysis"

def main_analysis(individuals_to_remove,
                  bodyparts_to_remove,
                  folder_path
                ):

    

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
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "nose_x", nose_x)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "nose_y", nose_y)


        # time visible
        visible = (~np.isnan(center_x)).astype(int)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "visible", visible)

        # speed, inklusive moving average und smoothing um jitter entgegen zu wirken
        speed = distance_travelled_arraybased(center_x, center_y)
        speed = moving_average(speed, window=int(FPS/2))
        # speed wird unter threshold auf 0 gesetzt (Maus ist immobile, Bewegung ist getrieben von Keypoint Jitter)
        speed = remove_distance_jitter(speed, thrsh=4)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "speed", speed)

        # cumulative distance travelled
        cumdist = np.nancumsum(speed)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "cumdist", cumdist)

        # immobile
        immobile = np.where(speed == 0, 1, 0)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "immobile", immobile)

        # immobile starts
        immobile_padded = np.r_[0, immobile, 0]
        immobile_diff = np.diff(immobile_padded)
        immobile_starts = np.where(immobile_diff == 1)[0]
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "immobile_starts", immobile_starts)

        # immobile bout lengths
        immobile_ends = np.where(immobile_diff == -1)[0]
        immobile_bout_lengths = immobile_ends - immobile_starts
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "immobile_bout_lengths", immobile_bout_lengths)

        # acceleration
        acc, acc_cm_s = acceleration(speed, FPS, PIXEL_PER_CM)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "acceleration", acc)

        # speed events
        count_speed_events, speed_event_frame_idx = acceleration_events(acc)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "speed_event_count", count_speed_events)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "speed_event_index", speed_event_frame_idx)

        # all trajectories (not regarding if a trajectory starts in the "entry area" of a module)
        all_traj, traj_slices, len_traj, start_traj = get_all_traj(center_x, center_y)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "trajectory_start", start_traj)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "trajectory_length", len_traj)

        # posture compactness
        pc = posture_compactness(working_df, individual, bodyparts, center_x, center_y)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "posture_compactness", pc)

        # body length
        bl = calculate_bodylength(working_df, individual, bodyparts)
        print("calculate_bodylength noch testen")
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "body_length", bl)

        # orientations
        front_x, front_y = mouse_center(working_df, individual, ["shoulder_left", "shoulder_right", "dorsal_1"], 3)
        rear_x, rear_y = mouse_center(working_df, individual, ["hip_left", "hip_right", "dorsal_4"], 3)

        orientations = get_orientation(
            front_x=front_x,
            front_y=front_y,
            rear_x=rear_x,
            rear_y=rear_y
        )

        print("orientations noch testen")
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "orientations", orientations)

        # trajectories
        traj_start, traj_end = get_all_traj(x=center_x, y=center_y, len_thr=30)
        print("traj noch testen")
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "center_traj_startidx", traj_start)
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "center_traj_endidx", traj_end)

        # thetas
        stepsize = 10
        thetas = theta_analysis(
            front_x = front_x,
            front_y = front_y,
            rear_x = rear_x,
            rear_y = rear_y,
            start_idxs = traj_start,
            end_idxs = traj_end,
            stepsize = stepsize
        )
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, f"thetas [stepsize {stepsize} frames]", thetas)
        print("thetas noch testen")

        # arc chord
        _, arc_chords = arc_chord_ratio(
            traj_start=traj_start,
            traj_end=traj_end,
            x_coords=center_x,
            y_coords=center_y,
            fragmentsize_divisor=3,
            speed_thr=3
        )
        metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "arc_chord", arc_chords)
        print("arc_chords testen")

        # social investigation


    # speichern

    save_metrics(metrics_df, folder_path)







