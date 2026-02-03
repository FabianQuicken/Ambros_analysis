import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from preprocessing import likelihood_filtering, likelihood_filtering_nans
from utils import euklidean_distance, fill_missing_values, shrink_rectangle, is_point_in_polygon, create_point
from config import PIXEL_PER_CM, ARENA_COORDS_TOP1, ARENA_COORDS_TOP2, FPS, ENTER_ZONE_COORDS

import matplotlib.pyplot as plt

def arc_chord_ratio(trajectory, fragmentsize_divisor = 3, speed_thr = 2):
    
    
    x = np.asarray(trajectory[0], dtype=float)
    y = np.asarray(trajectory[1], dtype=float)
    
    t_len = len(x)

    fragment_size = int(FPS/fragmentsize_divisor)


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
    
    tortuosity = []
    for f in fragments:
        start_end_dist = euklidean_distance(x1=f[0][0], y1=f[1][0], x2 = f[0][-1], y2 = f[1][-1])
        # hier summe der dist values berechnen
        distance_values = np.zeros((len(f[0])-1))
        for i in range(len(f[0])-1):
            distance_values[i] = euklidean_distance(x1=f[0][i],
                                                    y1=f[1][i],
                                                    x2=f[0][i+1],
                                                    y2=f[1][i+1])
        curve_length = sum(distance_values)

        # fragmente unter speed threshold (= Maus ist immobile) fallen raus
        speed = curve_length / (1/fragmentsize_divisor)  / PIXEL_PER_CM # cm/s
        if speed > speed_thr:
            continue

        tortuosity.append(curve_length / start_end_dist)
    
    #print(f"\nT = {np.mean(tortuosity)}")
    return np.mean(tortuosity)




def entry_exit_trajectories(entry_polygon, x_arrs, y_arrs, individuals, plot=False):
    """
    Detects arena entry and exit events for multiple individuals based on tracking
    data and an entry-zone polygon, and extracts the corresponding trajectory
    segments.

    For each individual, the function identifies contiguous tracking bouts
    (appearance → disappearance) in the coordinate time series. A bout is
    classified as an arena entry if the first valid coordinate lies within the
    given entry polygon, and as an arena exit if the last valid coordinate before
    disappearance lies within the same polygon. Entry and exit events are paired
    robustly in temporal order.

    For each valid entry–exit pair, the function:
    - Marks the entry and exit frame in binary indicator arrays
    - Stores the x/y trajectory segment between entry and exit (inclusive)
    - Optionally visualizes each trajectory segment for manual inspection

    Frames outside detected arena visits are filled with NaN in the trajectory
    arrays.

    Parameters
    ----------
    entry_polygon : array-like or polygon object
        Polygon defining the entry/exit zone (same coordinate system as x/y data).
    x_arrs : list of array-like
        List of x-coordinate arrays, one per individual. All arrays must have
        identical length (number of frames).
    y_arrs : list of array-like
        List of y-coordinate arrays, one per individual. Must match x_arrs in
        shape and coordinate system.
    individuals : list
        Identifiers (e.g. IDs or names) corresponding to each individual trajectory.

    Returns
    -------
    mice_enter : ndarray, shape (n_individuals, n_frames)
        Binary array marking arena entry frames (1 = entry, 0 = otherwise).
    mice_exit : ndarray, shape (n_individuals, n_frames)
        Binary array marking arena exit frames (1 = exit, 0 = otherwise).
    traj_x : ndarray, shape (n_individuals, n_frames)
        X-coordinates during arena visits; NaN outside entry–exit intervals.
    traj_y : ndarray, shape (n_individuals, n_frames)
        Y-coordinates during arena visits; NaN outside entry–exit intervals.
    all_traj : list of tuples (x,y) of ndarray
        contains all trajectories of all individuals 

    Raises
    ------
    ValueError
        If x and y arrays differ in length, if coordinate systems between data and
        polygon do not match (e.g. inverted y-axis), or if an entry cannot be paired
        with a subsequent exit.

    Notes
    -----
    - Entry and exit detection relies on transitions in valid tracking data
    (NaN → valid for entry, valid → NaN for exit).
    - The first and last frame are ignored as potential transitions to avoid
    edge artifacts.
    - This function assumes that all arena visits must pass through the
    entry polygon.
    """

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
    
    n_ind = len(individuals)
    n_frames = len(x_arrs[0])

    mice_enter = np.zeros((n_ind, n_frames), dtype=np.uint8)
    mice_exit  = np.zeros((n_ind, n_frames), dtype=np.uint8)

    traj_x = np.full((n_ind, n_frames), np.nan, dtype=float)
    traj_y = np.full((n_ind, n_frames), np.nan, dtype=float)

    all_traj = []

    for index, ind in enumerate(individuals):
        x = x_arrs[index]
        y = y_arrs[index]



        # x und y arrays müssen gleich lang sein
        if len(x) != len(y):
            raise ValueError("x and y arrays have different length.")

        # testen ob daten da sind für das jeweilige individum, sonst nächstes Ind
        valid = np.isfinite(x) & np.isfinite(y)
        # mind 1 sekunde insgesamt getrackt?
        if valid.sum() < FPS:
            continue

        # ggf werden die Daten transformiert (y-Invertiertung)
        # y der polygone und der Daten müssen also gleiches Vorzeichen haben
        example_y = np.nan
        for coord in y:
            if not np.isnan(coord):
                example_y = coord
                break
        polygon_y = ENTER_ZONE_COORDS[0][1]
        if np.sign(example_y) != np.sign(polygon_y):
            raise ValueError(
                f"\nY-axis mismatch detected:\n"
                f"Data y example: {example_y}\n"
                f"Polygon y: {polygon_y}\n"
                f"Make sure both use the same coordinate system (inverted or not).\n"
            )
        
        # um Randfälle (Maus wird schon im ersten Frame getrackt bzw noch im letzten) zu berechnen:
        if valid[0]:
            valid[0] = False
        if valid[-1]:
            valid[-1] = False

        # alle entry_polygon entries finden (sind arena entries)

        # finden wo coordinaten neu getrackt werden
        diff = np.diff(valid.astype(int))
        appearances = np.where(diff == 1)[0] + 1

        # neues tracking muss in entry polygon passieren
        entries = []
        for idx in appearances:
            point = create_point(x[idx], y[idx])
            if is_point_in_polygon(polygon=entry_polygon, point=point):
                entries.append(idx)
        entries = np.array(entries, dtype='int')

        # alle entry_polygon exits finden (sind arena exits)   
        disappearances = np.where(diff == -1)[0] + 1
        exits = []
        for idx in disappearances:
            point = create_point(x[idx-1], y[idx-1])
            if is_point_in_polygon(polygon=entry_polygon, point=point):
                exits.append(idx-1)
        exits = np.array(exits, dtype='int')

        # wenn alles klappt, müsste es für jeden entry einen exit geben
        if len(entries) != len(exits):
            print(f"\nEntry number ({len(entries)}) and exit number dont match ({len(exits)})")
        

        entries.sort()
        exits.sort()
        
        # robustes Pairing: für jeden entry den nächsten exit danach
        paired = []
        ex_ptr = 0
        for e in entries:
            while ex_ptr < len(exits) and exits[ex_ptr] <= e:
                ex_ptr += 1
            if ex_ptr >= len(exits):
                raise ValueError(f"{ind}: Entry at {e} has no subsequent exit.")
            paired.append((e, exits[ex_ptr]))
            ex_ptr += 1

        # markieren + trajectories schreiben
        for e, ex in paired:
            mice_enter[index, e] = 1
            mice_exit[index, ex] = 1
            traj_x[index, e:ex+1] = x[e:ex+1]
            traj_y[index, e:ex+1] = y[e:ex+1]

            traj = (x[e:ex+1], y[e:ex+1])
            arc_chord_ratio(traj)
            all_traj.append(traj)
            if plot:
                plot_trajectory_segment(x,y,e, ex)
    
        
    return mice_enter, mice_exit, traj_x, traj_y, all_traj







