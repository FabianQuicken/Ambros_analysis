import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from preprocessing import likelihood_filtering, likelihood_filtering_nans
from utils import euklidean_distance, fill_missing_values, shrink_rectangle, is_point_in_polygon, create_point
from config import PIXEL_PER_CM, ARENA_COORDS_TOP1, ARENA_COORDS_TOP2, FPS, ENTER_ZONE_COORDS

import matplotlib.pyplot as plt


def entry_exit_trajectories(entry_polygon, x_arrs, y_arrs, individuals):

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

    for index, ind in enumerate(individuals):
        x = x_arrs[index]
        y = y_arrs[index]

        # x und y arrays müssen gleich lang sein
        if len(x) != len(y):
            raise ValueError("x and y arrays have different length.")

        # testen ob daten da sind für das jeweilige individum, sonst nächstes Ind
        valid = np.isfinite(x) & np.isfinite(y)
        # mind 1s insgesamt getrackt?
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

            plot_trajectory_segment(x,y,e=1200, ex=1350)

        
    return mice_enter, mice_exit, traj_x, traj_y







def get_trajectories(individuals, x_arr, y_arr):
    pass
