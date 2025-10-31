import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm

def plot_mice_presence_states(mice_in_module, fps=30, title="Mouse presence in module"):
    """
    Plots the occupancy states (1, 2, 3, or at least 1 mouse present) over time.

    Parameters
    ----------
    mice_in_module : np.ndarray
        2D array of shape (n_mice, total_frames) with 0/1 values,
        indicating whether each mouse is present in the module in each frame.
    fps : int, optional
        Frames per second, used to convert x-axis into seconds (default: 30).
    title : str, optional
        Plot title.

    Returns
    -------
    None
        Displays a matplotlib figure showing the occupancy states over time.
    """

    # Anzahl Mäuse pro Frame zählen
    mice_per_frame = mice_in_module.sum(axis=0)

    # Zeitachse in Minuten
    time = np.arange(mice_per_frame.size) / fps / 60

    # Zustände berechnen
    one_mouse   = (mice_per_frame == 1).astype(int)
    two_mice    = (mice_per_frame == 2).astype(int)
    three_mice  = (mice_per_frame == 3).astype(int)
    at_least_one = (mice_per_frame >= 1).astype(int)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(time, at_least_one, color="gray", alpha=0.5, label="≥ 1 mouse")
    plt.plot(time, one_mouse, color="royalblue", label="1 mouse")
    plt.plot(time, two_mice + 1, color="orange", label="2 mice (offset +1)")
    plt.plot(time, three_mice + 2, color="crimson", label="3 mice (offset +2)")

    # Achsen & Layout
    plt.title(title)
    plt.xlabel("Time (min)")
    plt.ylabel("State")
    plt.yticks([0, 1, 2, 3], ["≥1 mouse", "1 mouse", "2 mice", "3 mice"])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_mouse_trajectory(center_x, center_y, title="Mouse trajectory", cmap="viridis"):
    """
    Plot the mouse trajectory with a time-based color gradient.

    Parameters
    ----------
    center_x : np.ndarray
        1D array of x-coordinates (may contain NaN values).
    center_y : np.ndarray
        1D array of y-coordinates (may contain NaN values).
    title : str, optional
        Title of the plot.
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis').

    Notes
    -----
    DeepLabCut uses image coordinates where the Y-axis increases downward.
    To match the video orientation, the Y values are inverted (multiplied by -1)
    before plotting.

    Returns
    -------
    None
        Displays a matplotlib plot showing the trajectory with a color gradient.
    """
    # Nur gültige Punkte behalten
    valid = ~np.isnan(center_x) & ~np.isnan(center_y)
    x = center_x[valid]
    y = center_y[valid]   

    if len(x) < 2:
        print("Not enough valid points to plot trajectory.")
        return

    # Liniensegmente für Farbverlauf
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Farbverlauf entlang der Zeit
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(segments)))
    lc.set_linewidth(2)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], color='green', s=60, label='Start', zorder=3)
    ax.scatter(x[-1], y[-1], color='red', s=60, label='End', zorder=3)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel("x position (px)")
    ax.set_ylabel("y position (px, inverted)")
    ax.legend()
    plt.tight_layout()
    plt.show()