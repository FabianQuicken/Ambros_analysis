import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_trace_transparent(values,
                  fps=30,
                  window_seconds=5,
                  color="red",
                  linewidth=2,
                  save_path=None,
                  dpi=200,
                  transparent=True):
    """
    Animate a 1D trace without axes (clean overlay style).

    Parameters
    ----------
    values : array-like
        One value per frame.
    fps : int
        Frames per second.
    window_seconds : float
        Width of visible window in seconds.
    color : str
        Line color (e.g. 'red', 'purple', 'mint', '#00ffaa').
    linewidth : float
        Line thickness.
    save_path : str or None
        If provided, animation will be saved (.mp4 or .gif).
    dpi : int
        Resolution for saving.
    transparent : bool
        Transparent background (useful for video overlay).
    """

    y = np.asarray(values, dtype=float)
    n = len(y)
    x = np.arange(n) / fps

    fig, ax = plt.subplots()

    # Remove everything
    ax.set_axis_off()
    fig.patch.set_alpha(0 if transparent else 1)
    ax.set_facecolor("none" if transparent else "black")

    (line,) = ax.plot([], [],
                      color=color,
                      linewidth=linewidth)

    # Fix y-limits
    finite = y[np.isfinite(y)]
    y_min, y_max = np.min(finite), np.max(finite)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    def update(i):
        current_time = x[i]
        left = max(0, current_time - window_seconds)
        right = current_time

        ax.set_xlim(left, right)

        mask = (x >= left) & (x <= right)
        line.set_data(x[mask], y[mask])

        return (line,)

    anim = FuncAnimation(
        fig,
        update,
        frames=n,
        interval=1000 / fps,
        blit=False,
        repeat=False
    )

    if save_path is not None:
        if save_path.endswith(".mp4"):
            anim.save(save_path,
                      writer="ffmpeg",
                      fps=fps,
                      dpi=dpi,
                      savefig_kwargs={"transparent": transparent})
        elif save_path.endswith(".gif"):
            anim.save(save_path,
                      writer="pillow",
                      fps=fps,
                      dpi=dpi)
        else:
            raise ValueError("Unsupported format. Use .mp4 or .gif")

        print(f"Saved to {save_path}")

    plt.show()
    return fig, ax, anim


def animate_trace(values, fps=30, window_seconds=10,
                  xlabel="Time (s)", ylabel="Value",
                  title="Animated trace", blit=True, color="blue", save_path="", dpi=200):
    """
    Animate a 1D trace (one value per frame) with a scrolling x-axis.

    Parameters
    ----------
    values : array-like
        One value per frame.
    fps : int
        Frames per second.
    window_seconds : float
        Width of visible x-axis window in seconds.
    """

    y = np.asarray(values, dtype=float)
    n = len(y)

    # Zeitachse in Sekunden
    x = np.arange(n) / fps
    window_size = window_seconds

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], lw=2, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Y-Limits einmal fixieren
    finite = y[np.isfinite(y)]
    y_min, y_max = np.min(finite), np.max(finite)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        current_time = x[i]

        # Fenster dynamisch setzen
        left = max(0, current_time - window_size)
        right = current_time

        ax.set_xlim(left, right)

        # Daten innerhalb des Fensters
        mask = (x >= left) & (x <= right)
        line.set_data(x[mask], y[mask])

        return (line,)

    interval_ms = 1000 / fps
    anim = FuncAnimation(
        fig, update, frames=n,
        init_func=init,
        interval=interval_ms,
        blit=False,   # wichtig: bei dynamischem set_xlim besser False
        repeat=False
    )

    if save_path is not None:
        if save_path.endswith(".mp4"):
            anim.save(save_path,
                      writer="ffmpeg",
                      fps=fps,
                      dpi=dpi)
        elif save_path.endswith(".gif"):
            anim.save(save_path,
                      writer="pillow",
                      fps=fps,
                      dpi=dpi)
        else:
            raise ValueError("Unsupported format. Use .mp4 or .gif")

        print(f"Saved to {save_path}")

    plt.show()
    return fig, ax, anim
