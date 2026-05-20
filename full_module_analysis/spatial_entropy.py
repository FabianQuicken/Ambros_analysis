import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt


def create_arena_mask(arena_coords, x_edges, y_edges):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])

    polygon = Path(arena_coords)
    arena_mask = polygon.contains_points(points)

    return arena_mask.reshape(len(x_centers), len(y_centers))


def spatial_entropy(
    center_x,
    center_y,
    x_edges,
    y_edges,
    arena_coords=None,
    arena_mask=None,
    return_details=False,
    plot_H=True,
    cmap="inferno"
):
    """
    Calculate normalized spatial entropy of arena occupancy.

    Parameters
    ----------
    center_x, center_y : array-like
        Per-frame center coordinates of one mouse.

    x_edges, y_edges : array-like
        Bin edges for np.histogram2d.

    arena_coords : list of tuple, optional
        Polygon coordinates defining the accessible arena area.
        Used to create arena_mask if arena_mask is not supplied.

    arena_mask : np.ndarray, optional
        Boolean mask with same shape as occupancy histogram.

    return_details : bool
        If True, also return histogram, occupancy probabilities and mask.

    Returns
    -------
    normalized_entropy : float
        Value between 0 and 1.

        1 = perfectly even spatial use
        0 = all occupancy concentrated in one bin
    """

    center_x = np.asarray(center_x)
    center_y = np.asarray(center_y)

    valid = ~np.isnan(center_x) & ~np.isnan(center_y)
    center_x = center_x[valid]
    center_y = center_y[valid]

    H, x_edges, y_edges = np.histogram2d(
        center_x,
        center_y,
        bins=[x_edges, y_edges]
    )

    print(len(H))
    print(H)

    if arena_mask is None:
        if arena_coords is not None:
            arena_mask = create_arena_mask(
                arena_coords,
                x_edges,
                y_edges
            )
        else:
            arena_mask = np.ones_like(H, dtype=bool)

    if arena_mask.shape != H.shape:
        raise ValueError(
            f"arena_mask shape {arena_mask.shape} does not match "
            f"histogram shape {H.shape}"
        )

    values = H[arena_mask]

    total = np.sum(values)

    if total == 0:
        if return_details:
            return {
                "normalized_entropy": np.nan,
                "entropy": np.nan,
                "max_entropy": np.nan,
                "histogram": H,
                "arena_mask": arena_mask,
                "occupancy": np.full_like(H, np.nan, dtype=float),
            }
        return np.nan

    p = values / total 
    p_nonzero = p[p > 0]

    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))

    n_accessible_bins = np.sum(arena_mask)
    max_entropy = np.log2(n_accessible_bins)

    normalized_entropy = entropy / max_entropy

    occupancy = np.full_like(H, np.nan, dtype=float)
    occupancy[arena_mask] = p

    # plot occupancy heatmap
    if plot_H:

        plot_data = occupancy.T

        plt.figure(figsize=(8, 6))

        im = plt.imshow(
            plot_data,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            extent=[
                x_edges[0],
                x_edges[-1],
                y_edges[0],
                y_edges[-1]
            ],
            aspect="equal"
        )

        plt.colorbar(
            im,
            label="Occupancy probability"
        )

        plt.xlabel("x position")
        plt.ylabel("y position")

        plt.title(
            f"Spatial occupancy\n"
            f"Normalized entropy = "
            f"{normalized_entropy:.3f}"
        )

        plt.tight_layout()
        plt.show()

    if return_details:
        return {
            "normalized_entropy": normalized_entropy,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "histogram": H,
            "arena_mask": arena_mask,
            "occupancy": occupancy,
            "n_accessible_bins": n_accessible_bins,
            "n_valid_frames": len(center_x),
        }

    return normalized_entropy