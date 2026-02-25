import numpy as np
from metrics import distance_travelled_arraybased
from utils import mouse_center, moving_average
from config import IMMOBILE_THRSH, FPS
import matplotlib.pyplot as plt
"""
Idee für Grooming:
"Kinetische Energy States/Levels" verschiedener Koordinaten berechnen
Beim Grooming bewegt sich viel der Kopf (Nose, Eyes, Ears, Headcenter)
Der hintere Teil (Hips, Tailbase) müsste sich eigentlich recht wenig bewegen



"""




def grooming_energy(df, scorer, individual, head_coords, trunk_coords,
                    *,
                    immobile_thr=IMMOBILE_THRSH,
                    fps=FPS,
                    energy_decay_per_frame=1.0,
                    reset_after_immobile_s=1.0,
                    reset_after_nan_s=1.0):
    """
    Computes simple 'energy' traces for head and trunk centers.

    Assumes df already likelihood-filtered + interpolated.
    """

    head_x, head_y = mouse_center(df, scorer, [individual], bodyparts=head_coords)
    head_x, head_y = head_x[0], head_y[0]

    trunk_x, trunk_y = mouse_center(df, scorer, [individual], bodyparts=trunk_coords)
    trunk_x, trunk_y = trunk_x[0], trunk_y[0]

    head_step = distance_travelled_arraybased(head_x, head_y)   # step per frame
    trunk_step = distance_travelled_arraybased(trunk_x, trunk_y)

    # optional: smooth here (better: smooth coords before step)
    head_step = moving_average(head_step, window=10)
    trunk_step = moving_average(trunk_step, window=10)

    reset_after_immobile_frames = int(round(reset_after_immobile_s * fps))
    reset_after_nan_frames = int(round(reset_after_nan_s * fps))

    def _energylevel(stepvalues):
        energy = 0.0
        immobile_frames = 0
        nan_frames = 0

        energy_arr = np.zeros(len(stepvalues), dtype=float)

        for i, v in enumerate(stepvalues):
            # 1) handle invisibility first
            if not np.isfinite(v):
                nan_frames += 1
                immobile_frames = 0  # optional: or keep it, but be explicit

                # either decay while invisible...
                energy = max(0.0, energy - energy_decay_per_frame)

                if nan_frames >= reset_after_nan_frames:
                    energy = 0.0

                energy_arr[i] = energy
                continue
            else:
                nan_frames = 0

            # 2) visible -> update energy based on activity
            if v > immobile_thr:
                energy += (v - immobile_thr)
                immobile_frames = 0
            else:
                immobile_frames += 1
                energy = max(0.0, energy - energy_decay_per_frame)
                if immobile_frames >= reset_after_immobile_frames:
                    energy = 0.0

            energy_arr[i] = energy

        return energy_arr

    head_energy = _energylevel(head_step)
    trunk_energy = _energylevel(trunk_step)

    return head_energy, trunk_energy, head_x, head_y, trunk_x, trunk_y
