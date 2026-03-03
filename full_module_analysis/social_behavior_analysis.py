import numpy as np
import pandas as pd

from utils import euklidean_distance
from config import PIXEL_PER_CM
from math import hypot

def social_investigation(df, scorer, individuals, bodyparts):

    all_nose_x = []
    all_nose_y = []

    all_x = []
    all_y = []

    for ind in individuals:
        nose_x = df.loc[:, (scorer, ind, ["nose"], ["x"])].to_numpy()
        nose_y = df.loc[:, (scorer, ind, ["nose"], ["y"])].to_numpy()

        all_nose_x.append(nose_x)
        all_nose_y.append(nose_y)

        x = df.loc[:, (scorer, ind, bodyparts, ["x"])].to_numpy()
        y = df.loc[:, (scorer, ind, bodyparts, ["y"])].to_numpy()

        all_x.append(x)
        all_y.append(y)

    social_investigation = np.zeros(len(all_nose_x[0]))



    for i in range(len(all_nose_x)): # jede Maus als Investigierende
        
        for p in range(len(all_x)):  # jede Maus als Ziel

            if not p == i:  
                

                for c in range(len(all_nose_x[i])):  # jedes Frame

                    closest_bp = 1000  # in px


                    for bp in range(len(bodyparts)):     # jeder Körperteil der Zielmaus
                        

                        dist = euklidean_distance(x1 = all_nose_x[i][c], y1 = all_nose_y[i][c], x2 = all_x[p][c][bp], y2=all_y[p][c][bp])
                        if dist < closest_bp:
                            closest_bp = dist

                    if closest_bp <= PIXEL_PER_CM*2:
                        social_investigation[c] = 1

    return social_investigation

"""
def detail_social_investigation(df, scorer, individuals, face_investigation = False, body_investigation = False, anogenital_investigation = False):

    face_bp = ["nose", "eye_left", "eye_right"]
    body_bp = ["lateral_left", "lateral_right", "hip_left", "hip_right", "dorsal_2", "dorsal_3"]
    anogenital_bp = ["tail_base", "dorsal_4"]


    if face_investigation:
        bodyparts = face_bp
    elif body_investigation:
        bodyparts = body_bp
    elif anogenital_investigation:
        bodyparts = anogenital_bp
    else:
        raise KeyError("No Mode for detailed social investigation detected. Set either face, body or anogenital investigation to 'True'")

    all_nose_x = []
    all_nose_y = []

    all_x = []
    all_y = []

    for ind in individuals:
        nose_x = df.loc[:, (scorer, ind, ["nose"], ["x"])].to_numpy()
        nose_y = df.loc[:, (scorer, ind, ["nose"], ["y"])].to_numpy()

        all_nose_x.append(nose_x)
        all_nose_y.append(nose_y)

        x = df.loc[:, (scorer, ind, bodyparts, ["x"])].to_numpy()
        y = df.loc[:, (scorer, ind, bodyparts, ["y"])].to_numpy()

        all_x.append(x)
        all_y.append(y)

    social_investigation = np.zeros(len(all_nose_x[0]))



    for i in range(len(all_nose_x)): # jede Maus als Investigierende
        
        for p in range(len(all_x)):  # jede Maus als Ziel

            if not p == i:  
                

                for c in range(len(all_nose_x[i])):  # jedes Frame

                    closest_bp = np.inf  # in px

                    for bp in range(len(bodyparts)):     # jeder Körperteil der Zielmaus

                        dist = euklidean_distance(x1 = all_nose_x[i][c], y1 = all_nose_y[i][c], x2 = all_x[p][c][bp], y2=all_y[p][c][bp])
                        if dist < closest_bp:
                            closest_bp = dist

                    if closest_bp <= PIXEL_PER_CM*2:
                        social_investigation[c] = 1

    return social_investigation
"""




def detail_social_investigation(
    df, scorer, individuals, pixel_per_cm,
    max_dist_cm=2.0,
    min_fragment_frames: int = 10,
):
    """
    Aggregiert Social-Investigation pro Frame als ZÄHLUNGEN über alle Individuen:
    - Pro investigierender Maus und Frame wird maximal EINE Kategorie gezählt (Priorität: Face > Anogenital > Body).
    - Gegenseitige Interaktionen (i->p und p->i) zählen getrennt und können so pro Frame doppelt zählen.

    Zusätzlich (optional):
    - Entfernt kurze Fragmente pro Kategorie: zusammenhängende Runs mit count>0,
      die kürzer als `min_fragment_frames` sind, werden auf 0 gesetzt.

    Returns
    -------
    {... wie vorher ...}
    """
    assert isinstance(scorer, str), "scorer muss ein String sein"

    # Kategorien (mit Priorität)
    face_bp       = ["nose", "eye_left", "eye_right"]
    anogenital_bp = ["tail_base", "dorsal_4"]
    body_bp       = ["lateral_left", "lateral_right", "hip_left", "hip_right", "dorsal_2", "dorsal_3"]
    categories = [("face", face_bp), ("anogenital", anogenital_bp), ("body", body_bp)]

    n_frames = len(df)
    n_ind = len(individuals)
    thr_px = pixel_per_cm * max_dist_cm

    # Investigator-Nasen laden
    nose_x = np.empty((n_ind, n_frames), dtype=float)
    nose_y = np.empty((n_ind, n_frames), dtype=float)
    for i, ind in enumerate(individuals):
        nose_x[i] = df.loc[:, (scorer, ind, "nose", "x")].to_numpy()
        nose_y[i] = df.loc[:, (scorer, ind, "nose", "y")].to_numpy()

    # Ziel-Koordinaten für alle benötigten Bodyparts laden
    needed_bps = {bp for _, bps in categories for bp in bps}
    coords = {ind: {} for ind in individuals}
    for ind in individuals:
        for bp in needed_bps:
            if (scorer, ind, bp, "x") in df.columns and (scorer, ind, bp, "y") in df.columns:
                x_arr = df.loc[:, (scorer, ind, bp, "x")].to_numpy()
                y_arr = df.loc[:, (scorer, ind, bp, "y")].to_numpy()
            else:
                x_arr = np.full(n_frames, np.nan)
                y_arr = np.full(n_frames, np.nan)
            coords[ind][bp] = (x_arr, y_arr)

    # Zähler pro Frame
    face_count       = np.zeros((n_ind, n_frames), dtype=int)
    anogenital_count = np.zeros((n_ind, n_frames), dtype=int)
    body_count       = np.zeros((n_ind, n_frames), dtype=int)

    # Hauptschleife: pro Frame und Investigator genau eine Kategorie (wenn vorhanden)
    for c in range(n_frames):
        for i, inv in enumerate(individuals):
            x1, y1 = nose_x[i, c], nose_y[i, c]
            if np.isnan(x1) or np.isnan(y1):
                continue

            assigned = False
            for cat_name, bps in categories:
                if assigned:
                    break
                for p, tgt in enumerate(individuals):
                    if p == i:
                        continue
                    closest = np.inf
                    for bp in bps:
                        x2, y2 = coords[tgt][bp][0][c], coords[tgt][bp][1][c]
                        if np.isnan(x2) or np.isnan(y2):
                            continue
                        d = euklidean_distance(x1=x1, x2=x2, y1=y1, y2=y2)
                        if d < closest:
                            closest = d
                            if closest <= thr_px:
                                if cat_name == "face":
                                    face_count[i][c] += 1
                                elif cat_name == "anogenital":
                                    anogenital_count[i][c] += 1
                                else:
                                    body_count[i][c] += 1
                                assigned = True
                                break
                    if assigned:
                        break
    

    # --- Fragment-Filter pro Kategorie: entferne kurze Runs mit count>0 ---
    def _remove_short_runs(count_arr: np.ndarray, min_len: int) -> np.ndarray:
        if min_len is None or min_len <= 1:
            return count_arr
        mask = count_arr > 0
        if not mask.any():
            return count_arr

        # Run-length encoding über True/False
        d = np.diff(mask.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends   = np.where(d == -1)[0]  # inkl.

        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask) - 1]

        out = count_arr.copy()
        for s, e in zip(starts, ends):
            if (e - s + 1) < min_len:
                out[s:e+1] = 0
        return out

    for i in range(len(individuals)):
        face_count[i] = _remove_short_runs(face_count[i], min_fragment_frames)
        body_count[i] = _remove_short_runs(body_count[i], min_fragment_frames)
        anogenital_count[i] = _remove_short_runs(anogenital_count[i], min_fragment_frames)

    face_count_per_frame = np.nansum(face_count, axis=0)
    body_count_per_frame = np.nansum(body_count, axis=0)
    anogenital_count_per_frame = np.nansum(anogenital_count, axis=0)

    presence_face       = (face_count_per_frame > 0).astype(np.uint8)
    presence_anogenital = (anogenital_count_per_frame > 0).astype(np.uint8)
    presence_body       = (body_count_per_frame > 0).astype(np.uint8)

    face_start_end = [[] for _ in individuals]
    anogenital_start_end = [[] for _ in individuals]
    body_start_end = [[] for _ in individuals]

    def _get_indices(arr):
        arr = (arr > 0).astype(np.int8)
        out_list = []

        diff = np.diff(arr)

        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0]

        # Falls Event direkt bei Frame 0 startet
        if arr[0] == 1:
            starts = np.r_[0, starts]

        # Falls Event am letzten Frame endet
        if arr[-1] == 1:
            ends = np.r_[ends, len(arr) - 1]

        for s, e in zip(starts, ends):
            out_list.append((int(s), int(e)))

        return out_list
    
    for i in range(len(individuals)):
        face_start_end[i] = _get_indices(face_count[i])
        anogenital_start_end[i] = _get_indices(anogenital_count[i])
        body_start_end[i] = _get_indices(body_count[i])

    return {
        "counts_per_frame": {
            "face": face_count_per_frame,
            "anogenital": anogenital_count_per_frame,
            "body": body_count_per_frame
        },
        "presence_per_frame": {
            "face": presence_face,
            "anogenital": presence_anogenital,
            "body": presence_body
        },
        "totals": {
            "face": int(face_count_per_frame.sum()),
            "anogenital": int(anogenital_count_per_frame.sum()),
            "body": int(body_count_per_frame.sum())
        },
        "individual_inv": {
            "face": face_count,
            "body": body_count,
            "anogenital": anogenital_count
        },
        "start_end_indices": {
            "face": face_start_end,
            "body": body_start_end,
            "anogenital": anogenital_start_end
        }
    }