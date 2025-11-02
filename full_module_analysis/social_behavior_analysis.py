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

                    closest_bp = 1000  # in px

                    for bp in range(len(bodyparts)):     # jeder Körperteil der Zielmaus

                        dist = euklidean_distance(x1 = all_nose_x[i][c], y1 = all_nose_y[i][c], x2 = all_x[p][c][bp], y2=all_y[p][c][bp])
                        if dist < closest_bp:
                            closest_bp = dist

                    if closest_bp <= PIXEL_PER_CM*2:
                        social_investigation[c] = 1

    return social_investigation


def detail_social_investigation2(df, scorer, individuals):

    face_bp = ["nose", "eye_left", "eye_right"]
    body_bp = ["lateral_left", "lateral_right", "hip_left", "hip_right", "dorsal_2", "dorsal_3"]
    anogenital_bp = ["tail_base", "dorsal_4"]

    fragmented_bp = [face_bp, anogenital_bp, body_bp]



    all_nose_x = []
    all_nose_y = []

    all_x = []
    all_y = []
    
    for bps in fragmented_bp:

        for ind in individuals:
            nose_x = df.loc[:, (scorer, ind, ["nose"], ["x"])].to_numpy()
            nose_y = df.loc[:, (scorer, ind, ["nose"], ["y"])].to_numpy()

            all_nose_x.append(nose_x)
            all_nose_y.append(nose_y)

            x = df.loc[:, (scorer, ind, bps, ["x"])].to_numpy()
            y = df.loc[:, (scorer, ind, bps, ["y"])].to_numpy()

            all_x.append(x)
            all_y.append(y)

    face_investigation = np.zeros(len(all_nose_x[0][0]))
    body_investigation = np.zeros(len(all_nose_x[0][0]))
    anogenital_investigation = np.zeros(len(all_nose_x[0][0]))

    for animal_part in range(len(all_nose_x)):  # einmal für face, anogenital, body

        for i in range(len(animal_part)): # jede Maus als Investigierende
            
            for p in range(len(all_x)):  # jede Maus als Ziel

                if not p == i:  # investigator und ziel müssen verschieden sein
                    

                    for c in range(len(all_nose_x[i])):  # jedes Frame

                        closest_bp = 1000  # in px

                        for bp in range(len(fragmented_bp[i])):     # jeder Körperteil der Zielmaus

                            dist = euklidean_distance(x1 = all_nose_x[i][c], y1 = all_nose_y[i][c], x2 = all_x[p][c][bp], y2=all_y[p][c][bp])
                            if dist < closest_bp:
                                closest_bp = dist

                        if closest_bp <= PIXEL_PER_CM*2:
                            social_investigation[c] = 1

    return social_investigation



def detail_social_investigation_gpt(df, scorer, individuals, pixel_per_cm, max_dist_cm=2.0):
    """
    Aggregiert Social-Investigation pro Frame als ZÄHLUNGEN über alle Individuen:
    - Pro investigierender Maus und Frame wird maximal EINE Kategorie gezählt (Priorität: Face > Anogenital > Body).
    - Gegenseitige Interaktionen (i->p und p->i) zählen getrennt und können so pro Frame doppelt zählen.
    
    Returns
    -------
    {
      "counts_per_frame": {
          "face": np.ndarray[int] (n_frames,),
          "anogenital": np.ndarray[int] (n_frames,),
          "body": np.ndarray[int] (n_frames,)
      },
      "presence_per_frame": {
          "face": np.ndarray[uint8] (n_frames,),        # 0/1
          "anogenital": np.ndarray[uint8] (n_frames,),  # 0/1
          "body": np.ndarray[uint8] (n_frames,)         # 0/1
      },
      "totals": {"face": int, "anogenital": int, "body": int}
    }
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
        nose_x[i] = df.loc[:, (scorer, ind, "nose", "x")].to_numpy().ravel()
        nose_y[i] = df.loc[:, (scorer, ind, "nose", "y")].to_numpy().ravel()

    # Ziel-Koordinaten für alle benötigten Bodyparts laden
    needed_bps = {bp for _, bps in categories for bp in bps}
    coords = {ind: {} for ind in individuals}
    for ind in individuals:
        for bp in needed_bps:
            if (scorer, ind, bp, "x") in df.columns and (scorer, ind, bp, "y") in df.columns:
                x_arr = df.loc[:, (scorer, ind, bp, "x")].to_numpy().ravel()
                y_arr = df.loc[:, (scorer, ind, bp, "y")].to_numpy().ravel()
            else:
                x_arr = np.full(n_frames, np.nan)
                y_arr = np.full(n_frames, np.nan)
            coords[ind][bp] = (x_arr, y_arr)

    # Zähler pro Frame
    face_count       = np.zeros(n_frames, dtype=int)
    anogenital_count = np.zeros(n_frames, dtype=int)
    body_count       = np.zeros(n_frames, dtype=int)

    # Hauptschleife: pro Frame und Investigator genau eine Kategorie (wenn vorhanden)
    for c in range(n_frames):
        for i, inv in enumerate(individuals):
            x1, y1 = nose_x[i, c], nose_y[i, c]
            if np.isnan(x1) or np.isnan(y1):
                continue

            # Kategorie-Priorität: Face -> Anogenital -> Body
            assigned = False
            for cat_name, bps in categories:
                if assigned:
                    break
                # Über alle Zielmäuse
                for p, tgt in enumerate(individuals):
                    if p == i:
                        continue
                    # kleinstes BP der Kategorie finden
                    closest = np.inf
                    for bp in bps:
                        x2, y2 = coords[tgt][bp][0][c], coords[tgt][bp][1][c]
                        if np.isnan(x2) or np.isnan(y2):
                            continue
                        d = hypot(x1 - x2, y1 - y2)
                        if d < closest:
                            closest = d
                            if closest <= thr_px:
                                # Treffer -> Kategorie zählen und Investigator für diesen Frame abschließen
                                if cat_name == "face":
                                    face_count[c] += 1
                                elif cat_name == "anogenital":
                                    anogenital_count[c] += 1
                                else:
                                    body_count[c] += 1
                                assigned = True
                                break
                    if assigned:
                        break

    presence_face       = (face_count > 0).astype(np.uint8)
    presence_anogenital = (anogenital_count > 0).astype(np.uint8)
    presence_body       = (body_count > 0).astype(np.uint8)

    return {
        "counts_per_frame": {
            "face": face_count,
            "anogenital": anogenital_count,
            "body": body_count
        },
        "presence_per_frame": {
            "face": presence_face,
            "anogenital": presence_anogenital,
            "body": presence_body
        },
        "totals": {
            "face": int(face_count.sum()),
            "anogenital": int(anogenital_count.sum()),
            "body": int(body_count.sum())
        }
    }