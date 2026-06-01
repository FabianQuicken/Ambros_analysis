import os
import glob
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main_multi_animal import multi_animal_main


basepath = Path(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\dlc_output")

groups = [
#    "germfree"
    "germfreeprop",
    "omm12",
    "omm12prop",
    "ommpgol",
]

paths = []

for group in groups:
    group_path = basepath / group

    for h5_file in group_path.glob("*_*_*_*/hab/*.h5"):
        paths.append(h5_file.parent)

    for h5_file in group_path.glob("*_*_*_*/top1/*.h5"):
        paths.append(h5_file.parent)

    for h5_file in group_path.glob("*_*_*_*/top2/*.h5"):
        paths.append(h5_file.parent)

paths = sorted(set(paths))


for path in paths:
    """
    if r"females_68_69_70\top2" not in str(path):
        print(str(path))
        continue
    """

    h5_paths = glob.glob(os.path.join(path, "*.h5"))

    # das dictionary mit Daten generieren
    habituation = path.name == "hab"
    print(habituation)
    print(path)

    def get_mice(path):
        from pathlib import Path

        path = Path(path)

        prefixes = ("females_", "males_")

        # passenden Ordner finden
        target_folder = next(
            p for p in path.parts if p.startswith(prefixes)
        )

        # Zahlen extrahieren
        numbers = target_folder.split("_", 1)[1]
        return numbers

    mice = get_mice(path)
    data_dictionary = multi_animal_main(str(path), habituation=habituation, social_inv=True, plot_heatmap=False, stitch_dataframes=True)

    n_frames = 216_000
    if habituation:
        n_frames = 54000

    # wie sind die individuals benannt?
    individuals = [
        "mouse_1",
        "mouse_2",
        "mouse_3"
    ]

    # alle metrics, die gespeichert werden sollen
    # müssen auf den shape n_ind, data passen
    metric_names = [ 
                "centers_x",
                "fronts_x",
                "rears_x",
                "nose_x",
                "centers_y",
                "fronts_y",
                "rears_y",
                "nose_y",
                "mice_presence",
                "mice_immobile",
                "immobile_bouts",
                "immobile_start",
                "mice_distances",
                "mice_cumdists",
                "mice_in_center",
                "thetas",
                "visit_len",
                "visit_start",
                "mice_accelerations",
                "speedevents",
                "face_inv",
                "body_inv",
                "anogenital_inv",
                "trajectories_x",
                "trajectories_y",
                "mean_arc_chord",
                "fragment_arc_chord",
                "orientations",
                "posture_compactness",
                "mice_bodylength",
                "mice_mean_likelihood"
    ]

    # helper function um aus dem filenamen später den multiindex zu basteln
    def parse_tokens(filepath: str):
        """
        Expects filename: <cond>_<sex>_<id1>_<id2>_<id3>_<module>.h5
        Example: germfree_females_30_45_46_hab.h5
        """
        stem = Path(filepath).stem  # removes .h5
        parts = stem.split("_")
        if len(parts) < 6:
            raise ValueError(f"Unexpected filename format: {stem}")

        level1 = parts[7]                 # parts[0]
        level2 = mice    # parts[2,3,4] together
        level3 = parts[8]                 # parts[1]
        if habituation:
            level4 = "hab"
        else:
            level4 = parts[11]                 #  (top1/top2)

        return level1, level2, level3, level4


    col_tuples = []
    for p in [h5_paths[0]]:
        l1, l2, l3, l4 = parse_tokens(p)
        for m in metric_names:
            for ind in individuals:
                col_tuples.append((l1, l2, l3, l4, m, ind))

    columns = pd.MultiIndex.from_tuples(
        col_tuples,
        names=["group", "mouse_ids", "sex", "condition", "metric", "individual"]
    )


    # emtpy df initialisieren
    df = pd.DataFrame(
        index=np.arange(n_frames),
        columns=columns,
        dtype=float
    )

    print(l1, l2, l3, l4)

    group = df.columns.levels[0]
    ids = df.columns.levels[1]
    sex = df.columns.levels[2]
    condition = df.columns.levels[3]
    metrics = df.columns.levels[4]


    for metric in metrics:
            data_dic = data_dictionary.copy()


            if metric in ("visit_len", "visit_start"):
                data = data_dic["visits"]
            elif metric in ("trajectories_x", "trajectories_y"):
                data = data_dic["trajectories"]
            else:
                data = data_dic[metric]


            if len(data) != 3:
                
                raise ValueError("Data shape not n_individuals")
            for i, ind in enumerate(individuals):
                
                if metric == "theta_dic":
                    data_singleanimal = data[ind]
                else:
                    data_singleanimal = data[i]

                if metric == "visit_len":
                    traj_lens = []
                    
                    for (start, length) in data_singleanimal:
                        traj_lens.append(length)                       
                    data_singleanimal = traj_lens

                if metric == "visit_start":
                    traj_starts = []
                    for (start, length) in data_singleanimal:
                        traj_starts.append(start)
                    data_singleanimal = traj_starts

                # data_dic["trajectories"] ist so aufgebaut: [([x1.1, x1.2, x1.3],[y1.1, y1.2, y1.3]), ([x2.1, x2.2],[y2.1, y2.2])]
                # x1.x sind die x-Werte des ersten Trajectories, x2.x die des Zweiten usw. 
                # um alles in einer .csv in einer spalte zu speichern, speichern wir jeweils x und y koordinaten in einer Spalte
                # und trennen einzelne trajectories mit einem "nan"
                if metric == "trajectories_x":
                    traj_xs = []
                    for t in data_singleanimal:
                        end = len(t[0])
                        counter = 0
                        while counter < end:
                            traj_xs.append(t[0][counter])
                            counter += 1
                        traj_xs.append(np.nan)
                    data_singleanimal = traj_xs
                if metric == "trajectories_y":
                    traj_ys = []
                    for t in data_singleanimal:
                        end = len(t[1])
                        counter = 0
                        while counter < end:
                            traj_ys.append(t[1][counter])
                            counter += 1
                        traj_ys.append(np.nan)
                    data_singleanimal = traj_ys
                
                print(metric)
                #print(data_singleanimal[0:2])
                if len(data_singleanimal) > n_frames:
                    data_singleanimal = data_singleanimal[0:n_frames]
                df.loc[0:len(data_singleanimal)-1, ([l1], [l2], [l3], [l4], [metric], [ind])] = data_singleanimal

                

    #print(df[0:10])
    safename = f"/single_mouse_datatest_{l1}_{l2}_{l3}_{l4}.csv"
    safepath = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest" + safename
    df.to_csv(safepath)
    #print(safepath)
