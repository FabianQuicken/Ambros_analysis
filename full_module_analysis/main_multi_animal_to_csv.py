import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from main_multi_animal import multi_animal_main

# analysiert habituation, top1 oder top2 Aufnahme eines Experiments und speichert den output der main_multi_animal in einer csv
paths = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_30_45_46\hab",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_30_45_46\top1",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_30_45_46\top2",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_68_69_70\hab",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_68_69_70\top1",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_68_69_70\top2",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_38_47_53\hab",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_38_47_53\top1",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_38_47_53\top2",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_53_55_61\hab",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_53_55_61\top1",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_53_55_61\top2"
]

paths = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\males_53_55_61\top2"
]
# hier path zu den zu analysierenden deeplabcut output files
#path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\germfree\females_30_45_46\hab"


for path in paths:

    paths = glob.glob(os.path.join(path, "*.h5"))

    # das dictionary mit Daten generieren
    if "hab" in path:
        habituation = True
    else:
        habituation = False

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
    data_dictionary = multi_animal_main(path, habituation=habituation, social_inv=True, plot_heatmap=False, stitch_dataframes=True)

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
                "mice_distances",
                "mice_cumdists",
                "mice_in_center",
                "thetas",
                "visit_len",
                "visit_start",
                "mice_accelerations",
                "face_inv",
                "body_inv",
                "anogenital_inv",
#                "trajectories_x",
#                "trajectories_y"
    #            "arc_chord"
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
            level4 = parts[11]                 # parts[5] (hab/top1/top2)

        return level1, level2, level3, level4


    col_tuples = []
    for p in [paths[0]]:
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
            #print(metric)
            if metric == "centers_x" or metric == "centers_y":
                data = data_dic["centers_xy"]
            elif metric == "nose_x" or metric == "nose_y":
                data = data_dic["nose_xy"]
            elif metric == "fronts_x" or metric == "fronts_y":
                data = data_dic["fronts_xy"]
            elif metric == "rears_x" or metric == "rears_y":
                data = data_dic["rears_xy"]
            elif metric == "visit_len" or metric == "visit_start":
                data = data_dic["visits"]
            elif metric == "trajectories_x"
            else:
                data = data_dic[metric]
            #print(len(data), data)
            print(metric)
            if len(data) != 3:
                
                raise ValueError("Data shape not n_individuals")
            for i, ind in enumerate(individuals):
                
                if metric == "theta_dic":
                    data_singleanimal = data[ind]
                else:
                    data_singleanimal = data[i]
                #print(metric, ind)
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

                if metric == "centers_x" or metric == "nose_x" or metric == "fronts_x" or metric == "rears_x":
                    xs = []
                    for (x, y) in data_singleanimal:
                        xs.append(x)
                    data_singleanimal = xs
                if metric == "centers_y" or metric == "nose_y" or metric == "fronts_y" or metric == "rears_y":
                    ys = []
                    for (x, y) in data_singleanimal:
                        ys.append(y)
                    data_singleanimal = ys
                
                
                
                if len(data_singleanimal) > n_frames:
                    data_singleanimal = data_singleanimal[0:n_frames]
                df.loc[0:len(data_singleanimal)-1, ([l1], [l2], [l3], [l4], [metric], [ind])] = data_singleanimal

                

    #print(df[0:10])
    safename = f"/single_mouse_datatest_{l1}_{l2}_{l3}_{l4}.csv"
    safepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data" + safename
    df.to_csv(safepath)
    #print(safepath)
