import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from save_load_dic import load_analysis
# -----------------------------
# 1) Inputs
# -----------------------------
basepath = r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\all"
paths = glob.glob(os.path.join(basepath, "*.joblib"))

n_frames = 216_000  # oder beliebig n
individuals = [
    "mouse_1",
    "mouse_2",
    "mouse_3"
]
metric_names = [
    # hier deine Metrik-Header rein
    "mice_presence",
    "mice_immobile",
    "mice_cumdists",
    "mice_in_center",
    "mice_accelerations",
    "thetas",
    "face_inv",
    "body_inv",
    "anogenital_inv",
    "trajectories",
    "arc_chord", 
    "center_x",
    "center_y"
]

metric_names = [
    "mice_immobile"
]

# -----------------------------
# 2) Helper: filename -> tokens
# -----------------------------
def parse_tokens(filepath: str):
    """
    Expects filename: <cond>_<sex>_<id1>_<id2>_<id3>_<module>.joblib
    Example: germfree_females_30_45_46_hab.joblib
    """
    stem = Path(filepath).stem  # removes .joblib
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {stem}")

    level1 = parts[0]                 # parts[0]
    level2 = "_".join(parts[2:5])     # parts[2,3,4] together
    level3 = parts[1]                 # parts[1]
    level4 = parts[5]                 # parts[5] (hab/top1/top2)

    return level1, level2, level3, level4

# -----------------------------
# 3) Build MultiIndex columns
# -----------------------------
col_tuples = []
for p in sorted(paths):
    l1, l2, l3, l4 = parse_tokens(p)
    for m in metric_names:
        for ind in individuals:
            col_tuples.append((l1, l2, l3, l4, m, ind))

columns = pd.MultiIndex.from_tuples(
    col_tuples,
    names=["group", "mouse_ids", "sex", "condition", "metric", "individual"]
)

# -----------------------------
# 4) Initialize empty DF (length n_frames)
# -----------------------------
df = pd.DataFrame(
    index=np.arange(n_frames),
    columns=columns,
    dtype=float
)

# optional: sort columns
#df = df.sort_index(axis=1)

group = df.columns.levels[0]
ids = df.columns.levels[1]
sex = df.columns.levels[2]
condition = df.columns.levels[3]
metrics = df.columns.levels[4]

# Alles auswählen:
df.loc[:, (group, ids, sex, condition, metrics)]
#print(df.shape)
#print(df.columns.names)
#print(df.columns[:10])

for path in paths:
    dic = load_analysis(path)

    parts = os.path.basename(path).split('_')

    g = parts[0]
    id = "_".join(parts[2:5]) 
    s = parts[1]
    cs = parts[5].split('.')
    c = cs[0]
              
    print("Data:",[g, id, s, c])

    for metric in metrics:
        #print(metric)
        if metric == "center_x" or metric == "center_y":
            data = dic["centers_xy"]
        else:
            data = dic[metric]
        #print(len(data), data)
        if len(data) != 3:
            raise ValueError("Data shape not n_individuals")
        for i, ind in enumerate(individuals):


            data_singleanimal = data[i]
            print(metric, ind)
            if metric == "trajectories":
                traj_lens = []
                for arr in data_singleanimal:
                    traj_lens.append(len(arr))
                data_singleanimal = traj_lens
            if metric == "center_x":
                xs = []
                for (x, y) in data_singleanimal:
                    xs.append(x)
                data_singleanimal = xs
            if metric == "center_y":
                ys = []
                for (x, y) in data_singleanimal:
                    ys.append(y)
                data_singleanimal = ys

            #print(data_singleanimal)
            if len(data_singleanimal) > n_frames:
                data_singleanimal = data_singleanimal[0:n_frames]
            df.loc[0:len(data_singleanimal)-1, ([g], [id], [s], [c], [metric], [ind])] = data_singleanimal
            print(metric, ind, " inserted")
            #print(len(df.loc[0:len(data)-1, ([g], [id], [s], [c], [metric])]))
            #print(len(data))
            #print(len(df))
            

print(df[0:10])
safename = f"/single_mouse_datatest_{metric_names[0]}.csv"
df.to_csv(r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\all" + safename)
