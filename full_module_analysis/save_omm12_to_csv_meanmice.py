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
metric_names = [
    # hier deine Metrik-Header rein
    "mice_per_frame",
    "all_face",
    "all_body",
    "all_anogenital",
    "mice_distances",
    "immobile_per_frame",
    "center_per_frame"
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
        col_tuples.append((l1, l2, l3, l4, m))

columns = pd.MultiIndex.from_tuples(
    col_tuples,
    names=["group", "mouse_ids", "sex", "condition", "metric"]
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
df = df.sort_index(axis=1)

group = df.columns.levels[0]
ids = df.columns.levels[1]
sex = df.columns.levels[2]
condition = df.columns.levels[3]
metrics = df.columns.levels[4]

print(group)
print(ids)
print(sex)
print(condition)
print(metrics)

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
              
    print([g, id, s, c])

    for metric in metrics:
        data = dic[metric]
        if metric == "mice_distances":
            data = np.nansum(data, axis=0)
        if len(data) > n_frames:
            data = data[0:n_frames]
        print(len(df.loc[0:len(data)-1, ([g], [id], [s], [c], [metric])]))
        print(len(data))
        print(len(df))
        df.loc[0:len(data)-1, ([g], [id], [s], [c], [metric])] = data

print(df[0:10])

df.to_csv(r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\all\data.csv")