import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\data.csv", header=[0,1,2,3,4], index_col=0)



idx = pd.IndexSlice
n_frames = 216_000

# germfree subset
germfree = df.loc[:n_frames-1, idx["germfree", :, :, :, "mice_per_frame"]]

ids = germfree.columns.get_level_values("mouse_ids").unique()

plt.figure(figsize=(12,6))

for mid in ids:

    top1 = germfree.loc[:, idx[:, mid, :, "top1", :]].to_numpy()
    top2 = germfree.loc[:, idx[:, mid, :, "top2", :]].to_numpy()

    # falls mehrere Spalten existieren → zusammenfassen
    #top1 = np.nansum(top1, axis=1)
    #top2 = np.nansum(top2, axis=1)

    diff = np.nancumsum(top1) - np.nancumsum(top2)

    plt.plot(diff, label=mid)

plt.xlabel("Frame")
plt.ylabel("Cumulative visits (top1 - top2)")
plt.title("Top1 vs Top2 module preference (germfree)")
plt.legend()
plt.show()

