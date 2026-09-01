import glob
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

sexes = ["males_", "females_"]
groups = ["germfree_", "germfreeprop_", "omm12_", "omm12prop_", "ommpgol_"]
conditions = ["top1", "top2"]

trajectory_filepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4\all_trajectories"
savepath = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4"

trajectory_files = glob.glob(os.path.join(trajectory_filepath, "*.xlsx"))





for sex in sexes:
    for group in groups:
        traj_x = []
        traj_y = []

        for condition in conditions:
            for file in tqdm(trajectory_files):

                if sex in file and group in file and condition in file:
                    df = pd.read_excel(file, sheet_name=None)
                    traj_x.append(df["Sheet1"]["traj_x"])
                    traj_y.append(df["Sheet1"]["traj_y"]*-1)


            figure = plt.figure(figsize=(8, 6))
            for (x, y) in zip(traj_x, traj_y):
                plt.plot(x, y, color="black", alpha=0.05, linewidth=1)
            plt.ylim(0, 1150)
            plt.xlim(0, 2000)

            plt.savefig(os.path.join(savepath, f"trajectories_{group}{sex}{condition}.pdf"), dpi=300)
            plt.savefig(os.path.join(savepath, f"trajectories_{group}{sex}{condition}.svg"))
#            plt.show()
