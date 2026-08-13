


from pathlib import Path
import pandas as pd 
import numpy as np
from tqdm import tqdm


data_path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
save_path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4\all_trajectories"

CSV_COLUMN_LEVELS = ["group", "mouse_ids", "sex", "condition", "metric", "individual"]


individuals = ["mouse_1", "mouse_2", "mouse_3"]


csv_folder = Path(data_path)
csv_paths = sorted(csv_folder.glob("*.csv"))






for csv_path in tqdm(csv_paths):

    if "hab" in csv_path.name:
        continue

    main_df = pd.read_csv(csv_path, header=[0, 1, 2, 3, 4, 5], index_col=0)
    main_df.columns = main_df.columns.set_names(CSV_COLUMN_LEVELS)

    # get metadata from the columns
    group = main_df.columns.get_level_values("group").unique()[0]
    sex = main_df.columns.get_level_values("sex").unique()[0]
    condition = main_df.columns.get_level_values("condition").unique()[0]
    mouse_id = main_df.columns.get_level_values("mouse_ids").unique()[0]



    for ind in individuals:

        traj_num = 1

        df = main_df.copy()

        centroid_x = df.loc[:, (slice(None), slice(None), slice(None), slice(None), "centers_x", ind)].to_numpy().ravel()
        centroid_y = df.loc[:, (slice(None), slice(None), slice(None), slice(None), "centers_y", ind)].to_numpy().ravel()
        visit_start = df.loc[:, (slice(None), slice(None), slice(None), slice(None), "visit_start", ind)].to_numpy().ravel()
        visit_len = df.loc[:, (slice(None), slice(None), slice(None), slice(None), "visit_len", ind)].to_numpy().ravel()



        for i, start in tqdm(enumerate(visit_start)):
            
            if not np.isnan(start):

                start = int(start)

                length = int(visit_len[i])
                traj_x = centroid_x[start:start+length]
                traj_y = centroid_y[start:start+length]

                traj = {
                    "traj_x": traj_x,
                    "traj_y": traj_y,
                }

                safename = group + "_" + sex + "_" + condition + "_" + mouse_id + "_" + ind + "_traj_" + str(traj_num) + ".xlsx"
                traj_num += 1   
                
                df = pd.DataFrame(traj)
                
                df.to_excel(Path(save_path) / safename, index=False)
