from barplot import plot_barplot
from load_multi_animal_csv import load_group_csvs
import numpy as np


csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
sex = "female"

germfree_df = load_group_csvs(csv_folder=csv_folder, group="germfree", metrics="mice_cumdists", sex=sex)

groups = germfree_df.columns.get_level_values("group").unique()
ids = germfree_df.columns.get_level_values("mouse_ids").unique()
sexes = germfree_df.columns.get_level_values("sex").unique()
conditions = germfree_df.columns.get_level_values("condition").unique()
metrics = germfree_df.columns.get_level_values("metric").unique()
individuals = germfree_df.columns.get_level_values("individual").unique()

print(groups, ids, sexes, conditions, metrics, individuals)
print(germfree_df[0:10])

for id in ids:
    for condition in conditions:
        for individual in individuals:
            d = germfree_df.loc[:, ("germfree", id, slice(None), condition, "mice_cumdists", individual)].to_numpy()
            print(id, condition, individual, np.nanmax(d))
            data["hab"][]

data = {
            "plot1": {
                "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
                "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]},
            },
            "plot2": {
                "group1": {"mean": 3, "sd": 0.5, "values": [2.5, 3, 3.5]},
                "group2": {"mean": 4, "sd": 0.8, "values": [3.2, 4, 4.8]},
            },
        }

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")