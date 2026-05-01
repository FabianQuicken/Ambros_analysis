from load_multi_animal_csv import load_group_csvs
import numpy as np

def create_data_dic(data_path, individuals, sex, group, metric, data_extraction_mode = "mean", dic=None, update_dic=False):
    df = load_group_csvs(csv_folder=data_path, group=group, metrics=metric, sex=sex)

    conditions = df.columns.get_level_values("condition").unique()
    ids = df.columns.get_level_values("mouse_ids").unique()

    if not update_dic:
        data = {}
    else:
        data = dic

    for condition in conditions:
        if not update_dic:
            data[condition +] = {}
        data[condition][group] = {}

    for condition in conditions:
        values = []
        for id in ids:
            for individual in individuals:
                d = df.loc[:, (group, id, slice(None), condition, "mice_cumdists", individual)].to_numpy()
                if data_extraction_mode == "mean":
                    values.append(np.nanmean(d))
                elif data_extraction_mode == "max":
                    values.append(np.nanmax(d))
                #print(id, condition, individual, np.nanmax(d))
        data[condition][group]["mean"] = np.nanmean(values)
        data[condition][group]["sd"] = np.nanstd(values)
        data[condition][group]["values"] = values

    return data