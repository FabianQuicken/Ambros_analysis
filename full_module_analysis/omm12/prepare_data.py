import numpy as np

from load_multi_animal_csv import load_group_csvs


def create_data_dic(
    data_path,
    individuals,
    sex,
    group,
    metric,
    data_extraction_mode="mean",
    data_transform=1,
    log10_transform=False,
    norm_to_time_present=False,
    dic=None,
    update_dic=False,
):
    df = load_group_csvs(csv_folder=data_path, group=group, metrics=[metric, "mice_presence"], sex=sex)

    conditions = df.columns.get_level_values("condition").unique()
    ids = df.columns.get_level_values("mouse_ids").unique()

    if not update_dic:
        data = {}
    else:
        data = dic

    for condition in conditions:
        if not update_dic:
            data[condition] = {}
        
        data[condition][group] = {}

    for condition in conditions:
        values = []
        for id in ids:
            for individual in individuals:
                d = df.loc[:, (group, id, slice(None), condition, metric, individual)].to_numpy()
                present = df.loc[:, (group, id, slice(None), condition, "mice_presence", individual)].to_numpy()
                if data_extraction_mode == "mean":
                    value = np.nanmean(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "max":
                    value = np.nanmax(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "sum":
                    if norm_to_time_present:
                        value = np.nansum(d) / np.nansum(present) * data_transform
                    else:
                        value = np.nansum(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "cumsum":
                    if norm_to_time_present:
                        value = np.nancumsum(d) / np.nancumsum(present) * data_transform
                    else:
                        value = np.nancumsum(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "raw":
                    value = d * data_transform
                    if log10_transform:
                        value = _log10_transform(value)
                    values.extend(value)
                elif data_extraction_mode == "len":
                    value = np.sum(~np.isnan(d))
                    
                    values.append(_log10_transform(value) if log10_transform else value)
                #print(id, condition, individual, np.nanmax(d))
        if data_extraction_mode == "cumsum":
            data[condition][group]["values"] = values
            data[condition][group]["mean"] = np.nan
            data[condition][group]["sd"] = np.nan
        else:
            data[condition][group]["mean"] = np.nanmean(values)
            data[condition][group]["sd"] = np.nanstd(values)
            data[condition][group]["values"] = values

    return data


def _log10_transform(value):
    values = np.asarray(value, dtype=float)
    transformed = np.full(values.shape, np.nan, dtype=float)
    positive_mask = values > 0
    transformed[positive_mask] = np.log10(values[positive_mask])

    if transformed.ndim == 0:
        return float(transformed)

    return transformed
