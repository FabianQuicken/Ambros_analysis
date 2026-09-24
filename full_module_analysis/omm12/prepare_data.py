import numpy as np

from load_multi_animal_csv import load_group_csvs

def create_data_list(
        data_path,
        individuals,
        sex,
        group,
        condition,
        metric
):
    df = load_group_csvs(csv_folder=data_path, group=group, metrics=[metric, "immobile_bouts"], sex=sex)
    ids = df.columns.get_level_values("mouse_ids").unique()

    data_names = []
    data_list = []

    for id in ids:
        for individual in individuals:
            if metric == "immobile_start":
                d = df.loc[:, (group, id, slice(None), condition, metric, individual)].to_numpy()
                lens = df.loc[:, (group, id, slice(None), condition, "immobile_bouts", individual)].to_numpy()
                d = np.where(lens > 30, d, np.nan)
            else:
                d = df.loc[:, (group, id, slice(None), condition, metric, individual)].to_numpy()
            event_indices = np.asarray(d, dtype=float).ravel()
            event_indices = event_indices[np.isfinite(event_indices)]
            data_list.append(event_indices)
            data_names.append(group)

    return data_names, data_list



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
    absolute_values=False,
    dic=None,
    update_dic=False,
    min_value=None,
    fraction_threshold=None,
    fraction_comparison=">",
):
    """Create the data dictionary, optionally keeping only values above a cutoff.

    ``min_value`` is applied to the raw metric data before extraction,
    transformation, or summary statistics. Values equal to or below the cutoff
    are replaced with NaN so they are ignored by the NaN-aware calculations.

    ``P90`` (or ``p90``) extracts the 90th percentile per individual.
    ``IQR`` (or ``iqr``) extracts the 75th minus the 25th percentile.
    ``fraction`` extracts the proportion of non-NaN samples satisfying
    ``fraction_comparison`` against ``fraction_threshold``. Comparisons may
    be ``>``, ``>=``, ``<``, ``<=``, ``==``, or ``!=``; the default is ``>``.
    The threshold uses metric units after absolute_values and min_value
    filtering, but before data_transform and log10_transform. An individual
    with no valid samples yields NaN. These modes apply data_transform and
    log10_transform to the extracted statistic, as with ``mean``; they do
    not use norm_to_time_present. They retain the 18,001-frame limit.

    ``raw_traces`` preserves full-length (frames, recordings) arrays, one per
    individual, ordered by mouse_ids then individuals. Unlike ``raw``, it does
    not flatten traces or apply the historical 18,001-frame limit.
    """
    data_extraction_mode = data_extraction_mode.lower()
    if data_extraction_mode == "fraction":
        comparisons = {
            ">": np.greater, ">=": np.greater_equal,
            "<": np.less, "<=": np.less_equal,
            "==": np.equal, "!=": np.not_equal,
        }
        if fraction_comparison not in comparisons:
            raise ValueError("fraction_comparison must be >, >=, <, <=, ==, or !=.")
        if (fraction_threshold is None or not np.isscalar(fraction_threshold)
                or not isinstance(fraction_threshold, (int, float, np.number))
                or not np.isreal(fraction_threshold) or not np.isfinite(fraction_threshold)):
            raise ValueError("fraction_threshold must be a finite real number.")
        compare = comparisons[fraction_comparison]

    df = load_group_csvs(csv_folder=data_path, group=group, metrics=[metric, "mice_presence", ""], sex=sex)

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
                frames = slice(None) if data_extraction_mode == "raw_traces" else slice(0, 18000)
                d = df.loc[frames, (group, id, slice(None), condition, metric, individual)].to_numpy()
                present = df.loc[:, (group, id, slice(None), condition, "mice_presence", individual)].to_numpy()
                if absolute_values:
                    d = np.abs(d)
                if min_value is not None:
                    d = np.where(d > min_value, d, np.nan)
                if data_extraction_mode == "mean":
                    value = np.nanmean(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "median":
                    value = np.nanmedian(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "max":
                    value = np.nanmax(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode in ("p90", "iqr", "fraction"):
                    valid = d[~np.isnan(d)]
                    if valid.size == 0:
                        value = np.nan
                    elif data_extraction_mode == "p90":
                        value = np.percentile(valid, 90)
                    elif data_extraction_mode == "iqr":
                        q25, q75 = np.percentile(valid, [25, 75])
                        value = q75 - q25
                    else:
                        value = np.mean(compare(valid, fraction_threshold))
                    value *= data_transform
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
                        value = 2 * value - 1
                    else:
                        value = np.nancumsum(d) * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "raw_traces":
                    value = d * data_transform
                    values.append(_log10_transform(value) if log10_transform else value)
                elif data_extraction_mode == "raw":
                    value = d * data_transform
                    if log10_transform:
                        value = _log10_transform(value)
                    values.extend(value)
                elif data_extraction_mode == "len":
                    value = np.sum(~np.isnan(d))
                    if norm_to_time_present:
                        value = value / np.nansum(present) * data_transform
                    
                    values.append(_log10_transform(value) if log10_transform else value)
                #print(id, condition, individual, np.nanmax(d))
        if data_extraction_mode in ("cumsum", "raw_traces"):
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
