from pathlib import Path
from tqdm import  tqdm
import os

import pandas as pd


CSV_COLUMN_LEVELS = ["group", "mouse_ids", "sex", "condition", "metric", "individual"]


def load_group_csvs(csv_folder, group, metrics, sex="both", pattern="*.csv"):
    """
    Load exported multi-animal CSV files and return one dataframe for one group.

    Parameters
    ----------
    csv_folder : str or Path
        Folder that contains the CSV files written by main_multi_animal_to_csv.py.
    group : str
        Group to keep, for example "germfree" or "omm12".
    metrics : str or list[str]
        Metric name(s) to keep, for example "mice_presence" or
        ["mice_presence", "mice_cumdists"].
    sex : str or list[str], default "both"
        Use "males", "females", "both", or a list like ["males", "females"].
    pattern : str, default "*.csv"
        Glob pattern for CSV files inside csv_folder.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing all matching columns from all matching CSV files.
        Columns keep the original MultiIndex levels:
        group, mouse_ids, sex, condition, metric, individual.
    """
    csv_folder = Path(csv_folder)
    csv_paths = sorted(csv_folder.glob(pattern))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_folder} with pattern {pattern!r}.")

    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)

    sex_filter = _normalize_sex_filter(sex)
    dataframes = []
    # reading in single dataframes...
    for csv_path in tqdm(csv_paths):
        if not _has_exact_token(filename=csv_path, token=group):
            continue
        df = pd.read_csv(csv_path, header=[0, 1, 2, 3, 4, 5], index_col=0)
        df.columns = df.columns.set_names(CSV_COLUMN_LEVELS)

        columns = df.columns
        keep = (
            (columns.get_level_values("group") == group)
            & columns.get_level_values("metric").isin(metrics)
        )

        if sex_filter is not None:
            keep &= columns.get_level_values("sex").isin(sex_filter)

        if keep.any():
            dataframes.append(df.loc[:, keep])

    if not dataframes:
        raise ValueError(
            f"No matching data found for group={group!r}, sex={sex!r}, metrics={metrics!r}."
        )

    group_df = pd.concat(dataframes, axis=1)
    group_df = group_df.loc[:, ~group_df.columns.duplicated()]
    group_df = group_df.sort_index(axis=1)

    return group_df


def _normalize_sex_filter(sex):
    if sex is None or sex == "both":
        return None

    if isinstance(sex, str):
        sex = [sex]

    normalized = []
    for value in sex:
        if value in ("male", "m"):
            normalized.append("males")
        elif value in ("female", "f"):
            normalized.append("females")
        else:
            normalized.append(value)

    return normalized

def _has_exact_token(filename, token=""):
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    parts = name.split("_")
    return token in parts
