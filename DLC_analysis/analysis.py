


# # # # # # # _________________________________________________________________________________________________________________

# Imports

# # # # # # # _________________________________________________________________________________________________________________

import glob as glob
import os

import numpy as np
import pandas as pd
import math

from tqdm import tqdm

from variables import FPS

# # # # # # # _________________________________________________________________________________________________________________

# Funktionen

# # # # # # # _________________________________________________________________________________________________________________

def load_experiment_data(folder_path):

    expected_working_df_headings = ["scorer", "individuals", "bodyparts", "coords"]

    expected_metrics_df_headings = ["name", "individuals", "metrics"]

    # read in correct file
    possible_files = glob.glob(os.path.join(folder_path, "*.h5"))
    experiment_file = None
    for file in possible_files:
        if "experiment" in os.path.basename(file):
            experiment_file = file
        else:
            raise ValueError(f"No experiment file found in {folder_path}")

    # read in dlc data and check if valid
    working_df = pd.read_hdf(experiment_file, key="pose")
    for heading in working_df.columns.names:
        if heading not in expected_working_df_headings:
            raise ValueError(f"Unexpected column heading found in dlc dataframe of {experiment_file}: {heading}")

    working_df = working_df.droplevel("scorer", axis=1)

    # read in empty metric df and check if valid
    metrics_df = pd.read_hdf(experiment_file, key="metrics")

    for heading in metrics_df.columns.names:
        if heading not in expected_metrics_df_headings:
            raise ValueError(f"Unexpected column heading found in metrics dataframe of {experiment_file}: {heading}")
        
    return working_df, metrics_df

def drop_dlc_columns(df, individuals=None, bodyparts=None):
    if individuals is not None:
        df = df.drop(columns=individuals, level="individuals")

    if bodyparts is not None:
        df = df.drop(columns=bodyparts, level="bodyparts")

    df.columns = df.columns.remove_unused_levels()

    return df

def add_metric_to_metric_df(metrics_df, name, individual, metric_name, metric_data):

    metrics_df = metrics_df.copy()

    if len(metric_data) != len(metrics_df):
        if len(metric_data) > len(metrics_df):
            raise ValueError(f"Unexpected metric length {len(metric_data)} does not fit into metrics df length of {len(metrics_df)}")
        else:
            array = np.full(len(metrics_df), np.nan)
            array[0:len(metric_data)] = metric_data
            metric_data = array


    metrics_df[(name, individual, metric_name)] = metric_data

    return metrics_df

def calculate_mean_likelihood(working_df, metrics_df, ind, bodyparts):

    likelihoods = working_df.loc[:, (ind, bodyparts, "likelihood")].to_numpy()

    # mean pro frame
    mean_likelihood = np.nanmean(likelihoods, axis=1)
    
    # speichern
    mean_likelihood = np.round(mean_likelihood, decimals=2)

    return mean_likelihood

# # # # # # # _________________________________________________________________________________________________________________

# Analysis Main 

# # # # # # # _________________________________________________________________________________________________________________

folder_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2\metric_analysis"

individuals_to_remove = None
bodyparts_to_remove = ["food1", "food2", "food3", "snicket"]

working_df, metrics_df = load_experiment_data(folder_path)

working_df = drop_dlc_columns(working_df, individuals_to_remove, bodyparts_to_remove)

name = metrics_df.columns.get_level_values("name").unique()

individuals = working_df.columns.get_level_values("individuals").unique()

bodyparts = working_df.columns.get_level_values("bodyparts").unique()



# einzene Metrics werden pro Individual erstellt und ins metric dataframe eingefügt
for individual in individuals:

    # mean likelihood als Maß für die Tracking Qualität
    mean_lh = calculate_mean_likelihood(working_df, metrics_df, individual, bodyparts)
    metrics_df = add_metric_to_metric_df(metrics_df, name, individual, "mean_likelihood", mean_lh)

    # speed

    # acceleration






