import numpy as np
import pandas as pd
import glob
import os
from config import DF_COLS

def interpolate_with_max_gap(df, max_gap=30, method="linear"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    #print(num_cols)

    # 1) Nur „echte“ Interpolation zwischen gültigen Punkten
    out[num_cols] = out[num_cols].interpolate(method=method,
                                              limit_direction="both",
                                              limit_area="inside")
    
    # 2) NaN-Runs > max_gap identifizieren und wieder auf NaN setzen
    for col in num_cols:
        s = df[col]  # Original mit NaNs
        # Gruppen-IDs zwischen Nicht-NaNs erstellen
        grp = s.notna().cumsum()
        # Länge jedes NaN-Runs
        run_len = s.isna().groupby(grp).transform("sum")
        # Maske: Positionen in zu langen NaN-Runs
        too_long = s.isna() & (run_len > max_gap)
        # Zurücksetzen
        out.loc[too_long, col] = np.nan
    
    return out

def likelihood_filtering_nans(df, likelihood_row_name=str, filter_val=0.3):
    """
    DeepLabCut provides a likelihood for the prediction of 
    each bodypart in each frame to be correct. Filtering predictions
    for the likelihood, replaces values in the entire row with NaN where the likelihood is below the filter_val.
    """
    df_filtered = df.copy()  # Make a copy to avoid modifying the original DataFrame
    filtered_rows = df_filtered[likelihood_row_name] < filter_val
    df_filtered.loc[filtered_rows] = np.nan
    num_replaced = filtered_rows.sum()
    #print(f"The filter replaced values in {num_replaced} rows with NaN out of a total of {len(df)} rows.")
    return df_filtered

def likelihood_filtering(df, likelihood_row_name=str, filter_val = 0.3):
    """
    DeepLabCut provides a likelihood for the prediction of 
    each bodypart in each frame to be correct. Filtering predictions
    for the likelihood, reduces false predictions in the dataset.
    """
    df_filtered = df.copy()
    df_filtered = df[df[likelihood_row_name] > filter_val]
    df_removed_rows = df[df[likelihood_row_name] < filter_val]
    #print(f"The filter removed {len(df_removed_rows)} rows of a total of {len(df)} rows.")
    return df_filtered

def transform_dlcdata(filepath, keypoints, new_indices):

    """
    Transforms DeepLabCut (DLC) CSV output into a clean and structured DataFrame.

    This function reads a DLC-generated .csv file, checks whether its structure matches the
    expected column names (`new_indices`), and extracts the specified bodyparts (`keypoints`).
    It discards the first three metadata rows (scorer, bodyparts, coordinate type),
    inverts the y-coordinates (to match conventional coordinate systems), and returns
    a float-typed DataFrame containing only the bodyparts of interest.

    Parameters
    ----------
    filepath : str
        Path to the DeepLabCut .csv file to be processed.

    keypoints : list of str
        List of bodyparts to extract from the DLC data (e.g., ['nose', 'centroid', 'food1']).

    new_indices : list of str
        List of column names corresponding to the DLC CSV file.
        Must match the structure of the file exactly (e.g., ['nose_x', 'nose_y', 'nose_likelihood', ...]).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing x, y (inverted), and likelihood columns for each selected bodypart.

    Raises
    ------
    ValueError
        If the file structure does not match the provided `new_indices`, or if parsing fails.

    Examples
    --------
    >>> df = transform_dlcdata("my_dlc_output.csv", ["nose", "centroid"], DF_COLS)
    >>> df.columns
    Index(['nose_x', 'nose_y', 'nose_likelihood', 'centroid_x', ...])
    """
    
    # Erst testen, ob die DLC Daten zu den erwarteten DF_COLS passen. Falls nicht, würden die Daten falsch benannt werden
    dlc_data_as_expected = True
    try:
        raw_dlc_df = pd.read_csv(filepath)
        bodypart_list = raw_dlc_df.iloc[0, :].tolist()
        bodypart_list.pop(0)

        for index, bodypart in enumerate(bodypart_list):
            if index >= len(new_indices):
                dlc_data_as_expected = False
                break
            if not new_indices[index].startswith(bodypart):
                dlc_data_as_expected = False
                break 

    except Exception as e:
        print(f"Error while checking DLC format: {e}")
        dlc_data_as_expected = False 

    if dlc_data_as_expected:
        
        # erstellt aus DeepLabCut .csv ein Dataframe mit einer sinnvolleren Indexzeile 
        df = pd.read_csv(filepath, names=new_indices)
        data = df.copy() # doesn't change the original DataFrame
        # dlc hat drei unnötige indexzeilen, einmal der Netzwerkname, dann die bodyparts und ob es sich um x/y koordinate oder likelihood handelt
        data = data.iloc[3:]
        data = data.astype(float)

        #dataframe kopieren & bodyparts of interest extrahieren
        empty_df = pd.DataFrame()
        bodypart_df = empty_df.copy()

        for bodypart in keypoints:
            bodypart_df[bodypart+"_x"] = data[bodypart+"_x"]
            bodypart_df[bodypart+"_y"] = data[bodypart+"_y"]*(-1)  # y invertieren da DLC y koordinaten aufsteigen
            bodypart_df[bodypart+"_likelihood"] = data[bodypart+"_likelihood"]

        return bodypart_df

    else:
        raise ValueError("Could not transform DLC Data. Check if DF_COLS fits your data structure.")

