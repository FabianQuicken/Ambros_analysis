import numpy as np
import pandas as pd
import glob
import os
from config import DF_COLS
from utils import mouse_center, euklidean_distance

def filter_id_overlays(overlay_inds, overlay_slices, scorer, bodyparts, df):
    """
    Resolve identity overlay artifacts between two individuals by invalidating
    duplicated tracking data within detected overlay segments.

    This function operates on overlay segments previously identified by
    `find_id_overlay`, where two individuals share identical center coordinates
    for contiguous frame ranges. Such overlays typically occur in multi-animal
    DeepLabCut tracking when fewer than the maximum number of animals are visible,
    causing a newly appearing identity to be assigned to an already tracked animal.

    For each overlay segment [start, end), the function determines which individual
    represents the physically continuous track and invalidates (sets to NaN) the
    other individual's data (x, y, likelihood) within the overlay segment.

    Decision logic (applied per overlay segment)
    ---------------------------------------------
    1. Presence check at frame `start - 1`:
       An individual is considered present if more than half of its x
       values across the specified bodyparts are finite.
    2. If only one individual was present at `start - 1`, that individual is kept and
       the other is invalidated within the overlay segment.
    3. If neither individual was present at `start - 1`, the second individual
       (`overlay_inds[1]`) is invalidated by convention.
    4. If both individuals were present at `start - 1`, continuity is used as a
       tie-breaker: the individual with the smaller mean Euclidean jump distance
       between frames `start - 1` and `start` (averaged across bodyparts) is kept,
       while the other is invalidated.

    Overlay segments are assumed to follow Python slicing conventions, i.e. each
    segment is defined as (start, end) with `end` being exclusive. Because `.loc`
    slicing is inclusive, invalidation is applied to frames [start, end - 1].

    Parameters
    ----------
    overlay_inds : sequence of str
        Exactly two individual identifiers (e.g. ["mouse1", "mouse2"]) involved in
        the overlay.
    overlay_slices : sequence of tuple(int, int)
        Overlay segments as (start, end) tuples, where `end` is exclusive.
    scorer : str
        Name of the DeepLabCut scorer used in the DataFrame.
    bodyparts : sequence of str
        Bodyparts considered for presence estimation and jump-distance computation.
    df : pandas.DataFrame
        DeepLabCut output DataFrame with a MultiIndex column structure
        (scorer, individual, bodypart, coordinate).

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with overlay artifacts removed. The input DataFrame
        is edited in place and returned for convenience.

    Notes
    -----
    - This function assumes that likelihood filtering and interpolation have already
      been applied upstream.
    - The method is designed to resolve true overlay artifacts and does not address
      identity swaps that occur without spatial overlap.
    - Jump distances are computed using the user-defined `euklidean_distance`
      utility function.

    See Also
    --------
    find_id_overlay : Detection of temporal overlay segments between individuals.
    mouse_center : Computation of per-frame center coordinates for individuals.
    """
    def mean_jump_distance(df, scorer, ind, bodyparts, prev_frame, cur_frame):
        dists = []

        for bp in bodyparts:
            x_prev, y_prev = df.loc[prev_frame, (scorer, ind, bp, ["x", "y"])]
            x_cur,  y_cur  = df.loc[cur_frame,  (scorer, ind, bp, ["x", "y"])]

            if np.isfinite(x_prev) and np.isfinite(y_prev) and np.isfinite(x_cur) and np.isfinite(y_cur):
                dists.append(
                    euklidean_distance(x_prev, y_prev, x_cur, y_cur)
                )

        return np.nanmean(dists) if len(dists) > 0 else np.inf

    # get data first
    for start, end in overlay_slices:

        ind1_previous_data = df.loc[start-1, (scorer, overlay_inds[0], bodyparts, ["x"])].to_numpy().ravel()
        ind2_previous_data = df.loc[start-1, (scorer, overlay_inds[1], bodyparts, ["x"])].to_numpy().ravel()

        ind1_previous_present = False
        ind2_previous_present = False

        # schauen ob die Mäuse vorher präsent waren
        if np.sum(np.isfinite(ind1_previous_data)) > len(ind1_previous_data) / 2:
            ind1_previous_present = True
        if np.sum(np.isfinite(ind2_previous_data)) > len(ind2_previous_data) / 2:
            ind2_previous_present = True

        # wenn ind 1 vorher präsent war, aber nicht ind 2, werden die daten von maus 2 mit nan ersetzt
        if ind1_previous_present and not ind2_previous_present:
            df.loc[start:end-1, (scorer, overlay_inds[1], bodyparts, ["x", "y", "likelihood"])] = np.nan
        # wenn ind 2 vorher präsent war, aber nicht ind 1, werden die daten von maus 1 mit nan ersetzt
        if ind2_previous_present and not ind1_previous_present:
            df.loc[start:end-1, (scorer, overlay_inds[0], bodyparts, ["x", "y", "likelihood"])] = np.nan
        # wenn keine von beiden vorher im modul waren, ist es egal welche deleted wird, wir deleted einfach das 2. ind
        if not ind2_previous_present and not ind1_previous_present:
            df.loc[start:end-1, (scorer, overlay_inds[1], bodyparts, ["x", "y", "likelihood"])] = np.nan
        # wenn beide präsent waren, schauen welche koordinaten weiter "springen" zwischen dem frame vor dem doppellabel und 
        # dem ersten doppelframe
        if ind1_previous_present and ind2_previous_present:
            
            jump_1 = mean_jump_distance(df, scorer, overlay_inds[0], bodyparts, start-1, start)
            jump_2 = mean_jump_distance(df, scorer, overlay_inds[1], bodyparts, start-1, start)

            if jump_1 <= jump_2:
                df.loc[start:end-1, (scorer, overlay_inds[1], bodyparts, ["x", "y", "likelihood"])] = np.nan
            else:
                df.loc[start:end-1, (scorer, overlay_inds[0], bodyparts, ["x", "y", "likelihood"])] = np.nan

    return df


def find_id_overlay(df, scorer, individuals, bodyparts):

    """
    Identify temporal segments in which two different individuals share identical
    center coordinates, indicating potential ID overlay artifacts in multi-animal
    DeepLabCut tracking data.

    For each pair of individuals (i < j), the function compares their center
    coordinates (x, y) frame-by-frame and detects contiguous frame ranges where
    both coordinates are exactly identical. Such segments typically arise when
    DeepLabCut assigns the same physical animal to multiple identities, often
    during identity swaps or interpolation artifacts.

    Center coordinates are obtained via `mouse_center(...)` and are assumed to be
    aligned in time across individuals.

    Parameters
    ----------
    df : pandas.DataFrame
        DeepLabCut output DataFrame with a MultiIndex column structure
        (scorer, individual, bodypart, coordinate).
    scorer : str
        Name of the DeepLabCut scorer used in the DataFrame.
    individuals : list of str
        List of individual identifiers to be compared pairwise.
    bodyparts : list of str
        Bodyparts used to compute the center position of each individual.

    Returns
    -------
    overlays_dic : dict
        Dictionary mapping individual pairs to lists of overlay segments.
        Keys are strings of the form "indA and indB".
        Values are lists of (start, end) tuples, where `start` is inclusive
        and `end` is exclusive, following Python slicing conventions
        (i.e., frames [start:end]).

        Example:
            {
                "mouse1 and mouse2": [(120, 145), (300, 312)],
                "mouse1 and mouse3": []
            }

    Notes
    -----
    - Overlay detection is based on exact equality of both x and y center coordinates.
      This is suitable when DLC outputs are quantized and when interpolated plateaus
      are considered indicative of identity overlays.
    - NaN values do not produce false positives, as NaN comparisons evaluate to False.
    - Only unique individual pairs are evaluated (i < j); reverse duplicates are omitted.
    - Segment boundaries are computed robustly, including overlays starting at the
      first frame or ending at the final frame.

    See Also
    --------
    mouse_center : Function used to compute per-frame center coordinates for each individual.
    """

    x_centers, y_centers = mouse_center(df, scorer, individuals, bodyparts)
    overlays_dic = {}


    for idx, ind in enumerate(individuals):
        counter = 0
        while counter < len(individuals):
            if counter > idx:
                overlays = []
                overlay_mask = np.where((x_centers[idx] == x_centers[counter]) & (y_centers[idx] == y_centers[counter]), 1, 0)

                if np.nansum(overlay_mask) > 0:
                    diff = np.diff(overlay_mask)
                    if overlay_mask[0] == 1:
                        starts = [0]
                        starts += list(np.where(diff == 1)[0] + 1)
                    else:
                        starts = list(np.where(diff == 1)[0] + 1)
                    ends = list(np.where(diff == -1)[0] + 1)
                    if overlay_mask[-1] == 1:
                        ends += [len(overlay_mask)]
                    for (start, end) in zip(starts, ends):
                        overlays.append((start, end))
                    overlays_dic[f"{ind} and {individuals[counter]}"] = overlays
                    
            counter += 1
    return overlays_dic


def ma_likelihood_filter(df, scorer, individuals, bodyparts, filter_value = 0.3):
    
    for ind in individuals:
        for bp in bodyparts:
            # likelihood als 1D-Array
            lh = df.loc[:, (scorer, ind, bp, "likelihood")]

            # Maske: True wo likelihood < threshold
            mask = lh < filter_value

            # x und y auf NaN setzen
            df.loc[mask, (scorer, ind, bp, "x")] = np.nan
            df.loc[mask, (scorer, ind, bp, "y")] = np.nan
            

    return df
            

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


def filter_prediction_fragments(
    df: pd.DataFrame,
    scorer: str,
    individuals,
    bodyparts,
    *,
    min_true_frames: int = 15,
    min_segment_frames: int | None = None,
    max_gap: int = 2,
    min_valid_fraction: float = 0.7,
    require_all_bodyparts: bool = False,
    set_likelihood_nan: bool = True,
) -> pd.DataFrame:
    """
    Remove short / fragmented prediction islands for each individual by setting x/y (and optionally likelihood) to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DeepLabCut-style MultiIndex columns: (scorer, individual, bodypart, coord) with coord in {"x","y","likelihood"}.
    scorer : str
        Scorer level key.
    individuals : list-like
        Individuals to process.
    bodyparts : list-like
        Bodyparts to consider for validity / NaN application.
    min_true_frames : int
        Minimum number of valid (True) frames inside a segment to keep it.
        Segments with fewer valid frames are removed.
    min_segment_frames : int | None
        Minimum overall segment span length (end-start+1) to keep it.
        If None, defaults to min_true_frames.
    max_gap : int
        Gaps (False runs) of length <= max_gap within a segment are bridged (treated as one fragment).
    min_valid_fraction : float
        Within a bridged segment, fraction of frames that must be valid to keep it.
        Helps remove "stuttering" predictions (valid-invalid-valid-...).
    require_all_bodyparts : bool
        If True: a frame is valid only if ALL bodyparts have finite x and y.
        If False: a frame is valid if ANY bodypart has finite x and y.
    set_likelihood_nan : bool
        If True, set likelihood to NaN for removed frames too (recommended).

    Returns
    -------
    df : pd.DataFrame
        Modified df with fragment frames set to NaN.
    """
    if min_segment_frames is None:
        min_segment_frames = min_true_frames
    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    n = len(df.index)
    if n == 0:
        return df

    removed_frames = 0
    for ind in individuals:
        # Build per-bodypart validity (finite x and y)
        bp_valid = []
        for bp in bodyparts:
            x = df.loc[:, (scorer, ind, bp, "x")].to_numpy()
            y = df.loc[:, (scorer, ind, bp, "y")].to_numpy()
            bp_valid.append(np.isfinite(x) & np.isfinite(y))

        bp_valid = np.vstack(bp_valid)  # shape: (n_bodyparts, n_frames)

        if require_all_bodyparts:
            frame_valid = np.all(bp_valid, axis=0)
        else:
            frame_valid = np.any(bp_valid, axis=0)

        true_idx = np.flatnonzero(frame_valid)
        if true_idx.size == 0:
            continue

        # Group valid frames into segments, bridging gaps up to max_gap
        segments = []
        start = true_idx[0]
        prev = true_idx[0]

        for i in true_idx[1:]:
            if (i - prev) <= (max_gap + 1):
                prev = i
            else:
                segments.append((start, prev))
                start = i
                prev = i
        segments.append((start, prev))

        # Decide which segments to remove
        remove_mask = np.zeros(n, dtype=bool)

        for s, e in segments:
            seg_len = (e - s + 1)
            seg_true = frame_valid[s:e+1].sum()
            valid_frac = seg_true / seg_len if seg_len > 0 else 0.0

            too_short = (seg_true < min_true_frames) or (seg_len < min_segment_frames)
            too_fragmented = (valid_frac < min_valid_fraction)

            if too_short or too_fragmented:
                remove_mask[s:e+1] = True

        if not remove_mask.any():
            continue
        
        removed_frames += sum(remove_mask)

        # Apply NaNs for removed frames across all bodyparts of this individual
        coords_to_nan = ["x", "y"] + (["likelihood"] if set_likelihood_nan else [])
        for bp in bodyparts:
            for coord in coords_to_nan:
                df.loc[remove_mask, (scorer, ind, bp, coord)] = np.nan

    return df, removed_frames