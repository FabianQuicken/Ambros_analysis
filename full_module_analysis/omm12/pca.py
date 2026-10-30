

import argparse
import glob
import os
import tqdm

FPS = 30

path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"


import pandas as pd
import numpy as np

def extract_metric(df, recording, metric, individual):
    """Extract one column, never mixing recordings or conditions."""
    key = (*recording, metric, individual)
    selected = df.loc[:, [key]]
    if selected.shape[1] != 1:
        raise ValueError(f"Expected one column for {key}, found {selected.shape[1]}")
    return selected.iloc[:, 0].to_numpy(dtype=float)


def calculate_bout_lengths(array):
    """Lengths in frames; missing samples break bouts rather than bridge gaps."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 1:
        raise ValueError("Bout calculations require a one-dimensional trace")
    if not np.isin(array[np.isfinite(array)], [0, 1]).all():
        raise ValueError("Bout traces must contain only 0, 1, or missing values")
    array = (array == 1).astype(int)
    diffs = np.diff(np.concatenate(([0], array, [0])))
    entries = np.where(diffs == 1)[0]
    exits = np.where(diffs == -1)[0]
    boutlengths = exits - entries
    return boutlengths

def analyze_file(file):
    """Return one feature row per recording and individual."""
    print(file)

    df = pd.read_csv(file, header=[0, 1, 2, 3, 4, 5], index_col=0)

    expected_levels = ["group", "mouse_ids", "sex", "condition", "metric", "individual"]
    if list(df.columns.names) != expected_levels:
        raise ValueError(f"Unexpected column levels: {df.columns.names}")
    rows = []
    for group, mouse_ids, sex, condition, individual in df.columns.droplevel("metric").unique():
        recording = (group, mouse_ids, sex, condition)
        presence = extract_metric(df, recording, "mice_presence", individual)
        immobile = extract_metric(df, recording, "mice_immobile", individual)
        immobile_bout_lengths = extract_metric(df, recording, "immobile_bouts", individual)
        distances = extract_metric(df, recording, "mice_distances", individual)
        accelerations = extract_metric(df, recording, "mice_accelerations", individual)
        face_inv = extract_metric(df, recording, "face_inv", individual)
        body_inv = extract_metric(df, recording, "body_inv", individual)
        anogenital_inv = extract_metric(df, recording, "anogenital_inv", individual)
        in_center = extract_metric(df, recording, "mice_in_center", individual)
        thetas = extract_metric(df, recording, "thetas", individual)
        visit_len = extract_metric(df, recording, "visit_len", individual)
        speedevents = extract_metric(df, recording, "speedevents", individual)
        px_exploration = extract_metric(df, recording, "px_exploration_score", individual)
        frag_mean_arc_chord = extract_metric(df, recording, "fragment_arc_chord", individual)
        traj_mean_arc_chord = extract_metric(df, recording, "mean_arc_chord", individual) # ist der mittelwert der einzelnen trajectories in frag_mean_arc_chord
        orientations = extract_metric(df, recording, "orientations", individual) # lassen wir erstmal raus, da es sich mit thetas ein wenig überschneidet
        posture_compactness = extract_metric(df, recording, "posture_compactness", individual)
        bodylengths = extract_metric(df, recording, "mice_bodylength", individual)
        mean_likelihood = extract_metric(df, recording, "mice_mean_likelihood", individual)

        # normalize some metrics to the presence 
        presence_fraction = np.nansum(presence) / len(presence) if len(presence) else np.nan
        sum_presence = np.nansum(presence)
        # Fractions and rates are undefined for a mouse never observed.
        if sum_presence == 0:
            sum_presence = np.nan
        minutes_present = sum_presence / FPS / 60
        immobile_fraction = np.nansum(immobile) / sum_presence
        face_inv_fraction = np.nansum(face_inv) / sum_presence
        body_inv_fraction = np.nansum(body_inv) / sum_presence
        anogenital_inv_fraction = np.nansum(anogenital_inv) / sum_presence

        center_boutlengths = calculate_bout_lengths(in_center)
        # center metrics for pca
        median_center_boutlength = np.median(center_boutlengths) if center_boutlengths.size else np.nan
        center_entries_per_minute = len(center_boutlengths) / minutes_present
        in_center_fraction = np.nansum(in_center) / sum_presence

        # investigation bout lengths
        face_inv_boutlengths = calculate_bout_lengths(face_inv)
        median_face_inv_boutlength = np.median(face_inv_boutlengths) if face_inv_boutlengths.size else np.nan
        body_inv_boutlengths = calculate_bout_lengths(body_inv)
        median_body_inv_boutlength = np.median(body_inv_boutlengths) if body_inv_boutlengths.size else np.nan
        anogenital_inv_boutlengths = calculate_bout_lengths(anogenital_inv)
        median_anogenital_inv_boutlength = np.median(anogenital_inv_boutlengths) if anogenital_inv_boutlengths.size else np.nan

        # distance related metrics
        median_speed = np.nanmedian(distances)
        p90_speed = np.nanpercentile(distances, 90)
        iqr_speed = np.nanpercentile(distances, 75) - np.nanpercentile(distances, 25)
        median_acceleration = np.nanmedian(np.absolute(accelerations))
        p95_acceleration = np.nanpercentile(np.absolute(accelerations), 95)
        speedevents_per_minute = np.count_nonzero(np.isfinite(speedevents)) / minutes_present
        median_immobile_bl = np.nanmedian(immobile_bout_lengths)

        # thetas
        thetas = thetas[~np.isnan(thetas)]
        median_theta = np.nanmedian(np.absolute(thetas))
        p90_theta = np.nanpercentile(np.absolute(thetas), 90)

        # get large turn fraction
        abs_thetas = np.absolute(thetas)
        large_turns = abs_thetas[abs_thetas>45]
        large_turn_fraction = len(large_turns) / len(thetas)

        # visit_len bei hab auf den gleichen Wert setzen
        if condition == "hab":
            median_visit_len = 54000
        else:
            median_visit_len = np.nanmedian(visit_len)

        # body metrics
        median_bodylength = np.nanmedian(bodylengths)
        median_posture_compactness = np.nanmedian(posture_compactness)

        # exploratory metrics
        px_exploration_score = np.nanmax(px_exploration)
        median_fragment_arc_chord = np.nanmedian(frag_mean_arc_chord)
        p90_fragment_arc_chord = np.nanpercentile(frag_mean_arc_chord, 90)


        

        rows.append({
            "group": group,
            "mouse_ids": mouse_ids,
            "sex": sex,
            "condition": condition,
            "individual": individual,
            "presence_fraction": presence_fraction,
            "immobile_fraction": immobile_fraction,
            "face_inv_fraction": face_inv_fraction,
            "body_inv_fraction": body_inv_fraction,
            "anogenital_inv_fraction": anogenital_inv_fraction,
            "median_center_boutlength": median_center_boutlength,
            "center_entries_per_minute": center_entries_per_minute,
            "in_center_fraction": in_center_fraction,
            "median_face_inv_boutlength": median_face_inv_boutlength,
            "median_body_inv_boutlength": median_body_inv_boutlength,
            "median_anogenital_inv_boutlength": median_anogenital_inv_boutlength,
            "median_speed": median_speed,
            "p90_speed": p90_speed,
            "iqr_speed": iqr_speed,
            "median_acceleration": median_acceleration,
            "p95_acceleration": p95_acceleration,
            "speedevents_per_minute": speedevents_per_minute,
            "median_theta": median_theta,
            "p90_theta": p90_theta,
            "large_turn_fraction": large_turn_fraction,
            "median_immobile_boutlength": median_immobile_bl,
            "median_visit_len": median_visit_len,
            "median_bodylength": median_bodylength,
            "median_posture_compactness": median_posture_compactness,
            "px_exploration_score": px_exploration_score,
            "median_arc_chord": median_fragment_arc_chord,
            "p90_arc_chord": p90_fragment_arc_chord

        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract per-mouse PCA features")
    parser.add_argument("files", nargs="*", help="CSV files (default: all CSVs in data folder)")
    args = parser.parse_args()
    files = args.files or glob.glob(os.path.join(path, "*.csv"))
    if not files:
        parser.error(f"No CSV files found in {path}")
    features = pd.concat([analyze_file(file) for file in tqdm.tqdm(files)], ignore_index=True)
    print(features.to_string(index=False))
    return features


if __name__ == "__main__":
    main()
