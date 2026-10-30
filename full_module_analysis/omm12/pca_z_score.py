

import argparse
import glob
import os
import tqdm

FPS = 30

path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def z_score_by_condition(features):
    """Standardize each metric within its condition; preserve metadata and NaNs."""
    metadata = ["group", "mouse_ids", "sex", "condition", "individual"]
    metrics = features.columns.difference(metadata, sort=False)
    result = features.copy()
    grouped = features.groupby("condition")[metrics]
    means = grouped.transform("mean")
    # Population standard deviation gives each nonconstant metric unit variance.
    stds = grouped.transform(lambda values: values.std(ddof=0))
    # Constant metrics become zero; missing observations remain missing.
    result[metrics] = (features[metrics] - means) / stds.mask(stds == 0, 1.0)
    return result


def pca_by_condition(features):
    """Fit one PCA per condition using the already standardized metrics."""
    metadata = ["group", "mouse_ids", "sex", "condition", "individual"]
    results = {}
    for condition in ["hab", "top1", "top2"]:
        subset = features.loc[features["condition"] == condition]
        values = subset.drop(columns=metadata)
        # Remove metrics without variance; impute missing z-scores with the mean (0).
        values = values.loc[:, values.var(ddof=0) > 0]
        missing = int(values.isna().sum().sum())
        values = values.fillna(0)
        if len(values) < 2 or values.shape[1] == 0:
            print(f"Skipping {condition}: need at least two observations and a varying metric.")
            continue

        pca = PCA(svd_solver="full")
        transformed = pca.fit_transform(values)
        components = [f"PC{i + 1}" for i in range(transformed.shape[1])]
        scores = subset[metadata].join(
            pd.DataFrame(transformed, index=subset.index, columns=components)
        )
        variance = pd.Series(pca.explained_variance_ratio_, index=components,
                             name="explained_variance_ratio")
        weights = pd.DataFrame(pca.components_.T, index=values.columns,
                               columns=components)
        results[condition] = {"pca": pca, "scores": scores,
                              "weights": weights, "explained_variance_ratio": variance}
        print(f"\nPCA {condition} ({missing} missing z-scores replaced with 0)")
        print(variance.to_string())
        print(scores.to_string(index=False))
    return results


def plot_pca_results(results):
    """Show scores, explained variance, and PC1/PC2 weights per condition."""
    if not results:
        return
    conditions = ["hab", "top1", "top2"]
    groups = pd.concat([r["scores"]["group"] for r in results.values()]).unique()
    cmap = plt.get_cmap("tab10")
    colors = {group: cmap(i % 10) for i, group in enumerate(groups)}

    fig_scores, axes_scores = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    fig_scores.suptitle("PCA scores (independent PCA per condition)")
    fig_variance, axes_variance = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    fig_variance.suptitle("Explained variance")

    for condition, ax_scores, ax_variance in zip(conditions, axes_scores, axes_variance):
        ax_scores.set_title(condition)
        ax_variance.set_title(condition)
        if condition not in results:
            for ax in [ax_scores, ax_variance]:
                ax.text(0.5, 0.5, "PCA unavailable", ha="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        result = results[condition]
        scores = result["scores"]
        variance = result["explained_variance_ratio"]
        if "PC2" in scores:
            for (group, sex), subset in scores.groupby(["group", "sex"], dropna=False):
                base = np.array(colors[group][:3])
                # Females: lighter group color; males: darker group color.
                sex_label = str(sex).strip().lower()
                if sex_label in {"f", "female", "females"}:
                    color = base + (1 - base) * 0.45
                elif sex_label in {"m", "male", "males"}:
                    color = base * 0.65
                else:
                    color = base
                ax_scores.scatter(subset["PC1"], subset["PC2"],
                                  color=color, label=f"{group} / {sex}", alpha=0.9)
            ax_scores.set_xlabel(f"PC1 ({variance['PC1']:.1%})")
            ax_scores.set_ylabel(f"PC2 ({variance['PC2']:.1%})")
            ax_scores.legend(title="Group / Sex")
        else:
            ax_scores.text(0.5, 0.5, "PC2 unavailable", ha="center", transform=ax_scores.transAxes)
            ax_scores.set_axis_off()

        positions = np.arange(1, len(variance) + 1)
        ax_variance.bar(positions, variance * 100, label="Individual")
        ax_variance.plot(positions, variance.cumsum() * 100, "o-",
                         color="black", label="Cumulative")
        ax_variance.set_xlabel("Principal component")
        ax_variance.set_ylabel("Explained variance (%)")
        ax_variance.set_xticks(positions)
        ax_variance.tick_params(axis="x", labelsize=8)
        ax_variance.set_ylim(0, 105)
        ax_variance.legend()

        weights = result["weights"]
        components = list(weights.columns[:2])
        fig_weights, axes_weights = plt.subplots(
            1, len(components), figsize=(14, max(4, 0.3 * len(weights))),
            squeeze=False, layout="constrained"
        )
        fig_weights.suptitle(f"{condition}: PCA component weights")
        for ax, component in zip(axes_weights[0], components):
            ordered = weights[component].sort_values()
            ax.barh(ordered.index, ordered.values)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{component} ({variance[component]:.1%})")
            ax.set_xlabel("Component weight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Extract per-mouse PCA features and z-score within each condition")
    parser.add_argument("files", nargs="*", help="CSV files (default: all CSVs in data folder)")
    args = parser.parse_args()
    files = args.files or glob.glob(os.path.join(path, "*.csv"))
    if not files:
        parser.error(f"No CSV files found in {path}")
    features = pd.concat([analyze_file(file) for file in tqdm.tqdm(files)], ignore_index=True)
    features = z_score_by_condition(features)
    print(features.to_string(index=False))
    pca_results = pca_by_condition(features)
    plot_pca_results(pca_results)
    return features, pca_results


if __name__ == "__main__":
    main()
