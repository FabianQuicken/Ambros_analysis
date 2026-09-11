from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

from analysis import load_experiment_data


def find_experiment_files(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Folder does not exist: {folder_path}"
        )

    experiment_files = [
        str(file)
        for file in folder_path.rglob("experiment.h5")
        if file.is_file()
    ]

    return sorted(experiment_files)

paths = find_experiment_files(r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli")

"""
for file in tqdm(paths):
    metrics_df = pd.read_hdf(file, key="metrics")

    name = metrics_df.columns.get_level_values("name").unique().item()

    
    individuals = metrics_df.columns.get_level_values("individuals").unique()

    metrics = metrics_df.columns.get_level_values("metrics").unique()

    exp_len = len(metrics_df)

    visible = metrics_df.loc[:, (name, individuals, "visible")].to_numpy()
    fraction_module = np.nansum(visible) / exp_len * 100

    visit_lens = metrics_df.loc[:, (name, individuals, "trajectory_length")].to_numpy()
    mean_visit_len = np.nanmean(visit_lens) / 30
    n_visits = len(visit_lens[visit_lens > 1])


    print(f"{individuals[0]} at day {name[0:10]} on camera {name[-4:]}")
    print("percent in module", np.round(fraction_module, decimals = 2))
    print("mean visit length in s", np.round(mean_visit_len, decimals = 1))
    print("n visits", n_visits)

"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def create_module_summary(paths, folder_path, fps=30):

    results = []

    for file in tqdm(paths):
        metrics_df = pd.read_hdf(file, key="metrics")

        names = metrics_df.columns.get_level_values("name").unique()

        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one name in {file}, found: {names.tolist()}"
            )

        name = names.item()

        individuals = (
            metrics_df.columns
            .get_level_values("individuals")
            .unique()
        )

        exp_len = len(metrics_df)

        for individual in individuals:

            visible = metrics_df.loc[
                :, (name, individual, "visible")
            ].to_numpy(dtype=float)

            fraction_module = (
                np.nansum(visible) / exp_len * 100
            )

            visit_lens = metrics_df.loc[
                :, (name, individual, "trajectory_length")
            ].to_numpy(dtype=float)

            # Nur tatsächliche Visits mit einer Länge > 1 Frame
            valid_visit_lens = visit_lens[
                np.isfinite(visit_lens) & (visit_lens > 1)
            ]

            n_visits = valid_visit_lens.size

            if n_visits > 0:
                mean_visit_len = np.mean(valid_visit_lens) / fps
            else:
                mean_visit_len = np.nan

            results.append({
                "maus": individual,
                "tag": name[:10],
                "kamera": name[-4:],
                "percent_in_module[%]": round(fraction_module, 2),
                "mean_visit_len[s]": round(mean_visit_len, 1),
                "n_visits": n_visits
            })

    summary_df = pd.DataFrame(
        results,
        columns=[
            "maus",
            "tag",
            "kamera",
            "percent_in_module[%]",
            "mean_visit_len[s]",
            "n_visits"
        ]
    )

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    output_path = folder_path / "module_summary.xlsx"

    summary_df.to_excel(
        output_path,
        index=False
    )

    print(f"Excel file saved to: {output_path}")

    return summary_df

create_module_summary(paths, r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli", 30)
    