import numpy as np
import pandas as pd
from scipy import stats


def compare_two_groups_to_excel(
    data_dict,
    group1_name="group1",
    group2_name="group2",
    output_path="statistical_results.xlsx",
    alpha=0.05
):
    """
    Vergleicht zwei ungepaarte Gruppen pro Datensatz statistisch.

    Erwartetes Format:
    data_dict = {
        "metric_name_1": {
            "group1": [..],
            "group2": [..]
        },
        "metric_name_2": {
            "group1": [..],
            "group2": [..]
        }
    }

    Output:
    Excel-Datei mit Rohdaten, Testname und p-Wert.
    """

    results = []

    for data_name, groups in data_dict.items():

        g1 = np.asarray(groups[group1_name], dtype=float)
        g2 = np.asarray(groups[group2_name], dtype=float)

        # NaNs entfernen
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        # Normalverteilung pro Gruppe testen
        shapiro_g1 = stats.shapiro(g1).pvalue if len(g1) >= 3 else np.nan
        shapiro_g2 = stats.shapiro(g2).pvalue if len(g2) >= 3 else np.nan

        normal_g1 = shapiro_g1 > alpha
        normal_g2 = shapiro_g2 > alpha
        both_normal = normal_g1 and normal_g2

        # Varianzhomogenität testen
        levene_p = stats.levene(g1, g2).pvalue
        equal_variance = levene_p > alpha

        # Passenden Test wählen
        if both_normal:
            if equal_variance:
                test_name = "Unpaired t-test"
                p_value = stats.ttest_ind(g1, g2, equal_var=True).pvalue
            else:
                test_name = "Welch's t-test"
                p_value = stats.ttest_ind(g1, g2, equal_var=False).pvalue
        else:
            test_name = "Mann-Whitney U test"
            p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided").pvalue

        results.append({
            "data_name": data_name,
            f"raw_data_{group1_name}": list(g1),
            f"raw_data_{group2_name}": list(g2),
            "shapiro_p_group1": shapiro_g1,
            "shapiro_p_group2": shapiro_g2,
            "levene_p": levene_p,
            "test_used": test_name,
            "p_value": p_value
        })

    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)

    return df