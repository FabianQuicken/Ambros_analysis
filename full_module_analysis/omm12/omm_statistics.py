import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def compare_two_groups_to_excel(
    data,
    group1,
    group2,
    metric_name,
    target_path,
    filename=None,
    alpha=0.05,
):
    """
    Compare two groups from create_data_dic output and save statistics to Excel.

    Parameters
    ----------
    data : dict
        Output from create_data_dic:
        data[condition][group]["values"] = raw values.
    group1, group2 : str
        Group names to compare.
    metric_name : str
        Name of the metric written to the Excel file.
    target_path : str or Path
        Folder where the Excel file is saved. If this ends with ".xlsx", it is
        treated as the full output file path.
    filename : str, optional
        Excel filename. Defaults to "<metric_name>_<group1>_vs_<group2>_stats.xlsx".
    alpha : float, default 0.05
        Significance threshold for normality and equal-variance checks.

    Returns
    -------
    pandas.DataFrame
        Summary table that was written to Excel.
    """
    output_path = _make_output_path(target_path, filename, metric_name, group1, group2)

    summary_rows = []
    raw_rows = []

    for condition, condition_data in data.items():
        if group1 not in condition_data or group2 not in condition_data:
            continue

        values1 = _clean_values(condition_data[group1].get("values", []))
        values2 = _clean_values(condition_data[group2].get("values", []))

        normality1 = _shapiro_normality(values1, alpha)
        normality2 = _shapiro_normality(values2, alpha)
        variance = _levene_equal_variance(values1, values2, alpha)
        test = _choose_and_run_test(values1, values2, normality1, normality2, variance)

        summary_rows.append(
            {
                "metric": metric_name,
                "condition": condition,
                "group1": group1,
                "group2": group2,
                "group1_n": len(values1),
                "group2_n": len(values2),
                "group1_values": _values_to_string(values1),
                "group2_values": _values_to_string(values2),
                "group1_normality_test": normality1["test"],
                "group1_normality_p": normality1["p_value"],
                "group1_normal": normality1["normal"],
                "group2_normality_test": normality2["test"],
                "group2_normality_p": normality2["p_value"],
                "group2_normal": normality2["normal"],
                "equal_variance_test": variance["test"],
                "equal_variance_p": variance["p_value"],
                "equal_variance": variance["equal_variance"],
                "test_used": test["test"],
                "statistic": test["statistic"],
                "p_value": test["p_value"],
            }
        )

        raw_rows.extend(_make_raw_rows(metric_name, condition, group1, values1))
        raw_rows.extend(_make_raw_rows(metric_name, condition, group2, values2))

    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        raw_df.to_excel(writer, sheet_name="raw_values", index=False)

    return summary_df


def _make_output_path(target_path, filename, metric_name, group1, group2):
    target_path = Path(target_path)

    if target_path.suffix.lower() == ".xlsx":
        return target_path

    if filename is None:
        safe_metric = _safe_filename(metric_name)
        safe_group1 = _safe_filename(group1)
        safe_group2 = _safe_filename(group2)
        filename = f"{safe_metric}_{safe_group1}_vs_{safe_group2}_stats.xlsx"

    return target_path / filename


def _clean_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _shapiro_normality(values, alpha):
    if len(values) < 3:
        return {
            "test": "not tested: n < 3",
            "p_value": np.nan,
            "normal": False,
        }

    if np.nanvar(values) == 0:
        return {
            "test": "not tested: variance is zero",
            "p_value": np.nan,
            "normal": False,
        }

    result = stats.shapiro(values)
    return {
        "test": "Shapiro-Wilk",
        "p_value": result.pvalue,
        "normal": result.pvalue >= alpha,
    }


def _levene_equal_variance(values1, values2, alpha):
    if len(values1) < 2 or len(values2) < 2:
        return {
            "test": "not tested: n < 2",
            "p_value": np.nan,
            "equal_variance": False,
        }

    variance1 = np.nanvar(values1)
    variance2 = np.nanvar(values2)
    if variance1 == 0 and variance2 == 0:
        return {
            "test": "not tested: both variances are zero",
            "p_value": np.nan,
            "equal_variance": True,
        }

    deviations1 = np.abs(values1 - np.nanmedian(values1))
    deviations2 = np.abs(values2 - np.nanmedian(values2))
    if np.nanvar(deviations1) == 0 and np.nanvar(deviations2) == 0:
        return {
            "test": "not tested: equal median deviations",
            "p_value": np.nan,
            "equal_variance": True,
        }

    result = stats.levene(values1, values2, center="median")
    return {
        "test": "Levene",
        "p_value": result.pvalue,
        "equal_variance": result.pvalue >= alpha,
    }


def _choose_and_run_test(values1, values2, normality1, normality2, variance):
    if len(values1) == 0 or len(values2) == 0:
        return {
            "test": "not tested: missing values",
            "statistic": np.nan,
            "p_value": np.nan,
        }

    both_normal = normality1["normal"] and normality2["normal"]

    if both_normal and variance["equal_variance"]:
        result = stats.ttest_ind(values1, values2, equal_var=True, nan_policy="omit")
        test_name = "independent t-test"
    elif both_normal:
        result = stats.ttest_ind(values1, values2, equal_var=False, nan_policy="omit")
        test_name = "Welch t-test"
    else:
        result = stats.mannwhitneyu(values1, values2, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "test": test_name,
        "statistic": result.statistic,
        "p_value": result.pvalue,
    }


def _make_raw_rows(metric_name, condition, group, values):
    return [
        {
            "metric": metric_name,
            "condition": condition,
            "group": group,
            "value_index": index,
            "value": value,
        }
        for index, value in enumerate(values, start=1)
    ]


def _values_to_string(values):
    return ", ".join(str(value) for value in values)


def _safe_filename(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))
