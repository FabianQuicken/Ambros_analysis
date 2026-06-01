from rasterplot import rasterplot
from prepare_data import create_data_list
from omm_statistics import compare_two_groups_to_excel




csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}

names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="female",
            group=group,
            condition="hab",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_hab_females.svg",
    ylabel=None,
    condition="hab",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)


names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="male",
            group=group,
            condition="hab",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_hab_males.svg",
    ylabel=None,
    condition="hab",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)


names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="female",
            group=group,
            condition="top1",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_top1_females.svg",
    ylabel=None,
    condition="top1",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)


names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="male",
            group=group,
            condition="top1",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_top1_males.svg",
    ylabel=None,
    condition="top1",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)

names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="female",
            group=group,
            condition="top2",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_top2_females.svg",
    ylabel=None,
    condition="top2",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)


names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex="male",
            group=group,
            condition="top2",
            metric="speedevents"
    )
    names.extend(n)
    data.extend(d)




rasterplot(
    data,
    names,
    30,
    x_time_unit="minutes",
    color=None,
    stylemode="dark",
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\rasterplot_top2_males.svg",
    ylabel=None,
    condition="top2",
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)