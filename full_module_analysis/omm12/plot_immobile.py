from violinplot import plot_violinplot
from barplot import plot_barplot
from prepare_data import create_data_dic, create_data_list
from rasterplot import rasterplot

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "immobile_bouts"
group = "germfree"
sex = "male"



immobile_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="sum", norm_to_time_present=True)
immobile_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=immobile_data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True)
immobile_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=immobile_data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True)
immobile_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=immobile_data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True)
immobile_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=immobile_data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True)

plot_barplot(
    data=immobile_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\immobile_time_" + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 1),
    ylabel="immobile time",
    stylemode="light",
)

"""
#bouts unter einer Sekunde werden rausgefiltert
bout_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False, min_value=30)
bout_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=bout_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False, min_value=30)
bout_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=bout_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False, min_value=30)
bout_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=bout_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False, min_value=30)
bout_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=bout_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False, min_value=30)  



plot_violinplot(
    data=bout_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\immobile_bout_lens_" + group + "_" + sex + "_violin.pdf",
    scatterdata=False,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 10),
    ylabel="immobile bout length [s]",
    stylemode="light",
)

names = []
data = []
groups = ["omm12", "omm12prop", "ommpgol", "germfree", "germfreeprop"]
condition = "hab"
sex = "female"
for group in groups:
    n, d = create_data_list(
            data_path=csv_folder,
            individuals=individuals,
            sex=sex,
            group=group,
            condition=condition,
            metric="immobile_start"
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
    savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4\rasterplot_immobilebouts_" + condition + "_" + sex + ".svg",
    ylabel=None,
    condition=condition,
    plotsize=(16, 9),
    fontsize=12,
    xlabel=None,
    xlim=None,
    ylim=None,
)
"""