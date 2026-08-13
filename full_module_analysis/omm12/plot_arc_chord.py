from barplot import plot_barplot
from violinplot import plot_violinplot
from prepare_data import create_data_dic

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "fragment_arc_chord"
sex = "female"

arc_chord_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="median")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="median")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="median")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="median")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="median")

plot_barplot(
    data=arc_chord_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=colors,
    conditions = ["hab", "top1", "top2"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\median_fragment_arc_chord_" + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(1, 1.2),
    ylabel="median arc chord ratio per fragment",
    stylemode="light",
)
"""
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="raw")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="raw")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="raw")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="raw")
arc_chord_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=arc_chord_data, update_dic=True, data_extraction_mode="raw")


plot_violinplot(
    data=arc_chord_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["hab", "top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\mean_arc_chord_" + "_" + sex + "_violin.pdf",
    scatterdata=False,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 5),
    ylabel="arc chord ratio per trajectory",
    stylemode="light",
    showmeans=True,
    showmedians=True,
)
"""