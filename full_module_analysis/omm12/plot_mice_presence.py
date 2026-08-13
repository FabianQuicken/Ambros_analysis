from barplot import plot_barplot
from prepare_data import create_data_dic

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "mice_presence"
sex = "male"

presence_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="sum", data_transform=(1/(30*3600*0.5)))
presence_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=presence_data, update_dic=True, data_extraction_mode="sum", data_transform=(1/(30*3600*0.5)))
presence_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=presence_data, update_dic=True, data_extraction_mode="sum", data_transform=(1/(30*3600*0.5)))
presence_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=presence_data, update_dic=True, data_extraction_mode="sum", data_transform=(1/(30*3600*0.5)))
presence_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=presence_data, update_dic=True, data_extraction_mode="sum", data_transform=(1/(30*3600*0.5)))

plot_barplot(
    data=presence_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=colors,
    conditions = ["top1", "top2"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\presence_time_" + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 1),
    ylabel="mice presence fraction",
    stylemode="light",
)
