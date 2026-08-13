from barplot import plot_barplot
from violinplot import plot_violinplot
from prepare_data import create_data_dic

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "speedevents"
sex = "male"

sum_speedevents_data = create_data_dic(csv_folder, individuals, sex, "germfree", metric, data_extraction_mode="len", norm_to_time_present=True, data_transform= (30*60*60))
sum_speedevents_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", metric, dic=sum_speedevents_data, update_dic=True, data_extraction_mode="len", norm_to_time_present=True, data_transform= (30*60*60))
sum_speedevents_data = create_data_dic(csv_folder, individuals, sex, "omm12", metric, dic=sum_speedevents_data, update_dic=True, data_extraction_mode="len", norm_to_time_present=True, data_transform= (30*60*60))
sum_speedevents_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", metric, dic=sum_speedevents_data, update_dic=True, data_extraction_mode="len", norm_to_time_present=True, data_transform= (30*60*60))
sum_speedevents_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", metric, dic=sum_speedevents_data, update_dic=True, data_extraction_mode="len", norm_to_time_present=True, data_transform= (30*60*60))

plot_barplot(
    data=sum_speedevents_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=colors,
    conditions = ["hab", "top2"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\sum_speedevents_hab_top2" + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 50),
    ylabel="speedevents",
    stylemode="light",
)

plot_barplot(
    data=sum_speedevents_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=colors,
    conditions = ["top1"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\sum_speedevents_top1" + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 300),
    ylabel="speedevents",
    stylemode="light",
)