from barplot import plot_barplot
from violinplot import plot_violinplot
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "visit_len"


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")


# # # GERMFREE VS GERMFREEPROP # # #

data = create_data_dic(csv_folder, individuals, "female", "germfree", metric, data_extraction_mode="raw", data_transform=(1/30))
data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", metric, dic=data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))

compare_two_groups_to_excel(data=data,group1="germfreeprop", group2="germfree", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visit_len_gf_vs_gfp_females_stats.xlsx")


plot_violinplot(data,
             colors =
             ["#D9D9D9", "#CCE6BB"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visitsviolin_gf_vs_gfp_female.pdf",
             ylabel="visit length (s)",
             scatterdata=False
             )