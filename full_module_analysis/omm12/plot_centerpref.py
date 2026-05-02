from barplot import plot_barplot
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel

PIXEL_PER_CM = 36.39


csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "mice_in_center"


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")

px_to_m = 1 / PIXEL_PER_CM / 100

# # # GERMFREE VS GERMFREEPROP # # #

data = create_data_dic(csv_folder, individuals, "male", "germfree", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "male", "germfreeprop", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="germfreeprop", group2="germfree", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_gf_vs_gfp_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#D9D9D9", "#CCE6BB"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_gf_vs_gfp_male.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )

data = create_data_dic(csv_folder, individuals, "female", "germfree", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="germfreeprop", group2="germfree", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_gf_vs_gfp_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#D9D9D9", "#CCE6BB"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_gf_vs_gfp_female.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )


# # # OMM12 VS OMM12PROP # # #


data = create_data_dic(csv_folder, individuals, "male", "omm12", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "male", "omm12prop", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="omm12prop", group2="omm12", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_omm12prop_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#bef49d"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_omm12prop_males.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )

data = create_data_dic(csv_folder, individuals, "female", "omm12", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "female", "omm12prop", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="omm12", group2="omm12prop", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_omm12prop_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#bef49d"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_omm12prop_females.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )

# # # OMM12 VS OMMPGOL # # #

data = create_data_dic(csv_folder, individuals, "male", "omm12", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "male", "ommpgol", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="omm12", group2="ommpgol", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_ommpgol_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#E58DF1"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_ommpgol_males.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )

data = create_data_dic(csv_folder, individuals, "female", "omm12", metric, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)
data = create_data_dic(csv_folder, individuals, "female", "ommpgol", metric, dic=data, update_dic=True, data_extraction_mode="sum", norm_to_time_present=True, data_transform=100)

compare_two_groups_to_excel(data=data,group1="omm12", group2="ommpgol", metric_name=metric, target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_ommpgol_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#E58DF1"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\centerpref_omm12_vs_ommpgol_females.pdf",
             ylabel="% in center",
             ylim=(0, 100)
             )