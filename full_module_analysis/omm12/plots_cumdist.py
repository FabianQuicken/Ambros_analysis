from barplot import plot_barplot
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel

PIXEL_PER_CM = 36.39


csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")

px_to_m = 1 / PIXEL_PER_CM / 100

# # # GERMFREE VS GERMFREEPROP # # #

data = create_data_dic(csv_folder, individuals, "male", "germfree", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "male", "germfreeprop", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="germfreeprop", group2="germfree", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_gf_vs_gfp_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#D9D9D9", "#CCE6BB"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_gf_vs_gfp_male.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 95)
             )

data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="germfreeprop", group2="germfree", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_gf_vs_gfp_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#D9D9D9", "#CCE6BB"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_gf_vs_gfp_female.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 95)
             )


# # # OMM12 VS OMM12PROP # # #


data = create_data_dic(csv_folder, individuals, "male", "omm12", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "male", "omm12prop", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="omm12prop", group2="omm12", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_omm12prop_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#bef49d"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_omm12prop_males.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 75)
             )

data = create_data_dic(csv_folder, individuals, "female", "omm12", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "female", "omm12prop", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="omm12", group2="omm12prop", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_omm12prop_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#bef49d"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_omm12prop_females.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 115)
             )

# # # OMM12 VS OMMPGOL # # #

data = create_data_dic(csv_folder, individuals, "male", "omm12", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "male", "ommpgol", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="omm12", group2="ommpgol", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_ommpgol_males_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#E58DF1"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_ommpgol_males.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 125)
             )

data = create_data_dic(csv_folder, individuals, "female", "omm12", "mice_cumdists", data_transform=px_to_m)
data = create_data_dic(csv_folder, individuals, "female", "ommpgol", "mice_cumdists", dic=data, update_dic=True, data_transform=px_to_m)

compare_two_groups_to_excel(data=data,group1="omm12", group2="ommpgol", metric_name="mice_cumdists", target_path=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_ommpgol_females_stats.xlsx")

plot_barplot(data,
             colors =
             ["#C0DEFC", "#E58DF1"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_omm12_vs_ommpgol_females.pdf",
             ylabel="Cumulative distance (m)",
             ylim=(0, 115)
             )


