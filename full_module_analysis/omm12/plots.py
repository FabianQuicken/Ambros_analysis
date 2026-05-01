from barplot import plot_barplot
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel



csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")

data = create_data_dic(csv_folder, individuals, "male", "germfreeprop", "mice_cumdists")
data = create_data_dic(csv_folder, individuals, "male", "germfree", "mice_cumdists", dic=data, update_dic=True)

compare_two_groups_to_excel(data, "germfreeprop", "germfree", r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4\stats.xlsx")

plot_barplot(data,
             colors =
             ["red", "green", "blue", "orange"],
             stylemode="dark",
             savepath=r"Z:\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumdist_gf_vs_gfp.svg"
             )




