from cumsum_plot import plot_cumsum
from prepare_data import create_data_dic

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
metric = "mice_in_center"

data = create_data_dic(csv_folder, individuals, "male", "omm12", metric, data_extraction_mode="cumsum", norm_to_time_present=True, data_transform=1)
data = create_data_dic(csv_folder, individuals, "male", "ommpgol", metric, dic=data, update_dic=True, data_extraction_mode="cumsum", norm_to_time_present=True, data_transform=1)






plot_cumsum(data,
             colors =
             ["#C0DEFC", "#E58DF1"],
             stylemode="light",
             savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\cumsumcenterpref_gf_vs_gfp_male.pdf",
             ylabel="% in center",
             ylim=(0, 1)
             )