from barplot import plot_barplot
from visitplot import visitplot
from ecdf_plot import plot_ecdf
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
x_metric = "visit_start"
y_metric = "visit_len"


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")


# # # GERMFREE VS GERMFREEPROP # # #


y_data = create_data_dic(csv_folder, individuals, "female", "germfree", y_metric, data_extraction_mode="raw", data_transform=(1/30), log10_transform=True)
y_data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", y_metric, dic=y_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=True)

plot_ecdf(
    data=y_data,
    colors=["#D9D9D9","#CCE6BB"],
    markers=None,
    stylemode="light",
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visits_gf_vs_gfp_female_ecdf.pdf",
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel=None,
    ylabel="ECDF",
    xlim=None,
    ylim=(0, 1),
    linewidth=2,
)


"""
x_data = create_data_dic(csv_folder, individuals, "female", "germfree", x_metric, data_extraction_mode="raw", data_transform=(1/30))
x_data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", x_metric, dic=x_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))

len_data = create_data_dic(csv_folder, individuals, "female", "germfree", x_metric, data_extraction_mode="len")
len_data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", x_metric, dic=len_data, update_dic=True, data_extraction_mode="len")

visitplot(
    ydata=y_data,
    xdata=x_data,
    y_logtransform=False,
    x_transform=1,
    colors=["#D9D9D9","#CCE6BB"],
    markers=None,
    stylemode="light",
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visits_gf_vs_gfp_female_dots.pdf",
    ylabel="visit length [s]",
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel="time [s]",
    xlim=None,
    ylim=None,
)

plot_barplot(
    data=x_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=["#D9D9D9","#CCE6BB"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visits_gf_vs_gfp_female_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    ylim=None,
    ylabel=None,
    stylemode="light",
)

plot_barplot(
    data=len_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    colors=["#D9D9D9","#CCE6BB"],
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\lenvisits_gf_vs_gfp_female.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    ylim=None,
    ylabel=None,
    stylemode="light",
)
"""