from barplot import plot_barplot
from violinplot import plot_violinplot
from visitplot import visitplot
from ecdf_plot import plot_ecdf
from prepare_data import create_data_dic
from omm_statistics import compare_two_groups_to_excel

csv_folder = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest"
individuals = ["mouse_1", "mouse_2", "mouse_3"]
colors = {"germfree": "#D9D9D9", "germfreeprop": "#CCE6BB", "omm12": "#C0DEFC", "omm12prop": "#bef49d", "ommpgol": "#E58DF1"}
x_metric = "visit_start"
y_metric = "visit_len"
group = "germfree"
sex = "female"


#data = create_data_dic(csv_folder, individuals, "female", "germfreeprop", "mice_cumdists")
#data = create_data_dic(csv_folder, individuals, "female", "germfree", "mice_cumdists", dic=data, update_dic=True)

#plot_barplot(data, colors = ["red", "green", "blue", "orange"], stylemode="dark")


# # # GERMFREE VS GERMFREEPROP # # #


y_data = create_data_dic(csv_folder, individuals, sex, "germfree", y_metric, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False)
y_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", y_metric, dic=y_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False)
y_data = create_data_dic(csv_folder, individuals, sex, "omm12", y_metric, dic=y_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False)
y_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", y_metric, dic=y_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False)
y_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", y_metric, dic=y_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30), log10_transform=False)

plot_ecdf(
    data=y_data,
    colors=colors,
    markers=None,
    stylemode="light",
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visits_" + group + "_" + sex + "_ecdf.pdf",
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel=None,
    ylabel="ECDF",
    xlim=None,
    ylim=(0, 1.1),
    linewidth=2,
)



x_data = create_data_dic(csv_folder, individuals, sex, "germfree", x_metric, data_extraction_mode="raw", data_transform=(1/30))
x_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", x_metric, dic=x_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))
x_data = create_data_dic(csv_folder, individuals, sex, "omm12", x_metric, dic=x_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))
x_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", x_metric, dic=x_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))
x_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", x_metric, dic=x_data, update_dic=True, data_extraction_mode="raw", data_transform=(1/30))

len_data = create_data_dic(csv_folder, individuals, sex, "germfree", x_metric, data_extraction_mode="len")
len_data = create_data_dic(csv_folder, individuals, sex, "germfreeprop", x_metric, dic=len_data, update_dic=True, data_extraction_mode="len")
len_data = create_data_dic(csv_folder, individuals, sex, "omm12", x_metric, dic=len_data, update_dic=True, data_extraction_mode="len")
len_data = create_data_dic(csv_folder, individuals, sex, "omm12prop", x_metric, dic=len_data, update_dic=True, data_extraction_mode="len")
len_data = create_data_dic(csv_folder, individuals, sex, "ommpgol", x_metric, dic=len_data, update_dic=True, data_extraction_mode="len")

visitplot(
    ydata=y_data,
    xdata=x_data,
    y_logtransform=False,
    x_transform=1,
    colors=colors,
    markers=None,
    stylemode="light",
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visits_" + group + "_" + sex + "_dots.pdf",
    ylabel="visit length [s]",
    condition="top2",
    plotsize=(8, 6),
    fontsize=12,
    xlabel="time [s]",
    xlim=None,
    ylim=None,
    linear_regression=True,
)

plot_barplot(
    data=y_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visit_lens_" + group + "_" + sex + "_bar.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 400),
    ylabel="visit length [s]",
    stylemode="light",
)

plot_violinplot(
    data=y_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\visit_lens_" + group + "_" + sex + "_violin.pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=5,
    ylim=(0, 400),
    ylabel="visit length [s]",
    stylemode="light",
)

plot_barplot(
    data=len_data,
    colormode="group",
    plotsize=(8, 6),
    fontsize=12,
    conditions= ["top1", "top2"],
    colors=colors,
    savepath=r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\Analysis4\numvisits_" + group + "_" + sex + ".pdf",
    scatterdata=True,
    scattercolors=None,
    scattermarkers=None,
    marker_size=35,
    ylim=(0,400),
    ylabel="n visits",
    stylemode="light",
)
