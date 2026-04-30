import matplotlib.pyplot as plt


def plot_barplot(data, colormode, plotsize, fontsize, colors, savepath, scatterdata, scattercolors, scattermarkers, colormode):
    # This barplot takes in data containing mean and SD values
    # data_example:
    # data = {
    #     "plot1": {
    #         "group1": {"mean": 5, "sd": 1, "values": [4, 5, 6]},
    #         "group2": {"mean": 7, "sd": 1.5, "values": [6, 7, 8]}
    #     },
    #     "plot2": {
    #         "group1": {"mean": 3, "sd": 0.5, "values": [2.5, 3, 3.5]},
    #         "group2": {"mean": 4, "sd": 0.8, "values": [3.2, 4, 4.8]}
    #     }
    # }
    # Dependent on the datashape, the function will create one or multiple barplot with error bars and optionally overlay scatter points for individual values.
    # The Title is based on "plot1", "plot2" etc. and the x-axis is based on "group1", "group2" etc.
    # If "values" are provided, the function will overlay scatter points for individual values on top of the barplot.
    # The colormode can be "group" or "plot", which determines how colors are assigned to the bars and scatter points.
    # The function will save the plot to the specified savepath.
    # The colormode will determine, wether the figure is created in light or darkmode, which will affect the background color and the color of the axes and text.
    pass