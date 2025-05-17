import matplotlib.pyplot as plt
import numpy as np

def cumsum_plot(data_list=list, labels=list, colors=list, plotname=str, x_label=str, y_label=str, save_as=str):
    # Kumulative Summe berechnen

    plotdata = []
    for data in data_list:
        cumulative_sum = np.nancumsum(data)
        plotdata.append(cumulative_sum)

    # Zeitachse erzeugen, basierend auf der Länge des längeren Arrays
    max_length = 0
    for data in data_list:
        if len(data) > max_length:
            max_length = len(data)
    time = np.arange(max_length)

    # Arrays auf gleiche Länge bringen (auffüllen mit dem letzten Wert)
    for i in range(len(plotdata)):
        if len(plotdata[i]) < max_length:
            plotdata[i] = np.pad(plotdata[i], (0, max_length - len(plotdata[i])), 'edge')

    # Plotten
    plt.figure(figsize=(10, 6))

    for i in range(len(plotdata)):
        
        plt.plot(time, plotdata[i], label=labels[i], linewidth=2, color=colors[i])

    plt.title(plotname)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_as, format='svg')
    plt.show()


def heatmap_plot(x_values = np.array, y_values = np.array, plotname = str, save_as = str, num_bins = 35, cmap = 'hot', plot_time_frame_hours = (None, None)):
    """
    This function plots a heatmap of x and y coordinates, e.g. of the snout. Pass the coordinates of a complete experiment, and they get filtered for all
    values that are not "0". Plotname and savepath need to be provided. Binsize is 50 per default. Colormap is "hot" per default.
    """

    if plot_time_frame_hours[1]:
        plot_time_frame_frames = (round(plot_time_frame_hours[0]*108000), round(plot_time_frame_hours[1]*108000))
        x_values = x_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]
        y_values = y_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]

    #Filter out (x, y) pairs where either is 0
    mask = (x_values != 0) & (y_values != 0)
    heatmap_x = x_values[mask]
    heatmap_y = y_values[mask]


    # get max x value and max y value to scale the heatmap
    x_max = max(heatmap_x)
    y_max = min(heatmap_y)

    y_max = round(y_max *-1)

    # calculate number of y-bins based on ratio between x and y axis
    y_bins = round((y_max / x_max) * num_bins)
    bins = (num_bins, y_bins)


    # create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(heatmap_x, heatmap_y, bins=bins)

    plt.figure(figsize=(8,6))
    #sns.heatmap(heatmap.T, cmap=cmap, square=True, cbar=True, xticklabels=True, yticklabels=True)
    
    plt.imshow(heatmap.T, origin='lower', cmap=cmap,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect=1)  # Set aspect ratio 
    
    plt.colorbar(label='Frames')
    plt.title(plotname)
    plt.savefig(save_as, format='svg')
    plt.show()


def cumsum_plot_average(data_stim_modul, data_con_modul, savename=""):

    

    # cut arrays to shortest length (all experiments should be roughly the same length anyway)
    min_length_modul1 = min(map(len, data_stim_modul))
    min_length_modul2 = min(map(len, data_con_modul))
    overall_min = min(min_length_modul1, min_length_modul2)

    data_stim_modul = [arr[:overall_min] for arr in data_stim_modul]
    data_con_modul = [arr[:overall_min] for arr in data_con_modul]

    data_stim_modul = np.array(data_stim_modul)
    data_con_modul = np.array(data_con_modul)

    # cumsum berechnen für die daten
    for i in range(len(data_stim_modul)):
        data_stim_modul[i] = np.nancumsum(data_stim_modul[i])

    for i in range(len(data_con_modul)):
        data_con_modul[i] = np.nancumsum(data_con_modul[i])



    # get mean and std of the data
    mean_module1 = np.mean(data_stim_modul, axis=0)
    std_module1 = np.std(data_stim_modul, axis=0)
    mean_module2 = np.mean(data_con_modul, axis=0)
    std_module2 = np.std(data_con_modul, axis=0)

    # Umrechnung von Frames in Minuten (30 fps)
    fps = 30
    minutes_factor = fps * 60
    x_values = np.arange(len(mean_module1)) / minutes_factor

    mean_module1 /= minutes_factor
    std_module1  /= minutes_factor
    mean_module2 /= minutes_factor
    std_module2  /= minutes_factor

    # z. B. jeden 30. Wert nehmen → 1 Wert/Sekunde bei 30 fps
    step = 30
    x_values = x_values[::step]
    mean_module1 = mean_module1[::step]
    std_module1 = std_module1[::step]
    mean_module2 = mean_module2[::step]
    std_module2 = std_module2[::step]


    fig, ax = plt.subplots()

    # Hintergrundfarben setzen
    fig.patch.set_facecolor('black')   # Hintergrund der gesamten Figur
    ax.set_facecolor('black')          # Hintergrund des Plots

    # plot module 1
    ax.plot(x_values, mean_module1, label="module 1", color="darkgrey")
    ax.fill_between(
        x_values,
        mean_module1 - std_module1,
        mean_module1 + std_module1,
        color="grey",
        alpha=0.3,
    )

    # plot module 2
    ax.plot(x_values, mean_module2, label="module 2", color="white")
    ax.fill_between(
        x_values,
        mean_module2 - std_module2,
        mean_module2 + std_module2,
        color="white",
        alpha=0.3,
    )

    # Achsenformatierung
    ax.set_xlabel("experiment time [min]", color='white')
    ax.set_ylabel("cumulative time [min]", color='white')
    ax.set_title("cumulative time spent", color='white')

    # Achsen und Legende anpassen für bessere Lesbarkeit auf schwarzem Hintergrund
    ax.tick_params(colors='white')           # Achsenbeschriftungen weiß
    ax.spines['bottom'].set_color('white')   # Achsenlinien weiß
    ax.spines['top'].set_color('white') 
    ax.spines['left'].set_color('white') 
    ax.spines['right'].set_color('white') 
    ax.yaxis.label.set_color('white')        # y-Achsentitel
    ax.xaxis.label.set_color('white')        # x-Achsentitel
    ax.title.set_color('white')              # Titel
    ax.set_ylim(0, 130)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    #plt.savefig(f"{savename}.jpg", format='jpg')
    #plt.savefig(f"{savename}.svg", format='svg')
    plt.show()