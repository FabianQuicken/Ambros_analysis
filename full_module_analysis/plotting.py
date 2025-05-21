import matplotlib.pyplot as plt
import numpy as np
from config import FPS


def visits_multi_histogram(data_list, xmax=None, xlabel="visit length [s]", datalabels=["Dataset 1", "Dataset 2"], plotname="", save_as="", bin_width=6.0, zoom_in=False):
    """
    Plots a histogram of visit lengths for multiple datasets with black background and white axes.

    Parameters:
    - data_list: List of arrays or objects with '.all_visits' attribute
    - xmax: Upper x-axis limit; if None, it is determined from the data
    - xlabel: Label for the x-axis
    - plotname: Title of the plot
    - save_as: File path to save the figure
    - bin_width: Width of histogram bins
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Farben für die einzelnen Datensätze
    colors = ['cyan', 'magenta', 'cyan', 'magenta', 'green']

    # Daten extrahieren und FPS anwenden
    visit_arrays = []
    for data in data_list:
        try:
            visits = np.array(data.all_visits) / FPS
        except:
            visits = np.array(data) / FPS
        visit_arrays.append(visits)

    # xmax bestimmen
    if xmax is None:
        xmax = max([np.max(arr) for arr in visit_arrays])
    bins = np.arange(0, xmax + bin_width, bin_width)

    # Plot vorbereiten
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Histogramme plotten
    for i, visits in enumerate(visit_arrays):
        plt.hist(
            x=visits,
            bins=bins,
            color=colors[i % len(colors)],
            edgecolor='white',
            alpha=0.6,
            label=datalabels[i]
        )

    # Achsen, Titel, Legende, etc.
    plt.xlabel(xlabel, color='white', fontsize=12)
    plt.ylabel("n visits", color='white', fontsize=12)
    plt.title(plotname, color='white', fontsize=14)
    plt.xlim(0, xmax)
    if zoom_in:
        plt.ylim(-0.5, 10)
        if save_as:
            save_as = save_as + '_zoomin'
    else:
        plt.ylim(-10, 280)


    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as+'.svg', format='svg')
    plt.show()

def visits_histogram(data, xmax=None, xlabel="visit length [s]", plotname="", save_as="", bin_width=6.0):
    """
    Plots a histogram of 'all_visits' from a data object with black background and white axes,
    and consistent bar width regardless of data length.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Daten vorbereiten
    try:
        data_array = np.array(data.all_visits) / FPS
    except:
        data_array = np.array(data) / FPS

    # Bin-Grenzen festlegen (z. B. 0–xmax mit fixer Breite)
    if xmax is None:
        xmax = np.max(data_array)
    bins = np.arange(0, xmax + bin_width, bin_width)

    # Plot vorbereiten
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Histogram zeichnen mit festen Bins
    plt.hist(
        x=data_array,
        bins=bins,
        color=('red' if 'con' in save_as else 'yellow'),
        edgecolor='white'
    )

    # Achsen, Titel, etc.
    plt.xlabel(xlabel, color='white', fontsize=12)
    plt.ylabel("Frequency", color='white', fontsize=12)
    plt.title(plotname, color='white', fontsize=14)
    plt.xlim(0, xmax)
    plt.ylim(0,250)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, format='svg')
    plt.show()




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


def cumsum_plot_average(data_stim_modul, data_con_modul, ymax=None, label1="modul1", label2="modul2",title="", savename=""):

    

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

    # Umrechnung von Frames in Minuten 
    minutes_factor = FPS * 60
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
    ax.plot(x_values, mean_module1, label=label1, color="yellow")
    ax.fill_between(
        x_values,
        mean_module1 - std_module1,
        mean_module1 + std_module1,
        color="yellow",
        alpha=0.3,
    )

    # plot module 2
    ax.plot(x_values, mean_module2, label=label2, color="red")
    ax.fill_between(
        x_values,
        mean_module2 - std_module2,
        mean_module2 + std_module2,
        color="red",
        alpha=0.3,
    )

    # Achsenformatierung
    ax.set_xlabel("experiment time [min]", color='white')
    ax.set_ylabel("cumulative time [min]", color='white')
    ax.set_title(title, color='white')

    # Achsen und Legende anpassen für bessere Lesbarkeit auf schwarzem Hintergrund
    ax.tick_params(colors='white')           # Achsenbeschriftungen weiß
    ax.spines['bottom'].set_color('white')   # Achsenlinien weiß
    ax.spines['top'].set_color('white') 
    ax.spines['left'].set_color('white') 
    ax.spines['right'].set_color('white') 
    ax.yaxis.label.set_color('white')        # y-Achsentitel
    ax.xaxis.label.set_color('white')        # x-Achsentitel
    ax.title.set_color('white')              # Titel
    if ymax:
        ax.set_ylim(0, ymax*1.1 / (FPS*60))
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    #plt.savefig(f"{savename}.jpg", format='jpg')
    plt.savefig(f"{savename}.svg", format='svg')
    plt.show()



def plot_grouped_barplot_with_black_bg(hab1_stimulus, hab1_control,
                                       exp1_stimulus, exp1_control,
                                       hab2_stimulus, hab2_control,
                                       exp2_stimulus, exp2_control,
                                       convert_to_min = False,
                                       ymax = None,
                                       title="Grouped Barplot", ylabel="Value", savename=None):
    """
    Plots grouped barplots for stimulus vs. control across four conditions
    with a black background and white axes and text.
    """

    # Mittelwerte und Standardabweichungen berechnen
    data = [
        (np.mean(hab1_control), np.std(hab1_control), np.mean(hab1_stimulus), np.std(hab1_stimulus)),
        (np.mean(exp1_control), np.std(exp1_control), np.mean(exp1_stimulus), np.std(exp1_stimulus)),
        (np.mean(hab2_control), np.std(hab2_control), np.mean(hab2_stimulus), np.std(hab2_stimulus)),
        (np.mean(exp2_control), np.std(exp2_control), np.mean(exp2_stimulus), np.std(exp2_stimulus)),
    ]

    group_labels = ['HAB1', 'EXP1', 'HAB2', 'EXP2']

    bar_width = 0.35
    indices = np.arange(len(data))

    control_means = [d[0] for d in data]
    control_stds = [d[1] for d in data]
    stimulus_means = [d[2] for d in data]
    stimulus_stds = [d[3] for d in data]

    if convert_to_min:
        # Umrechnung von Frames in Minuten 
        minutes_factor = FPS * 60

        control_means /= minutes_factor
        control_stds  /= minutes_factor
        stimulus_means /= minutes_factor
        stimulus_stds  /= minutes_factor

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Control links, Stimulus rechts
    ax.bar(indices - bar_width/2, control_means, width=bar_width, yerr=control_stds,
           label='Control', color='gray', capsize=5, error_kw=dict(ecolor='white'))
    ax.bar(indices + bar_width/2, stimulus_means, width=bar_width, yerr=stimulus_stds,
           label='Stimulus', color='orange', capsize=5, error_kw=dict(ecolor='white'))

    ax.set_xticks(indices)
    ax.set_xticklabels(group_labels, color='white', fontsize=12)
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    if ymax:
        if convert_to_min:
            ax.set_ylim(0, ymax*1.1 / (FPS*60))
        else:
            ax.set_ylim(0, ymax*1.1)

    plt.tight_layout()

    if savename:
        plt.savefig(f"{savename}.svg", format='svg')
    plt.show()

def plot_stimulus_over_days(hab1_stimulus, exp1_stimulus, hab2_stimulus, exp2_stimulus, mice, convert_to_min=False, ymax=None, title="Stimulus Module Data per Mouse", ylabel="Time (s)", savename=None):
    """
    Plots individual stimulus module data points for each mouse across experimental days
    and connects each mouse's data with a line. Assumes same mouse order across lists.
    """

    hab1_stimulus = np.array(hab1_stimulus)
    exp1_stimulus = np.array(exp1_stimulus)
    hab2_stimulus = np.array(hab2_stimulus)
    exp2_stimulus = np.array(exp2_stimulus)



    if convert_to_min:
        minutes_factor = FPS * 60

        hab1_stimulus /= minutes_factor
        exp1_stimulus /= minutes_factor
        hab2_stimulus /= minutes_factor
        exp2_stimulus /= minutes_factor

    # Namen für die Versuchstage
    x_labels = ['HAB1', 'EXP1', 'HAB2', 'EXP2']
    x = np.arange(len(x_labels))

    # Anzahl Mäuse = Länge einer der Listen
    n_mice = len(hab1_stimulus)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot pro Maus
    for i in range(n_mice):
        y_values = [
            hab1_stimulus[i],
            exp1_stimulus[i],
            hab2_stimulus[i],
            exp2_stimulus[i]
        ]
        ax.plot(x, y_values, marker='o', linestyle='-', label=mice[i])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, color='white', fontsize=12)
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9, loc='best')
    if ymax:
        if convert_to_min:
            ax.set_ylim(0, ymax*1.1 / (FPS*60))
        else:
            ax.set_ylim(0, ymax*1.1)

    plt.tight_layout()
    if savename:
        plt.savefig(f"{savename}.svg", format='svg')
    plt.show()