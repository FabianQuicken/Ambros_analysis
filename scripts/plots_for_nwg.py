import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from df_columns import df_cols

def cumsum_plot_nwg(data_module1, data_module2, savename=""):

    # Umrechnung: cumsum und in Minuten
    fps = 30
    minutes_factor = fps * 60

    data_module1 = [np.nancumsum(trace) / minutes_factor for trace in data_module1]
    data_module2 = [np.nancumsum(trace) / minutes_factor for trace in data_module2]

    # X-Werte (Zeit in Minuten)
    max_len = max(len(trace) for trace in data_module1 + data_module2)
    x_values = np.arange(max_len) / minutes_factor

    # Plot vorbereiten
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Modul 1 (gelb)
    for trace in data_module1:
        ax.plot(x_values[:len(trace)], trace, color="darkgrey", alpha=0.8, label="module 1")

    # Modul 2 (rot)
    for trace in data_module2:
        ax.plot(x_values[:len(trace)], trace, color="white", alpha=0.8, label="module 2")

    # Nur einmalige Labels in Legende
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), facecolor='black', edgecolor='white', labelcolor='white')

    # Achsenformatierung
    ax.set_xlabel("experiment time [min]", color='white')
    ax.set_ylabel("cumulative time [min]", color='white')
    ax.set_title("cumulative time spent", color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.set_ylim(0, 130)

    # Speichern & Anzeigen
    plt.tight_layout()
    plt.savefig(f"{savename}_single.jpg", format='jpg', facecolor=fig.get_facecolor())
    plt.savefig(f"{savename}_single.svg", format='svg', facecolor=fig.get_facecolor())
    plt.show()


def cumsum_plot_average_nwg(data_stim_modul, data_con_modul, savename=""):

    # cumsum berechnen für die daten
    for i in range(len(data_module1)):
        data_module1[i] = np.nancumsum(data_module1[i])

    for i in range(len(data_module2)):
        data_module2[i] = np.nancumsum(data_module2[i])



    # get mean and std of the data
    mean_module1 = np.mean(data_module1, axis=0)
    std_module1 = np.std(data_module1, axis=0)
    mean_module2 = np.mean(data_module2, axis=0)
    std_module2 = np.std(data_module2, axis=0)

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

    plt.savefig(f"{savename}.jpg", format='jpg')
    plt.savefig(f"{savename}.svg", format='svg')
    plt.show()


def deg_barplot_nwg(data_module1, data_module2, savename = ""):

    # Mittelwerte berechnen für jeden Index der inneren Listen
    avg_modul1 = np.mean(data_module1, axis=0)
    avg_modul2 = np.mean(data_module2, axis=0)

    # Plot-Vorbereitung
    labels = ['rearing', 'drinking', 'grooming']
    x = np.arange(len(labels))  # Position der Gruppen auf der X-Achse
    width = 0.35  # Breite der Balken

    fig, ax = plt.subplots()

    # Hintergrundfarben setzen
    fig.patch.set_facecolor('black')   # Hintergrund der gesamten Figur
    ax.set_facecolor('black')          # Hintergrund des Plots


    balken1 = ax.bar(x - width/2, avg_modul1, width, label='module 1', color="grey")
    balken2 = ax.bar(x + width/2, avg_modul2, width, label='module 2', color="white")

    # Scatter Punkte & Verbindungslinien
    for i in range(3):  # Für jeden Wert (0-2)
        for j in range(4):  # Für jede Messung (0, 1)
            x1 = x[i] - width/2
            x2 = x[i] + width/2
            y1 = data_module1[j][i]
            y2 = data_module2[j][i]

            # Linie zwischen den Punkten
            ax.plot([x1, x2], [y1, y2], color='white', linestyle='-', alpha=0.6, zorder=4)

            # Punkte plotten
            ax.scatter(x1, y1, color='grey', edgecolors='white', zorder=5)
            ax.scatter(x2, y2, color='white', edgecolors='grey', zorder=5)

    # Labels, Titel, Legende
    ax.set_ylabel('time [%]')
    ax.set_title('stimulus vs control')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Achsen und Legende anpassen für bessere Lesbarkeit auf schwarzem Hintergrund
    ax.tick_params(colors='white')           # Achsenbeschriftungen weiß
    ax.spines['bottom'].set_color('white')   # Achsenlinien weiß
    ax.spines['top'].set_color('white') 
    ax.spines['left'].set_color('white') 
    ax.spines['right'].set_color('white') 
    ax.yaxis.label.set_color('white')        # y-Achsentitel
    ax.xaxis.label.set_color('white')        # x-Achsentitel
    ax.title.set_color('white')              # Titel
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    """
    # Werte über den Balken anzeigen
    for balken in [balken1, balken2]:
        for rect in balken:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Abstand nach oben
                        textcoords="offset points",
                        ha='center', va='bottom')
    """
    plt.tight_layout()
    plt.savefig(savename+'.jpg', format='jpg')
    plt.savefig(savename+'.svg', format='svg')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def dlc_barplot_nwg(data_module1, data_module2, savename=""):
    # Feature-Namen & Einheiten
    labels = [
        'time at food\n[%]',
        'average speed\n[px/frame]',
        'average visits\n[/hour]',
        'time in module\n[%]'
    ]

    # Farben
    color1 = "grey"
    color2 = "white"
    edge1 = "white"
    edge2 = "grey"

    # Daten vorbereiten
    daten1 = np.array(data_module1)
    daten2 = np.array(data_module2)

    n_metrics = daten1.shape[1]
    width = 0.2  # 👉 schmalere Balken
    x_center = 0  # zentrierte X-Position für das Balkenpaar

    fig, axs = plt.subplots(1, n_metrics, figsize=(2 * n_metrics, 5), sharey=False)
    fig.patch.set_facecolor('black')  # Gesamter Hintergrund

    y_lims = [
        (0, 5.25),     # für Metrik 0
        (0, 8.4),     # für Metrik 1
        (0, 42),    # für Metrik 2
        (0, 52.5)     # für Metrik 3
    ]

    for i in range(n_metrics):
        ax = axs[i]
        ax.set_facecolor('black')

        # Werte extrahieren
        vals1 = daten1[:, i]
        vals2 = daten2[:, i]

        # Mittelwerte
        avg1 = np.mean(vals1)
        avg2 = np.mean(vals2)

        # Balken (direkt nebeneinander)
        bar1_x = x_center - width / 2
        bar2_x = x_center + width / 2

        ax.bar(bar1_x, avg1, width, color=color1, alpha=0.7)
        ax.bar(bar2_x, avg2, width, color=color2, alpha=0.7)

        # Scatter-Punkte + Linien
        for j in range(len(vals1)):
            ax.plot([bar1_x, bar2_x], [vals1[j], vals2[j]],
                    color='white', linestyle='-', alpha=0.6, zorder=4)
            ax.scatter(bar1_x, vals1[j], color=color1, edgecolors=edge1, zorder=5)
            ax.scatter(bar2_x, vals2[j], color=color2, edgecolors=edge2, zorder=5)

        # 👉 Y-Achsenlimit setzen
        ax.set_ylim(y_lims[i])

        # Achsen & Titel
        ax.set_xticks([])
        ax.set_title(labels[i], color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        #ax.spines['top'].set_color('white') 
        ax.spines['left'].set_color('white') 
        #ax.spines['right'].set_color('white') 
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')


    # Layout und Speichern
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savename:
        plt.savefig(f"{savename}.jpg", format='jpg', facecolor=fig.get_facecolor())
        plt.savefig(f"{savename}.svg", format='svg', facecolor=fig.get_facecolor())
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def heatmap_dual_plot(
    x1, y1, x2, y2,
    plotname="Heatmap Vergleich",
    save_as="heatmap_vergleich.svg",
    num_bins=35,
    cmap='hot',
    plot_time_frame_hours=(None, None)
):
    def process_coordinates(x_vals, y_vals):
        if plot_time_frame_hours[1] is not None:
            frames = (round(plot_time_frame_hours[0]*108000), round(plot_time_frame_hours[1]*108000))
            x_vals = x_vals[frames[0]:frames[1]]
            y_vals = y_vals[frames[0]:frames[1]]
        mask = (x_vals != 0) & (y_vals != 0)
        return x_vals[mask], y_vals[mask]

    # Vorverarbeitung
    x1, y1 = process_coordinates(x1, y1)
    x2, y2 = process_coordinates(x2, y2)

    # Gemeinsames Bin-Setup
    x_max = max(np.max(x1), np.max(x2))
    y_max = abs(min(np.min(y1), np.min(y2)))
    y_bins = round((y_max / x_max) * num_bins)
    bins = (num_bins, y_bins)

    # Histogramme berechnen
    heatmap1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=bins)
    heatmap2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=bins)

    # In Minuten umrechnen (30 fps → 1800 Frames = 1 Minute)
    heatmap1 /= 1800
    heatmap2 /= 1800

    # Gemeinsames vmax für beide Heatmaps
    vmax = max(np.max(heatmap1), np.max(heatmap2))

    # Plot mit dunklem Hintergrund
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')  # Hintergrund der Figur

    titles = ["Module 1", "Module 2"]
    imgs = []

    for i, (heatmap, xedges, yedges, ax) in enumerate(zip(
        [heatmap1, heatmap2],
        [xedges1, xedges2],
        [yedges1, yedges2],
        axs
    )):
        ax.set_facecolor('black')  # Achsenhintergrund
        img = ax.imshow(
            heatmap.T, origin='lower', cmap=cmap,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect=1, vmin=0, vmax=vmax
        )
        ax.set_title(titles[i], color='white')
        ax.set_xlabel("X-coordinates", color='white')
        ax.set_ylabel("Y-coordinates", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        imgs.append(img)

    # Colorbar rechts daneben
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label("time spend [min]", color='white')

    fig.suptitle(plotname, color='white')
    plt.subplots_adjust(right=0.9)
    plt.savefig(f"{save_as}.svg", format='svg', facecolor=fig.get_facecolor())
    plt.savefig(f"{save_as}.jpg", format='jpg', facecolor=fig.get_facecolor())
    plt.show()
