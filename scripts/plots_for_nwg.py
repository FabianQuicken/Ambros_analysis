import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from df_columns import df_cols


def cumsum_plot_nwg(data_module1, data_module2, savename=""):

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


    fig, ax = plt.subplots()

    # Hintergrundfarben setzen
    fig.patch.set_facecolor('black')   # Hintergrund der gesamten Figur
    ax.set_facecolor('black')          # Hintergrund des Plots

    # plot module 1
    ax.plot(x_values, mean_module1, label="Modul1", color="yellow")
    ax.fill_between(
        x_values,
        mean_module1 - std_module1,
        mean_module1 + std_module1,
        color="yellow",
        alpha=0.3,
    )

    # plot module 2
    ax.plot(x_values, mean_module2, label="Modul2", color="red")
    ax.fill_between(
        x_values,
        mean_module2 - std_module2,
        mean_module2 + std_module2,
        color="red",
        alpha=0.3,
    )
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

    plt.savefig(savename, format='svg')
    plt.show()