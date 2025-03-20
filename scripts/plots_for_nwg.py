import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from df_columns import df_cols


def cumsum_plot_nwg(data_module1, data_module2, savename=""):

    # get mean and std of the data
    mean_module1 = np.mean(data_module1)
    std_module1 = np.std(data_module1)
    mean_module2 = np.mean(data_module2)
    std_module2 = np.std(data_module2)


    plt.figure()

    # plot module 1
    plt.plot(mean_module1, label="Modul1", color="yellow")
    plt.fill_between(
            range(len(mean_module1)),
            mean_module1 - std_module1,
            mean_module1 + std_module1,
            color="yellow",
            alpha=0.3,
            label="Standard Deviation 1"
    )

    # plot module 2
    plt.plot(mean_module2, label="Modul2", color="red")
    plt.fill_between(
            range(len(mean_module2)),
            mean_module2 - std_module2,
            mean_module2 + std_module2,
            color="yellow",
            alpha=0.3,
            label="Standard Deviation 2"
    )

    plt.show()