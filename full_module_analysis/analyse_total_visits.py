import h5py
import numpy as np
import pandas as pd
import glob
import os
from dataclasses import dataclass
from h5_handling import load_modulevariables_from_h5
from plotting import plot_grouped_barplot_with_black_bg, plot_stimulus_over_days


path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"
#path_ho = "//fileserver2.bio2.rwth-aachen.de/AG Spehr BigData/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"

#path = path_ho

file_list = glob.glob(os.path.join(path, '*.h5'))
file_list.sort()

modul_data_list = []

for file in file_list:
    
    modul_data_list.append(load_modulevariables_from_h5(file))


mice = ["mouse_15", "mouse_17", "mouse_18", "mouse_5785"]


mouse7_data = [data for data in modul_data_list if data.mouse == "mouse_15"]
mouse21_data = [data for data in modul_data_list if "17" in data.mouse]
mouse73_data = [data for data in modul_data_list if "18" in data.mouse]
mouse75_data = [data for data in modul_data_list if "5785" in data.mouse]
data = [mouse7_data, mouse21_data, mouse73_data, mouse75_data]


hab1_stimulus_module_data = []
hab2_stimulus_module_data = []
exp1_stimulus_module_data = []
exp2_stimulus_module_data = []

hab1_control_module_data = []
hab2_control_module_data = []
exp1_control_module_data = []
exp2_control_module_data = []

# max_value für y-Achsenbeschränkung berechnen
all_data = []
for mouse_data in data:
    for stats in mouse_data:
        all_data.append(stats.visits_per_hour)

max_value = max(all_data)


for mouse_data in data:
    # first get the paradigm (stimulus side)

    if mouse_data[2].modulnumber == 1 and mouse_data[2].is_stimulus_module:
        first_stim_modul = 1
    else:
        first_stim_modul = 2

    # daten sortieren
    if len(mouse_data) == 8:
        if first_stim_modul == 1:
            hab1_stimulus_module_data.append(mouse_data[0].visits_per_hour)
            hab1_control_module_data.append(mouse_data[1].visits_per_hour)

            hab2_stimulus_module_data.append(mouse_data[4].visits_per_hour)
            hab2_control_module_data.append(mouse_data[5].visits_per_hour)
            
            exp1_stimulus_module_data.append(mouse_data[2].visits_per_hour)
            exp1_control_module_data.append(mouse_data[3].visits_per_hour)

            exp2_stimulus_module_data.append(mouse_data[7].visits_per_hour)
            exp2_control_module_data.append(mouse_data[6].visits_per_hour)


        elif first_stim_modul == 2:
            hab1_stimulus_module_data.append(mouse_data[1].visits_per_hour)
            hab1_control_module_data.append(mouse_data[0].visits_per_hour)

            hab2_stimulus_module_data.append(mouse_data[5].visits_per_hour)
            hab2_control_module_data.append(mouse_data[4].visits_per_hour)

            exp1_stimulus_module_data.append(mouse_data[3].visits_per_hour)
            exp1_control_module_data.append(mouse_data[2].visits_per_hour)  

            exp2_stimulus_module_data.append(mouse_data[6].visits_per_hour)
            exp2_control_module_data.append(mouse_data[7].visits_per_hour)


plot_grouped_barplot_with_black_bg(
    hab1_stimulus=hab1_stimulus_module_data,
    hab1_control=hab1_control_module_data,
    exp1_stimulus=exp1_stimulus_module_data,
    exp1_control=exp1_control_module_data,
    hab2_stimulus=hab2_stimulus_module_data,
    hab2_control=hab2_control_module_data,
    exp2_stimulus=exp2_stimulus_module_data,
    exp2_control=exp2_control_module_data,
    ymax=max_value,
    title="Visits per hour - Females",
    ylabel="#visits",
    savename=f'{path}visits_per_hour_females'
)

plot_stimulus_over_days(
    hab1_stimulus=hab1_stimulus_module_data,
    exp1_stimulus=exp1_stimulus_module_data,
    hab2_stimulus=hab2_stimulus_module_data,
    exp2_stimulus=exp2_stimulus_module_data,
    mice=mice,
    ymax=max_value,
    title="Visits per hour (Stimulus) - Females - single mice",
    ylabel="#visits",
    savename=f'{path}visits_per_hour_stim_females'
)

plot_stimulus_over_days(
    hab1_stimulus=hab1_control_module_data,
    exp1_stimulus=exp1_control_module_data,
    hab2_stimulus=hab2_control_module_data,
    exp2_stimulus=exp2_control_module_data,
    mice=mice,
    ymax=max_value,
    title="Visits per hour (Control) - Females - single mice",
    ylabel="#visits",
    savename=f'{path}visits_per_hour_con_females'
)