import h5py
import numpy as np
import pandas as pd
import glob
import os
from dataclasses import dataclass
from h5_handling import load_modulevariables_from_h5
from plotting import cumsum_plot_average, plot_stimulus_over_days


path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"
path_ho = "//fileserver2.bio2.rwth-aachen.de/AG Spehr BigData/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"

path = path_ho

file_list = glob.glob(os.path.join(path, '*.h5'))
file_list.sort()

modul_data_list = []

for file in file_list:
    
    modul_data_list.append(load_modulevariables_from_h5(file))


mice = ["mouse_15", "mouse_17", "mouse_18", "mouse_5785"]

#print(modul_data_list)

mouse15_data = [data for data in modul_data_list if "15" in data.mouse]
mouse17_data = [data for data in modul_data_list if "17" in data.mouse]
mouse18_data = [data for data in modul_data_list if "18" in data.mouse]
mouse5785_data = [data for data in modul_data_list if "5785" in data.mouse]
data = [mouse15_data, mouse17_data, mouse18_data, mouse5785_data]

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
        all_data.append(sum(stats.maus_in_modul_über_zeit))

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
            hab1_stimulus_module_data.append(mouse_data[0].maus_in_modul_über_zeit)
            hab1_control_module_data.append(mouse_data[1].maus_in_modul_über_zeit)

            hab2_stimulus_module_data.append(mouse_data[4].maus_in_modul_über_zeit)
            hab2_control_module_data.append(mouse_data[5].maus_in_modul_über_zeit)
            
            exp1_stimulus_module_data.append(mouse_data[2].maus_in_modul_über_zeit)
            exp1_control_module_data.append(mouse_data[3].maus_in_modul_über_zeit)

            exp2_stimulus_module_data.append(mouse_data[7].maus_in_modul_über_zeit)
            exp2_control_module_data.append(mouse_data[6].maus_in_modul_über_zeit)


        elif first_stim_modul == 2:
            hab1_stimulus_module_data.append(mouse_data[1].maus_in_modul_über_zeit)
            hab1_control_module_data.append(mouse_data[0].maus_in_modul_über_zeit)

            hab2_stimulus_module_data.append(mouse_data[5].maus_in_modul_über_zeit)
            hab2_control_module_data.append(mouse_data[4].maus_in_modul_über_zeit)

            exp1_stimulus_module_data.append(mouse_data[3].maus_in_modul_über_zeit)
            exp1_control_module_data.append(mouse_data[2].maus_in_modul_über_zeit)  

            exp2_stimulus_module_data.append(mouse_data[6].maus_in_modul_über_zeit)
            exp2_control_module_data.append(mouse_data[7].maus_in_modul_über_zeit)


#cumsum_plot_average(data_stim_modul=hab1_stimulus_module_data, data_con_modul=hab1_control_module_data, ymax=max_value, label1='stimulus module in exp1', label2='control module in exp1', title= 'HAB1 - females - cumsum time', savename=f'{path}cumsum_hab1_females')
#cumsum_plot_average(data_stim_modul=exp1_stimulus_module_data, data_con_modul=exp1_control_module_data, ymax=max_value, label1='stimulus', label2='control', title= 'EXP1 - females - cumsum time', savename=f'{path}cumsum_exp1_females')
#cumsum_plot_average(data_stim_modul=hab2_stimulus_module_data, data_con_modul=hab2_control_module_data, ymax=max_value, label1='stimulus module in exp1', label2='control module in exp1', title= 'HAB2 - females - cumsum time', savename=f'{path}cumsum_hab2_females')
#cumsum_plot_average(data_stim_modul=exp2_stimulus_module_data, data_con_modul=exp2_control_module_data, ymax=max_value, label1='stimulus', label2='control', title= 'EXP2 - females - cumsum time', savename=f'{path}cumsum_exp2_females')


hab1_stimulus_module_data = [sum(array) for array in hab1_stimulus_module_data]
hab2_stimulus_module_data = [sum(array) for array in hab2_stimulus_module_data]
exp1_stimulus_module_data = [sum(array) for array in exp1_stimulus_module_data]
exp2_stimulus_module_data = [sum(array) for array in exp2_stimulus_module_data]

hab1_control_module_data = [sum(array) for array in hab1_control_module_data]
hab2_control_module_data = [sum(array) for array in hab2_control_module_data]
exp1_control_module_data = [sum(array) for array in exp1_control_module_data]
exp2_control_module_data = [sum(array) for array in exp2_control_module_data]

plot_stimulus_over_days(
    hab1_stimulus=hab1_stimulus_module_data,
    exp1_stimulus=exp1_stimulus_module_data,
    hab2_stimulus=hab2_stimulus_module_data,
    exp2_stimulus=exp2_stimulus_module_data,
    mice=mice,
    convert_to_min=True,
    ymax=max_value,
    title="Total Time (Stimulus) - Females - single mice",
    ylabel="Time (min)",
    savename=f'{path}total_time_stim_females'
)

plot_stimulus_over_days(
    hab1_stimulus=hab1_control_module_data,
    exp1_stimulus=exp1_control_module_data,
    hab2_stimulus=hab2_control_module_data,
    exp2_stimulus=exp2_control_module_data,
    mice=mice,
    convert_to_min=True,
    ymax=max_value,
    title="Total Time (Control) - Females - single mice",
    ylabel="Time (min)",
    savename=f'{path}total_time_con_females'
)