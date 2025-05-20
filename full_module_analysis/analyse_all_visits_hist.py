import h5py
import numpy as np
import pandas as pd
import glob
import os
from dataclasses import dataclass
from h5_handling import load_modulevariables_from_h5
from plotting import visits_histogram


path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/male_mice_female_stimuli/h5_files/"
#path_ho = "//fileserver2.bio2.rwth-aachen.de/AG Spehr BigData/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"

#path = path_ho

file_list = glob.glob(os.path.join(path, '*.h5'))
file_list.sort()

modul_data_list = []

for file in file_list:
    
    modul_data_list.append(load_modulevariables_from_h5(file))


mice = ["mouse_7", "mouse_21", "mouse_73", "mouse_75"]


mouse7_data = [data for data in modul_data_list if data.mouse == "mouse_7"]
mouse21_data = [data for data in modul_data_list if "21" in data.mouse]
mouse73_data = [data for data in modul_data_list if "73" in data.mouse]
mouse75_data = [data for data in modul_data_list if "75" in data.mouse]
data = [mouse7_data, mouse21_data, mouse73_data, mouse75_data]

hab1_stimulus_module_data = []
hab2_stimulus_module_data = []
exp1_stimulus_module_data = []
exp2_stimulus_module_data = []

hab1_control_module_data = []
hab2_control_module_data = []
exp1_control_module_data = []
exp2_control_module_data = []

# max_value für x-Achsenbeschränkung berechnen
all_data = []
for mouse_data in data:
    for stats in mouse_data:
        for visit in stats.all_visits:
            all_data.append(visit)
# findet längsten visit
max_value = max(all_data) / 30
print(max_value)


for mouse_data in data:
    # first get the paradigm (stimulus side)

    if mouse_data[2].modulnumber == 1 and mouse_data[2].is_stimulus_module:
        first_stim_modul = 1
    else:
        first_stim_modul = 2

    # daten sortieren
    if len(mouse_data) == 8:
        if first_stim_modul == 1:
            for visit in mouse_data[0].all_visits:
                hab1_stimulus_module_data.append(visit)

            for visit in mouse_data[1].all_visits:
                hab1_control_module_data.append(visit)

            for visit in mouse_data[4].all_visits:
                hab2_stimulus_module_data.append(visit)

            for visit in mouse_data[5].all_visits:
                hab2_control_module_data.append(visit)
            
            for visit in mouse_data[2].all_visits:
                exp1_stimulus_module_data.append(visit)

            for visit in mouse_data[3].all_visits:
                exp1_control_module_data.append(visit)

            for visit in mouse_data[7].all_visits:
                exp2_stimulus_module_data.append(visit)

            for visit in mouse_data[6].all_visits:
                exp2_control_module_data.append(visit)


        elif first_stim_modul == 2:

            for visit in mouse_data[1].all_visits:
                hab1_stimulus_module_data.append(visit)

            for visit in mouse_data[0].all_visits:
                hab1_control_module_data.append(visit)

            for visit in mouse_data[5].all_visits:
                hab2_stimulus_module_data.append(visit)

            for visit in mouse_data[4].all_visits:
                hab2_control_module_data.append(visit)

            for visit in mouse_data[3].all_visits:
                exp1_stimulus_module_data.append(visit)

            for visit in mouse_data[2].all_visits:
                exp1_control_module_data.append(visit)  

            for visit in mouse_data[6].all_visits:
                exp2_stimulus_module_data.append(visit)

            for visit in mouse_data[7].all_visits:
                exp2_control_module_data.append(visit)


    
visits_histogram(data=hab1_control_module_data, xmax=max_value, plotname="hab1 con - male", save_as=f'{path}visits_hist_males_hab1_con.svg')
visits_histogram(data=hab1_stimulus_module_data, xmax=max_value, plotname="hab1 stim - male", save_as=f'{path}visits_hist_males_hab1_stim.svg')
visits_histogram(data=hab2_control_module_data, xmax=max_value, plotname="hab2 con - male", save_as=f'{path}visits_hist_males_hab2_con.svg')
visits_histogram(data=hab2_stimulus_module_data, xmax=max_value, plotname="hab2 stim - male", save_as=f'{path}visits_hist_males_hab2_stim.svg')
visits_histogram(data=exp1_control_module_data, xmax=max_value, plotname="exp1 con - male", save_as=f'{path}visits_hist_males_exp1_con.svg')
visits_histogram(data=exp1_stimulus_module_data, xmax=max_value, plotname="exp1 stim - male", save_as=f'{path}visits_hist_males_exp1_stim.svg')
visits_histogram(data=exp2_control_module_data, xmax=max_value, plotname="exp2 con - male", save_as=f'{path}visits_hist_males_exp2_con.svg')
visits_histogram(data=exp2_stimulus_module_data, xmax=max_value, plotname="exp2 stim - male", save_as=f'{path}visits_hist_males_exp2_stim.svg')