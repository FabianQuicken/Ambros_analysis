import h5py
import numpy as np
import pandas as pd
import glob
import os
from dataclasses import dataclass
from h5_handling import load_modulevariables_from_h5


path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/h5_files/"

file_list = glob.glob(os.path.join(path, '*.h5'))

modul_data_list = []

for file in file_list:
    
    modul_data_list.append(load_modulevariables_from_h5(file))


mice = ["mouse_15", "mouse_17", "mouse_18", "mouse_5785"]

#print(modul_data_list)

mouse15_data = [data for data in modul_data_list if "15" in data.mouse]
mouse17_data = [data for data in modul_data_list if "17" in data.mouse]
mouse18_data = [data for data in modul_data_list if "18" in data.mouse]
mouse5785_data = [data for data in modul_data_list if "5785" in data.mouse]

