import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from h5_handling import load_modulevariables_from_h5

Variables = load_modulevariables_from_h5(file_path="Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/code_test/dataframe_transformation/test.h5")

print(Variables.exp_duration_frames)