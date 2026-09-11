import preprocessing
import os

FPS = 30
path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
path = r"C:\Users\Fabian\Desktop\Transfer\analysis_testing"

exp_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\male_mice_female_stimuli"

mice = [
    r"mouse_42\2026_08_17",
    r"mouse_42\2026_08_18",
    r"mouse_42\2026_08_19",
    r"mouse_42\2026_08_20",
    r"mouse_60\2026_08_24",
    r"mouse_60\2026_08_25",
    r"mouse_60\2026_08_26",
    r"mouse_60\2026_08_27",
    r"mouse_307\2026_08_31",
    r"mouse_307\2026_09_01",
    r"mouse_307\2026_09_02",
    r"mouse_307\2026_09_03",
    
]

cams = [r"top1", r"top2"]

for mouse in mice:
    for cam in cams:
        path = os.path.join(exp_path, mouse, cam)

        preprocessing.main_preprocessing(path, FPS, exp_len_seconds=18000, filter_value=0.6)