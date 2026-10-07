import preprocessing
import os

FPS = 30
path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
path = r"C:\Users\Fabian\Desktop\Transfer\analysis_testing"

exp_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli"

mice = [
    r"mouse_5785\2025_05_05",
    r"mouse_5785\2025_05_06",
    r"mouse_5785\2025_05_07",
    r"mouse_5785\2025_05_08"
]

cams = [r"top1", r"top2"]

for mouse in mice:
    for cam in cams:
        path = os.path.join(exp_path, mouse, cam)

        preprocessing.main_preprocessing(path, FPS, exp_len_seconds=18000, filter_value=0.6)