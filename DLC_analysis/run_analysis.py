from analysis import main_analysis
import os

folder_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2\metric_analysis"

experiment_folder = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse"
mouse = r"mouse2"
date = r"2024_12_17"
camera = r"top2"
file_folder  = r"metric_analysis"

print(os.path.join(experiment_folder, mouse, date, camera, file_folder))


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

exp_h5_folder = r"metric_analysis"

for mouse in mice:
    for cam in cams:

        path = os.path.join(exp_path, mouse, cam, exp_h5_folder)


        individuals_to_remove = None
        bodyparts_to_remove = None

        main_analysis(individuals_to_remove,
                    bodyparts_to_remove,
                    path)



            


