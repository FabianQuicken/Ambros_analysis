from analysis import main_analysis
import os

folder_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2\metric_analysis"

experiment_folder = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse"
mouse = r"mouse2"
date = r"2024_12_17"
camera = r"top2"
file_folder  = r"metric_analysis"

print(os.path.join(experiment_folder, mouse, date, camera, file_folder))


exp_path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\female_mice_male_stimuli"

# die ausgeklammerten alle noch mal machen !!!
mice = [
#    r"mouse_15\2025_04_22",
#    r"mouse_15\2025_04_23",
#    r"mouse_15\2025_04_24",
#    r"mouse_15\2025_04_25",
#    r"mouse_17\2025_04_14",
#    r"mouse_17\2025_04_15",
#    r"mouse_17\2025_04_16",
#    r"mouse_17\2025_04_17",
#    r"mouse_18\2025_04_26",
#    r"mouse_18\2025_04_28",
#    r"mouse_18\2025_04_29",
#    r"mouse_18\2025_04_30",
#    r"mouse_47\2026_06_30",
#    r"mouse_47\2026_07_01",
#    r"mouse_47\2026_07_02",
#    r"mouse_47\2026_07_03",
#    r"mouse_48\2026_07_21",
#    r"mouse_48\2026_07_22",
#    r"mouse_48\2026_07_23",
#    r"mouse_48\2026_07_24",
#    r"mouse_49\2026_08_10",
#    r"mouse_49\2026_08_11",
#    r"mouse_49\2026_08_12",
#    r"mouse_49\2026_08_13",
#    r"mouse_67\2026_06_23",
#    r"mouse_67\2026_06_24",
#    r"mouse_67\2026_06_25",
#    r"mouse_67\2026_06_26",
    r"mouse_105\2025_10_27",
    r"mouse_105\2025_10_28",
    r"mouse_105\2025_10_29",
    r"mouse_105\2025_10_30",
#    r"mouse_5785\2025_05_05",
#    r"mouse_5785\2025_05_06",
#    r"mouse_5785\2025_05_07",
#    r"mouse_5785\2025_05_08"
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



            


