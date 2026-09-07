import preprocessing

FPS = 30
path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\dlc_output\omm12\females_54_57_60\hab"
preprocessing.main_preprocessing(path, FPS, exp_len_seconds=36000)