import preprocessing

FPS = 30
path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\analyse\mouse_2\2024_12_17\top2"
path = r"C:\Users\Fabian\Desktop\Transfer\analysis_testing"
preprocessing.main_preprocessing(path, FPS, exp_len_seconds=36000, filter_value=0.6)