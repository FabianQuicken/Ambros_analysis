import preprocessing
import os

FPS = 30
"""



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

"""

path = r"\\fileserver2.bio2.rwth-aachen.de\AG Spehr BigData\n2023_odor_related_behavior\2026_sick_behavior\raw\c1\baseline\top1"

preprocessing.main_preprocessing(path, FPS, 440000, video_filetype=".avi", filter_value=0.6)