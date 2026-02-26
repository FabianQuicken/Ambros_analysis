from main_multi_animal import multi_animal_main
import matplotlib.pyplot as plt
from config import PIXEL_PER_CM, FPS
import numpy as np

top1 = r"\top1"
top2 = r"\top2"
hab = r"\top1\hab"


germfree = [
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\females_30_45_46",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\females_68_69_70",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\males_38_47_53",
            r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfree\males_53_55_61"
]

germfreeprop = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\females_37_44_55",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\females_52_56_62",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\germfreeprop\males_34_38_42"
]

omm12 = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\females_31_36_59",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\females_54_57_60",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\males_41_43_58",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12\males_83_86_71"
]

omm12prop = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\females_32_35_37",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\females_75_78_82",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\males_60_64_66",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\omm12prop\males_73_74_77"
]

ommpgol = [
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_33_47_48",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_72_76_79",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\males_41_44_51",
    r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\males_80_81_87"
]


testpaths_grp1 = [
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\f1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\f2",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\m1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp1\m2"
]
testpaths_grp2 = [
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\f1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\f2",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\m1",
    r"C:\Users\quicken\Code\Ambros_analysis\code_test\ommfull\grp2\m2"
]

groups = [testpaths_grp1, testpaths_grp2]
test_colors = ["red", "blue"]


mode = hab
cumdists = []
colors = []
for group in groups:
    for path in group:
        path = path + mode
        dic = multi_animal_main(path)
        cumdists.append(dic["cumdist"])
        if "grp1" in path:
            colors.append(test_colors[0])
        elif "grp2" in path:
            colors.append(test_colors[1])

    min_len = min([len(arr) for arr in cumdists])
    for i, arr in enumerate(cumdists):
        arr = arr / PIXEL_PER_CM
        time_sec = np.arange(min_len) / FPS
        plt.plot(time_sec, arr[0:min_len], color = colors[i])
plt.xlabel()
plt.show()