from scipy.stats import mannwhitneyu, wilcoxon
import numpy as np


# petri dish investigation
stim_day1 = [838, 397, 17, 619, 836, 135]
con_day1 = [810, 273, 516, 496, 269, 183]

stim_day2 = [679, 647, 1282, 774, 1116, 770]
con_day2 = [1580, 593, 1242, 1053, 909, 1382]

stim_day3 = [1133, 2258, 1294, 1970, 1397, 640]
con_day3 = [1085, 287, 1019, 763, 1006, 391]


print("Stim day comparisons")
print(mannwhitneyu(x=stim_day1, y=stim_day2))
print(mannwhitneyu(x=stim_day1, y=stim_day3))
print(mannwhitneyu(x=stim_day2, y=stim_day3))

print("Con day comparisons")
print(mannwhitneyu(x=con_day1, y=con_day2))
print(mannwhitneyu(x=con_day1, y=con_day3))
print(mannwhitneyu(x=con_day2, y=con_day3))

print("Stim vs con day")
print(mannwhitneyu(x=stim_day1, y=con_day1))
print(mannwhitneyu(x=stim_day2, y=con_day2))
print(mannwhitneyu(x=stim_day3, y=con_day3))


# time present
stim_day1 = [0.264, 0.251, 0.09, 0.216, 0.276, 0.031]
con_day1 = [0.345, 0.181, 0.128, 0.212, 0.186, 0.150]
stimpref_day1 = np.asarray(stim_day1) - np.asarray(con_day1)

stim_day2 = [0.200, 0.220, 0.267, 0.280, 0.269, 0.206]
con_day2 = [0.315, 0.230, 0.267, 0.229, 0.232, 0.298]
stimpref_day2 = np.asarray(stim_day2) - np.asarray(con_day2)

stim_day3 = [0.250, 0.245, 0.243, 0.292, 0.290, 0.194]
con_day3 = [0.285, 0.166, 0.232, 0.246, 0.136, 0.185]
stimpref_day3 = np.asarray(stim_day3) - np.asarray(con_day3)

stim_day2_corrected = np.array(stim_day2) - np.array(stim_day1)
con_day2_corrected = np.array(con_day2) - np.array(con_day1)

stim_day3_corrected = np.array(stim_day3) - np.array(stim_day1)
con_day3_corrected = np.array(con_day3) - np.array(con_day1)

day_2_pref_corrected = stim_day2_corrected - con_day2_corrected
day_3_pref_corrected = stim_day3_corrected - con_day3_corrected

# Steigt die Stimpref von Day2 zu Day3? 
print("Is there a difference of baseline corrected stimpref vs day3 and day2?")
stat, p = mannwhitneyu(x=day_2_pref_corrected, y=day_3_pref_corrected)
print("p:", p)

# Gibt es eine preference zu stimulus seite?
print("Is there a Baseline corrected preference to stimulus module day 2?")
stat, p = wilcoxon(day_2_pref_corrected, alternative="greater")
print("p:", p)
print("Is there a Baseline corrected preference to stimulus module day 3?")
stat, p = wilcoxon(day_3_pref_corrected, alternative="greater")
print("p:", p)

print("Is there a difference in raw stimulus module preference between day2 and hab(Day1)?")
stat, p = mannwhitneyu(x=stimpref_day1, y=stimpref_day2)
print("p:", p)

print("Is there a difference in raw stimulus module preference between day3 and hab(Day1)?")
stat, p = mannwhitneyu(x=stimpref_day3, y=stimpref_day1)
print("p:", p)

print("Is there a raw stimmodule pref at day1?")
stat, p = wilcoxon(stimpref_day1, alternative="greater")
print("p:", p)
print("Is there a raw stimmodule at day2?")
stat, p = wilcoxon(stimpref_day2, alternative="greater")
print("p:", p)
print("Is there a raw stimmodule at day3?")
stat, p = wilcoxon(stimpref_day3, alternative="greater")
print("p:", p)
