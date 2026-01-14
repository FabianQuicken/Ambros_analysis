"""
Analyseziele:
- Aufenthaltsdauer im Stimulusmodul vs Kontrollmodul Gesamt
- Aufenthaltsdauer im Stimulusmodul vs Kontrollmodul pro Visit
- Interaktionsdauer Stimulus vs Kontrolle
- Strecke/Zeit in Stimulusmodul vs Kontrollmodul

Plots: 
- Eventplots für jede einzelne Maus

Output: 
- Metriken werden in eine Excel Datei geschrieben

Generelles:
- Jeder Run analysiert eine Maus
- Die Daten aller Mäuse sind im Ordner "raw" über 3 Subordner "Day1", "Day2", "Day3" verteilt
- Kameraöffnen markiert Experimentstart
- dann enstehen über 20min Videoschnipsel, die zeitlich eingeordnet werden müssen
- "none_none" im Namen markiert dishes ohne stimuli (Day1 und Day3)
- Beispiel um die Benennung am Stimulustag 2 zu verstehen --> "153darcin_152_hepes": 
    - Wenn darcin vor hepes modul1 = Stimulus, sonst modul2 = Stimulus
    - Modul1 Stimulus = Urin Maus 153 1:1 Darcin(in HEPES) 
    - Modul2 Kontrolle = Urin Maus 152 1:1 HEPES
- likelihood Filterung muss stattfinden, da die Mäuse nicht immer im Modul sind 
"""


import numpy as np
import pandas as pd
import tqdm
import os
import glob
import matplotlib.pyplot as plt


FPS = 30
PIXEL_PER_CM = 36.39
all_mice = ["109", "121", "122", "125"]
mouse = "111"

"""
Daten einlesen und in Stimulus und Kontrolle sortieren
"""

exp_path = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\raw"

day1_files = sorted(glob.glob(os.path.join(exp_path + "/Day1/" + mouse, '*.h5')))
day2_files = sorted(glob.glob(os.path.join(exp_path + "/Day2/" + mouse, '*.h5')))
day3_files = sorted(glob.glob(os.path.join(exp_path + "/Day2/" + mouse, '*.h5')))

m1_d1_files = [file for file in day1_files if "top1" in file]
m2_d1_files = [file for file in day1_files if "top2" in file]

m1_d2_files = [file for file in day2_files if "top1" in file]
m2_d2_files = [file for file in day2_files if "top2" in file]

m1_d3_files = [file for file in day3_files if "top1" in file]
m2_d3_files = [file for file in day3_files if "top2" in file]


stim_modul = None
if "darcin" in os.path.basename(m1_d2_files[0])[43:52]:
    stim_modul = 1
elif "hepes" in os.path.basename(m1_d2_files[0])[43:52]: 
    stim_modul = 2
else:
    raise NameError(f"Filename seems to be incorrect.\nFilename: {m1_d2_files[0]}")


if stim_modul == 1:
    stim_data = [m1_d1_files, m1_d2_files, m1_d3_files]
    con_data = [m2_d1_files, m2_d2_files, m2_d3_files]
else:
    con_data = [m1_d1_files, m1_d2_files, m1_d3_files]
    stim_data = [m2_d1_files, m2_d2_files, m2_d3_files]


    





