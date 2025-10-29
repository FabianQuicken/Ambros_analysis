"""

# hier entsteht der analyseteil für ein Multi Animal Experiment

# identity = False, d.h. es wird keine individuelle Maus analysiert, sondern die aufgenommene Kohorte als ganzes

# für identity = True, e.g. balb/c Maus + c57/bl6 Maus vermutlich eigene main.py später

# zur struktur: da die Multianimal dateien fehler beinhalten (id switches, fehlende daten, id overlays usw.), sollte die Analyse mit preprocessed Daten passieren, 
# nicht mit den raw dlc output files

# für non-social metrics macht es vermutlich sinn, über die Mäuse zu iterieren und die Daten am Ende zusammenzufassen - so könnte die single animal pipeline hier genutzt werden

# neue social metrics (e.g. social interaction) könnte separat analysiert werden

# nach Möglichkeit sollte die Pipeline mit verschieden vielen Individuen funktionieren


"""

# externe imports
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

# interne imports
from config import FPS, PIXEL_PER_CM, LIKELIHOOD_THRESHOLD, DF_COLS
from utils import euklidean_distance, fill_missing_values, time_to_seconds, convert_videostart_to_experiment_length, calculate_experiment_length
from metadata import module_has_stimulus_ma

# struktur zum speichern erstellen
@dataclass
class ModuleVariables:
    # arrays
    exp_duration_frames: np.ndarray
    distance_over_time: np.ndarray
    one_mouse_over_time: np.ndarray
    two_mice_over_time: np.ndarray
    three_mice_over_time: np.ndarray
    minimum_one_mouse_over_time: np.ndarray
    one_mouse_center_over_time: np.ndarray
    two_mice_center_over_time: np.ndarray
    three_mice_center_over_time: np.ndarray
    minimum_one_mouse_center_over_time: np.ndarray
    all_visits: np.ndarray

    # scalars
    speed_pixel_frame: float
    center_crossings: int
    all_mice_center_time: int
    visits_per_hour: float
    mean_visit_time: float

    # coords
    nose_coords_x_y: tuple

    # metadata
    date: str
    is_stimulus_module: bool
    start_time: str
    end_time: str
    mice: list
    modulnumber: int

# files für ein modul werden eingelesen (von einem Experimenttag)
path = r"C:\Users\quicken\Code\Ambros_analysis\code_test\multi_animal_analysis_test_exp\top1"
#path = r"C:\Users\Fabian\Code\Ambros_analysis\code_test"
file_list = glob.glob(os.path.join(path, '*.h5'))

"""
Beschneiden der Filelist für die Testruns
"""
file_list = file_list[10:13]



# information über die Anzahl individuen extrahieren, um Variablen zu initialisieren
first_df = pd.read_hdf(file_list[0])



individuals = first_df.columns.levels[1]




# dauer des Experiments wird berechnet, based on start und end video Zeitdaten
exp_duration_frames, startzeit, endzeit, date = calculate_experiment_length(first_file=file_list[0], last_file=file_list[-1])


# auslesen, ob das Modul einen Stimulus beinhaltet
is_stimulus_side, mouse_cohort, modulnumber = module_has_stimulus_ma(file_list[0])


# leere variablen und variablen die mit der Experimentlänge zu tun haben (und während der analyse stück für stück befüllt werden) einführen lel
mice_in_module = np.zeros((len(individuals), len(exp_duration_frames)), dtype=int)
min_one_mouse_in_module = exp_duration_frames.copy()

mice_in_center = np.zeros((len(individuals), len(exp_duration_frames)), dtype=int)
min_one_mouse_in_center = exp_duration_frames.copy()

distance_over_time = exp_duration_frames.copy()

nose_x_values_over_time = exp_duration_frames.copy()
nose_y_values_over_time = exp_duration_frames.copy()

# Weitere Variablen
distance_in_px = 0
sum_min_one_mouse_center = 0

all_visits = []
sum_visits = 0

"""
Ideen für weitere Analysemetriken:
# path length to straight-line distance (path meander)
# acceleration / deceleration events
# proportion of stationary vs moving time
# entropy of position distribution
# time to reach center after entry
# average distance from module entry 
# inter-entry intervals (social following)
# inter-individual distances over time
# investigation time (how long in total, what bodyparts)
# sequential occupancy (do mice leave together)
# autocorrelation of visits (do visits cluster in bursts)
# time to last entry
# exploration index (unique area visited / total arena)
# grooming-like stationary bouts (low speed + posture change)

"""

# iteration über jede videofile
for file in tqdm(file_list):
    

    # an welchem Zeitpunkt der Experimenttdauer befindet sich diese File
    time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS


    read_in_df = pd.read_hdf(file)
    df = read_in_df.copy()

    scorer = df.columns.levels[0][0]
    individuals = df.columns.levels[1]
    bodyparts = df.columns.levels[2]

    mouse_1_data = df.loc[:, (scorer, individuals[0], ["nose"], ["x", "y"])]
    

    # mouse present analyse: für die maximale anzahl an mäusen angepasst (1 mouse in modul, 2 mice in module, 3 mice in module)
    for index, ind in enumerate(individuals):

        # nose data extrahieren für die präsenz analyse
        nose_data = df.loc[:, (scorer, ind, ["nose"], ["x", "y", "likelihood"])].to_numpy()

        # über die likelihood checken, ob die Maus im Modul ist
        ind_is_present = (nose_data[:,2] >= 0.2).astype(int)

        # speichern der information im Kontext des gesamten Experiments
        for i in range(len(ind_is_present)):
            mice_in_module[index][i+(time_position_in_frames-1)] = ind_is_present[i]


# berechnen, ob mindestens eine Maus präsent ist
min_one_mouse_in_module = mice_in_module.any(axis=0).astype(int)
mice_per_frame = mice_in_module.sum(axis=0)
print(min_one_mouse_in_module.sum())
    # visit number etwas schwieriger, weil eine maus mehrmals das modul verlassen und betreten kann während einem video - maybe die entry zone entries --> arena entries zählen?
    # alle abgeschlossenen trajectories sammeln und speichern, am besten einmal alle zusammen und dann für jede maus einzeln in passender, zeitlicher relation


    # zurückgelegte strecke berechnen: einzelne mäuse addieren


    # maus in center analyse: ebenfalls für die maximale anzahl an mäusen angepasst


# center crossings über alle Mäuse zählen
# mean visit time
# visits per hour
# strecke über zeit = mean geschwindigkeit, sollte auf eine maus runtergerechnet werden (zwei mäuse = doppelt so viel strecke pro zeit, also Summe des 2 Maus arrays *2 und summe des 3 Maus arrays * 3)


# speichern als h5

# plots generieren





