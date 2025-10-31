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
import math
import warnings

# interne imports
from config import FPS, PIXEL_PER_CM, LIKELIHOOD_THRESHOLD, DF_COLS, ARENA_COORDS, ENTER_ZONE_COORDS
from utils import euklidean_distance, fill_missing_values, time_to_seconds, convert_videostart_to_experiment_length, calculate_experiment_length, is_point_in_polygon, create_point, create_polygon, shrink_rectangle
from metadata import module_has_stimulus_ma
from chatgpt_plots import plot_mice_presence_states, plot_mouse_trajectory
from preprocessing import interpolate_with_max_gap

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
    filenames: list




# arena und eingangsbereich des moduls als polygon definieren
enter_zone_polygon = create_polygon(ENTER_ZONE_COORDS)
arena_polygon = create_polygon(ARENA_COORDS)

# files für ein modul werden eingelesen (von einem Experimenttag)
path = r"C:\Users\quicken\Code\Ambros_analysis\code_test\for_labelled_video"
#path = r"C:\Users\Fabian\Code\Ambros_analysis\code_test"
file_list = glob.glob(os.path.join(path, '*.h5'))

"""
Beschneiden der Filelist für die Testruns
"""
#file_list = file_list[10:13]



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

filenames = []

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
    
    filenames.append(os.path.basename(file))

    # an welchem Zeitpunkt der Experimenttdauer befindet sich diese File
    time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS


    read_in_df = pd.read_hdf(file)
    df = read_in_df.copy()

    # kleinere fehlende Fragmente werden interpoliert, e.g. wenn Mäuse sich gegenseitig überdecken oder Keypoints fehlen
    df = interpolate_with_max_gap(df)

    scorer = df.columns.levels[0][0]
    individuals = df.columns.levels[1]
    bodyparts = df.columns.levels[2]

    mouse_1_data = df.loc[:, (scorer, individuals[0], ["nose"], ["x", "y"])]
    

    # mouse present analyse: für die maximale anzahl an mäusen angepasst (1 mouse in modul, 2 mice in module, 3 mice in module)
    for index, ind in enumerate(individuals):

        # nose data extrahieren für die präsenz analyse
        nose_data = df.loc[:, (scorer, ind, ["nose"], ["x", "y", "likelihood"])].to_numpy()

        # über die likelihood checken, ob die Maus im Modul ist
        ind_is_present = (nose_data[:,2] >= 0.05).astype(int)

        # speichern der information im Kontext des gesamten Experiments
        for i in range(len(ind_is_present)):
            mice_in_module[index][i+(time_position_in_frames-1)] = ind_is_present[i]



    # visit number etwas schwieriger, weil eine maus mehrmals das modul verlassen und betreten kann während einem video - maybe die entry zone entries --> arena entries zählen?
    for index, ind in enumerate(individuals):

        # x arrays für alle bodyparts eines individuums 
        arr_x = df.loc[:, (scorer, ind, bodyparts, ["x", "y"])].values[:,::2]
        # y arrays für alle bodyparts eines individuums
        arr_y = df.loc[:, (scorer, ind, bodyparts, ["x", "y"])].values[:,1::2] * -1

        # einmal die länge des Arrays sowie die anzahl  verschiedener Bodyparts extrahieren
        n_frames, n_bp = arr_x.shape

        # mindestens ein Drittel der Bodyparts sollte vorhanden sein, um das Center der Maus zu berechnen
        min_bodyparts = math.ceil(n_bp / 3)

        # valid sind alle einträge, wo sowohl x und y für einen Bodypart vorhanden sind
        valid = (~np.isnan(arr_x)) & (~np.isnan(arr_y))       # (n_frames, n_bp)
        # summe aller erfolgreich getrackten bodyparts pro zeile
        valid_counts = valid.sum(axis=1)                      # (n_frames,)

        """
        # checkt, ob die maus überhaupt im Video ist
        valid_x = ~np.isnan(arr_x)
        valid_y = ~np.isnan(arr_y)
        valid_points = valid_x & valid_y
        has_valid = valid_points.sum() > 0
        """
        

        # center der maus als mean aller punkte bestimmen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centroid_x = np.nanmean(arr_x, axis=1)
            centroid_y = np.nanmean(arr_y, axis=1)

        # Wenn weniger als die Hälfte der Bodyparts getrackt wurden, wird der center wert auf nan gesetzt
        too_few = valid_counts < min_bodyparts
        centroid_x[too_few] = np.nan
        centroid_y[too_few] = np.nan

        # mouse in center analyse
        mouse_in_center = np.zeros(len(centroid_x))
        # arena polygon verkleinern, damit es zum center polygon wird
        center_coords = shrink_rectangle(ARENA_COORDS, scale=0.6)
        center_polygon = create_polygon(center_coords)

        for i in range(len(mouse_in_center)):
            # point object erstellen
            point = create_point(x=centroid_x[i], y=centroid_y[i])
            # testen, ob point im polygon liegt oder nicht
            mouse_in_center[i] = is_point_in_polygon(polygon = center_polygon, point=point)

        # speichern der information im Kontext des gesamten Experiments
        for i in range(len(mouse_in_center)):
            mice_in_center[index][i+(time_position_in_frames-1)] = mouse_in_center[i]


        """
        if '2025_10_08_13_07_18_mice_c1_exp1_male_none_top1' in file:
            plot_mouse_trajectory(center_x=centroid_x, center_y=centroid_y)
        """

        
        # analyse der entries und exits




    # alle abgeschlossenen trajectories sammeln und speichern, am besten einmal alle zusammen und dann für jede maus einzeln in passender, zeitlicher relation


    # zurückgelegte strecke berechnen: einzelne mäuse addieren


    # maus in center analyse: ebenfalls für die maximale anzahl an mäusen angepasst


# berechnen, ob mindestens eine Maus präsent ist 
min_one_mouse_in_module = mice_in_module.any(axis=0).astype(int)
# berechnen, wie viele mäuse pro frame im Bild sind
mice_per_frame = mice_in_module.sum(axis=0)
"""
plot_mice_presence_states(mice_in_module=mice_in_module)
"""

# berechnen, ob mindestens eine Maus im center ist
min_one_mouse_in_center = mice_in_center.any(axis=0).astype(int)
# berechnen, wieviele mäuse pro frame im Center sind
mice_center_per_frame = mice_in_center.sum(axis=0)
"""
plot_mice_presence_states(mice_in_module=mice_in_center, title = 'Mice in Center')
"""

from create_labelled_video import create_labelled_video
create_labelled_video(video_path=r"C:\Users\quicken\Code\Ambros_analysis\code_test\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_40439818DLC_HrnetW32_multi_animal_pretrainedOct24shuffle1_detector_best-270_snapshot_best-120_el_id_p0_labeled.mp4",
                      output_path=r"C:\Users\quicken\Code\Ambros_analysis\code_test\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_labelled.mp4",
                      num_mice=mice_per_frame,
                      num_mice_center=mice_center_per_frame)

# center crossings über alle Mäuse zählen
# mean visit time
# visits per hour
# strecke über zeit = mean geschwindigkeit, sollte auf eine maus runtergerechnet werden (zwei mäuse = doppelt so viel strecke pro zeit, also Summe des 2 Maus arrays *2 und summe des 3 Maus arrays * 3)


# speichern als h5

# plots generieren





