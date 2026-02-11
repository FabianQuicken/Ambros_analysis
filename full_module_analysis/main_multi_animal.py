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
import scipy as sc

# interne imports
from config import FPS, PIXEL_PER_CM, LIKELIHOOD_THRESHOLD, DF_COLS, ARENA_COORDS, ENTER_ZONE_COORDS
from metrics import distance_travelled_arraybased
from utils import euklidean_distance, fill_missing_values, time_to_seconds, moving_average
from utils import convert_videostart_to_experiment_length, calculate_experiment_length
from utils import is_point_in_polygon, create_point, create_polygon, shrink_rectangle, mouse_center
from metadata import module_has_stimulus_ma
from chatgpt_plots import plot_mice_presence_states, plot_mouse_trajectory
from preprocessing import interpolate_with_max_gap, ma_likelihood_filter, find_id_overlay, filter_id_overlays
from social_behavior_analysis import social_investigation, detail_social_investigation
from trajectory_metrics import entry_exit_trajectories, arc_chord_ratio, get_all_traj, theta_analysis
from plotting import polar_angle_histogram
from animated_plots import animate_trace

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
#path = r"C:\Users\quicken\Code\Ambros_analysis\code_test\trajectory_immobile"
path = r"Z:\n2023_odor_related_behavior\2025_omm_mice\analyse tests"
path_ho = r"C:\Users\Fabian\Code\Ambros_analysis\code_test\ma_unfamiliar"
ho = False
if ho:
    path=path_ho
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

mice_distances = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
distance_over_time = mice_distances.copy()

nose_x_values_over_time = exp_duration_frames.copy()
nose_y_values_over_time = exp_duration_frames.copy()

# Weitere Variablen
distance_in_px = 0
sum_min_one_mouse_center = 0

all_visits = []
sum_visits = 0

# speichert alle einzelnen trajectories
trajectories = []
# speichert alle arc/chord ratios einzelner trajectories
all_arc_chord = []

social_inv = None


filenames = []

stitch_dataframes = False
if stitch_dataframes:
    def stitch_dfs_realtime(dfs, start_frames, total_frames):
        """
        Stitches multiple dataframes into a single realtime-aligned dataframe.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            DataFrames with identical columns.
        start_frames : list[int]
            Start frame of each dataframe in global time.
        total_frames : int
            Total number of frames of the experiment.

        Returns
        -------
        pd.DataFrame
            Stitched dataframe with NaNs for gaps.
        """
        master = pd.DataFrame(
            index=np.arange(total_frames),
            columns=dfs[0].columns,
            dtype=float
        )

        for df, start in zip(dfs, start_frames):
            end = start + len(df)
            master.iloc[start:end] = df.to_numpy()

        existing = master.iloc[start:end]
        has_overlap = existing.notna().any().any()
        if has_overlap:
            print("\nOverlap detected")

        return master
    
    def stitch_dfs_no_overlap(dfs, start_seconds, fps=FPS):
        """
        Stitches dataframes by enforcing continuous frame order.
        Overlaps caused by coarse timestamps are removed.
        """

        stitched = []
        current_end = 0

        for df, sec in zip(dfs, start_seconds):
            theoretical_start = int(sec * fps)

            start = max(theoretical_start, current_end)
            df = df.copy()
            df.index = np.arange(start, start + len(df))

            stitched.append(df)
            current_end = start + len(df)

        return pd.concat(stitched).sort_index()

    dfs = [pd.read_hdf(file) for file in file_list]
    t_frames = len(exp_duration_frames)
    s_frames = [convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS for file in file_list]
    master_df = stitch_dfs_realtime(dfs,
                             s_frames,
                             t_frames)
    #master_df = stitch_dfs_no_overlap(dfs,
    #                                  [convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) for file in file_list]
    #                                  )


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

if stitch_dataframes:
    file_list = [file_list[0]]
# iteration über jede videofile
for file in tqdm(file_list):

    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # DATAFRAME EINLESEN UND VORBEREITEN # # # # # # # # # # # # # # # # # 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    filenames.append(os.path.basename(file))

    # an welchem Zeitpunkt der Experimenttdauer befindet sich diese File
    time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS

    # einlesen und copy anlegen
    if stitch_dataframes:
        df = master_df.copy()
    else:
        read_in_df = pd.read_hdf(file)
        df = read_in_df.copy()

    # überschriften des df auslesen
    scorer = df.columns.levels[0][0]
    individuals = df.columns.levels[1]
    bodyparts = df.columns.levels[2]

    # likelihood filter, um sehr unwahrscheinliche predictions rauszunehmen
    df = ma_likelihood_filter(df, scorer, individuals, bodyparts, filter_value=0.3)
    
    # kleinere fehlende Fragmente werden interpoliert, e.g. wenn Mäuse sich gegenseitig überdecken oder Keypoints fehlen
    df = interpolate_with_max_gap(df)

    # selten treten ID Overlays auf, DeepLabCut labelt die selbe Maus zwei Mal mit den exakt selben Koordinaten
    overlays_dic = find_id_overlay(df, scorer, individuals, bodyparts)
    if overlays_dic:
        
        for entry in overlays_dic:
            overlay_inds = []
            for ind in individuals:
                if ind in entry:
                    overlay_inds.append(ind)

            df = filter_id_overlays(overlay_inds=overlay_inds,
                               overlay_slices=overlays_dic[entry],
                               scorer=scorer,
                               bodyparts=bodyparts,
                               df=df)

    # y invertieren, da DLC Bildkoordinaten nutzt (y=0 ist oberer Bildrand)
    df.loc[:, (scorer, individuals, bodyparts, ["y"])] *= -1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # Mouse Center + Damit zusammenhängende Analysen # # # # # # # # # # # # # # 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # jeweilige mouse center berechnen (shape n_ind, n_frames)
    all_centroid_x, all_centroid_y = mouse_center(df, scorer, individuals, bodyparts, min_bodyparts = math.ceil(len(bodyparts) / 3))

    # # # # # distance travelled # # # # # 

    for index, ind in enumerate(individuals):

        dist_values = distance_travelled_arraybased(x_arr=all_centroid_x[index],
                                                    y_arr=all_centroid_y[index])
        


        dist_values = moving_average(data=dist_values, window=10)    

        is_immobile = np.where(dist_values < 4, 1, 0)

        #print(f"\n Dist Traveled: {sum(dist_values)}")
        print(f"\n Total immobile frames: {sum(is_immobile) / len(is_immobile)}")

        animated_plot = False
        if animated_plot:
            colors = ["purple", "green", "red"]
            color = colors[index]

            animate_trace(dist_values[0:1800],
                fps=30,
                window_seconds=1,
                color=color,
                save_path=path+f"/trace_animation_{ind}.mp4")
            
            
        
        for i in range(len(dist_values)):
            if not np.isnan(dist_values[i]):
                mice_distances[index][i+(time_position_in_frames-1)] = dist_values[i]

        cum_dist = np.nancumsum(dist_values)
        for i in range(len(dist_values)):
            distance_over_time[index][i+(time_position_in_frames-1)] = cum_dist[i]

    # # # # # arena center analyse # # # # # 

    for index, ind in enumerate(individuals):


        centroid_x = all_centroid_x[index]
        centroid_y = all_centroid_y[index]


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
        #print(f"Mouse in center for {sum(mouse_in_center)} frames")
        # speichern der information im Kontext des gesamten Experiments
        for i in range(len(mouse_in_center)):
            mice_in_center[index][i+(time_position_in_frames-1)] = mouse_in_center[i]
    

    # # # # # trajectory analyse # # # # # 
    
    # alle trajectories, auch von mäusen die bereits in der Kammer sind bei Videostart
    all_traj, traj_slices = get_all_traj(x_arrs = all_centroid_x, y_arrs=all_centroid_y, individuals=individuals)
    

    # trajectories von Mäusen, die im Video die Kamera betreten und wieder verlassen
    e_ex_all_traj = entry_exit_trajectories(x_arrs=all_centroid_x,
                                            y_arrs=all_centroid_y,
                                            traj_slices=traj_slices,
                                            individuals=individuals,
                                            entry_polygon=enter_zone_polygon,
                                            plot=False)

    # # # WARNUNG # # # 
    # DAS SIND IM MOMENT NUR DIE VISITS, DIE IN STARTZONE BEGINNEN/ENDEN
    # TIERE DIE IM GANZEN VIDEO IM MODUL SIND, TRAGEN EINEN VISIT IN VIDEOLÄNGE BEI
    # # # WARNUNG # # #

    # visits info hinzufügen
    for ind in individuals:
        sum_visits += len(e_ex_all_traj[ind])

        for x,y in e_ex_all_traj[ind]:
            all_visits.append(len(x))



            
    """
    for ind in e_ex_all_traj:
        for traj in e_ex_all_traj[ind]:

            for t in traj:
                all_visits.append(len(t))
        sum_visits += len(e_ex_all_traj[ind])
    """
    # front & rear auf wirbelsäule der Maus, um Richtungsänderung zu berechnen
    front_center_x, front_center_y = mouse_center(df,
                                                  scorer, individuals,
                                                  ["shoulder_left", "shoulder_right", "dorsal_1"],
                                                  min_bodyparts=3)
    
    rear_center_x, rear_center_y = mouse_center(df,
                                                scorer,
                                                individuals,
                                                ["hip_left", "hip_right", "dorsal_4"],
                                                min_bodyparts=3)
    
    theta_list, theta_dic = theta_analysis(individuals=individuals,
                   front_x=front_center_x,
                   front_y=front_center_y,
                   rear_x=rear_center_x,
                   rear_y=rear_center_y,
                   slices=traj_slices)
    
    polarplot = False

    if polarplot:
        print(len(theta_list))
        ax, hist, bins = polar_angle_histogram(
            theta_list,
            n_bins=36,
            angle_cutoff=20,
            density=True,
            title="Turning angle distribution"
            )

        plt.show()

    # mittlere arc/chord ratio je trajectory für alle trajectories
    for t in all_traj:
        trajectories.append(t)
        all_arc_chord.append(arc_chord_ratio(trajectory=t))
    


    # visit number etwas schwieriger, weil eine maus mehrmals das modul verlassen und betreten kann während einem video - maybe die entry zone entries --> arena entries zählen?
    for index, ind in enumerate(individuals):
        pass


    


        """
        if '2025_10_08_13_07_18_mice_c1_exp1_male_none_top1' in file:
            plot_mouse_trajectory(center_x=centroid_x, center_y=centroid_y)
        """


    # mouse present analyse: für die maximale anzahl an mäusen angepasst (1 mouse in modul, 2 mice in module, 3 mice in module)
    for index, ind in enumerate(individuals):

        # nose data extrahieren für die präsenz analyse
        nose_data = df.loc[:, (scorer, ind, ["nose"], "x")].to_numpy()

        # über finite koordinaten checken, ob die maus im modul ist (bezieht interpolation mit ein)
        ind_is_present = np.zeros(len(nose_data)).astype(int)
        for idx, x_coord in enumerate(nose_data):
            if np.isfinite(x_coord):
                ind_is_present[idx] = 1
        # speichern der information im Kontext des gesamten Experiments
        for i in range(len(ind_is_present)):
            mice_in_module[index][i+(time_position_in_frames-1)] = ind_is_present[i]




    # social investigation analyse
    social_inv = social_investigation(df, scorer, individuals, bodyparts)

    social_inv_details = detail_social_investigation(df, scorer, individuals, pixel_per_cm=PIXEL_PER_CM, max_dist_cm=2)
    
    # checkt ob jeweils face, body oder anogenital investigation pro frame (keine doppelzählung, also zB anogenital_inv individual 1 -> individual 2 und invividual 2 -> individual 3 wird hier nicht beides gezählt)
    face_inv = social_inv_details["presence_per_frame"]["face"]
    body_inv = social_inv_details["presence_per_frame"]["body"]
    anogenital_inv = social_inv_details["presence_per_frame"]["anogenital"]

    # hier die summen, um auch gleichzeitige investigation gleicher bodyparts zu finden
    sum_face_inv = social_inv_details["totals"]["face"]
    sum_body_inv = social_inv_details["totals"]["body"]
    sum_anogenital_inv = social_inv_details["totals"]["anogenital"]


    
    




# berechnen, ob mindestens eine Maus präsent ist 
min_one_mouse_in_module = mice_in_module.any(axis=0).astype(int)
# berechnen, wie viele mäuse pro frame im Bild sind
mice_per_frame = mice_in_module.sum(axis=0)
# distanz, die pro frame zurückgelegt wird (addiert mehrere Mäuse)
distance_per_frame = mice_distances.sum(axis=0)
# kumulative distanz pro frame
cumdist_per_frame = np.nancumsum(distance_per_frame)
# full dist
distance_in_px = cumdist_per_frame[-1]

"""
plot_mice_presence_states(mice_in_module=mice_in_module)
"""

# berechnen, ob mindestens eine Maus im center ist
min_one_mouse_in_center = mice_in_center.any(axis=0).astype(int)
# berechnen, wieviele mäuse pro frame im Center sind
mice_center_per_frame = mice_in_center.sum(axis=0)

#print("\n Metrics:")
#print(np.nansum(mice_center_per_frame))
#print(np.nansum(mice_in_module))
#print(distance_in_px)

#plot_mice_presence_states(mice_in_module=mice_in_center, title = 'Mice in Center')


create_labelled_video = False
if create_labelled_video:
    from create_labelled_video import create_labelled_video
    create_labelled_video(video_path=r'C:\Users\Fabian\Code\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_40439818DLC_HrnetW32_multi_animal_pretrainedOct24shuffle1_detector_best-270_snapshot_best-120_el_id_p0_labeled.mp4',
                      output_path=r'C:\Users\Fabian\Code\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_labelled.mp4',
                      metric1=test["presence_per_frame"]["body"],
                      text1="Body Investigation",
                      metric2=test["presence_per_frame"]["anogenital"],
                      text2="Anogenital Investigation")

create_labelled_video_modular = False
if create_labelled_video_modular:
    from create_labelled_video import create_labelled_video_modular
    create_labelled_video_modular(video_path=r'C:\Users\quicken\Code\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_40439818DLC_HrnetW32_multi_animal_pretrainedOct24shuffle1_detector_best-270_snapshot_best-120_el_id_p0_labeled.mp4',
                      output_path=r'C:\Users\quicken\Code\2025_10_08_13_07_18_mice_c1_exp1_male_none_top1_labelled.mp4',
                      metrics=[
                          ("Mice in Video:", mice_per_frame),
                          ("Mice in Center", mice_center_per_frame),
                          ("Social Investigation:", social_inv),
                          ("Face Investigation:", face_inv),
                          ("Body Investigation:", body_inv),
                          ("Anogenital Investigation:", anogenital_inv)
                      ],
                      row_gap=20,
                      scale_factor=1.0
    )



# center crossings über alle Mäuse zählen
# mean visit time
# visits per hour
# strecke über zeit = mean geschwindigkeit, sollte auf eine maus runtergerechnet werden (zwei mäuse = doppelt so viel strecke pro zeit, also Summe des 2 Maus arrays *2 und summe des 3 Maus arrays * 3)


# speichern als h5

# plots generieren