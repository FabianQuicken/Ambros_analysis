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
from config import FPS, PIXEL_PER_CM, LIKELIHOOD_THRESHOLD, DF_COLS, ARENA_COORDS, ENTER_ZONE_COORDS, IMMOBILE_THRSH
from metrics import distance_travelled_arraybased, acceleration, acceleration_events
from utils import euklidean_distance, fill_missing_values, time_to_seconds, moving_average, remove_distance_jitter
from utils import convert_videostart_to_experiment_length, calculate_experiment_length
from utils import is_point_in_polygon, create_point, create_polygon, shrink_rectangle, mouse_center
from metadata import module_has_stimulus_ma
from chatgpt_plots import plot_mice_presence_states, plot_mouse_trajectory
from preprocessing import interpolate_with_max_gap, ma_likelihood_filter, find_id_overlay, filter_id_overlays, filter_prediction_fragments
from social_behavior_analysis import social_investigation, detail_social_investigation
from trajectory_metrics import entry_exit_trajectories, arc_chord_ratio, get_all_traj, theta_analysis
from plotting import polar_angle_histogram
from animated_plots import animate_trace
from video_gen import overlay_two_points_line_and_theta_segments, overlay_metric_at_centers, overlay_rolling_plot_at_centers
from grooming import grooming_energy
from plotting import heatmap_plot
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
habituation = False
social_inv = False
path = r"C:\Users\quicken\Code\Ambros_analysis\code_test\soial_behavior"
path_ho = r"C:\Users\Fabian\Code\Ambros_analysis\code_test\ma_unfamiliar"
ho = False
if ho:
    path=path_ho
#path = r"C:\Users\Fabian\Code\Ambros_analysis\code_test"
def multi_animal_main(path, habituation=False, social_inv=False, plot_heatmap=False, stitch_dataframes=True):
    file_list = glob.glob(os.path.join(path, '*.h5'))
    file_list.sort()


    """
    Beschneiden der Filelist für die Testruns
    """
    #print(file_list)
    #file_list = [file_list[2]]


    # information über die Anzahl individuen extrahieren, um Variablen zu initialisieren
    first_df = pd.read_hdf(file_list[0])
    individuals = first_df.columns.levels[1]

    # dauer des Experiments wird berechnet, based on start und end video Zeitdaten
    exp_duration_frames, startzeit, endzeit, date = calculate_experiment_length(first_file=file_list[0], last_file=file_list[-1])
    total_exp_frames = exp_duration_frames
    if habituation:
        exp_duration_frames = exp_duration_frames[0:min([54000, len(exp_duration_frames)])]

    # auslesen, ob das Modul einen Stimulus beinhaltet
    is_stimulus_side, mouse_cohort, modulnumber = module_has_stimulus_ma(file_list[0])


    # leere variablen und variablen die mit der Experimentlänge zu tun haben (und während der analyse stück für stück befüllt werden) einführen lel
    

    # Weitere Variablen
    all_accelerations = []
    acc_events_count = 0
    distance_in_px = 0
    sum_min_one_mouse_center = 0

    all_visits = []
    sum_visits = 0
    start_len_visits = []


    # speichert alle einzelnen trajectories
    trajectories = []
    # speichert alle arc/chord ratios einzelner trajectories
    all_arc_chord = []

    # # # Zusammenfassende Metriken: shape (n_ind, n_frames) # # #
    mice_in_module = np.zeros((len(individuals), len(exp_duration_frames)), dtype=int)
    mice_in_center = np.zeros((len(individuals), len(exp_duration_frames)), dtype=int)
    immobile_over_time = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=int)
    mice_distances = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    cumdist_over_time = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    mice_accelerations = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    mice_center_x = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    mice_center_y = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    face_inv = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    body_inv = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)
    anogenital_inv = np.full((len(individuals), len(exp_duration_frames)), np.nan, dtype=float)

    # # # Zusammenfassende Metriken: shape (n_frames), mit zeros gefüllter array # # # 
    min_one_mouse_in_module = exp_duration_frames.copy()
    min_one_mouse_in_center = exp_duration_frames.copy()

    # # # Zusammenfassende Metriken: shape (n_ind, n_values) # # #
    thetas = [[] for _ in range(len(individuals))]
    trajectories = [[] for _ in range(len(individuals))]
    all_arc_chord = [[] for _ in range(len(individuals))]
    centers_xy = [[] for _ in range(len(individuals))]
    fronts_xy = [[] for _ in range(len(individuals))]
    rears_xy = [[] for _ in range(len(individuals))]
    nose_xy = [[] for _ in range(len(individuals))]

    # # # Zusammenhängende Metriken: Start & End Indices von Events {ind: [(start, end), (start, end), ...]}
    start_end_visits = {ind: [] for ind in individuals}
    start_end_immobile = {ind: [] for ind in individuals}
    start_end_center = {ind: [] for ind in individuals}
    start_end_acc = {ind: [] for ind in individuals}
    start_end_faceing = {ind: [] for ind in individuals}
    start_end_bodyinv = {ind: [] for ind in individuals}
    start_end_anogenitalinv = {ind: [] for ind in individuals}
    



    filenames = []

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
        
        def rename_individuals(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
            """
            Rename individual identifiers in df.columns using `mapping`.

            Supports:
            - MultiIndex columns with a level named 'individual' OR positional level 1 (DLC-typisch).
            - Flat string columns (best-effort substring replacement).

            Parameters
            ----------
            df : pd.DataFrame
            mapping : dict
                e.g. {"ind1":"mouse_1","ind2":"mouse_2","ind3":"mouse_3","ind4":"mouse_4",
                    "mouse1":"mouse_1","mouse2":"mouse_2","mouse3":"mouse_3","mouse4":"mouse_4"}

            Returns
            -------
            pd.DataFrame
            """
            if df is None or df.empty:
                return df

            out = df.copy()

            if isinstance(out.columns, pd.MultiIndex):
                names = list(out.columns.names)

                # determine which level is the individual level
                if "individual" in names:
                    lvl = names.index("individual")
                else:
                    # common DLC: (scorer, individual, bodypart, coord)
                    lvl = 1 if out.columns.nlevels >= 2 else None

                if lvl is None:
                    return out

                cols = out.columns
                tuples = list(cols)

                new_tuples = []
                for t in tuples:
                    t = list(t)
                    old = t[lvl]
                    t[lvl] = mapping.get(old, old)
                    new_tuples.append(tuple(t))

                out.columns = pd.MultiIndex.from_tuples(new_tuples, names=cols.names)
                return out

            # flat columns: do string replace token-wise
            new_cols = []
            for c in out.columns:
                s = str(c)
                for k, v in mapping.items():
                    # replace as token or substring; keep it simple but predictable
                    if k in s:
                        s = s.replace(k, v)
                new_cols.append(s)
            out.columns = new_cols
            return out


        def _cols_for_individual(cols, individual):
            if isinstance(cols, pd.MultiIndex):
                if "individual" in cols.names:
                    return cols.get_level_values("individual") == individual
                return cols.get_level_values(1) == individual
            ind = str(individual)
            return np.array([ind in str(c) for c in cols], dtype=bool)


        def fold_extra_individual(
            df: pd.DataFrame,
            *,
            preferred=("mouse_1", "mouse_2", "mouse_3"),
            extra="mouse_4",
        ) -> pd.DataFrame:
            """
            Move data from `extra` individual into the first preferred individual that has
            no data in that row (all NaN across its columns), then drop `extra` columns.
            """
            if df is None or df.empty:
                return df

            extra_mask = _cols_for_individual(df.columns, extra)
            extra_cols = df.columns[extra_mask]
            if len(extra_cols) == 0:
                return df

            df2 = df.copy()

            for tgt in preferred:
                tgt_mask = _cols_for_individual(df2.columns, tgt)
                tgt_cols = df2.columns[tgt_mask]
                if len(tgt_cols) == 0:
                    continue
                if len(tgt_cols) != len(extra_cols):
                    # can't block-copy safely if layout differs
                    continue

                tgt_all_nan = df2.loc[:, tgt_cols].isna().all(axis=1)
                extra_has_any = df2.loc[:, extra_cols].notna().any(axis=1)
                rows_to_move = tgt_all_nan & extra_has_any

                if rows_to_move.any():
                    df2.loc[rows_to_move, tgt_cols] = df2.loc[rows_to_move, extra_cols].to_numpy()
                    df2.loc[rows_to_move, extra_cols] = np.nan
                    break

            # drop extra either way (as requested)
            df2 = df2.drop(columns=extra_cols)
            return df2


        def normalize_df_individual_layout(df: pd.DataFrame) -> pd.DataFrame:
            """
            Canonicalize individual naming to mouse_1..mouse_4, then fold/drop mouse_4.
            Handles inputs like [mouse1, mouse2, mouse3] and [ind1..ind4].
            """
            mapping = {
                # ind layout
                "ind1": "mouse_1", "ind2": "mouse_2", "ind3": "mouse_3", "ind4": "mouse_4",
                # mouse layout without underscore
                "mouse1": "mouse_1", "mouse2": "mouse_2", "mouse3": "mouse_3", "mouse4": "mouse_4",
                # already canonical forms map to themselves implicitly
            }
            df = rename_individuals(df, mapping=mapping)
            df = fold_extra_individual(df, preferred=("mouse_1", "mouse_2", "mouse_3"), extra="mouse_4")
            return df


        def stitch_dfs_realtime(dfs, start_frames, total_frames):
            """
            Stitches multiple dataframes into a single realtime-aligned dataframe.

            - Normalizes individual names (ind1.. -> mouse_1.., mouse1 -> mouse_1, etc.)
            - Folds mouse_4 into mouse_1..mouse_3 where target rows are empty
            - Drops mouse_4 columns
            - Aligns all dfs by union of columns to avoid positional mis-write
            """
            norm_dfs = [normalize_df_individual_layout(df) for df in dfs]

            # union columns across all dfs
            all_cols = norm_dfs[0].columns
            for df in norm_dfs[1:]:
                all_cols = all_cols.union(df.columns)

            master = pd.DataFrame(index=np.arange(total_frames), columns=all_cols, dtype=float)

            for df, start in zip(norm_dfs, start_frames):
                end = start + len(df)
                block = df.reindex(columns=all_cols)

                existing = master.iloc[start:end]
                if existing.notna().any().any():
                    print("\nOverlap detected")

                master.iloc[start:end] = block.to_numpy()

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
        t_frames = len(total_exp_frames)
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
        
        #print("\n Preprocessing started...")
        filenames.append(os.path.basename(file))

        # an welchem Zeitpunkt der Experimenttdauer befindet sich diese File
        time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS

        # einlesen und copy anlegen
        if stitch_dataframes:
            df = master_df.copy()
            if habituation:
                df = df[0:54000]
        else:
            read_in_df = pd.read_hdf(file)
            df = read_in_df.copy()

        # überschriften des df auslesen
        scorer = df.columns.levels[0][0]
        individuals = df.columns.levels[1]
        bodyparts = df.columns.levels[2]

        # fragment size filter
        df, n_removedframes = filter_prediction_fragments(df, scorer, individuals, bodyparts)

        if n_removedframes > 0:
            warnings.warn(f"\nFragment filter removed {n_removedframes} out of a total of {len(df)} frames. Checking predictions is recommended.")

        # likelihood filter, um sehr unwahrscheinliche predictions rauszunehmen
        df = ma_likelihood_filter(df, scorer, individuals, bodyparts, filter_value=0.3)
        
        # kleinere fehlende Fragmente werden interpoliert, e.g. wenn Mäuse sich gegenseitig überdecken oder Keypoints fehlen
        df = interpolate_with_max_gap(df)

        # selten treten ID Overlays auf, DeepLabCut labelt die selbe Maus zwei Mal mit den exakt selben Koordinaten
        overlays_dic = find_id_overlay(df, scorer, individuals, bodyparts)
        if overlays_dic:
            
            # jedes overlay event wird einzeln behandelt
            for entry in overlays_dic:

                # die individuals die overlayen werden genommen
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
        print("\n Getting Mouse Center...")
        # jeweilige mouse center berechnen (shape n_ind, n_frames)
        all_centroid_x, all_centroid_y = mouse_center(df, scorer, individuals, bodyparts, min_bodyparts = math.ceil(len(bodyparts) / 3))
        
        for i in range(len(individuals)):
            for j, (x, y) in enumerate(zip( all_centroid_x[i], all_centroid_y[i])):
                #centers_xy[i][j] = (x, y*-1)
                centers_xy[i].append((x, y*-1))

        head_energy = []
        trunk_energy = []
        head_xs = []
        head_ys = []
        trunk_xs = []
        trunk_ys = []

        for i, ind in enumerate(individuals):
            head_coords = ['eye_left', 'eye_right', 'nose']
            trunk_coords = ['tail_base', 'dorsal_4']
            center_coords = ["dorsal_1", "dorsal_2", "dorsal_3", "lateral_left", "lateral_right"]
            head, trunk, head_x, head_y, trunk_x, trunk_y = grooming_energy(df = df,
                        scorer = scorer,
                        individual = ind,
                        head_coords=head_coords,
                        trunk_coords=trunk_coords
                        )
            head_energy.append(head)
            trunk_energy.append(trunk)
            head_xs.append(head_x)
            head_ys.append(head_y)
            trunk_xs.append(trunk_x)
            trunk_ys.append(trunk_y)
            
        plot_energy = False
        if plot_energy:
            ind_idx = 2
            slice = (0, 8900)
            xy = []
            for x, y in zip(trunk_xs[ind_idx], trunk_ys[ind_idx]):
                xy.append((x,y*-1))
            overlay_metric_at_centers(in_video_path=r"C:\Users\quicken\Code\energyhead.mp4",
                                    out_video_path=r"C:\Users\quicken\Code\energytrunk.mp4",
                                    centers_xy=xy,
                                            metric=trunk_energy[ind_idx][slice[0]:slice[1]],
                                            unit="",
                                            label="Energy",
                                            draw_center_marker=True,
                                            font_scale=0.9
                                            )

        # # # # #  mouse present analyse  # # # # # 
        print("\n Calculating time mice are present...")
        for index, ind in enumerate(individuals):

            # invidivuelle x-daten reichen
            center_x_data = all_centroid_x[index]
            center_y_data = all_centroid_y[index]

            # über finite koordinaten checken, ob die maus im modul ist
            ind_is_present = np.zeros(len(center_x_data)).astype(int)
            for idx, coord in enumerate(center_x_data):
                if np.isfinite(coord):
                    ind_is_present[idx] = 1

            # speichern der information im Kontext des gesamten Experiments
            for i in range(len(ind_is_present)):
                mice_in_module[index][i+(time_position_in_frames-1)] = ind_is_present[i]
                mice_center_x[index][i+(time_position_in_frames-1)] = center_x_data[i]
                mice_center_y[index][i+(time_position_in_frames-1)] = center_y_data[i]


        # # # # #  distance & speed analysis  # # # # # 
        cumdists = []
        dists = []
        immobiles = []
        accelerations = []
        accelerations_count_arrays = []
        
        for index, ind in enumerate(individuals):
            print(f"\n Getting all distance, speed and mobility for {ind}...")
            dist_values = distance_travelled_arraybased(x_arr=all_centroid_x[index],
                                                        y_arr=all_centroid_y[index]
                                                        )
            

            # smoothing
            dist_values = moving_average(data=dist_values,
                                        window=10
                                        )    

            # Werte unter dem Threshold werden auf 0 gesetzt und als "immobile" angesehen
            dist_values = remove_distance_jitter(dist_values=dist_values, thrsh=IMMOBILE_THRSH)

            a_px_frame, a_cm_s = acceleration(dist_values)
            for i in range (len(a_cm_s)):
                if not np.isnan(a_cm_s[i]):
                    mice_accelerations[index][i+(time_position_in_frames-1)] = a_cm_s[i]


            is_immobile = np.where(dist_values == 0, 1, 0)

            

            for i in range(len(is_immobile)):
                if not np.isnan(is_immobile[i]):
                    immobile_over_time[index][i+(time_position_in_frames-1)] = is_immobile[i]
            

            #print(f"\n Dist Traveled: {sum(dist_values)}")
            #print(f"\n Total immobile frames: {sum(is_immobile) / len(is_immobile)}")
            #print(len(is_immobile))
            #print(len(immobile_over_time[index]))
            #plt.plot(all_centroid_x[index][0:10])
            #plt.show()

            animated_plot = False

            if animated_plot:
                colors = ["purple", "green", "red"]
                color = colors[index]

                animate_trace(dist_values[0:1800],
                    fps=30,
                    window_seconds=5,
                    color=color,
                    save_path=path+f"/trace_animation_{ind}_speed.mp4"
                    )
                
                
            
            for i in range(len(dist_values)):
                if not np.isnan(dist_values[i]):
                    mice_distances[index][i+(time_position_in_frames-1)] = dist_values[i]

        
            cum_dist = np.nancumsum(dist_values)
            for i in range(len(dist_values)):
                cumdist_over_time[index][i+(time_position_in_frames-1)] = cum_dist[i]
            
            # für video generation
            immobiles.append(is_immobile)
            cumdists.append(cum_dist)
            dists.append(dist_values)
            accelerations.append(a_px_frame)
            all_accelerations += list(a_cm_s)
            ac, ac_array = acceleration_events(a=a_px_frame)
            accelerations_count_arrays.append(ac_array)

            acc_events_count += ac

            if stitch_dataframes:
                pass
                


        acc = False
        if acc:
            ind_idx = 1
            slice = (0, 8900)
            xy = []
            for x, y in zip(all_centroid_x[ind_idx], all_centroid_y[ind_idx]):
                xy.append((x,y*-1))
            overlay_metric_at_centers(in_video_path=r"C:\Users\quicken\Code\rollingplottest3.mp4",
                                            out_video_path=r"C:\Users\quicken\Code\acc_count2.mp4",
                                            centers_xy=xy,
                                            metric=accelerations_count_arrays[ind_idx][0][slice[0]:slice[1]],
                                            unit="",
                                            label="Events",
                                            draw_center_marker=False,
                                            font_scale=0.9,
                                            color_mask=accelerations_count_arrays[ind_idx][1][slice[0]:slice[1]]
                                            )

        rolling_plot = False
        if rolling_plot:
            ind_idx = 2
            slice = (0,8900)
            xy = []
            for x, y in zip(all_centroid_x[ind_idx], all_centroid_y[ind_idx]):
                xy.append((x,y*-1))
            overlay_rolling_plot_at_centers(
                in_video_path=r"C:\Users\quicken\Code\rollingplottest2.mp4",
                out_video_path=r"C:\Users\quicken\Code\rollingplottest3.mp4",
                centers_xy=xy[slice[0]:slice[1]],
                metric=accelerations[ind_idx][slice[0]:slice[1]],
                fps = 30,
                scale_mode="global",
                line_color_bgr=(0, 0, 255)

            )
        # # # # # arena center analyse # # # # # 

        for index, ind in enumerate(individuals):
            #print(f"\n Calculating center time for {ind}...")

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
        #print("\n Getting all trajectories...")
        # alle trajectories, auch von mäusen die bereits in der Kammer sind bei Videostart
        all_traj, traj_slices, start_len_t = get_all_traj(x_arrs = all_centroid_x, y_arrs=all_centroid_y, individuals=individuals)
        
        start_len_visits += start_len_t
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
        
        


        #print("\n Calculating thetas...")
        theta_list, theta_dic = theta_analysis(individuals=individuals,
                    front_x=front_center_x,
                    front_y=front_center_y,
                    rear_x=rear_center_x,
                    rear_y=rear_center_y,
                    slices=traj_slices)
        
        for i in range(len(individuals)):
            for j, (x, y) in enumerate(zip(front_center_x[i], front_center_y[i])):
                fronts_xy[i].append((x, y*-1))
            for j, (x, y) in enumerate(zip(rear_center_x[i], rear_center_y[i])):
                rears_xy[i].append((x, y*-1))

            thetas[i] += theta_list[i]

        xy1 = []
        xy2 = []


        for i in range(len(front_center_x[2])):
            xy1.append((front_center_x[2][i], front_center_y[2][i]*-1))
            xy2.append((rear_center_x[2][i], rear_center_y[2][i]*-1))
        theta_vid = False
        if theta_vid:
            overlay_two_points_line_and_theta_segments(in_video_path=r"C:\Users\quicken\Code\testfadeind2.avi",
                                            out_video_path=r"C:\Users\quicken\Code\testfadeind3.avi",
                                            xy1=xy1,
                                            xy2=xy2,
                                            theta_segments=theta_dic[individuals[2]],
                                            trail_len=10
                                            )
        
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




        

        print("\n Investigating social investigation...")
        print(os.path.basename(file))

        # nose coordinaten für videoplots speichern
        for i, ind in enumerate(individuals):
            nose_x = df.loc[:, (scorer, ind, "nose", "x")].to_numpy()
            nose_y = df.loc[:, (scorer, ind, "nose", "y")].to_numpy()
            for j, (x, y) in enumerate(zip(nose_x, nose_y)):
                nose_xy[i].append((x, y*-1))

        
        
        # social investigation analyse
        #social_inv = social_investigation(df, scorer, individuals, bodyparts)
        if social_inv:
            social_inv_details = detail_social_investigation(df, scorer, individuals, pixel_per_cm=PIXEL_PER_CM, max_dist_cm=2)
            # shape n_ind, n_frames
            face_inv = social_inv_details["individual_inv"]["face"]
            body_inv = social_inv_details["individual_inv"]["body"]
            anogenital_inv = social_inv_details["individual_inv"]["anogenital"]

            all_face = social_inv_details["counts_per_frame"]["face"]
            all_body = social_inv_details["counts_per_frame"]["body"]
            all_anogenital = social_inv_details["counts_per_frame"]["anogenital"]

            # hier die summen, um auch gleichzeitige investigation gleicher bodyparts zu finden
            sum_face_inv = social_inv_details["totals"]["face"]
            sum_body_inv = social_inv_details["totals"]["body"]
            sum_anogenital_inv = social_inv_details["totals"]["anogenital"]

            # eventindices sammeln
            
            if stitch_dataframes:
                for i, ind in enumerate(individuals):
                    start_end_faceing[ind] = social_inv_details["start_end_indices"]["face"][i]
                    print(ind)
                    print(social_inv_details["start_end_indices"]["face"][i])
                    start_end_bodyinv[ind] = social_inv_details["start_end_indices"]["body"][i]
                    start_end_anogenitalinv[ind] = social_inv_details["start_end_indices"]["anogenital"][i]
            

        # # # plots # # # 
        if stitch_dataframes and plot_heatmap:
            heatmap_x = np.concatenate([arr for arr in all_centroid_x])
            heatmap_y = np.concatenate([arr for arr in all_centroid_y])
            if habituation:
                savename = path + r"\heatmap_hab.svg"
            elif "top1" in path:
                savename = path + r"\heatmap_top1_exp.svg"
            else:
                savename = path + r"\heatmap_top2_exp.svg"
            heatmap_plot(x_values=heatmap_x, y_values=heatmap_y, plotname="heatmaptest", save_as=savename)


        
        




    # berechnen, ob mindestens eine Maus präsent ist 
    min_one_mouse_in_module = np.any(mice_in_module, axis=0).astype(int)
    # berechnen, wie viele mäuse pro frame im Bild sind
    mice_per_frame = np.nansum(mice_in_module, axis=0)
    # berechnen, wie viele mäuse pro frame immobile sind
    immobile_per_frame = np.nansum(immobile_over_time, axis=0)
    # distanz, die pro frame zurückgelegt wird (addiert mehrere Mäuse)
    distance_per_frame = np.nansum(mice_distances, axis=0)
    # kumulative distanz pro frame
    cumdist_per_frame = np.nancumsum(distance_per_frame)
    # full dist
    distance_in_px = cumdist_per_frame[-1]

    di = 0 
    for array in mice_distances:
        di += np.nansum(array)
    print(di)


    # berechnen, ob mindestens eine Maus im center ist
    min_one_mouse_in_center = mice_in_center.any(axis=0).astype(int)
    # berechnen, wieviele mäuse pro frame im Center sind
    mice_center_per_frame = mice_in_center.sum(axis=0)

    # normalisierte Werte auf Anwesenheit
    frames_with_mice = np.nansum(mice_per_frame)
    immobile_proportion = np.nansum(immobile_per_frame) / frames_with_mice
    in_center_proportion = np.nansum(mice_center_per_frame) / frames_with_mice
    distance_norm_px = distance_in_px / np.nansum(mice_per_frame)

    # social behavior werden auf frames mit mindestens 2 Mäusen normalisiert
    min_two_mice = (mice_per_frame > 1).astype(np.uint8)
    min_three_mice = (mice_per_frame > 2).astype(np.uint8)


    """
    plot_mice_presence_states(mice_in_module=mice_in_module)
    

    print("\nExperiment overview")
    print(f"Experiment duration: {len(exp_duration_frames)}")
    print(f"Mice per frame / total frames: {np.nansum(mice_per_frame) / len(exp_duration_frames)}")
    print(f"Amount with at least 1 mouse: {np.nansum(min_one_mouse_in_module) / len(exp_duration_frames)}")
    print(f"Amount with at least 2 mice: {np.nansum(min_two_mice) / len(exp_duration_frames)}")
    print(f"Amount with 3 mice: {np.nansum(min_three_mice) / len(exp_duration_frames)}")
    print(f"Immobile Amount: {immobile_proportion}")
    print(f"Center Amount: {in_center_proportion}")
    print(f"Distance / individual presence time in px: {distance_norm_px}")
    print(f"Total Face Inv: {social_inv_details["totals"]["face"] / np.nansum(min_two_mice)}")
    print(f"Total Body Inv: {social_inv_details["totals"]["body"] / np.nansum(min_two_mice)}")
    print(f"Total Anogenital Inv: {social_inv_details["totals"]["anogenital"] / np.nansum(min_two_mice)}")
    """
    def export_summary_metrics_to_excel(
        path: str,
        exp_duration_frames,
        mice_per_frame,
        min_one_mouse_in_module,
        min_two_mice,
        min_three_mice,
        immobile_proportion,
        in_center_proportion,
        distance_norm_px,
        social_inv_details,
        visits,
        filename: str = "Metriken.xlsx",
        sheet_name: str = "Metriken",
    ):
        """
        Exportiert Summary-Metriken in eine Excel-Datei.

        Speichert eine Tabelle mit genau 1 Zeile (ein Experiment) und klar benannten Spalten.

        Parameters
        ----------
        path : str
            Zielordner.
        exp_duration_frames : array-like
            Frames (oder irgendein Array, dessen Länge die Experimentdauer in Frames repräsentiert).
        mice_per_frame : array-like
            Anzahl Mäuse pro Frame (kann NaNs enthalten).
        min_one_mouse_in_module : array-like (0/1 oder bool)
        min_two_mice : array-like (0/1 oder bool)
        min_three_mice : array-like (0/1 oder bool)
        immobile_proportion : float
        in_center_proportion : float
        distance_norm_px : array-like oder float
            "Distance / individual presence time in px" (falls array, wird es als JSON-ähnlicher String gespeichert).
        social_inv_details : dict
            Erwartet Struktur: social_inv_details["totals"]["face"/"body"/"anogenital"].
        filename : str
            Excel-Dateiname.
        sheet_name : str
            Excel-Sheetname.

        Returns
        -------
        str
            Voller Speicherpfad zur Excel-Datei.
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        n_frames = int(len(exp_duration_frames)) if exp_duration_frames is not None else 0

        def safe_div(num, den):
            num = float(num) if np.isfinite(num) else np.nan
            den = float(den) if np.isfinite(den) else np.nan
            if den == 0 or not np.isfinite(den):
                return np.nan
            return num / den

        mice_per_frame_mean = safe_div(np.nansum(mice_per_frame), n_frames) if n_frames else np.nan
        p_min_one = safe_div(np.nansum(min_one_mouse_in_module), n_frames) if n_frames else np.nan
        p_min_two = safe_div(np.nansum(min_two_mice), n_frames) if n_frames else np.nan
        p_three = safe_div(np.nansum(min_three_mice), n_frames) if n_frames else np.nan

        n_frames_min_two = float(np.nansum(min_two_mice))  # Nenner für social inv normalization

        # Social investigations normalized by frames with >=2 mice 
        face_norm = safe_div(social_inv_details["totals"]["face"], n_frames_min_two)
        body_norm = safe_div(social_inv_details["totals"]["body"], n_frames_min_two)
        anogen_norm = safe_div(social_inv_details["totals"]["anogenital"], n_frames_min_two)
    

        # distance_norm_px kann float oder array sein → wir speichern beides sinnvoll
        if isinstance(distance_norm_px, (list, tuple, np.ndarray)):
            distance_norm_px_val = np.asarray(distance_norm_px)
            # falls 1 Wert -> float, sonst als String
            if distance_norm_px_val.size == 1:
                distance_norm_px_out = float(distance_norm_px_val.ravel()[0])
            else:
                distance_norm_px_out = distance_norm_px_val.tolist()
        else:
            distance_norm_px_out = float(distance_norm_px) if np.isfinite(distance_norm_px) else np.nan

        row = {
            "experiment_duration_frames": n_frames,
            "mice_per_frame_mean": mice_per_frame_mean,
            "p_frames_min_1_mouse": p_min_one,
            "p_frames_min_2_mice": p_min_two,
            "p_frames_3_mice": p_three,
            "immobile_proportion": float(immobile_proportion) if np.isfinite(immobile_proportion) else np.nan,
            "in_center_proportion": float(in_center_proportion) if np.isfinite(in_center_proportion) else np.nan,
            "distance_norm_px": distance_norm_px_out,
            "social_face_per_min2_frame": face_norm,
            "social_body_per_min2_frame": body_norm,
            "social_anogenital_per_min2_frame": anogen_norm,
            "n_frames_min_2_mice": n_frames_min_two,
            "visits": visits
        }

        df = pd.DataFrame([row])

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        return save_path
    """
    save_path = export_summary_metrics_to_excel(
        path=path,
        exp_duration_frames=exp_duration_frames,
        mice_per_frame=mice_per_frame,
        min_one_mouse_in_module=min_one_mouse_in_module,
        min_two_mice=min_two_mice,
        min_three_mice=min_three_mice,
        immobile_proportion=immobile_proportion,
        in_center_proportion=in_center_proportion,
        distance_norm_px=distance_norm_px,
        social_inv_details=social_inv_details,
        visits=sum_visits
    )
    """
    #print("Saved:", save_path)
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
                        metric1=social_inv_details["presence_per_frame"]["body"],
                        text1="Body Investigation",
                        metric2=social_inv_details["presence_per_frame"]["anogenital"],
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

    return {"individuals": individuals,
            "exp_len": len(exp_duration_frames),
            "centers_xy": centers_xy,
            "fronts_xy": fronts_xy,
            "rears_xy": rears_xy,
            "nose_xy": nose_xy,
            "mice_per_frame": mice_per_frame,
            "mice_presence": mice_in_module,
            "immobile_per_frame": immobile_per_frame,
            "mice_immobile": immobile_over_time,
            "mice_distances": mice_distances,
            "mice_cumdists": cumdist_over_time,
            "cumdist": cumdist_per_frame,
            "mice_in_center": mice_in_center,
            "center_per_frame": mice_center_per_frame,
            "thetas": thetas,
            "theta_dic": theta_dic,
            "visits": start_len_visits,
            "accelerations": all_accelerations,
            "face_inv": face_inv,
            "body_inv": body_inv,
            "anogenital_inv": anogenital_inv,
            "face_inv_idx": start_end_faceing,
            "all_face": all_face,
            "all_body": all_body,
            "all_anogenital": all_anogenital
    }

# center crossings über alle Mäuse zählen
# mean visit time
# visits per hour
# strecke über zeit = mean geschwindigkeit, sollte auf eine maus runtergerechnet werden (zwei mäuse = doppelt so viel strecke pro zeit, also Summe des 2 Maus arrays *2 und summe des 3 Maus arrays * 3)


# speichern als h5

# plots generieren



# # # # TESTING # # # # 
metrics = multi_animal_main(path=r'C:\Users\quicken\Code\Ambros_analysis\code_test\soial_behavior',
                  habituation=False,
                  social_inv=True)

individuals = metrics["individuals"]
exp_len = metrics["exp_len"]

for ind in metrics["individuals"]:
    print(metrics["face_inv_idx"][ind])

create =  False
if create:
    from create_labelled_video import create_labelled_video_modular
    create_labelled_video_modular(video_path=r'C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4',
                            output_path=r'C:\Users\quicken\Code\social_behavior_analysistest\socialbehavior.mp4',
                            metrics=[
                                ("Face Investigation:", metrics["face_inv"][0]),
                                ("Body Investigation:", metrics["body_inv"][0]),
                                ("Anogenital Investigation:", metrics["anogenital_inv"][0])
                            ],
                            row_gap=20,
                            scale_factor=1.0
    )
create_center = True
if create_center:
    shrinked_coords = shrink_rectangle(ARENA_COORDS, scale=0.6)
    arena_coords = []
    for (x, y) in shrinked_coords:
        arena_coords.append((x, y*-1))
    overlay_metric_at_centers(in_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4",
                            out_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\faceinvoverlay.mp4",
                            centers_xy = metrics["centers_xy"][0],
                            metric = [np.nancumsum(metrics["face_inv"][0]) / np.nancumsum(metrics["mice_presence"][0]) *100],
                            color_mask = metrics["face_inv"][0],
                            unit=["% ind1 faceinv"],
                            marker_radius=8,
                            marker_color_bgr=(245, 66, 194),
                            font_scale=0.8,
                            draw_rect=True,
                            rect_xy = arena_coords,
                            draw_circle=True,
                            circle_xy=metrics["nose_xy"][0],
                            circle_radius=PIXEL_PER_CM*2.0,
                            circle_fill_bgr=(245, 66, 194),
                            circle_outline_bgr=(245, 66, 194)
                            )

thetaplot = False
if thetaplot:
    
    overlay_two_points_line_and_theta_segments(in_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4",
                                                out_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\thetas.mp4",
                                                xy1=metrics["fronts_xy"][0],
                                                xy2=metrics["rears_xy"][0],
                                                theta_segments=metrics["theta_dic"][individuals[0]],
                                                trail_len=10
                                                )
    
