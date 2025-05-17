import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import h5py

from config import FPS, PIXEL_PER_CM, LIKELIHOOD_THRESHOLD, DF_COLS
from utils import euklidean_distance, fill_missing_values, time_to_seconds, convert_videostart_to_experiment_length, calculate_experiment_length
from metrics import distance_bodypart_object, distance_travelled, investigation_time, get_food_coordinates
from plotting import heatmap_plot, cumsum_plot
from metadata import module_is_stimulus_side
from preprocessing import transform_dlcdata
from h5_handling import save_modulevariables_to_h5, load_modulevariables_from_h5

@dataclass
class ModuleVariables:
    exp_duration_frames: int
    strecke_über_zeit: np.array
    maus_in_modul_über_zeit: np.array
    maus_an_food_percent: float
    strecke_pixel_frame: float
    visits_per_hour: float
    mean_visit_time: float
    zeit_in_modul_prozent: float
    nose_coords_x_y: tuple
    date: str
    is_stimulus_module: bool
    start_time: str
    end_time: str
    mouse: str
    modulnumber: int


def analyze_one_module(path, bodyparts_to_extract = ["nose", "centroid", "food1"]):
    
    # csv's einlesen und nach uhrzeit(name) sortieren
    file_list = glob.glob(os.path.join(path, '*.csv'))
    file_list.sort()

    # experimentdauer in frames
    exp_duration_frames, startzeit, endzeit, date = calculate_experiment_length(first_file=file_list[0], last_file=file_list[-1])

    # speichert als boolean ob das modul einen stimulus beinhaltet
    is_stimulus_side, mouse, modulnumber = module_is_stimulus_side(file_list[0])

    #variablen of interest einführen
    maus_in_modul_über_zeit = exp_duration_frames.copy()
    maus_an_snicket_über_zeit = exp_duration_frames.copy()
    strecke_über_zeit = exp_duration_frames.copy()
    nose_x_values_over_time = exp_duration_frames.copy()
    nose_y_values_over_time = exp_duration_frames.copy()

    maus_in_modul_in_frames = 0
    maus_am_snicket_in_frames = 0
    strecke_in_pixeln = 0
    maus_an_food = 0

    #visits in module
    all_visits = []
    num_visits = 0

    # leere food data preparen und mit nans füllen, damit später fehlende food predictions interpoliert werden können
    food_x_values_over_time = exp_duration_frames.copy()
    food_x_values_over_time[food_x_values_over_time == 0] = np.nan
    food_y_values_over_time = exp_duration_frames.copy()
    food_y_values_over_time[food_y_values_over_time == 0] = np.nan


    for file in tqdm(file_list):
        
        # dataframe erstellen und transformieren
        bodypart_df = transform_dlcdata(file, bodyparts_to_extract, DF_COLS)


        """
        
        Mouse Present Analyse
        
        """
        
        #berechnen ob maus present: insgesamt und over time
        mouse_present_calculation_df = bodypart_df.copy()

        mouse_snout_likelihood_arr = mouse_present_calculation_df["nose_likelihood"]
        
        mouse_present_arr = np.zeros(len(mouse_snout_likelihood_arr))

        # wenn die schnauze während dem video mindestens einmal getracked wurde, war die maus während dem Video im Modul
        if max(mouse_snout_likelihood_arr) > 0.95:
            num_visits +=1
    
        # markiert jeden frame, in dem die Maus vermutlich anwesend war
        for i in range(len(mouse_present_arr)):
            if mouse_snout_likelihood_arr.iloc[i] > 0.3:
                mouse_present_arr[i] = 1


        # addiert zu der gesamtzeit, die Maus im Modul verbracht hat über alle Videos
        maus_in_modul_in_frames += np.nansum(mouse_present_arr) 
        # Länge des Visits in diesem Video gespeichert, um später mean visit length zu berechnen
        all_visits.append(np.nansum(mouse_present_arr))

        # an welchem Zeitpunkt der Experimenttdauer befindet sich dieses Video
        time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * FPS
        
        # mouse present data von diesem Video in Kontext der gesamten Experimentdauer einsetzen
        for i in range(len(mouse_present_arr)):
            maus_in_modul_über_zeit[i+(time_position_in_frames-1)] = mouse_present_arr[i]


    
        """

         #berechnen ob maus nah am snicket (faktor = 1 = 34.77 pixel) (wieder insgesamt und over time)
        distance_mouse_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose", object = "snicket")

        mouse_is_investigating, factor = investigation_time(distance_values=distance_mouse_nose_snicket, factor=3)


        maus_am_snicket_in_frames += np.nansum(mouse_is_investigating)

        for i in range(len(mouse_is_investigating)):
            maus_an_snicket_über_zeit[i+(time_position_in_frames-1)] = mouse_is_investigating[i]

        """

        """
        
        Distance Travelled Analyse

        """

        #zurückgelegte Strecke berechnen (in pixeln)
        maus_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid")

        strecke_in_pixeln += np.nansum(maus_distance_travelled)

        for i in range(len(maus_distance_travelled)):
            strecke_über_zeit[i+(time_position_in_frames-1)] = maus_distance_travelled[i]

        """
        
        Food Interaktion Analyse

        """


        # Koordinaten für die Heatmap extrahieren und speichern:
        for i in range(len(bodypart_df["nose_x"])):
            nose_x_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_x"].iloc[i]
        
            nose_y_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_y"].iloc[i]

        
        food_x, food_y = get_food_coordinates(df = bodypart_df.copy(), food_likelihood_row = 'food1_likelihood')


        for i in range(len(food_x)):
            food_x_values_over_time[i+(time_position_in_frames-1)] = food_x[i]
        
            food_y_values_over_time[i+(time_position_in_frames-1)] = food_y[i]

        # distanz zwischen nose und food berechnen um food interaction zu bestimmen
        for i in range(len(food_x)-1):
                distance = euklidean_distance(x1=bodypart_df["nose_x"].iloc[i],
                                                        y1=bodypart_df["nose_y"].iloc[i],
                                                        x2=food_x[i],
                                                        y2=food_y[i])
                # sollte etwa 1 cm entsprechen, später anpassen!!!!!
                if distance <= 35:
                    maus_an_food +=1
                # !!!!!!!!!!!!!!!!


    # food koordinaten interpolieren

    #food_x_values_over_time = fill_missing_values(food_x_values_over_time)
    #food_y_values_over_time = fill_missing_values(food_y_values_over_time)
    """
    # food koordinaten plotten zur kontrolle
    plt.figure()
    plt.plot(food_x_values_over_time)
    plt.plot(food_y_values_over_time)
    plt.show()
    """

    # wie lange ist ein visit im schnitt
    mean_visit_time = np.mean(all_visits) / FPS

    # food interaktion zählen
    maus_an_food_percent = maus_an_food/len(exp_duration_frames)*100

    # das hier wäre die Strecke über die Zeit
    strecke_pixel_frame = strecke_in_pixeln/sum(maus_in_modul_über_zeit)
    visits_per_hour = num_visits / (len(exp_duration_frames)/30/3600)

    zeit_in_modul_prozent = sum(maus_in_modul_über_zeit) / len(exp_duration_frames) * 100

    nose_coords = (nose_x_values_over_time, nose_y_values_over_time)


    module_vars = ModuleVariables(
        exp_duration_frames = len(exp_duration_frames),
        strecke_über_zeit = strecke_über_zeit,
        maus_in_modul_über_zeit = maus_in_modul_über_zeit,
        maus_an_food_percent = maus_an_food_percent,
        strecke_pixel_frame = strecke_pixel_frame,
        visits_per_hour = visits_per_hour,
        mean_visit_time = mean_visit_time,
        zeit_in_modul_prozent = zeit_in_modul_prozent,
        nose_coords_x_y = nose_coords,
        date = date,
        is_stimulus_module = is_stimulus_side,
        start_time = startzeit,
        end_time = endzeit,
        mouse = mouse,
        modulnumber = modulnumber
    )

    return module_vars


project_path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/female_mice_male_stimuli/"

mouse = "mouse_15"

dates = ["2025_04_22", "2025_04_23", "2025_04_24", "2025_04_25"]


for date in dates:
    Modul1Variables = analyze_one_module(path=f"{project_path}{mouse}/{date}/top1/")
    Modul2Variables = analyze_one_module(path=f"{project_path}{mouse}/{date}/top2/")

    name_modul1_h5 = f"{Modul1Variables.date}_top1_{('stimulus' if Modul1Variables.is_stimulus_module else 'control')}.h5"
    name_modul2_h5 = f"{Modul2Variables.date}_top2_{('stimulus' if Modul2Variables.is_stimulus_module else 'control')}.h5"


    save_modulevariables_to_h5(file_path=f"{project_path}{mouse}/{date}/{name_modul1_h5}",
                            data = Modul1Variables)
    save_modulevariables_to_h5(file_path=f"{project_path}{mouse}/{date}/{name_modul2_h5}",
                            data = Modul2Variables)





#save_modulevariables_to_h5(file_path=f"{experiment_day_path}test.h5",
#                            data = Modul1Variables)





"""
cumsum_plot(data_list=[modul1_maus_an_snicket_über_zeit,modul2_maus_an_snicket_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Maus an Snicket",
            x_label = "Experimentdauer in Frames",
            y_label= "Maus am Snicket in Frames",
            save_as= f"{experiment_day_path}maus_an_snicket.svg"
            )


cumsum_plot(data_list=[Modul1Variables.maus_in_modul_über_zeit,Modul2Variables.maus_in_modul_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Maus in Modul",
            x_label = "Experimentdauer in Frames",
            y_label= "Maus in Modul in Frames",
            save_as= f"{experiment_day_path}maus_in_modul.svg"
            )

cumsum_plot(data_list=[Modul1Variables.strecke_über_zeit,Modul2Variables.strecke_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Zurückgelegte Strecke pro Modul",
            x_label = "Experimentdauer in Frames",
            y_label= "Strecke in Pixeln",
            save_as= f"{experiment_day_path}maus_strecke.svg"
            )



heatmap_plot(x_values=Modul1Variables.nose_coords_x_y[0], y_values=Modul1Variables.nose_coords_x_y[1], plotname="Heatmap Modul 1", save_as=f"{experiment_day_path}heatmap_modul1.svg", num_bins=12)


heatmap_plot(x_values=Modul2Variables.nose_coords_x_y[0], y_values=Modul2Variables.nose_coords_x_y[1], plotname="Heatmap Modul 2", save_as=f"{experiment_day_path}heatmap_modul2.svg", num_bins=12)
"""
"""

deg_file_path = "E:/Fabi_Setup/In_Soundchamber/behaviors_urine_validation_deepethogram/DATA/2025_03_10_mouse_7_habituation_side1_40357253_stitched/2025_03_10_mouse_7_habituation_side1_40357253_stitched_predictions.csv"

deg_behaviors = ['rearing"', "drinking", "grooming"]

for behavior in deg_behaviors:
    bouts, sum = analyze_deg_file(deg_file_path, behavior)
    print(bouts)
    print(sum)
"""


"""

# # # Göttingen Stuff # # #

folder = "D:/Uni Transfer/Göttingen NWG 2025/poster_data/hab/"

deg_path = "D:/Uni Transfer/Göttingen NWG 2025/poster_data/deg_hab_data/"

mice = ["mouse7", "mouse75", "mouse7b", "mouse75b"]




def create_cumsum_plot_for_nwg():

    cumsum_data_modul1 = []
    cumsum_data_modul2 = []

    dlc_behavior_data_modul1 = []
    dlc_behavior_data_modul2 = []

    for mouse in mice:

        modul1_maus_an_snicket_über_zeit, modul1_maus_in_modul_über_zeit, modul1_strecke_über_zeit, modul1_nose_coords, modul1_dlc_data = analyze_one_module(path=f"{folder}{mouse}/top1/")
        modul2_maus_an_snicket_über_zeit, modul2_maus_in_modul_über_zeit, modul2_strecke_über_zeit, modul2_nose_coords, modul2_dlc_data = analyze_one_module(path=f"{folder}{mouse}/top2/")

        print(modul1_dlc_data)
        print(modul2_dlc_data)

        cumsum_data_modul1.append(modul1_maus_in_modul_über_zeit)
        cumsum_data_modul2.append(modul2_maus_in_modul_über_zeit)

        dlc_behavior_data_modul1.append(modul1_dlc_data)
        dlc_behavior_data_modul2.append(modul2_dlc_data)

        #heatmap_dual_plot(x1 = modul1_nose_coords[0], y1 = modul1_nose_coords[1], x2 = modul2_nose_coords[0], y2 = modul2_nose_coords[1], plotname = f"Heatmap {mouse}", save_as = f"{folder}{mouse}_heatmap", num_bins=12)


    # cut arrays to shortest length (all experiments should be roughly the same length anyway)
    min_length_modul1 = min(map(len, cumsum_data_modul1))
    min_length_modul2 = min(map(len, cumsum_data_modul2))
    overall_min = min(min_length_modul1, min_length_modul2)

    cumsum_data_modul1 = [arr[:overall_min] for arr in cumsum_data_modul1]
    cumsum_data_modul2 = [arr[:overall_min] for arr in cumsum_data_modul2]

    #cumsum_plot_nwg(data_module1=cumsum_data_modul1, data_module2=cumsum_data_modul2, savename=f"{folder}cumsum")

    cumsum_plot_average_nwg(data_module1=cumsum_data_modul1, data_module2=cumsum_data_modul2, savename=f"{folder}cumsum_step")



    dlc_barplot_nwg(data_module1=dlc_behavior_data_modul1, data_module2=dlc_behavior_data_modul2, savename=f"{folder}dlc_behaviors")




#create_cumsum_plot_for_nwg()

def create_deg_barplot_for_nwg(exp_day):

    behavior_data_modul1 = []
    behavior_data_modul2 = []

    for mouse in mice:

        # nimmt csv für eine Maus aus dem jeweigen experimenttag ordner
        path_stim = deg_path + mouse + "/" + "side1" + "/"
        files_stim = glob.glob(os.path.join(path_stim, '*.csv'))

        path_con = deg_path + mouse + "/" + "side2" + "/"
        files_con = glob.glob(os.path.join(path_con, '*.csv'))
        
        
        # geht über jede csv im ordner
        for file in files_stim:
            
            # öffnet die csv als dataframe
            df = pd.read_csv(file)
            working_df = df.copy()

            # get behavior data normalized to experiment length
            rearing = np.sum(working_df['rearing"']) / len(working_df['rearing"']) * 100
            drinking = np.sum(working_df['drinking']) / len(working_df['rearing"']) * 100
            grooming = np.sum(working_df['grooming']) / len(working_df['rearing"']) * 100

            data = [rearing, drinking, grooming]
            behavior_data_modul1.append(data)

        for file in files_con:
            
            # öffnet die csv als dataframe
            df = pd.read_csv(file)
            working_df = df.copy()

            # get behavior data normalized to experiment length
            rearing = np.sum(working_df['rearing"']) / len(working_df['rearing"']) * 100
            drinking = np.sum(working_df['drinking']) / len(working_df['rearing"']) * 100
            grooming = np.sum(working_df['grooming']) / len(working_df['rearing"']) * 100

            data = [rearing, drinking, grooming]
            behavior_data_modul2.append(data)

            
            
            # nach modulen trennen
            if 'side1' in file:

                data = [rearing, drinking, grooming]
                behavior_data_modul1.append(data)


            if 'side2' in file:

                data = [rearing, drinking, grooming]
                behavior_data_modul2.append(data)
                
            

            # nach stimulus trennen
            if 'stimulus' in file:

                data = [rearing, drinking, grooming]
                behavior_data_modul1.append(data)

            else:

                data = [rearing, drinking, grooming]
                behavior_data_modul2.append(data)
           

    print(behavior_data_modul1)
    print(behavior_data_modul2)

    deg_barplot_nwg(data_module1=behavior_data_modul1, data_module2=behavior_data_modul2, savename=f"{deg_path}{exp_day}_deg")

create_deg_barplot_for_nwg(exp_day="hab")


cumsum_data_modul1 = np.array(cumsum_data_modul1)
cumsum_data_modul2 = np.array(cumsum_data_modul2)



cumsum_plot_nwg(data_module1=cumsum_data_modul1, data_module2=cumsum_data_modul2, savename=f"{folder}cumsum.svg")


# # # create ethograms for example behaviors


# Load the data
file_path = "D:/Uni Transfer/Göttingen NWG 2025/poster_data/stiched_evaluation_video/stiched_evaluation_video_predictions.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = [col.strip().replace('"', '') for col in df.columns]

behavior_name = "drinking"

# Auswahl der Events
behavior = df[f"{behavior_name}"].iloc[15350:15550]

# Event-Indizes
event_indices = np.where(behavior == 1)[0]

# Plot
fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')

ax.eventplot(event_indices, orientation='horizontal', colors='white',
             lineoffsets=0, linelengths=0.5)

ax.set_xlim(0, len(behavior))
ax.set_xlabel('frame', color='white')
ax.set_title(f'{behavior_name}', color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_yticks([])
ax.spines['bottom'].set_color('white')
plt.savefig(f"D:/Uni Transfer/Göttingen NWG 2025/poster_data/stiched_evaluation_video/{behavior_name}.svg", format='svg', facecolor=fig.get_facecolor())

plt.tight_layout()
plt.show()
"""