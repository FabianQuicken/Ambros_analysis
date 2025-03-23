import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from df_columns import df_cols
from plots_for_nwg import cumsum_plot_nwg, deg_barplot_nwg, dlc_barplot_nwg, heatmap_dual_plot, cumsum_plot_average_nwg


def cumsum_plot(data_list=list, labels=list, colors=list, plotname=str, x_label=str, y_label=str, save_as=str):
    # Kumulative Summe berechnen

    plotdata = []
    for data in data_list:
        cumulative_sum = np.nancumsum(data)
        plotdata.append(cumulative_sum)



    # Zeitachse erzeugen, basierend auf der Länge des längeren Arrays
    max_length = 0
    for data in data_list:
        if len(data) > max_length:
            max_length = len(data)
    time = np.arange(max_length)

    # Arrays auf gleiche Länge bringen (auffüllen mit dem letzten Wert)
    for i in range(len(plotdata)):
        if len(plotdata[i]) < max_length:
            plotdata[i] = np.pad(plotdata[i], (0, max_length - len(plotdata[i])), 'edge')

    # Plotten
    plt.figure(figsize=(10, 6))

    for i in range(len(plotdata)):
        
        plt.plot(time, plotdata[i], label=labels[i], linewidth=2, color=colors[i])

    plt.title(plotname)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_as, format='svg')
    plt.show()




def heatmap_plot(x_values = np.array, y_values = np.array, plotname = str, save_as = str, num_bins = 35, cmap = 'hot', plot_time_frame_hours = (None, None)):
    """
    This function plots a heatmap of x and y coordinates, e.g. of the snout. Pass the coordinates of a complete experiment, and they get filtered for all
    values that are not "0". Plotname and savepath need to be provided. Binsize is 50 per default. Colormap is "hot" per default.
    """

    if plot_time_frame_hours[1]:
        plot_time_frame_frames = (round(plot_time_frame_hours[0]*108000), round(plot_time_frame_hours[1]*108000))
        x_values = x_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]
        y_values = y_values[plot_time_frame_frames[0]:plot_time_frame_frames[1]]

    #Filter out (x, y) pairs where either is 0
    mask = (x_values != 0) & (y_values != 0)
    heatmap_x = x_values[mask]
    heatmap_y = y_values[mask]


    # get max x value and max y value to scale the heatmap
    x_max = max(heatmap_x)
    y_max = min(heatmap_y)

    y_max = round(y_max *-1)



    # calculate number of y-bins based on ratio between x and y axis
    y_bins = round((y_max / x_max) * num_bins)
    bins = (num_bins, y_bins)


    # create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(heatmap_x, heatmap_y, bins=bins)

    plt.figure(figsize=(8,6))
    #sns.heatmap(heatmap.T, cmap=cmap, square=True, cbar=True, xticklabels=True, yticklabels=True)
    
    plt.imshow(heatmap.T, origin='lower', cmap=cmap,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect=1)  # Set aspect ratio 
    
    plt.colorbar(label='Frames')
    plt.title(plotname)
    plt.savefig(save_as, format='svg')
    plt.show()


def analyze_deg_file(file_path, behavior = str):

    
    """
    
    Nimmt eine .csv file von DeepEthogram und wertet ein angegebenes Behavior aus.
    Returned einen array der die längen aller behavior bouts der datei enthält. [20, 50, 30, 23, ...]
    
    """

    df = pd.read_csv(file_path)
    # copy um ursprüngliche file nicht zu verändern
    working_df = df.copy()
    
    try:
        data = working_df[behavior]
    except:
        print("Behavior not found. Did you name it correctly?")
        raise NameError

    behavior_data = []
    counter = 0
    for i in range(len(data)):
        
        
        # count bout length and append bout to behavior data, when ending
        if data[i] == 1:
            counter += 1

        if data[i] == 1 and data[i+1] == 0:
            counter += 1
            behavior_data.append(counter)
            counter = 0

    return np.array(behavior_data), np.sum(data)/len(data)
        
        



def analyze_one_module(path):

    def likelihood_filtering_nans(df, likelihood_row_name=str, filter_val=0.95):
        """
        DeepLabCut provides a likelihood for the prediction of 
        each bodypart in each frame to be correct. Filtering predictions
        for the likelihood, replaces values in the entire row with NaN where the likelihood is below the filter_val.
        """
        df_filtered = df.copy()  # Make a copy to avoid modifying the original DataFrame
        filtered_rows = df_filtered[likelihood_row_name] < filter_val
        df_filtered.loc[filtered_rows] = np.nan
        num_replaced = filtered_rows.sum()
        #print(f"The filter replaced values in {num_replaced} rows with NaN out of a total of {len(df)} rows.")
        return df_filtered

    def likelihood_filtering(df, likelihood_row_name=str, filter_val = 0.95):
        """
        DeepLabCut provides a likelihood for the prediction of 
        each bodypart in each frame to be correct. Filtering predictions
        for the likelihood, reduces false predictions in the dataset.
        """
        df_filtered = df.copy()
        df_filtered = df[df[likelihood_row_name] > filter_val]
        df_removed_rows = df[df[likelihood_row_name] < filter_val]
        #print(f"The filter removed {len(df_removed_rows)} rows of a total of {len(df)} rows.")
        return df_filtered

    def euklidean_distance(x1, y1, x2, y2):
        """
        This func returns the euklidean distance between two points.
        (x1, y1) and (x2, y2) are the cartesian coordinates of the points.
        """
        if np.isnan(x1):
            distance = np.nan
        elif np.isnan(x2):
            distance = np.nan
        else:
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        return distance
    
    def fill_missing_values(array):
        """
        Replacing np.nans with a linear interpolation. Takes and returns an array.
        """
        nan_indices = np.isnan(array)
        array[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), array[~nan_indices])
        return array

    def distance_bodypart_object(df, bodypart=str, object=str, filter_object = True):
        """
        Takes a Dataframe, a bodypart and an object as strings,
        to calculate the distance between both.
        Note: Df gets likelihood filtered for bodypart first.
        Object should not move during recording, since
        the first good prediction will be set to the object location.
        """
        data = df.copy()
        #print(f"\nGet distance {bodypart} to {object}...")
        #print(f"Filtering {bodypart} for object distance calculation...")
        data = likelihood_filtering_nans(df=data, 
                                    likelihood_row_name=bodypart+"_likelihood")
        bodypart_x = data[bodypart+"_x"]
        bodypart_y = data[bodypart+"_y"]
        
        data = df.copy()

        #print("Filtering for a good object prediction...")
        data = likelihood_filtering(df=data, 
                                    likelihood_row_name=object+"_likelihood",
                                    filter_val=0.95)
        object_x = data[object+"_x"]
        object_y = data[object+"_y"]
        

        bodypart_x = np.array(bodypart_x)
        bodypart_y = np.array(bodypart_y)
        object_x = np.array(object_x)
        object_y = np.array(object_y)

        if np.nansum(object_x) > 0:

            object_x = object_x[0]
            object_y = object_y[0]


            distance_values = np.zeros((len(bodypart_x)))
            for i in range(len(bodypart_x)-1):
                distance_values[i] = euklidean_distance(x1=bodypart_x[i],
                                                        y1=bodypart_y[i],
                                                        x2=object_x,
                                                        y2=object_y)
            return distance_values
        
        else:
            return np.zeros((len(bodypart_x)))

    def investigation_time(distance_values, factor = 1):
        distance_values = distance_values.copy()
        pixel_per_cm = 34.77406
        radius_threshold = factor * pixel_per_cm
        is_investigating = np.zeros((len(distance_values)))
        for i in range(len(distance_values)-1):
            if distance_values[i] < radius_threshold:
                is_investigating[i] = 1
            elif np.isnan(distance_values[i]):
                is_investigating[i] = np.nan
        return is_investigating, factor

    def distance_travelled(df,bodypart=str):
        """
        Takes a Dataframe and a bodypart as input
        calculates the distance of a keypoint
        between consequetive frames in m.
        Note: Likelihood filtering gets applied for the bodypart.
        """
        pixel_per_cm = 34.77
        data = df.copy()
        #print("\nGet distance values...")
        #print(f"Filtering {bodypart} for distance calculation...")
        data = likelihood_filtering_nans(df=data, 
                                    likelihood_row_name=bodypart+"_likelihood")
        
        bodypart_x = data[bodypart+"_x"]
        bodypart_y = data[bodypart+"_y"]

        bodypart_x = np.array(bodypart_x) #transforms bodypart data into np array for easier calculation
        bodypart_y = np.array(bodypart_y)

        distance_values = np.zeros((len(bodypart_x)))
        for i in range(len(bodypart_x)-1):
            distance_values[i] = euklidean_distance(x1=bodypart_x[i],
                                                    y1=bodypart_y[i],
                                                    x2=bodypart_x[i+1],
                                                    y2=bodypart_y[i+1])
            
            #distance_values[i] = distance_values[i] / (pixel_per_cm*100) # umrechnung in meter
        return distance_values

    # Wandle Zeit in Sekunden seit Mitternacht um
    def time_to_seconds(time_str):
        hours, minutes, seconds = map(int, time_str.split("_"))
        return hours * 3600 + minutes * 60 + seconds

    def convert_videostart_to_experiment_length(first_file, filename):
        if not filename:
            return 0

        # Nur den Dateinamen ohne Path nehmen
        first_file = os.path.splitext(os.path.basename(first_file))[0]
        this_file = os.path.splitext(os.path.basename(filename))[0]

        # Extrahiere Zeit aus den Dateinamen
        first_time = first_file[11:19]  # Erwarte Format HH_MM_SS
        this_time = this_file[11:19]

        start_seconds = time_to_seconds(first_time)
        current_seconds = time_to_seconds(this_time)

        # Berechne die Differenz in Sekunden
        return current_seconds - start_seconds


    # csv's einlesen und nach uhrzeit(name) sortieren
    #path = "E:/Fabi_Setup/single_animal/analyse_test/"
    path = path
    file_list = glob.glob(os.path.join(path, '*.csv'))
    file_list.sort()

    
    # # # erstmal experimentdauer berechnen # # #
    experiment_dauer_in_s = 0

    name_first_file = file_list[0]
    name_first_file = os.path.splitext(os.path.basename(name_first_file))[0]
    name_last_file = file_list[-1]
    name_last_file = os.path.splitext(os.path.basename(name_last_file))[0]

    df_last_file = pd.read_csv(file_list[-1])

    #Zeit immer an selber stelle
    startzeit = name_first_file[11:19] #platzhalter Zahlen
    endzeit = name_last_file[11:19]


    start_in_s = time_to_seconds(startzeit)
    ende_in_s = time_to_seconds(endzeit)

    experiment_dauer_in_s = ende_in_s - start_in_s 

    # gesamte experimentdauer in frames
    fps = 30
    exp_duration_frames = np.zeros(experiment_dauer_in_s * fps + len(df_last_file)) 


    #variablen of interest einführen
    maus_in_modul_über_zeit = exp_duration_frames.copy()
    maus_an_snicket_über_zeit = exp_duration_frames.copy()
    strecke_über_zeit = exp_duration_frames.copy()

    maus_in_modul_in_frames = 0
    maus_am_snicket_in_frames = 0
    strecke_in_pixeln = 0
    maus_an_food = 0

    #visits in module
    num_visits = 0

    nose_x_values_over_time = exp_duration_frames.copy()
    nose_y_values_over_time = exp_duration_frames.copy()

    # leere food data preparen und mit nans füllen, damit später fehlende food predictions interpoliert werden können
    food_x_values_over_time = exp_duration_frames.copy()
    food_x_values_over_time[food_x_values_over_time == 0] = np.nan
    food_y_values_over_time = exp_duration_frames.copy()
    food_y_values_over_time[food_y_values_over_time == 0] = np.nan


    for file in tqdm(file_list):
        
        # dataframe erstellen und schneiden

        """
        df-cols richtig wählen!!!!
        
        """
        df = pd.read_csv(file, names=df_cols)
        data = df.copy()
        data = data.iloc[3:]
        data = data.astype(float)

        #dataframe kopieren & bodyparts of interest extrahieren
        empty_df = pd.DataFrame()
        bodypart_df = empty_df.copy()
        bodyparts_to_extract = ["nose", "centroid", "food1"]

        #print(data['nose_x'])

        for bodypart in bodyparts_to_extract:
            bodypart_df[bodypart+"_x"] = data[bodypart+"_x"]
            bodypart_df[bodypart+"_y"] = data[bodypart+"_y"]*(-1)  # y invertieren da DLC y koordinaten aufsteigen
            bodypart_df[bodypart+"_likelihood"] = data[bodypart+"_likelihood"]


        #berechnen ob maus present: insgesamt und over time
        mouse_present_calculation_df = bodypart_df.copy()

        mouse_snout_likelihood_arr = mouse_present_calculation_df["nose_likelihood"]
        
        mouse_present_arr = np.zeros(len(mouse_snout_likelihood_arr))

        # if snout is detected with high likelihood, a mouse was present in the module
        if max(mouse_snout_likelihood_arr) > 0.95:
            num_visits +=1
    

        for i in range(len(mouse_present_arr)):
            # check um anzahl der modul visits zu zählen
            
            if mouse_snout_likelihood_arr.iloc[i] > 0.3:
                mouse_present_arr[i] = 1


        #insgesamt
        maus_in_modul_in_frames += np.nansum(mouse_present_arr) 

        #over time
        time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * fps
        
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

        #zurückgelegte Strecke berechnen (in pixeln)
        maus_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid")

        strecke_in_pixeln += np.nansum(maus_distance_travelled)

        for i in range(len(maus_distance_travelled)):
            strecke_über_zeit[i+(time_position_in_frames-1)] = maus_distance_travelled[i]


        # Koordinaten für die Heatmap extrahieren und speichern:
        for i in range(len(bodypart_df["nose_x"])):
            nose_x_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_x"].iloc[i]
        
            nose_y_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_y"].iloc[i]

        """
        Maus am Food mit DLC auswerten:
        - Schnauze in der Nähe der Food Koordinate?
        - Wenn food von maus verdeckt, letzte Food Koordinate
        - letzte food koordinate nur nehmen, wenn maus im käfig ist; ansonsten food = nicht detected
        - als kontrolle die reine food detection plotten über die Zeit, auch interessant für ggf food bewegung
        
        """        
        # Koordinaten für die Heatmap extrahieren und speichern:
        food_likelihood_filtered_df = likelihood_filtering_nans(bodypart_df, likelihood_row_name='food1_likelihood',filter_val=0.7)
        food_x = food_likelihood_filtered_df["food1_x"]
        food_y = food_likelihood_filtered_df["food1_y"]

        # vorhandene food koordinaten nutzen um zu interpolieren, wenn vorhanden
        food_x = np.array(food_x)
        food_y = np.array(food_y)
        try:
            food_x = fill_missing_values(food_x)
            food_y = fill_missing_values(food_y)
        except:
            #print("No food data found for interpolation.")
            pass


        for i in range(len(food_likelihood_filtered_df["food1_x"])):
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
    # food interaktion zählen
    maus_an_food_percent = maus_an_food/len(exp_duration_frames)*100

    # das hier wäre die Strecke über die Zeit
    strecke_pixel_frame = strecke_in_pixeln/sum(maus_in_modul_über_zeit)
    visits_per_hour = num_visits / (len(exp_duration_frames)/30/3600)

    zeit_in_modul_prozent = sum(maus_in_modul_über_zeit) / len(exp_duration_frames) * 100


    dlc_data = [maus_an_food_percent, strecke_pixel_frame, visits_per_hour, zeit_in_modul_prozent]

    return maus_an_snicket_über_zeit, maus_in_modul_über_zeit, strecke_über_zeit, (nose_x_values_over_time, nose_y_values_over_time), dlc_data


"""

experiment_day_path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/mouse_75/2025_03_17/"

modul1_maus_an_snicket_über_zeit, modul1_maus_in_modul_über_zeit, modul1_strecke_über_zeit, modul1_nose_coords = analyze_one_module(path=f"{experiment_day_path}top1/")

modul2_maus_an_snicket_über_zeit, modul2_maus_in_modul_über_zeit, modul2_strecke_über_zeit, modul2_nose_coords = analyze_one_module(path=f"{experiment_day_path}top2/")





cumsum_plot(data_list=[modul1_maus_an_snicket_über_zeit,modul2_maus_an_snicket_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Maus an Snicket",
            x_label = "Experimentdauer in Frames",
            y_label= "Maus am Snicket in Frames",
            save_as= f"{experiment_day_path}maus_an_snicket.svg"
            )


cumsum_plot(data_list=[modul1_maus_in_modul_über_zeit,modul2_maus_in_modul_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Maus in Modul",
            x_label = "Experimentdauer in Frames",
            y_label= "Maus in Modul in Frames",
            save_as= f"{experiment_day_path}maus_in_modul.svg"
            )

cumsum_plot(data_list=[modul1_strecke_über_zeit,modul2_strecke_über_zeit],
            labels=["modul 1", "modul 2"],
            colors=["blue", "red"],
            plotname="Zurückgelegte Strecke pro Modul",
            x_label = "Experimentdauer in Frames",
            y_label= "Strecke in Pixeln",
            save_as= f"{experiment_day_path}maus_strecke.svg"
            )



heatmap_plot(x_values=modul1_nose_coords[0], y_values=modul1_nose_coords[1], plotname="Heatmap Modul 1", save_as=f"{experiment_day_path}heatmap_modul1.svg", num_bins=12)


heatmap_plot(x_values=modul2_nose_coords[0], y_values=modul2_nose_coords[1], plotname="Heatmap Modul 2", save_as=f"{experiment_day_path}heatmap_modul2.svg", num_bins=12)
"""

"""
deg_file_path = "E:/Fabi_Setup/In_Soundchamber/behaviors_urine_validation_deepethogram/DATA/2025_03_10_mouse_7_habituation_side1_40357253_stitched/2025_03_10_mouse_7_habituation_side1_40357253_stitched_predictions.csv"

deg_behaviors = ['rearing"', "drinking", "grooming"]

for behavior in deg_behaviors:
    bouts, sum = analyze_deg_file(deg_file_path, behavior)
    print(bouts)
    print(sum)

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

            """
            
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
            """

    print(behavior_data_modul1)
    print(behavior_data_modul2)

    deg_barplot_nwg(data_module1=behavior_data_modul1, data_module2=behavior_data_modul2, savename=f"{deg_path}{exp_day}_deg")

create_deg_barplot_for_nwg(exp_day="hab")

"""
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