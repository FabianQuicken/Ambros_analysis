import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


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

    # adjust the heatmap size to the original data, since x-axis of the video is larger than y-axis

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

    def distance_bodypart_object(df, bodypart=str, object=str):
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

    # format um das dataframe umzuschreiben
    df_cols = ("nose_x", "nose_y", "nose_likelihood",
                "leftear_x", "leftear_y", "leftear_likelihood",
                "rightear_x", "rightear_y", "rightear_likelihood",
                "spine1_x", "spine1_y", "spine1_likelihood",
                "spine2_x", "spine2_y", "spine2_likelihood",
                "centroid_x", "centroid_y", "centroid_likelihood",
                "spine3_x", "spine3_y", "spine3_likelihood",
                "spine4_x", "spine4_y", "spine4_likelihood",
                "tail1_x", "tail1_y", "tail1_likelihood",
                "tail2_x", "tail2_y", "tail2_likelihood",
                "tail3_x", "tail3_y", "tail3_likelihood",
                "snicket_x", "snicket_y", "snicket_likelihood",
                "food1_x", "food1_y", "food1_likelihood",
                "food2_x", "food2_y", "food2_likelihood",
                "food3_x", "food3_y", "food3_likelihood")
    
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
   

    #variablen of interest einführen
    # array mit len(experimentdauer in s * fps) WICHTIG 
    fps = 30
    maus_in_modul_über_zeit = np.zeros(experiment_dauer_in_s * fps + len(df_last_file)) 
    maus_an_snicket_über_zeit = maus_in_modul_über_zeit.copy()
    strecke_über_zeit = maus_in_modul_über_zeit.copy()

    maus_in_modul_in_frames = 0
    maus_am_snicket_in_frames = 0
    strecke_in_pixeln = 0

    nose_x_values_over_time = maus_in_modul_über_zeit.copy()
    nose_y_values_over_time = maus_in_modul_über_zeit.copy()


    for file in tqdm(file_list):
        
        # dataframe erstellen und schneiden
        df = pd.read_csv(file, names=df_cols)
        data = df.copy()
        data = data.iloc[3:]
        data = data.astype(float)

        #dataframe kopieren & bodyparts of interest extrahieren
        empty_df = pd.DataFrame()
        bodypart_df = empty_df.copy()
        bodyparts_to_extract = ["nose", "centroid", "snicket"]

        

        for bodypart in bodyparts_to_extract:
            bodypart_df[bodypart+"_x"] = data[bodypart+"_x"]
            bodypart_df[bodypart+"_y"] = data[bodypart+"_y"]*(-1)  # y invertieren da DLC y koordinaten aufsteigen
            bodypart_df[bodypart+"_likelihood"] = data[bodypart+"_likelihood"]


        #berechnen ob maus present: insgesamt und over time
        mouse_present_calculation_df = bodypart_df.copy()

        mouse_snout_likelihood_arr = mouse_present_calculation_df["nose_likelihood"]
    
        
        mouse_present_arr = np.zeros(len(mouse_snout_likelihood_arr))

        for i in range(len(mouse_present_arr)):
            if mouse_snout_likelihood_arr.iloc[i] > 0.3:
                mouse_present_arr[i] = 1

        #insgesamt
        maus_in_modul_in_frames += np.nansum(mouse_present_arr) 

        #over time
        time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * fps
        
        for i in range(len(mouse_present_arr)):
            maus_in_modul_über_zeit[i+(time_position_in_frames-1)] = mouse_present_arr[i]


    


         #berechnen ob maus nah am snicket (faktor = 1 = 34.77 pixel) (wieder insgesamt und over time)
        distance_mouse_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose", object = "snicket")

        mouse_is_investigating, factor = investigation_time(distance_values=distance_mouse_nose_snicket, factor=3)


        maus_am_snicket_in_frames += np.nansum(mouse_is_investigating)

        for i in range(len(mouse_is_investigating)):
            maus_an_snicket_über_zeit[i+(time_position_in_frames-1)] = mouse_is_investigating[i]


        #zurückgelegte Strecke berechnen (in pixeln)
        maus_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid")

        strecke_in_pixeln += np.nansum(maus_distance_travelled)

        for i in range(len(maus_distance_travelled)):
            strecke_über_zeit[i+(time_position_in_frames-1)] = maus_distance_travelled[i]


        # Koordinaten für die Heatmap extrahieren und speichern:
        for i in range(len(bodypart_df["nose_x"])):
            nose_x_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_x"].iloc[i]
            nose_y_values_over_time[i+(time_position_in_frames-1)] = bodypart_df["nose_y"].iloc[i]




    return maus_an_snicket_über_zeit, maus_in_modul_über_zeit, strecke_über_zeit, (nose_x_values_over_time, nose_y_values_over_time)




experiment_day_path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/mouse_10/2025_02_13/"

modul1_maus_an_snicket_über_zeit, modul1_maus_in_modul_über_zeit, modul1_strecke_über_zeit, modul1_nose_coords = analyze_one_module(path=f"{experiment_day_path}top1/")

modul2_maus_an_snicket_über_zeit, modul2_maus_in_modul_über_zeit, modul2_strecke_über_zeit, modul2_nose_coords = analyze_one_module(path=f"{experiment_day_path}top2/")


"""
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

"""

heatmap_plot(x_values=modul1_nose_coords[0], y_values=modul1_nose_coords[1], plotname="Heatmap Modul 1", save_as=f"{experiment_day_path}heatmap_modul1.svg", num_bins=12, plot_time_frame_hours=(0,5))
heatmap_plot(x_values=modul2_nose_coords[0], y_values=modul2_nose_coords[1], plotname="Heatmap Modul 2", save_as=f"{experiment_day_path}heatmap_modul2.svg", num_bins=12, plot_time_frame_hours=(0,5))
