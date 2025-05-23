# # # Analyse des ersten Experiments - richtige Analysepipeline kommt später # # #

"""
Planung

ich habe...
- 2 topview kameras für beide Module
- 1 Experimenttag mit Interaktion in den Modulen 
- female urin in modul 2
- wasser in modul 1

Analyseziele:
- analyse erst mal einzeln laufen lassen pro modul
- Zeit in Modulen
- Zeit am Snicket
- Zeit in Modulen gebinned über die gesamtexperimentdauer
- stecke zurückgelegt je modul

Code Schritte:
- csv's einlesen
- liste an csv's nach uhrzeit sortieren
- dataframe erstellen
- dataframe schneiden (erste Zeilen removen)
- keypoints of interest extrahieren: individual1, snout, centroid; individual2, snout, centroid; snicket
- koordinaten invertieren
- separate variablen erstellen wo die ergebnisse gespeichert werden
    - int maus_in_modul_in_frames
    - int maus_am_snicket_in_frames
    - int strecke_in_pixeln
    - list maus_in_modul_über_zeit # sollte len(differenz uhrzeit 1. Video - letztes Video in frames) haben

- AB HIER: nur noch mit dataframe.copy arbeiten für JEDEN schritt
- counter auf 0 -> individual1 snout_likelihood als array und individual2 snout_likelihood als array nehmen -> über arraylänge iterieren, wenn bei min einem likelihood hoch, counter +1 -> maus_in_modul + counter
- counter auf 0 -> likelihood filtern individual1 snout -> individual1 snoutx/snouty array + 1 snicket coord als int -> euklidean distance snout-snicket -> counter +1 wenn unter thresh -> maus am snicket + counter
- counter auf 0 -> likelihood filtern individual2 snout -> individual2 snoutx/snouty array + 1 snicket coord als int -> euklidean distance snout-snicket -> counter +1 wenn unter thresh -> maus am snicket + counter
- likelihood filtern individual1 centroid -> array individual1 centroid x/y -> euklidean distance i/i+1 in distance_array -> sum distance_array + strecke_in_pixeln
- likelihood filtern individual2 centroid -> array individual2 centroid x/y -> euklidean distance i/i+1 in distance_array -> sum distance_array + strecke_in_pixeln 

"""

import glob
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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
        print(f"The filter replaced values in {num_replaced} rows with NaN out of a total of {len(df)} rows.")
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
        print(f"The filter removed {len(df_removed_rows)} rows of a total of {len(df)} rows.")
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
        print(f"\nGet distance {bodypart} to {object}...")
        print(f"Filtering {bodypart} for object distance calculation...")
        data = likelihood_filtering_nans(df=data, 
                                    likelihood_row_name=bodypart+"_likelihood")
        bodypart_x = data[bodypart+"_x"]
        bodypart_y = data[bodypart+"_y"]
        

        print("Filtering for a good object prediction...")
        data = likelihood_filtering(df=data, 
                                    likelihood_row_name=object+"_likelihood",
                                    filter_val=0.95)
        object_x = data[object+"_x"]
        object_y = data[object+"_y"]


        bodypart_x = np.array(bodypart_x)
        bodypart_y = np.array(bodypart_y)
        object_x = np.array(object_x)
        object_y = np.array(object_y)
        object_x = object_x[0]
        object_y = object_y[0]


        distance_values = np.zeros((len(bodypart_x)))
        for i in range(len(bodypart_x)-1):
            distance_values[i] = euklidean_distance(x1=bodypart_x[i],
                                                    y1=bodypart_y[i],
                                                    x2=object_x,
                                                    y2=object_y)
        return distance_values

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
        print("\nGet distance values...")
        print(f"Filtering {bodypart} for distance calculation...")
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

        # Extrahiere Zeit aus den Dateinamen
        first_time = first_file[39:47]  # Erwarte Format HH_MM_SS
        this_time = filename[39:47]

        start_seconds = time_to_seconds(first_time)
        current_seconds = time_to_seconds(this_time)

        # Berechne die Differenz in Sekunden
        return current_seconds - start_seconds




    # csv's einlesen und nach uhrzeit(name) sortieren
    #path = "E:/Fabi_Setup/fileserver_transfer/top1/"
    path = path
    file_list = glob.glob(os.path.join(path, '*.csv'))
    file_list.sort()

    # format um das dataframe umzuschreiben
    df_cols = ("nose1_x", "nose1_y", "nose1_likelihood",
                "leftear1_x", "leftear1_y", "leftear1_likelihood",
                "rightear1_x", "rightear1_y", "rightear1_likelihood",
                "spine11_x", "spine11_y", "spine11_likelihood",
                "spine21_x", "spine21_y", "spine21_likelihood",
                "centroid1_x", "centroid1_y", "centroid1_likelihood",
                "spine31_x", "spine31_y", "spine31_likelihood",
                "spine41_x", "spine41_y", "spine41_likelihood",
                "tail11_x", "tail11_y", "tail11_likelihood",
                "tail21_x", "tail21_y", "tail21_likelihood",
                "tail31_x", "tail31_y", "tail31_likelihood",
                "nose2_x", "nose2_y", "nose2_likelihood",
                "leftear2_x", "leftear2_y", "leftear2_likelihood",
                "rightear2_x", "rightear2_y", "rightear2_likelihood",
                "spine12_x", "spine12_y", "spine12_likelihood",
                "spine22_x", "spine22_y", "spine22_likelihood",
                "centroid2_x", "centroid2_y", "centroid2_likelihood",
                "spine32_x", "spine32_y", "spine32_likelihood",
                "spine42_x", "spine42_y", "spine42_likelihood",
                "tail12_x", "tail12_y", "tail12_likelihood",
                "tail22_x", "tail22_y", "tail22_likelihood",
                "tail32_x", "tail32_y", "tail32_likelihood",
                "snicket_x", "snicket_y", "snicket_likelihood")


    # # # erstmal experimentdauer berechnen # # #
    experiment_dauer_in_s = 0

    name_first_file = file_list[0]
    name_last_file = file_list[-1]

    df_last_file = pd.read_csv(file_list[-1])



    #Zeit immer an selber stelle
    startzeit = name_first_file[39:47] #platzhalter Zahlen
    endzeit = name_last_file[39:47]

    start_in_s = time_to_seconds(startzeit)
    ende_in_s = time_to_seconds(endzeit)

    experiment_dauer_in_s = ende_in_s - start_in_s 

    #variablen of interest einführen
    # array mit len(experimentdauer in s * fps) WICHTIG 
    fps = 30
    maus_in_modul_über_zeit = np.zeros(experiment_dauer_in_s * fps + len(df_last_file)) 
    maus_an_snicket_über_zeit = maus_in_modul_über_zeit.copy()



    maus_in_modul_in_frames = 0
    maus_am_snicket_in_frames = 0
    strecke_in_pixeln = 0


    """
    #csv einlesen
    df = pd.read_csv(file_list[0], names=df_cols)
    data = df.copy()
    data = data.iloc[4:]
    data = data.astype(float)

    #dataframe kopieren & bodyparts of interest extrahieren
    empty_df = pd.DataFrame()
    bodypart_df = empty_df.copy()
    bodyparts_to_extract = ["nose1", "centroid1", "nose2", "centroid2", "snicket"]

    for bodypart in bodyparts_to_extract:
        bodypart_df[bodypart+"_x"] = data[bodypart+"_x"]
        bodypart_df[bodypart+"_y"] = data[bodypart+"_y"]*(-1)  # y invertieren da DLC y koordinaten aufsteigen
        bodypart_df[bodypart+"_likelihood"] = data[bodypart+"_likelihood"]
    """




    """


    #berechnen ob maus nah am snicket (faktor = 1 = 34.77 pixel)
    distance_mouse1_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose1", object = "snicket")
    distance_mouse2_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose2", object = "snicket")

    mouse1_is_investigating, factor = investigation_time(distance_values=distance_mouse1_nose_snicket, factor=2)
    mouse2_is_investigating, factor = investigation_time(distance_values=distance_mouse2_nose_snicket, factor=2)

    maus_am_snicket_in_frames += np.nansum(mouse1_is_investigating)
    maus_am_snicket_in_frames += np.nansum(mouse2_is_investigating)

    #zurückgelegte Strecke berechnen (in pixeln)
    maus1_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid1")
    maus2_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid2")

    strecke_in_pixeln += np.nansum(maus1_distance_travelled)
    strecke_in_pixeln += np.nansum(maus2_distance_travelled)

    """


    for file in file_list:
        
        # dataframe erstellen und schneiden
        df = pd.read_csv(file, names=df_cols)
        data = df.copy()
        data = data.iloc[4:]
        data = data.astype(float)

        #dataframe kopieren & bodyparts of interest extrahieren
        empty_df = pd.DataFrame()
        bodypart_df = empty_df.copy()
        bodyparts_to_extract = ["nose1", "centroid1", "nose2", "centroid2", "snicket"]

        for bodypart in bodyparts_to_extract:
            bodypart_df[bodypart+"_x"] = data[bodypart+"_x"]
            bodypart_df[bodypart+"_y"] = data[bodypart+"_y"]*(-1)  # y invertieren da DLC y koordinaten aufsteigen
            bodypart_df[bodypart+"_likelihood"] = data[bodypart+"_likelihood"]

        #berechnen ob maus present: insgesamt und over time
        mouse_present_calculation_df = bodypart_df.copy()

        mouse1_snout_likelihood_arr = mouse_present_calculation_df["nose1_likelihood"]
        mouse2_snout_likelihood_arr = mouse_present_calculation_df["nose2_likelihood"]
        
        mouse_present_arr = np.zeros(len(mouse1_snout_likelihood_arr))

        for i in range(len(mouse_present_arr)):
            if mouse1_snout_likelihood_arr[i] > 0.3 or mouse2_snout_likelihood_arr[i] > 0.3:
                mouse_present_arr[i] = 1

        #insgesamt
        maus_in_modul_in_frames += np.nansum(mouse_present_arr) 

        #over time
        time_position_in_frames = convert_videostart_to_experiment_length(first_file=file_list[0], filename=file) * fps
        
        for i in range(len(mouse_present_arr)):
            maus_in_modul_über_zeit[i+(time_position_in_frames-1)] = mouse_present_arr[i]

    


        #berechnen ob maus nah am snicket (faktor = 1 = 34.77 pixel) (wieder insgesamt und over time)
        distance_mouse1_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose1", object = "snicket")
        distance_mouse2_nose_snicket = distance_bodypart_object(df = bodypart_df, bodypart = "nose2", object = "snicket")

        mouse1_is_investigating, factor = investigation_time(distance_values=distance_mouse1_nose_snicket, factor=3)
        mouse2_is_investigating, factor = investigation_time(distance_values=distance_mouse2_nose_snicket, factor=3)

        any_mouse_investigating = np.zeros(len(mouse1_is_investigating))
        for i in range(len(any_mouse_investigating)):
            if mouse1_is_investigating[i] == 1 or mouse2_is_investigating[i] == 1:
                any_mouse_investigating[i] = 1

        maus_am_snicket_in_frames += np.nansum(any_mouse_investigating)

        for i in range(len(any_mouse_investigating)):
            maus_an_snicket_über_zeit[i+(time_position_in_frames-1)] = any_mouse_investigating[i]

        #zurückgelegte Strecke berechnen (in pixeln)
        maus1_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid1")
        maus2_distance_travelled = distance_travelled(df=bodypart_df, bodypart="centroid2")

        strecke_in_pixeln += np.nansum(maus1_distance_travelled)
        strecke_in_pixeln += np.nansum(maus2_distance_travelled)

    return maus_an_snicket_über_zeit, maus_in_modul_über_zeit, strecke_in_pixeln


modul1_maus_an_snicket_über_zeit, modul1_maus_in_modul_über_zeit, modul1_strecke_in_pixeln = analyze_one_module(path="C:/transfer/pu_analyse/top1/")

modul2_maus_an_snicket_über_zeit, modul2_maus_in_modul_über_zeit, modul2_strecke_in_pixeln = analyze_one_module(path="C:/transfer/pu_analyse/top2/")


# Kumulative Summe berechnen
cumulative_sum1 = np.cumsum(modul1_maus_an_snicket_über_zeit)
cumulative_sum2 = np.cumsum(modul2_maus_an_snicket_über_zeit)

# Zeitachse erzeugen, basierend auf der Länge des längeren Arrays
max_length = max(len(modul1_maus_an_snicket_über_zeit), len(modul2_maus_an_snicket_über_zeit))
time = np.arange(max_length)

# Arrays auf gleiche Länge bringen (auffüllen mit dem letzten Wert)
if len(cumulative_sum1) < max_length:
    cumulative_sum1 = np.pad(cumulative_sum1, (0, max_length - len(cumulative_sum1)), 'edge')
if len(cumulative_sum2) < max_length:
    cumulative_sum2 = np.pad(cumulative_sum2, (0, max_length - len(cumulative_sum2)), 'edge')


# Plotten
plt.figure(figsize=(10, 6))
plt.plot(time, cumulative_sum1, label="Modul1", linewidth=2)
plt.plot(time, cumulative_sum2, label="Modul2", linewidth=2, color='red')
plt.title("Maus an Snicket (3cm Radius)")
plt.xlabel("Zeit in frames")
plt.ylabel("Kumulative Summe")
plt.grid(True)
plt.legend()
plt.savefig(f"C:/transfer/pu_analyse/maus_an_snicket.svg", format='svg')
plt.show()