import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from preprocessing import likelihood_filtering, likelihood_filtering_nans
from utils import euklidean_distance, fill_missing_values, shrink_rectangle
from config import PIXEL_PER_CM, ARENA_COORDS_TOP1, ARENA_COORDS_TOP2

def social_investigation(df, scorer, individuals, bodyparts):

    all_nose_x = []
    all_nose_y = []

    all_x = []
    all_y = []

    for ind in individuals:
        nose_x = df.loc[:, (scorer, ind, ["nose"], ["x"])].to_numpy()
        nose_y = df.loc[:, (scorer, ind, ["nose"], ["y"])].to_numpy()

        all_nose_x.append(nose_x)
        all_nose_y.append(nose_y)

        x = df.loc[:, (scorer, ind, bodyparts, ["x"])].to_numpy()
        y = df.loc[:, (scorer, ind, bodyparts, ["y"])].to_numpy()

        all_x.append(x)
        all_y.append(y)

    social_investigation = np.zeros(len(all_nose_x[0]))



    for i in range(len(all_nose_x)): # jede Maus als Investigierende
        
        for p in range(len(all_x)):  # jede Maus als Ziel

            if not p == i:  
                

                for c in range(len(all_nose_x[i])):  # jedes Frame

                    closest_bp = 1000  # in px

                    for bp in range(len(all_x)):     # jeder Körperteil der Zielmaus

                        dist = euklidean_distance(x1 = all_nose_x[i][c], y1 = all_nose_y[i][c], x2 = all_x[p][c][bp], y2=all_y[p][c][bp])
                        if dist < closest_bp:
                            closest_bp = dist

                    if closest_bp <= PIXEL_PER_CM*3:
                        social_investigation[c] = 1

    return social_investigation


                    
                       
    

def time_in_center(df,bodypart, module):

    """
    Determines for each video frame whether the specified bodypart is within the center area of the arena.

    This function takes DeepLabCut tracking data and checks whether the x/y coordinates of the specified bodypart 
    fall within a scaled-down center region of the arena, defined by fixed polygon coordinates. The function returns 
    a binary array indicating whether the bodypart is in the center at each time point (frame).

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing DeepLabCut tracking data with columns named '<bodypart>_x' and '<bodypart>_y'.
    bodypart : str
        The name of the bodypart to be checked (e.g., 'nose', 'center', 'tailbase').
    module : int
        The module number (1 or 2) which determines which arena coordinates to use.

    Returns
    -------
    numpy.ndarray
        A 1D array of integers (0 or 1) with the same length as the input data, where 1 indicates that the bodypart 
        was within the center area of the arena in that frame.

    Raises
    ------
    ValueError
        If an invalid module number is provided (not 1 or 2).

    Notes
    -----
    The center area is defined by scaling the arena rectangle coordinates down to 60% of their original size, 
    centered around the arena midpoint. The arena coordinates must be defined globally as ARENA_COORDS_TOP1 and ARENA_COORDS_TOP2.
    """

    # Richtige Arena Koordinaten wählen
    if module == 1:
        arena_coords = ARENA_COORDS_TOP1
    elif module == 2:
        arena_coords = ARENA_COORDS_TOP2
    else:
        raise ValueError(f"Ungültiges Modul: {module}")

    # DeepLabCut Koordinaten extrahieren
    mouse_x_coords = df[bodypart+'_x']
    mouse_y_coords = df[bodypart+'_y']

    # Center Area der Arena definieren
    center_coords = shrink_rectangle(arena_coords, scale=0.6)
    center_polygon = Polygon(center_coords)

    # Leeren Ergebnisarray erstellen
    mouse_coords_in_center = np.zeros(len(mouse_x_coords), dtype=int)

    # Für jeden Frame testen, ob die Maus im Center ist
    for i in range(len(mouse_x_coords)):
        point = Point(mouse_x_coords.iloc[i], mouse_y_coords.iloc[i])
        if center_polygon.contains(point):
            mouse_coords_in_center[i] = 1

    return mouse_coords_in_center
    

def count_center_crossings(center_array, min_frames_between=10):
    """
    Zählt, wie oft die Maus das Zentrum durchquert hat, basierend auf einer binären Zeitreihe (0 = außerhalb, 1 = im Zentrum).

    Parameters
    ----------
    center_array : np.ndarray
        1D binäres Array mit 0 (außerhalb) und 1 (im Zentrum) pro Frame.
    min_frames_between : int, optional
        Minimale Anzahl an Frames zwischen zwei Übertritten, um Rauschen oder Zittern zu ignorieren.

    Returns
    -------
    int
        Anzahl der Mittendurchquerungen.
    """
    in_center = np.array(center_array, dtype=int)
    transitions = np.diff(in_center)
    entries = np.where(transitions == 1)[0] + 1  # +1 wegen diff-Verschiebung

    # Optional: Rauschen filtern – z. B. schnelle hin-und-her Wechsel innerhalb weniger Frames ignorieren
    if len(entries) == 0:
        return 0

    filtered_entries = [entries[0]]
    for i in range(1, len(entries)):
        if entries[i] - filtered_entries[-1] >= min_frames_between:
            filtered_entries.append(entries[i])

    return len(filtered_entries)

def mean_visit_time():
    pass

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
    radius_threshold = factor * PIXEL_PER_CM
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

def get_food_coordinates(df, food_likelihood_row):
    """
    Maus am Food mit DLC auswerten:
    - Schnauze in der Nähe der Food Koordinate?
    - Wenn food von maus verdeckt, letzte Food Koordinate
    - letzte food koordinate nur nehmen, wenn maus im käfig ist; ansonsten food = nicht detected
    - als kontrolle die reine food detection plotten über die Zeit, auch interessant für ggf food bewegung
        
    """        
    # Koordinaten für die Heatmap extrahieren und speichern:
    food_likelihood_filtered_df = likelihood_filtering_nans(df, likelihood_row_name='food1_likelihood',filter_val=0.7)
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

    return food_x, food_y