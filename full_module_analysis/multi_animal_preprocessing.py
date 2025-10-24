"""
multi animal DLC files contain several errors, that can and should be corrected prior to running the behavior analysis

first problem: when trained for 3 mice, DLC tries to find 3 mice: wrong predictions on shadows, objects (especially dark), and multi labelled mice are the cause of this
solutions:

1. delete double-labels: try to only delete the added mouse, if it appears on an already present other mouse
2. based on keypoint distances, delete predictions with wird keypoint distance relations (caused by predictions on not-mouse-shaped objects)
3. Idea: Flag entrances as valid (location), if entrance is not valid, delete consecutive data (but this could also delete when mouse enters during this time)

second problem: missing predictions
solution: 
1. interpolation of missing chunks of a certain threshold

"""

# externe imports
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import glob
import os 
import seaborn as sns

# lokale imports
from utils import euklidean_distance
from config import FPS

# eigene funktionen fürs preprocessing
def create_polygon(polygon_coords=list):
    return Polygon(polygon_coords)

def create_point(x, y):
    return Point(x, y)

def is_point_in_polygon(polygon, point):
    return polygon.contains(point)

def interpolate_with_max_gap(df, max_gap=30, method="linear"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    print(num_cols)

    # 1) Nur „echte“ Interpolation zwischen gültigen Punkten
    out[num_cols] = out[num_cols].interpolate(method=method,
                                              limit_direction="both",
                                              limit_area="inside")
    
    # 2) NaN-Runs > max_gap identifizieren und wieder auf NaN setzen
    for col in num_cols:
        s = df[col]  # Original mit NaNs
        # Gruppen-IDs zwischen Nicht-NaNs erstellen
        grp = s.notna().cumsum()
        # Länge jedes NaN-Runs
        run_len = s.isna().groupby(grp).transform("sum")
        # Maske: Positionen in zu langen NaN-Runs
        too_long = s.isna() & (run_len > max_gap)
        # Zurücksetzen
        out.loc[too_long, col] = np.nan
    
    return out

# feste variablen 
arena_coords = [(110,20), (1870,25), (1860,1070), (110,1070)]
enter_zone_coords = [(1700,430),(1900,430),(1900,670),(1700,670)]


# datei wird importiert (später als Liste einlesen)
path = r"C:\Users\quicken\Code\Ambros_analysis\code_test"
filepath = r"C:\Users\quicken\Code\Ambros_analysis\code_test\2024_10_30_14_19_32_testing_top1_40439818_mice-12-17_water_furineDLC_DekrW18_multi_animal_frame_extractionOct16shuffle1_snapshot_best-390_el.h5"




# der folgende codeabschnitt muss später geädnert werden, um über eine Liste von files iterieren zu können 

# dataframe einlesen
df = pd.read_hdf(filepath)

# information aus der header Zeile entnehmen, um einzelne Teile bearbeiten zu können
scorer = [df.columns.levels[0][0]]
individuals = [df.columns.levels[1].to_list()]
bodyparts = [df.columns.levels[2].to_list()]


# da die h5 verändert werden, wird einmal die ursprüngliche Version gespeichert
filename = os.path.basename(filepath)
df.to_hdf((path + "/" + filename + '_old.h5'), key = 'tracks')





"""
for idx, df in enumerate(df_list):
    filename = os.path.basename(file_list[idx])
    filename = filename.rsplit('.')

    # build savepath
    save_as = os.path.join(path, filename[0] + '_old.h5')
    df.to_hdf(save_as, key='tracks')
"""