import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from preprocessing import likelihood_filtering, likelihood_filtering_nans
from utils import euklidean_distance, fill_missing_values, shrink_rectangle
from config import PIXEL_PER_CM, ARENA_COORDS_TOP1, ARENA_COORDS_TOP2




def entry_or_exit_analysis(entry_polygon, arena_polygon, df, scorer, individual):
    pass