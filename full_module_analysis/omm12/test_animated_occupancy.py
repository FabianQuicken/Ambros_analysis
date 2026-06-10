from animated_occupancy_plot import create_animated_occupancy_plot
import pandas as pd

csv = r"\\fileserver2\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest\single_mouse_datatest_germfree_68_69_70_females_top2.csv"


df = pd.read_csv(csv, header=[0, 1, 2, 3, 4, 5], index_col=0)

idx = pd.IndexSlice
x = df.loc[:, idx["germfree", slice(None), "females", "top2", "nose_x", "mouse_1"]].to_numpy()
y = df.loc[:, idx["germfree", slice(None), "females", "top2", "nose_y", "mouse_1"]].to_numpy()

create_animated_occupancy_plot(
    x,
    y,
    rectangle_coords=[(128,-22), (1862,-23), (1855,-1065), (136,-1058)],
    binnumber=5,
    normalizemode="realtime",
    experimentlength=20,
    windowsize=100,
    savefolder=r"\\fileserver2\AG Spehr BigData\n2023_odor_related_behavior\2025_omm_mice\behavior_data_betatest",
    create_svg=True,
    original_image_size=(1920,1080),

)
print("Done")