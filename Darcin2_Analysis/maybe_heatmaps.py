"""

# # # Falls Heatmaps # # #

bodyparts = ["left_ear", "nose", "right_ear"]
dish_bp = "dish"

# data for heatmap
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day1_top1"
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day1_top2"
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day2_top1"
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day2_top2"
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day3_top1"
heatmap_data_con1 = r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\day3_top2"


file_list = glob.glob(os.path.join(heatmap_data_con1, '*.csv'))
file_list.sort()

dfs = []
for file in file_list:

    df = pd.read_csv(file, header=[0, 1, 2])



    for bp in bodyparts:

        lh = df.loc[:, (slice(None), bp, "likelihood")].to_numpy().ravel()
        df.loc[lh < 0.6, (slice(None), bp, ["x", "y", "likelihood"])] = np.nan

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.to_csv(r"Z:\n2023_odor_related_behavior\2025_darcin\Darcin2\for FENS 2026\Heatmaps\test.csv", index=False)


"""