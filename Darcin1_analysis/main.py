import numpy as np
import pandas as pd
import tqdm
import os
import glob
import matplotlib.pyplot as plt
from create_labelled_video import create_labelled_video

FPS = 30
PIXEL_PER_CM = 36.39

def plot_event_data(data, labels, max_len=36000, marker_frame=19000, save_as=None):
    """
    Gestapelter Eventplot (Rasterplot) für binäre Arrays (0/1).
    Zeichnet optional eine rote vertikale Linie (marker_frame).
    """
    fig = plt.figure(figsize=(10, len(data) * 1.2))
    fig.patch.set_facecolor("black")     # gesamte Figure schwarz
    ax = plt.gca()
    ax.set_facecolor("black")            # Plotbereich schwarz

    for i, (arr, label) in enumerate(zip(data, labels)):
        arr = np.asarray(arr[:max_len])
        event_idx = np.flatnonzero(arr == 1)
        plt.eventplot(event_idx, lineoffsets=i, linelengths=0.8, color='magenta')

    plt.yticks(range(len(labels)), labels, color='white')
    plt.xlabel("Frame", color='white')
    plt.title("Eventplot der Stimulus-Interaktionen", color='white')
    plt.tick_params(colors='white')
    plt.grid(False)

    if marker_frame is not None:
        plt.axvline(marker_frame, color='red', linestyle='--', linewidth=2)

    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_psth(data, fps=30, bin_s=10.0, max_len=36000,
              show_individual=False, smooth_bins=2,
              aggregate='mean', marker_time_s=None, title="PSTH",
              y_mode="percent",
              save_as=None):
    """
    y_mode: 'percent' -> Anteil aktiver Zeit pro Bin [%]
            'seconds' -> aktive Sekunden pro Bin
            'count'   -> aktive Frames (Counts) pro Bin
    """
    assert aggregate in {'mean','sum','median'}
    assert y_mode in {'percent','seconds','count'}

    bin_frames = max(1, int(round(bin_s * fps)))
    n_bins = int(np.ceil(max_len / bin_frames))
    bin_edges_f = np.arange(0, n_bins + 1) * bin_frames
    time_s = (bin_edges_f[:-1] + bin_edges_f[1:]) / 2.0 / fps

    def _to_metric(arr):
        a = np.asarray(arr[:max_len], dtype=float)
        pad = n_bins * bin_frames - a.size
        if pad > 0:
            a = np.pad(a, (0, pad), mode='constant', constant_values=0)
        counts = a.reshape(n_bins, bin_frames).sum(axis=1)

        if y_mode == 'count':
            y = counts
            ylabel = "Aktive Frames / Bin"
        elif y_mode == 'seconds':
            y = counts / fps
            ylabel = "Aktive Zeit pro Bin (s)"
        else:  # 'percent'
            y = (counts / bin_frames) * 100.0
            ylabel = "Aktivitätsanteil pro Bin (%)"
        return y, ylabel

    Ys, ylabel = zip(*[_to_metric(arr) for arr in data])
    Ys = np.vstack(Ys)

    if smooth_bins and smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        Ys = np.apply_along_axis(lambda x: np.convolve(x, k, mode='same'), 1, Ys)

    if aggregate == 'mean':
        agg = Ys.mean(axis=0)
        err = Ys.std(axis=0) / np.sqrt(Ys.shape[0])
    elif aggregate == 'sum':
        agg, err = Ys.sum(axis=0), None
    else:
        agg, err = np.median(Ys, axis=0), None

    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor("black")
    ax = plt.gca()
    ax.set_facecolor("black")

    if show_individual:
        for y in Ys:
            plt.plot(time_s, y, alpha=0.3, color='magenta', linewidth=1)

    plt.plot(time_s, agg, linewidth=2.5, color='magenta', label=f"{aggregate.title()}")

    if err is not None:
        plt.fill_between(time_s, agg - err, agg + err, alpha=0.25, color='white', label="± SEM")

    if marker_time_s is not None:
        plt.axvline(marker_time_s, color='red', linestyle='--', linewidth=2, label="Marker")

    plt.xlabel("Zeit (s)", color='white')
    plt.ylabel(ylabel[0], color='white')
    plt.title(title, color='white')
    leg = plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('white')
    plt.grid(False)
    plt.axvline(19000/30, color='red', linestyle='--', linewidth=2)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


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


def interpolate_with_max_gap(df, max_gap=30, method="linear"):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    #print(num_cols)

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


def analyse_darcin_exp(path, dist_tresh):

    file_list = glob.glob(os.path.join(path, '*.h5'))
    file_list.sort()

    n_mice = len(file_list)/2
    stim_inv_data = []
    con_inv_data = []
    exp_data = []
    hab_data = []
    previous_mouse = ""
    for file_index, file in enumerate(file_list):
        read_in_df = pd.read_hdf(file)
        df = read_in_df.copy()

        mouse = os.path.basename(file)[34:36]
        

        df = interpolate_with_max_gap(df)

        scorer = df.columns.levels[0][0]
        bodyparts = df.columns.levels[1]

        def get_dish_coords(df, scorer, dish_name):

            dish_likelihood = df.loc[:, (scorer, [dish_name], ["likelihood"])].to_numpy().ravel()

            best_likelihood_frame = 0
            max_likelihood = 0

            for index, likelihood in enumerate(dish_likelihood):
                if likelihood > max_likelihood:
                    
                    
                    # best likelihood wird nur genommen, wenn die Schale auf der richtigen Seite getrackt wurde (es gibt prediction switches mit hoher likelihood)
                    # left dish x-Werte sollten bei etwa 730 sein, right dish x-Werte bei etwa 1430
                    dish_x_candidate = df.loc[index, (scorer, [dish_name], ["x"])].to_numpy().item()
                    if dish_name == "left_dish":
                        if dish_x_candidate < 1000:
                            best_likelihood_frame = index
                            max_likelihood = likelihood
                    if dish_name == "right_dish":
                        if dish_x_candidate > 1000:
                            best_likelihood_frame = index
                            max_likelihood = likelihood

            dish_x = df.loc[best_likelihood_frame, (scorer, [dish_name], ["x"])].to_numpy().item()
            dish_y = df.loc[best_likelihood_frame, (scorer, [dish_name], ["y"])].to_numpy().item()

            #print(dish_x, dish_y)
            """
            if dish_name == "left_dish":
                if dish_x > 1000:
                    raise ValueError(f"Left Dish Prediction probably on right dish. Coords: {dish_x}, {dish_y}")
            elif dish_name == "right_dish":
                if dish_x < 1000:
                    raise ValueError(f"Right Dish Prediction probably on left dish. Coords: {dish_x}, {dish_y}")
            """  
            return dish_x, dish_y

        #print(file)
        left_dish_x, left_dish_y = get_dish_coords(df, scorer, "left_dish")
        right_dish_x, right_dish_y = get_dish_coords(df, scorer, "right_dish")

        ld_inv = np.zeros(len(df))
        rd_inv = np.zeros(len(df))

        nose_x = df.loc[:, (scorer, ["nose"], ["x"])].to_numpy().ravel()
        nose_y = df.loc[:, (scorer, ["nose"], ["y"])].to_numpy().ravel()

        for i in range(len(nose_x)):
            ld_dist = euklidean_distance(x1=left_dish_x, y1=left_dish_y, x2=nose_x[i], y2=nose_y[i])
            if ld_dist <= dist_tresh:
                ld_inv[i] = 1
            rd_dist = euklidean_distance(x1=right_dish_x, y1=right_dish_y, x2=nose_x[i], y2=nose_y[i])
            if rd_dist <= dist_tresh:
                rd_inv[i] = 1
        if "stim_con" in os.path.basename(file):
            #print(os.path.basename(file))
            #print("stim left")
            stim_inv = ld_inv
            con_inv = rd_inv
        elif "con_stim" in os.path.basename(file):
            #print(os.path.basename(file))
            #print("stim right")
            stim_inv = rd_inv
            con_inv = ld_inv

        if previous_mouse != mouse:
            #print("appending...")
            stim_inv_data.append(stim_inv)
            con_inv_data.append(con_inv)

        elif previous_mouse == mouse:
            #print("concatenates...")
            halfed_ind = int((file_index+1)/2-1)
            #print(halfed_ind)
            stim_inv_data[halfed_ind] = np.concatenate((stim_inv_data[halfed_ind], stim_inv))  
            con_inv_data[halfed_ind] = np.concatenate((con_inv_data[halfed_ind], con_inv))
        
        if "exp" in os.path.basename(file):
            exp_data.append((stim_inv, con_inv))
        elif "hab" in os.path.basename(file):
            hab_data.append((stim_inv, con_inv))

        previous_mouse = mouse

        
    return stim_inv_data, con_inv_data, exp_data, hab_data
            





path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\raw\Darcin1\exp2"


stim_inv_data, con_inv_data, exp_data, hab_data = analyse_darcin_exp(path=path, dist_tresh=PIXEL_PER_CM*2.5)

labels = ["mouse_54", "mouse_55", "mouse_52", "mouse_56", "mouse_58"]

plot_save = r"Z:\n2023_odor_related_behavior\other\Vorträge\251114_Data_Club"
#plot_event_data(stim_inv_data, labels, save_as=plot_save+"\eventplot_stim_exp2.svg")
#plot_event_data(con_inv_data, labels, save_as=plot_save+"\eventplot_con_exp2.svg")
#plot_psth(data=stim_inv_data, save_as=plot_save+"\psthplot_stim_exp2.svg")
#plot_psth(data=con_inv_data, save_as=plot_save+"\pstplot_con_exp2.svg")

for index, arr in enumerate(con_inv_data):
    diff = sum(stim_inv_data[index][19000:38000]) / (sum(con_inv_data[index][19000:38000]))
    print(diff)



#create_labelled_video(video_path=r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\raw\Darcin1\exp1\2025_09_27_12_14_56_Darcin1_mouse_54_exp_stim_con_top1_40439818.avi", output_path=r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\raw\Darcin1\exp1\2025_09_27_12_14_56_Darcin1_mouse_54_exp_stim_con_top1_40439818_labeled.avi", metric2=exp_data[0][0], metric1=exp_data[0][1], text2= "Stim Inv", text1 = "Con Inv")


"""
file_list = glob.glob(os.path.join(path, '*.h5'))
file_list.sort()

read_in_df = pd.read_hdf(file_list[0])
df = read_in_df.copy()

scorer = df.columns.levels[0][0]
bodyparts = df.columns.levels[1]





def get_dish_coords(df, scorer, dish_name):

    dish_likelihood = df.loc[:, (scorer, [dish_name], ["likelihood"])].to_numpy().ravel()

    best_likelihood_frame = 0
    max_likelihood = 0

    for index, likelihood in enumerate(dish_likelihood):
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_likelihood_frame = index

    dish_x = df.loc[best_likelihood_frame, (scorer, [dish_name], ["x"])].to_numpy().item()
    dish_y = df.loc[best_likelihood_frame, (scorer, [dish_name], ["y"])].to_numpy().item()

    return dish_x, dish_y

left_dish_x, left_dish_y = get_dish_coords(df, scorer, "left_dish")
right_dish_x, right_dish_y = get_dish_coords(df, scorer, "right_dish")

print(left_dish_x, left_dish_y)
print(right_dish_x, right_dish_y)
    
if left_dish_x > 1000:
    raise ValueError("Left Dish Prediction probably on right dish.")
elif right_dish_x < 1000:
    raise ValueError("Right Dish Prediction probably on left dish.")

"""