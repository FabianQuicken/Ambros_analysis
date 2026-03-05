import matplotlib.pyplot as plt
from save_load_dic import load_analysis
import glob as glob
import os
from tqdm import tqdm
import numpy as np
from omm12_helper import build_plot_inputs_from_module_dicts, plot_grouped_category_scatter


# # # meta infos # # # 

# Welche Gruppen sollen zusammen geplottet werden?
gf_gfprop = ["germfree_", "germfreeprop"]
gf_omm12 = ["germfree_", "omm12_"]
omm12_omm12prop_ommpgol = ["omm12_", "omm12prop", "ommpgol"]
groups = [gf_gfprop, gf_omm12, omm12_omm12prop_ommpgol]
group_names = ["gf_vs_gfprop", "gf_vs_omm12", "omm12_vs_omm12prop_vs_ommpgol"]
# Jede Gruppe hat eine eigene Farbe
group_colors = {
    "germfree": "black",
    "germfreeprop": "brown",
    "omm12": "darkorange",
    "omm12prop": "forestgreen",
    "ommpgol": "darkviolet"
}

# Es gibt verschiedene Sexes
sexes = ["male", "female"]
# Diese sollten eigene Formen haben
sex_layout = {
    "male": {"scatterplot": "circle", "lineplot": "solid"},
    "female": {"scatterplot": "triangle", "lineplot": "dotted"}
}


# hier liegen die habituation results
hab_basepath = r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\hab"
hab_paths = glob.glob(os.path.join(hab_basepath, '*.joblib'))

# hier liegen die top1 results
top1_basepath = r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\top1"
top1_paths = glob.glob(os.path.join(top1_basepath, '*.joblib'))

# hier liegen die top2 results
top2_basepath = r"C:\Users\quicken\Code\Ambros_analysis\OMM_analysis\top2"
top2_paths = glob.glob(os.path.join(top2_basepath, '*.joblib'))


# Zwischnespeicher layout für die Daten
to_analyse = gf_gfprop
need_hab = True
need_top1 = True
need_top2 = True

hab_data = {}
top1_data = {}
top2_data = {}

if need_hab:
    for g in to_analyse:
        hab_data[g] = {}
        for path in tqdm(hab_paths):
            basename = os.path.basename(path)
            if g in basename:
                print(f"loading data: {basename}")
                hab_data[g][basename] = load_analysis(path)


if need_top1:
    for g in to_analyse:
        top1_data[g] = {}
        for path in tqdm(top1_paths):
            basename = os.path.basename(path)
            if g in basename:
                print(f"loading data: {basename}")
                top1_data[g][basename] = load_analysis(path)

if need_top2:
    for g in to_analyse:
        top2_data[g] = {}
        for path in tqdm(top2_paths):
            basename = os.path.basename(path)
            if g in basename:
                print(f"loading data: {basename}")
                top2_data[g][basename] = load_analysis(path)

def get_key_data(groups_to_analyze, data, key):
    out = {}
    for g in groups_to_analyze:
        for basename in data[g]:
            key_data = data[g][basename][key]
            out[basename] = key_data
    return out

# # # plot social behavior # # # 
# - - - - - - -- - - - - - - - #
plt_social = False
if plt_social:
    # benötigte Daten sammeln
    hab_mice_per_frame = get_key_data(to_analyse, hab_data, "mice_per_frame")
    top1_mice_per_frame = get_key_data(to_analyse, top1_data, "mice_per_frame")
    top2_mice_per_frame = get_key_data(to_analyse, top2_data, "mice_per_frame")

    hab_face_inv = get_key_data(to_analyse, hab_data, "all_face")
    hab_body_inv = get_key_data(to_analyse, hab_data, "all_body")
    hab_anogenital_inv = get_key_data(to_analyse, hab_data, "all_anogenital")

    top1_face_inv = get_key_data(to_analyse, top1_data, "all_face")
    top1_body_inv = get_key_data(to_analyse, top1_data, "all_body")
    top1_anogenital_inv = get_key_data(to_analyse, top1_data, "all_anogenital")

    top2_face_inv = get_key_data(to_analyse, top2_data, "all_face")
    top2_body_inv = get_key_data(to_analyse, top2_data, "all_body")
    top2_anogenital_inv = get_key_data(to_analyse, top2_data, "all_anogenital")

    plot_data, group_style = build_plot_inputs_from_module_dicts(
        hab_mice_per_frame=hab_mice_per_frame,
        hab_face_inv=hab_face_inv,
        hab_body_inv=hab_body_inv,
        hab_anogenital_inv=hab_anogenital_inv,

        top1_mice_per_frame=top1_mice_per_frame,
        top1_face_inv=top1_face_inv,
        top1_body_inv=top1_body_inv,
        top1_anogenital_inv=top1_anogenital_inv,

        top2_mice_per_frame=top2_mice_per_frame,
        top2_face_inv=top2_face_inv,
        top2_body_inv=top2_body_inv,
        top2_anogenital_inv=top2_anogenital_inv,

        group_colors=group_colors,
        sex_layout=sex_layout,
    )

    # Reihenfolge in der Legende/Plot (optional)
    group_order = list(plot_data.keys())  # oder custom sort

    fig, ax = plot_grouped_category_scatter(
        plot_data,
        categories=("Hab", "Top1", "Top2"),
        group_order=group_order,
        group_style=group_style,
        y_label="Social investigation / pair-seconds (a.u.)",  # oder in %
        ylim=None,  # falls nicht 0-100
        group_offset_span=0.30,
        point_jitter=0.02,
        show_means=False,
    )
    plt.show()
    """


    for entry in hab_mice_per_frame:
        # zum plotten benötigte arrays nehmen und vorbereiten
        mice_per_frame = hab_mice_per_frame[entry]
        face_inv = hab_face_inv[entry]
        body_inv = hab_body_inv[entry]
        anogenital_inv = hab_anogenital_inv[entry]

        # paarsekunden berechnen
        p_seconds = 0
        for i in mice_per_frame:
            if not np.isfinite(i) or i <=1:
                continue
            elif i == 2:
                p_seconds += 1
            elif i == 3:
                p_seconds += 3

        # summe des investigation behaviors
        inv_sum = np.nansum(face_inv) + np.nansum(body_inv) + np.nansum(anogenital_inv)

        # Anteil berechnen
        a = inv_sum / p_seconds    
    """

     

