import matplotlib.pyplot as plt
from save_load_dic import load_analysis
import glob as glob
import os
from tqdm import tqdm
import numpy as np

# # # PLOT FUNCTIONS # # # 
def plot_grouped_category_scatter(
    data,
    *,
    categories=("Hab", "Top1", "Top2"),
    group_order=None,
    group_style=None,
    # layout
    base_positions=None,
    group_offset_span=0.26,   # total width occupied by all groups within a category
    point_jitter=0.02,        # random x-jitter per point
    # aesthetics
    point_size=45,
    alpha=0.85,
    y_label="%",
    title=None,
    ylim=(0, 100),
    show_means=False,
    mean_linewidth=2.0,
    ax=None,
    seed=0,
):
    """
    Scatter plot with categorical x-axis (Hab/Top1/Top2) and multiple groups per category.

    Parameters
    ----------
    data : dict
        Nested mapping: data[group][category] -> 1D array-like of y-values (percent).
        Example:
            data = {
              "GF":   {"Hab":[...], "Top1":[...], "Top2":[...]},
              "OMM":  {"Hab":[...], "Top1":[...], "Top2":[...]},
            }

    categories : tuple[str]
        Category labels on x-axis.

    group_order : list[str] | None
        Plotting order of groups. If None, uses data.keys() order.

    group_style : dict | None
        Mapping group -> dict(marker="^", color="C0", label="...").
        You can set any matplotlib scatter kwargs here (edgecolors, linewidths, etc.)

    group_offset_span : float
        Total horizontal span within one category used to separate groups.

    point_jitter : float
        Random horizontal jitter around each group's offset (helps separate overlapping points).

    show_means : bool
        If True, draw a short horizontal line per group/category at the mean.

    ax : matplotlib.axes.Axes | None
        Provide axis or create new.

    seed : int
        RNG seed for deterministic jitter.
    """
    rng = np.random.default_rng(seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure

    if base_positions is None:
        base_positions = np.arange(len(categories), dtype=float)

    if group_order is None:
        group_order = list(data.keys())

    n_groups = len(group_order)
    if n_groups == 0:
        raise ValueError("No groups found in `data`.")

    # offsets centered around each category position:
    # e.g. for 3 groups -> [-span/2, 0, +span/2] (scaled)
    if n_groups == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-group_offset_span / 2, group_offset_span / 2, n_groups)

    # default styles
    if group_style is None:
        group_style = {}
    def _style_for(g, i):
        st = dict(marker="o", color=f"C{i}", label=g)
        st.update(group_style.get(g, {}))
        return st

    # plot
    for gi, g in enumerate(group_order):
        st = _style_for(g, gi)

        for ci, cat in enumerate(categories):
            y = np.asarray(data[g].get(cat, []), dtype=float)
            y = y[~np.isnan(y)]
            if y.size == 0:
                continue

            x0 = base_positions[ci] + offsets[gi]
            x = x0 + rng.uniform(-point_jitter, point_jitter, size=y.size)

            ax.scatter(
                x, y,
                s=point_size,
                alpha=alpha,
                marker=st.get("marker", "o"),
                c=st.get("color", f"C{gi}"),
                edgecolors=st.get("edgecolors", "none"),
                linewidths=st.get("linewidths", 0.0),
                label=st.get("label", g) if ci == 0 else None,  # legend once per group
                zorder=3,
            )

            if show_means:
                m = float(np.nanmean(y))
                ax.plot(
                    [x0 - 0.03, x0 + 0.03], [m, m],
                    linewidth=mean_linewidth,
                    color=st.get("color", f"C{gi}"),
                    zorder=4,
                )

    # axes formatting
    ax.set_xticks(base_positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", alpha=0.25, zorder=0)

    # make x-limits a bit roomy
    ax.set_xlim(base_positions.min() - 0.6, base_positions.max() + 0.6)

    if title:
        ax.set_title(title)

    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    return fig, ax



# # # Sortieren # # # 
import re

def parse_entry(entry: str):
    """
    Erwartet z.B. 'germfree_females_30_45_46_hab'
    -> group='germfree', sex='female', module='Hab'
    """
    s = entry.lower()

    # module am Ende
    module_map = {"hab": "Hab", "top1": "Top1", "top2": "Top2"}
    module_token = s.split("_")[-1]
    if module_token not in module_map:
        raise ValueError(f"Unknown module token '{module_token}' in entry='{entry}'")
    module = module_map[module_token]

    # sex irgendwo drin (females/males oder female/male)
    # Wir suchen bewusst nach whole tokens.
    tokens = s.split("_")
    sex = None
    if "females" in tokens or "female" in tokens:
        sex = "female"
    elif "males" in tokens or "male" in tokens:
        sex = "male"
    else:
        raise ValueError(f"Could not infer sex from entry='{entry}'")

    # group = erstes token (bei dir scheint es so zu sein: germfree, germfreeprop, ...)
    group = tokens[0]
    return group, sex, module

def mpl_marker_from_layout(sex_layout_scatter: str) -> str:
    m = sex_layout_scatter.lower()
    if m == "circle":
        return "o"
    if m == "triangle":
        return "^"
    # optional: erweitern
    raise ValueError(f"Unknown scatter marker '{sex_layout_scatter}' (expected 'circle' or 'triangle').")


import numpy as np

def compute_social_metric_per_entry(
    mice_per_frame: np.ndarray,
    face_inv: np.ndarray,
    body_inv: np.ndarray,
    anogenital_inv: np.ndarray,
):
    """
    Rechnet:
    p_seconds = sum(1 wenn 2 Mäuse), sum(3 wenn 3 Mäuse)  (pro Frame als 'Sekunden' angenommen)
    inv_sum = Summe face+body+anogenital
    a = inv_sum / p_seconds
    """
    # p_seconds
    p_seconds = 0.0
    for i in mice_per_frame:
        if not np.isfinite(i) or i <= 1:
            continue
        elif i == 2:
            p_seconds += 1
        elif i == 3:
            p_seconds += 3

    inv_sum = float(np.nansum(face_inv) + np.nansum(body_inv) + np.nansum(anogenital_inv))

    if p_seconds <= 0:
        return np.nan  # oder 0.0, je nachdem was du willst
    return inv_sum / p_seconds


def build_plot_inputs_from_module_dicts(
    *,
    hab_mice_per_frame, hab_face_inv, hab_body_inv, hab_anogenital_inv,
    top1_mice_per_frame, top1_face_inv, top1_body_inv, top1_anogenital_inv,
    top2_mice_per_frame, top2_face_inv, top2_body_inv, top2_anogenital_inv,
    group_colors,
    sex_layout,
):
    """
    Returns
    -------
    plot_data : dict
        plot_data[group__sex][module] -> list of y values (a)
    group_style : dict
        group_style[group__sex] -> dict(color=..., marker=..., label=...)
    """

    # sammle alles in einer Struktur: module_name -> dicts
    module_sources = {
        "Hab":  (hab_mice_per_frame,  hab_face_inv,  hab_body_inv,  hab_anogenital_inv),
        "Top1": (top1_mice_per_frame, top1_face_inv, top1_body_inv, top1_anogenital_inv),
        "Top2": (top2_mice_per_frame, top2_face_inv, top2_body_inv, top2_anogenital_inv),
    }

    plot_data = {}    # group__sex -> {Hab:[...], Top1:[...], Top2:[...]}
    group_style = {}  # group__sex -> style dict

    for module_name, (mpf_dict, face_dict, body_dict, anog_dict) in module_sources.items():
        for entry in mpf_dict.keys():
            # entry parsen
            group, sex, module_from_entry = parse_entry(entry)

            # sanity check: entry sollte zum iterierten module passen
            if module_from_entry != module_name:
                # Falls deine dicts strikt getrennt sind, sollte das nie passieren.
                # Wenn doch: einfach skippen statt crashen.
                continue

            # metric berechnen
            a = compute_social_metric_per_entry(
                mice_per_frame=np.asarray(mpf_dict[entry]),
                face_inv=np.asarray(face_dict[entry]),
                body_inv=np.asarray(body_dict[entry]),
                anogenital_inv=np.asarray(anog_dict[entry]),
            )

            plot_group_key = f"{group}__{sex}"

            # plot_data initialisieren
            if plot_group_key not in plot_data:
                plot_data[plot_group_key] = {"Hab": [], "Top1": [], "Top2": []}

            plot_data[plot_group_key][module_name].append(a)

            # style nur einmal anlegen
            if plot_group_key not in group_style:
                if group not in group_colors:
                    raise KeyError(f"group '{group}' not in group_colors. Known: {list(group_colors)}")

                marker = mpl_marker_from_layout(sex_layout[sex]["scatterplot"])
                group_style[plot_group_key] = {
                    "color": group_colors[group],
                    "marker": marker,
                    "label": f"{group} ({sex})"
                }

    return plot_data, group_style