import os

def module_is_stimulus_side(path):
    basename = os.path.basename(path)
    parts = basename.split('_')
    mouse = parts[6] + '_' + parts[7]
    modulnumber = None

    if 'top1' in basename:
        modulnumber = 1
    elif 'top2' in basename:
        modulnumber = 2
    elif 'top3' in basename:
        modulnumber = 3

    # habituation versuche haben keine stimulus side, daher ist der string kürzer
    if 'top' in parts[9]:
        return False, mouse, modulnumber
    # control als tag für die seite, auf der kein Stimulus ist (da stimuli verschiedene benennungen haben)
    elif 'control' in parts[9] and 'top1' in parts[10]:
        return True, mouse, modulnumber
    elif 'control' in parts[8] and 'top2' in parts[10]:
        return True, mouse, modulnumber
    else:
        return False, mouse, modulnumber
    

    

def module_has_stimulus_ma(path):
    """
    Determine whether a module contains a stimulus based on the experiment file name.

    This function parses a file path corresponding to a multi-animal (MA) recording
    and extracts information about the module setup. It checks if the module
    (either top1 or top2) was assigned a stimulus based on specific filename tokens.
    The filename is expected to follow a structured naming convention where
    information such as mouse cohort, habituation status, and stimulus identifiers
    are encoded as underscore-separated fields.

    Parameters
    ----------
    path : str
        Full path to the experiment file (e.g.,
        ".../2025_10_08_12_00_48_mice_c1_exp1_male_none_top2_40405188DLC_HrnetW32_multi_animal_pretrainedOct24shuffle1_detector_best-270_snapshot_best-120_el.h5").

    Returns
    -------
    stimulus_module : bool
        True if the given module contains a stimulus, otherwise False.
    mouse_cohort : str
        Identifier of the mouse cohort (extracted from the filename, e.g., 'c1').
    modulnumber : int
        Module number inferred from the filename (1 for 'top1', 2 for 'top2').

    Raises
    ------
    KeyError
        If the filename does not contain a valid module identifier ('top1' or 'top2').

    Notes
    -----
    The function assumes a specific filename structure:
    [date]_[time]_..._[mouse_cohort]_[hab/exp]_[stimulus1]_[stimulus2]_[camera_position]
    
    For example:
    "2025_10_08_12_00_48_mice_c1_exp1_male_none_top2_40405188DLC_HrnetW32_multi_animal_pretrainedOct24shuffle1_detector_best-270_snapshot_best-120_el.h5"
    → will yield (False, 'c1', 2)
    """
    stimulus_module = False
    basename = os.path.basename(path)
    parts = basename.split('_')

    mouse_cohort = parts[7]
    is_hab = True if parts[8] == 'hab' else False

    if parts[11] == "top1":
        modulnumber = 1
        if not parts[9].lower() == 'none':
            stimulus_module  = True 
    elif parts[11] == "top2":
        modulnumber = 2
        if not parts[10].lower() == 'none':
            stimulus_module  = True 
    else:
        raise KeyError(f"Unexpected module number entry in filename: {parts[11]}")
    
    return stimulus_module, mouse_cohort, modulnumber
