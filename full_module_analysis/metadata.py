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
    

    

