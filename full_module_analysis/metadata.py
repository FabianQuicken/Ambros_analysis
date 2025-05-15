import os

def module_is_stimulus_side(path):
    basename = os.path.basename(path)
    parts = basename.split('_')
    # habituation versuche haben keine stimulus side, daher ist der string kürzer
    if 'top' in parts[9]:
        return False
    # control als tag für die seite, auf der kein Stimulus ist (da stimuli verschiedene benennungen haben)
    elif 'control' in parts[9] and 'top1' in parts[10]:
        return True
    elif 'control' in parts[8] and 'top2' in parts[10]:
        return True 
    else:
        return False
    

    

