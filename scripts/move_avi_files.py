# This script moves avi files from one direction to another. They will be sorted to a top1, top2, side1, side2 folder respective to their name.

import shutil
import glob
import os

def move_files(source, destination):

    """
    Moves .avi files from the source directory into subdirectories at the destination based on keywords ('top1', 'top2', 'side1', 'side2') in the filenames.
    
    Preconditions:
        - Subdirectories named 'top1', 'top2', 'side1', 'side2' exist in the destination directory.
        - The filenames must contain a date string matching the destination directory name. The destination directory folder is just named as date.
    
    Args:
        source (str): Path to the source directory containing .avi files.
        destination (str): Path to the destination directory where files will be moved.
    
    Raises:
        Prints feedback on how many files are moved.
        Prints warning messages if destination is not empty or if file dates do not match.
    """

    file_list = glob.glob(os.path.join(source, '*.avi'))
    

    # test if there are files at the destination already
    destination_clear = True
    possible_destinations = ['top1', 'top2', 'side1', 'side2']
    for dest in possible_destinations:
        are_here_files = glob.glob(os.path.join(destination + '/' + dest, '*.avi'))
        if len(are_here_files) > 0:
            # sets flag to false, if avi files are at the destination
            destination_clear = False

    # test if destination directory has the correct date
    date_matches = True
    for file in file_list:
        if destination[-10:] not in file:
            # sets flag to false, if date doesn't match
            date_matches = False
    
    # moves files if destination is clear
    files_not_sorted = []
    if destination_clear and date_matches:
        files_moved = 0
        for file in file_list:
            file_moved = False
            for i in range(len(possible_destinations)):
                if possible_destinations[i] in file:
                    shutil.move(file, destination + '/' + possible_destinations[i])
                    file_moved = True
                    files_moved += 1
            if not file_moved:
                files_not_sorted.append(file)
            
        print(f"Du musstest {files_moved} Dateien nicht manuell verschieben. Geile Scheisse wa?")

    # gives feedback if moving was not possible
    if not destination_clear:
        print("Da sind schon Dateien, yikes!")
    if not date_matches:
        print("Dat Datum vom Ordner passt nicht zum Video, hast du dich vertippt du Larry?")
    if len(files_not_sorted) > 0:
        print(f"Bei folgenden Dateien hängt der Haussegen schief und daher wurden sie nicht verschoben: {files_not_sorted}")



files_are_at = r'G:/Fabi transfer/Videos/mouse21_hab2'

files_go_to = r'Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/raw/male_mice_female_stimuli/mouse_21/2025_05_15'

move_files(source=files_are_at, destination=files_go_to)