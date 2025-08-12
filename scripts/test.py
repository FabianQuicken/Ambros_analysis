import os
import glob

# Pfad zu den AVI-Dateien
path = r"Z:\n2023_odor_related_behavior\2023_behavior_setup_seminatural_odor_presentation\raw\female_mice_male_stimuli_plus_ventilation\mouse_5778\2025_07_23\top2"  # aktuelles Verzeichnis, ggf. anpassen
avi_files = glob.glob(os.path.join(path, "*.avi"))

for file_path in avi_files:
    dirname, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    parts = name.split("_")
    try:
        idx_control = parts.index("control")
        idx_male = parts.index("male")

        # Positionen tauschen
        parts[idx_control], parts[idx_male] = parts[idx_male], parts[idx_control]

        new_name = "_".join(parts) + ext
        new_path = os.path.join(dirname, new_name)

        os.rename(file_path, new_path)
        print(f"Renamed:\n  {filename}\n→ {new_name}")
    except ValueError:
        print(f"⚠ '{filename}' enthält nicht beide Begriffe 'control' und 'male' – übersprungen.")
