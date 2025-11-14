import os

# === Einstellungen ===
folder = r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_33_47_48\side1"   # Zielordner
old_part = "_omm12prop_"                 # Text, der ersetzt werden soll
new_part = "_ommpgol_"                 # Neuer Text

# === Dateien umbenennen ===
for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)
    if os.path.isfile(old_path) and old_part in filename:
        new_filename = filename.replace(old_part, new_part)
        new_path = os.path.join(folder, new_filename)
        os.rename(old_path, new_path)
        print(f"✔️ {filename} → {new_filename}")