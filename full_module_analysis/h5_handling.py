import h5py
from dataclasses import dataclass
import numpy as np

@dataclass
class ModuleVariables:
    exp_duration_frames: np.ndarray
    strecke_über_zeit: np.ndarray
    maus_in_modul_über_zeit: np.ndarray
    maus_an_food_percent: float
    strecke_pixel_frame: float
    visits_per_hour: float
    mean_visit_time: float
    zeit_in_modul_prozent: float
    nose_coords_x_y: tuple
    date: str
    is_stimulus_module: bool
    start_time: str
    end_time: str

def save_modulevariables_to_h5(file_path, data):
    """
    Speichert eine Instanz der Klasse ModuleVariables als komprimierte HDF5-Datei (.h5).

    Die Funktion legt alle großen numerischen Arrays als Datasets ab (mit GZIP-Kompression),
    während einfache Metadaten wie Strings, Floats und Booleans als HDF5-Attribute gespeichert werden.
    Dies ermöglicht eine effiziente und strukturierte Persistierung von Verhaltensdaten
    für spätere Analysen und Visualisierungen.

    Args:
        file_path (str): Vollständiger Pfad zur Zieldatei inklusive Dateiname (.h5).
        data (ModuleVariables): Die zu speichernden Analyseergebnisse eines Moduls.

    Raises:
        OSError: Falls die Datei nicht geschrieben werden kann oder ungültige Daten enthalten sind.
    """
    with h5py.File(file_path, 'w') as f:
        # Arrays
        f.create_dataset("strecke_über_zeit", data=data.strecke_über_zeit, compression="gzip")
        f.create_dataset("maus_in_modul_über_zeit", data=data.maus_in_modul_über_zeit, compression="gzip")
        f.create_dataset("nose_coords_x", data=data.nose_coords_x_y[0], compression="gzip")
        f.create_dataset("nose_coords_y", data=data.nose_coords_x_y[1], compression="gzip")

        # Metadaten als Attribute
        f.attrs["exp_duration_frames"] = data.exp_duration_frames
        f.attrs["maus_an_food_percent"] = data.maus_an_food_percent
        f.attrs["strecke_pixel_frame"] = data.strecke_pixel_frame
        f.attrs["visits_per_hour"] = data.visits_per_hour
        f.attrs["mean_visit_time"] = data.mean_visit_time
        f.attrs["zeit_in_modul_prozent"] = data.zeit_in_modul_prozent
        f.attrs["date"] = data.date
        f.attrs["is_stimulus_module"] = int(data.is_stimulus_module)  # bools als int
        f.attrs["start_time"] = data.start_time
        f.attrs["end_time"] = data.end_time

def load_modulevariables_from_h5(file_path: str) -> ModuleVariables:
    """
    Lädt eine zuvor gespeicherte HDF5-Datei (.h5) und rekonstruiert ein ModuleVariables-Objekt.

    Diese Funktion liest sowohl die gespeicherten Arrays (z.B. Koordinaten, Zeitreihen)
    als auch alle zugehörigen Metadaten (z.B. Datumsinformationen, Stimulusstatus) ein.
    Sie wird typischerweise verwendet, um Analyseergebnisse ohne Neuberechnung
    erneut zu visualisieren oder weiterzuverarbeiten.

    Args:
        file_path (str): Pfad zur .h5-Datei, die ein gespeichertes Modul enthält.

    Returns:
        ModuleVariables: Ein vollständig rekonstruiertes Dataclass-Objekt mit allen Feldern.

    Raises:
        OSError: Falls die Datei nicht geöffnet oder gelesen werden kann.
        KeyError: Wenn erwartete Felder/Datasets fehlen.
    """
    with h5py.File(file_path, 'r') as f:
        return ModuleVariables(
            exp_duration_frames=int(f.attrs["exp_duration_frames"]),
            strecke_über_zeit=f["strecke_über_zeit"][:],
            maus_in_modul_über_zeit=f["maus_in_modul_über_zeit"][:],
            nose_coords_x_y=(f["nose_coords_x"][:], f["nose_coords_y"][:]),
            maus_an_food_percent=float(f.attrs["maus_an_food_percent"]),
            strecke_pixel_frame=float(f.attrs["strecke_pixel_frame"]),
            visits_per_hour=float(f.attrs["visits_per_hour"]),
            mean_visit_time=float(f.attrs["mean_visit_time"]),
            zeit_in_modul_prozent=float(f.attrs["zeit_in_modul_prozent"]),
            date=str(f.attrs["date"]),
            is_stimulus_module=bool(f.attrs["is_stimulus_module"]),
            start_time=str(f.attrs["start_time"]),
            end_time=str(f.attrs["end_time"])
        )