#!/usr/bin/env python3
"""
Verschiebt alle .avi-Dateien in 'done/', wenn im gleichen Ordner
mindestens eine .csv existiert, deren Name den AVI-Stem enthält.
Direkt in VS Code ausführbar. Ohne Argument öffnet sich ein Ordner-Dialog.
"""

from pathlib import Path
import shutil
import sys

def pick_folder_via_dialog() -> Path | None:
    """Einfache Ordnerauswahl per Dialog (tkinter ist in der Stdlib)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Ordner mit AVI/CSV wählen")
        root.update()
        root.destroy()
        return Path(path) if path else None
    except Exception as e:
        print(f"Ordnerdialog nicht verfügbar ({e}).")
        return None

def safe_move(src: Path, dst_dir: Path) -> Path:
    """Verschiebe src nach dst_dir; bei Namenskollision Zähler anhängen."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        shutil.move(str(src), str(target))
        return target
    stem, suffix = src.stem, src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        i += 1

def main(folder: Path):
    folder = folder.resolve()
    if not folder.is_dir():
        print(f"❌ Kein gültiger Ordner: {folder}")
        sys.exit(1)

    done_dir = folder / "done"

    # Alle CSV-Dateinamen (lowercase) für case-insensitive Matching
    csv_names = [p.name.lower() for p in folder.glob("*.csv")]

    # .avi und .AVI berücksichtigen (nicht rekursiv)
    avi_files = list(folder.glob("*.avi")) + list(folder.glob("*.AVI"))

    if not avi_files:
        print("ℹ️ Keine AVI-Dateien gefunden.")
        return

    moved = skipped = 0
    for avi in avi_files:
        stem = avi.stem.lower()
        has_match = any(stem in csv_name for csv_name in csv_names)
        if has_match:
            new_path = safe_move(avi, done_dir)
            print(f"✅ Verschoben: {avi.name}  →  {new_path.relative_to(folder)}")
            moved += 1
        else:
            print(f"⏭️  Keine passende CSV gefunden, bleibt: {avi.name}")
            skipped += 1

    print(f"\nFertig. Verschoben: {moved}, verblieben: {skipped}. Zielordner: {done_dir}")

if __name__ == "__main__":
    # 1) Pfad aus Argument, wenn vorhanden
    if len(sys.argv) >= 2:
        main(Path(sys.argv[1]))
    else:
        # 2) Sonst Ordner-Dialog; falls nicht möglich, Konsoleingabe
        chosen = pick_folder_via_dialog()
        if chosen is None:
            try:
                entered = input("Pfad zum Ordner eingeben: ").strip('"').strip("'").strip()
                chosen = Path(entered) if entered else None
            except KeyboardInterrupt:
                chosen = None
        if chosen:
            main(chosen)
        else:
            print("Abgebrochen – kein Ordner gewählt.")
