import pretty_midi

# Erstelle ein neues MIDI-Objekt
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

# Noteninformationen (Tonhöhe, Startzeit, Endzeit in Sekunden)
notes = [
    ('D4', 0.0, 1.2),   # Viertel
    ('F4', 1.2, 1.8),   # Achtel
    ('G#4', 1.8, 2.4),  # Achtel
    ('A4', 2.4, 3.6),   # Viertel
    ('F4', 3.6, 4.8),   # Viertel
    ('D4', 4.8, 5.4),   # Achtel
    ('C#4', 5.4, 6.0),  # Achtel
    ('D4', 6.0, 7.2),   # Viertel
]

# Füge die Noten dem Instrument hinzu
for pitch_name, start, end in notes:
    pitch = pretty_midi.note_name_to_number(pitch_name)
    note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
    instrument.notes.append(note)

# Füge das Instrument der MIDI-Datei hinzu
midi.instruments.append(instrument)

# Speichere die Datei
midi.write("strahd_leitmotiv.mid")
print("MIDI gespeichert als 'strahd_leitmotiv.mid'")