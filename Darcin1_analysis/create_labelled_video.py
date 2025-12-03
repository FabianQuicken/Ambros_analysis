import cv2
import numpy as np
import sys


def create_labelled_video(video_path, output_path, metric1, text1, metric2, text2):
    # ===============================
    # Einstellungen
    # ===============================



    scale_factor = 1   # ~1/9 Pixel -> ~10x kleinere Datei
    PADDING = 12
    TEXT_COLOR = (255,255,255)
    BOX_COLOR = (0,0,0)
    BOX_INNER_PAD = 6
    # ===============================


    # -------- Text Overlay Funktionen --------
    def _compute_font_params(frame_width):
        font_scale = max(0.5, frame_width / 1280 * 1.2)
        thickness = max(1, int(round(font_scale + 0.6)))
        return font_scale, thickness

    def put_label(frame, text, x, y):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale, thickness = _compute_font_params(w)
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(frame, (x-BOX_INNER_PAD, y-th-BOX_INNER_PAD),
                    (x+tw+BOX_INNER_PAD, y+base+BOX_INNER_PAD),
                    BOX_COLOR, cv2.FILLED)

        cv2.putText(frame, text, (x, y),
                    font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    def put_text_top_right(frame, text, pad=PADDING):
        h, w = frame.shape[:2]
        font_scale, thickness = _compute_font_params(w)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        x = w - pad - tw
        y = pad + th
        put_label(frame, text, x, y)

    def put_text_top_left(frame, text, pad=PADDING):
        h, w = frame.shape[:2]
        font_scale, thickness = _compute_font_params(w)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        x = pad
        y = pad + th
        put_label(frame, text, x, y)


    # -------- Video vorbereiten --------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit("Cannot open Video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30


    # Erste Frame lesen zum Dimensionscheck
    ok, frame = cap.read()
    if not ok:
        sys.exit("no frames")

    orig_h, orig_w = frame.shape[:2]
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    # zurück zum Frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"   Start Export...")
    print(f"   Resolution: {orig_w}x{orig_h}")
    print(f"   New Resolution:     {new_w}x{new_h}")
    print(f"   Output: {output_path}\n")

    # -------- Hauptloop (ohne Anzeige) --------
    for frame_idx in range(total_frames):

        ok, frame = cap.read()
        if not ok:
            break

        total_val = int(metric1[frame_idx])
        center_val = int(metric2[frame_idx])

        put_text_top_right(frame, f"{text1} {total_val}")
        put_text_top_left(frame, f"{text2} {center_val}")

        frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(frame_small)

        if frame_idx % 200 == 0:
            print(f"   {frame_idx}/{total_frames} Frames processed...")

    cap.release()
    out.release()

    print("\n✅ Fertig! Video gespeichert als:")
    print(output_path)


def create_labelled_video_modular(
    video_path,
    output_path,
    metrics,
    *,
    scale_factor=0.33,
    pad=12,
    row_gap=6,
    text_color=(255, 255, 255),
    box_color=(0, 0, 0),
    box_inner_pad=6,
    progress_every=200,
    max_frames=None,
):
    """
    Zeichnet beliebig viele Metriken links-oben untereinander ins Video.

    Parameters
    ----------
    video_path : str
    output_path : str
    metrics : list
        Liste von Items im Format:
          - ("Label", series)  -> Default-Format: int(series[t])
          - ("Label", series, fmt)  -> fmt kann z.B. eine format()-taugliche str oder callable sein,
            z.B. "{:.2f}", lambda v: f"{v:.1f} cm", etc.
        'series' muss indexierbar sein (len==n_frames) und numerisch.
    scale_factor : float
        Resizing (z.B. 0.33 für ~1/9 Pixel).
    pad : int
        Außenabstand zur oberen linken Ecke.
    row_gap : int
        Vertikaler Abstand zwischen den Textzeilen (zusätzlich zur Texthöhe).
    text_color : tuple[int,int,int]
        BGR-Farbe des Texts.
    box_color : tuple[int,int,int]
        BGR-Farbe der Hintergrundbox.
    box_inner_pad : int
        Innenabstand der Hintergrundbox um den Text.
    progress_every : int
        Alle `progress_every` Frames Fortschritt ausgeben. Setze None/0 für kein Logging.
    max_frames : int | None
        Optional: beschränkt die Anzahl verarbeiteter Frames (Debug/Tests).

    Returns
    -------
    None
    """

    # -------- Hilfsfunktionen --------
    def _compute_font_params(frame_width):
        font_scale = max(0.5, frame_width / 1280 * 1.2)
        thickness = max(1, int(round(font_scale + 0.6)))
        return font_scale, thickness

    def _line_metrics(font, font_scale, thickness):
        # Einmalige "Referenz"-Zeichenhöhe und Baseline für konsistente Zeilenabstände
        (tw, th), base = cv2.getTextSize("Ag", font, font_scale, thickness)
        return th, base

    def _format_value(v, fmt):
        if callable(fmt):
            return fmt(v)
        elif isinstance(fmt, str):
            try:
                return fmt.format(v)
            except Exception:
                return str(v)
        else:
            # Default: integer falls ganzzahlig, sonst float mit 2 Nachkommastellen
            if isinstance(v, (int, np.integer)):
                return f"{int(v)}"
            try:
                iv = int(v)
                if float(v) == iv:
                    return f"{iv}"
            except Exception:
                pass
            return f"{float(v):.2f}"

    def _put_label(frame, text, x, y, font, font_scale, thickness):
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x - box_inner_pad, y - th - box_inner_pad),
                      (x + tw + box_inner_pad, y + base + box_inner_pad),
                      box_color, cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # -------- Eingaben prüfen/vereinheitlichen --------
    norm_metrics = []
    for m in metrics:
        if len(m) == 2:
            label, series = m
            fmt = None
        elif len(m) == 3:
            label, series, fmt = m
        else:
            raise ValueError("Jedes metrics-Item muss ('Label', series) oder ('Label', series, fmt) sein.")
        norm_metrics.append((str(label), series, fmt))

    if len(norm_metrics) == 0:
        raise ValueError("metrics darf nicht leer sein.")

    # -------- Video öffnen --------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Kann Video nicht öffnen: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Erste Frame lesen (für Dimensionen & Fontparam)
    ok, frame0 = cap.read()
    if not ok:
        sys.exit("Keine Frames im Video.")

    orig_h, orig_w = frame0.shape[:2]
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    # Zurückspulen
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Font-Settings (aus Breite ableiten)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = _compute_font_params(orig_w)
    th_ref, base_ref = _line_metrics(font, font_scale, thickness)  # Referenzhöhe

    # Länge der Serien validieren
    for label, series, fmt in norm_metrics:
        if len(series) < total_frames:
            raise ValueError(f"Serie für '{label}' hat weniger Einträge ({len(series)}) als Video Frames ({total_frames}).")

    print("   Start Export...")
    print(f"   Resolution: {orig_w}x{orig_h}")
    print(f"   New Resolution:     {new_w}x{new_h}")
    print(f"   Output: {output_path}\n")

    # -------- Hauptloop --------
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # Zeilenweise links-oben schreiben
        # y-Position: pad + (row_index+1)*th_ref + row_index*row_gap (Baseline-Logik)
        for row_idx, (label, series, fmt) in enumerate(norm_metrics):
            val = series[frame_idx]
            text = f"{label} {_format_value(val, fmt)}"

            x = pad
            y = pad + (row_idx + 1) * th_ref + row_idx * row_gap  # Baseline
            _put_label(frame, text, x, y, font, font_scale, thickness)

        # Resize & schreiben
        frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(frame_small)

        if progress_every and frame_idx % progress_every == 0:
            print(f"   {frame_idx}/{total_frames} Frames processed...")

    cap.release()
    out.release()
    print("\n✅ Fertig! Video gespeichert als:")
    print(output_path)