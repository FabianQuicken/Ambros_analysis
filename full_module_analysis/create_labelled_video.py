import cv2
import numpy as np
import sys


def create_labelled_video(video_path, output_path, num_mice, num_mice_center):
    # ===============================
    # Einstellungen
    # ===============================



    scale_factor = 0.33   # ~1/9 Pixel -> ~10x kleinere Datei
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

        total_val = int(num_mice[frame_idx])
        center_val = int(num_mice_center[frame_idx])

        put_text_top_right(frame, f"Number of mice {total_val}")
        put_text_top_left(frame, f"Mice in Center {center_val}")

        frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(frame_small)

        if frame_idx % 200 == 0:
            print(f"   {frame_idx}/{total_frames} Frames processed...")

    cap.release()
    out.release()

    print("\n✅ Fertig! Video gespeichert als:")
    print(output_path)
