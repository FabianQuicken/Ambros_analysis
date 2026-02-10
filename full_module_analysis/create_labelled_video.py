import cv2
import numpy as np
import sys

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class OverlayStyle:
    point_radius: int = 6
    point_thickness: int = -1  # -1 = filled
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 2
    line_spacing: int = 22
    pad: int = 10
    box_alpha: float = 0.35
    box_offset: Tuple[int, int] = (25, -25)  # (dx, dy) from center
    box_max_width: int = 340
    box_min_width: int = 220


def _to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _default_palette(n: int) -> List[Tuple[int, int, int]]:
    base = [
        (66, 133, 244),  # blue
        (234, 67, 53),   # red
        (251, 188, 5),   # yellow
        (52, 168, 83),   # green
        (171, 71, 188),  # purple
        (0, 172, 193),   # cyan
        (255, 112, 67),  # orange
        (124, 179, 66),  # lime
    ]
    return [base[i % len(base)] for i in range(n)]


def stack_centers_from_bodyparts(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: Optional[np.ndarray] = None,
    likelihood_thresh: float = 0.3,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute per-individual center coordinates as mean over bodyparts.

    Expected shapes
    ---------------
    x, y:           (n_ind, n_frames, n_bodyparts)
    likelihood:     (n_ind, n_frames, n_bodyparts)  (optional)

    Returns
    -------
    centers: (n_ind, n_frames, 2)  with [x_center, y_center]
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("x and y must have shape (n_ind, n_frames, n_bodyparts) and match exactly.")

    if likelihood is None:
        # simple mean ignoring NaNs
        cx = np.nanmean(x, axis=2)
        cy = np.nanmean(y, axis=2)
        return np.stack([cx, cy], axis=2)

    l = np.asarray(likelihood, dtype=float)
    if l.shape != x.shape:
        raise ValueError("likelihood must have the same shape as x/y: (n_ind, n_frames, n_bodyparts)")

    # mask low-confidence points
    valid = l >= likelihood_thresh
    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    if not weighted:
        cx = np.nanmean(x_masked, axis=2)
        cy = np.nanmean(y_masked, axis=2)
        return np.stack([cx, cy], axis=2)

    # weighted mean (avoid nan propagation)
    w = np.where(valid, l, 0.0)
    wx = np.where(np.isfinite(x), x, 0.0) * w
    wy = np.where(np.isfinite(y), y, 0.0) * w

    wsum = np.sum(w, axis=2)
    # where wsum==0 -> nan
    cx = np.where(wsum > 0, np.sum(wx, axis=2) / wsum, np.nan)
    cy = np.where(wsum > 0, np.sum(wy, axis=2) / wsum, np.nan)

    return np.stack([cx, cy], axis=2)


def overlay_metrics_on_video_arrays(
    video_in_path: str,
    video_out_path: str,
    individuals: List[str],
    centers_nif2: np.ndarray,                 # (n_ind, n_frames, 2)
    metrics_nif: Dict[str, np.ndarray],       # each (n_ind, n_frames) OR (n_frames,)
    metric_formats: Optional[Dict[str, str]] = None,
    fps_out: Optional[float] = None,
    colors_rgb: Optional[Dict[str, Tuple[int, int, int]]] = None,
    style: OverlayStyle = OverlayStyle(),
    codec_fourcc: Optional[str] = None,
    draw_trails: bool = False,
    trail_len: int = 25,
    trail_thickness: int = 2,
) -> None:
    """
    Annotate video with per-individual center points and arbitrary per-frame metrics.

    Your native format is supported:
    - centers: (n_ind, n_frames, 2)
    - metrics: (n_ind, n_frames) or (n_frames,) for global metrics

    Output video is written to `video_out_path`.
    """

    if metric_formats is None:
        metric_formats = {}
    default_fmt = "{:.3g}"

    centers = np.asarray(centers_nif2, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 2:
        raise ValueError("centers_nif2 must have shape (n_ind, n_frames, 2).")

    n_ind, n_frames = centers.shape[0], centers.shape[1]
    if len(individuals) != n_ind:
        raise ValueError(f"len(individuals)={len(individuals)} does not match centers n_ind={n_ind}.")

    # Open video
    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_out if fps_out is not None else (fps_in if fps_in and fps_in > 0 else 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ext = os.path.splitext(video_out_path)[1].lower()
    if codec_fourcc is None:
        codec_fourcc = "mp4v" if ext == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)

    out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open VideoWriter for {video_out_path}. "
            f"Try codec_fourcc='mp4v' (mp4) or 'XVID' (avi)."
        )

    # colors
    if colors_rgb is None:
        pal = _default_palette(n_ind)
        colors_rgb = {individuals[i]: pal[i] for i in range(n_ind)}

    trails = {ind: [] for ind in individuals}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx >= n_frames:
            # Safety: stop if your arrays are shorter than the video
            break

        overlay = frame.copy()

        for i, ind in enumerate(individuals):
            rgb = colors_rgb[ind]
            color = _to_bgr(rgb)

            x, y = centers[i, frame_idx, 0], centers[i, frame_idx, 1]
            if np.isnan(x) or np.isnan(y):
                continue

            xi, yi = int(round(float(x))), int(round(float(y)))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            # trails
            if draw_trails:
                trails[ind].append((xi, yi))
                if len(trails[ind]) > trail_len:
                    trails[ind] = trails[ind][-trail_len:]
                if len(trails[ind]) >= 2:
                    cv2.polylines(
                        overlay,
                        [np.array(trails[ind], dtype=np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=trail_thickness,
                    )

            # point
            cv2.circle(
                overlay,
                (xi, yi),
                style.point_radius,
                color,
                style.point_thickness,
                lineType=cv2.LINE_AA,
            )

            # metric lines (per ind)
            lines = []
            for name, arr in metrics_nif.items():
                arr = np.asarray(arr)

                if arr.ndim == 1:
                    v = arr[frame_idx]
                elif arr.ndim == 2:
                    # (n_ind, n_frames)
                    v = arr[i, frame_idx]
                else:
                    raise ValueError(f"Metric '{name}' has ndim={arr.ndim}. Expected 1 or 2.")

                fmt = metric_formats.get(name, default_fmt)
                if isinstance(v, (float, np.floating, int, np.integer)):
                    if isinstance(v, (float, np.floating)) and np.isnan(v):
                        v_str = "nan"
                    else:
                        v_str = fmt.format(v)
                else:
                    v_str = str(v)

                lines.append(f"{name}: {v_str}")

            lines.append(f"x: {xi}, y: {yi}")

            # measure box
            max_tw = 0
            for txt in lines:
                (tw, _), _ = cv2.getTextSize(txt, style.font, style.font_scale, style.font_thickness)
                max_tw = max(max_tw, tw)

            box_w = int(np.clip(max_tw + 2 * style.pad, style.box_min_width, style.box_max_width))
            box_h = int(len(lines) * style.line_spacing + style.pad)

            dx, dy = style.box_offset
            bx = int(np.clip(xi + dx, 0, w - box_w))
            by = int(np.clip(yi + dy - box_h, 0, h - box_h))

            # box
            cv2.rectangle(
                overlay,
                (bx, by),
                (bx + box_w, by + box_h),
                (255, 255, 255),
                thickness=-1,
            )

            # text
            tx = bx + style.pad
            ty = by + style.pad + style.line_spacing - 6
            for txt in lines:
                cv2.putText(
                    overlay,
                    txt,
                    (tx, ty),
                    style.font,
                    style.font_scale,
                    color,
                    style.font_thickness,
                    lineType=cv2.LINE_AA,
                )
                ty += style.line_spacing

        # blend
        cv2.addWeighted(overlay, style.box_alpha, frame, 1.0 - style.box_alpha, 0, frame)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()



def create_labelled_video(video_path, output_path, metric1, text1, metric2, text2):
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