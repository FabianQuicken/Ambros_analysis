import os
import cv2
import numpy as np
from typing import Iterable, Dict, Any, Optional, List

def overlay_two_points_line_and_theta_segments(
    in_video_path: str,
    out_video_path: str,
    xy1: np.ndarray,
    xy2: np.ndarray,
    theta_segments: Optional[List[Dict[str, Any]]] = None,
    *,
    # drawing
    point_radius: int = 5,
    line_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_offset_px: int = 10,
    theta_fmt: str = "{:.1f}°",
    skip_if_nan: bool = True,
    codec: str = "mp4v",
    progress: bool = True,
    # trail / fade-out
    trail_len: int = 0,
    trail_alpha_max: float = 0.55,
    trail_alpha_min: float = 0.05,
    # theta control
    draw_theta: bool = True,
    theta_inclusive_end: bool = True,   # True: apply to t0<=i<=t1, False: t0<=i<t1
    theta_overlap: str = "last",        # "last" or "first"
):
    """
    Render an annotated video with two points + connecting line per frame and an optional
    theta text that is applied over frame intervals.

    Parameters
    ----------
    xy1, xy2 : (n_frames, 2)
        Pixel coordinates per frame.
    theta_segments : list[dict] or None
        Each dict must contain keys:
            - "frame t0": int  (start frame)
            - "frame t1": int  (end frame)
            - "theta": float
        Theta is displayed for frames in [t0, t1] if theta_inclusive_end=True,
        else in [t0, t1).
    theta_overlap : {"last","first"}
        If segments overlap:
            - "last": later segments overwrite earlier ones
            - "first": earlier segments keep priority

    Returns
    -------
    dict with metadata about processed frames.
    """
    xy1 = np.asarray(xy1)
    xy2 = np.asarray(xy2)

    if xy1.ndim != 2 or xy1.shape[1] != 2:
        raise ValueError(f"xy1 must be shape (n_frames, 2), got {xy1.shape}")
    if xy2.ndim != 2 or xy2.shape[1] != 2:
        raise ValueError(f"xy2 must be shape (n_frames, 2), got {xy2.shape}")
    if trail_len < 0:
        raise ValueError("trail_len must be >= 0")
    if not (0.0 <= trail_alpha_min <= 1.0 and 0.0 <= trail_alpha_max <= 1.0):
        raise ValueError("trail_alpha_min/max must be in [0, 1]")
    if trail_alpha_min > trail_alpha_max:
        raise ValueError("trail_alpha_min must be <= trail_alpha_max")
    if theta_overlap not in ("last", "first"):
        raise ValueError("theta_overlap must be 'last' or 'first'")

    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_used = min(n_frames_video, xy1.shape[0], xy2.shape[0])
    if n_frames_used <= 0:
        cap.release()
        raise ValueError("No frames to process (check video and array lengths).")

    # --- Build theta_per_frame from segments ---
    theta_per_frame = [None] * n_frames_used
    if draw_theta and theta_segments:
        def _get(seg, key):
            if key not in seg:
                raise KeyError(f"theta segment missing key '{key}': {seg}")
            return seg[key]

        # determine fill strategy
        def _assign_range(t0, t1, th):
            if theta_inclusive_end:
                end = t1
            else:
                end = t1 - 1  # last included frame

            if end < t0:
                return  # empty interval

            start = max(0, int(t0))
            end   = min(n_frames_used - 1, int(end))

            if theta_overlap == "last":
                for i in range(start, end + 1):
                    theta_per_frame[i] = th
            else:  # "first"
                for i in range(start, end + 1):
                    if theta_per_frame[i] is None:
                        theta_per_frame[i] = th

        for seg in theta_segments:
            t0 = int(_get(seg, "frame t0"))
            t1 = int(_get(seg, "frame t1"))
            th = _get(seg, "theta")

            if not np.isfinite(th):
                continue

            # if user accidentally gives reversed frames, normalize
            if t1 < t0:
                t0, t1 = t1, t0

            _assign_range(t0, t1, float(th))

    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_video_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    def _finite_xy(xy):
        return np.isfinite(xy[0]) and np.isfinite(xy[1])

    def _draw_line_points(img, p1, p2):
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)
        cv2.circle(img, (x1, y1), point_radius, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), point_radius, (0, 0, 255), -1)
        return x1, y1, x2, y2

    for i in range(n_frames_used):
        ok, frame = cap.read()
        if not ok:
            break

        # trail: past frames only
        if trail_len > 0 and i > 0:
            overlay = np.zeros_like(frame, dtype=np.uint8)
            start = max(0, i - trail_len)
            idxs = list(range(start, i))
            n = len(idxs)

            for k, j in enumerate(idxs):
                p1j, p2j = xy1[j], xy2[j]
                if skip_if_nan and (not (_finite_xy(p1j) and _finite_xy(p2j))):
                    continue

                t = 0.0 if n == 1 else (k / (n - 1))  # oldest->newest
                a = trail_alpha_min + t * (trail_alpha_max - trail_alpha_min)

                tmp = np.zeros_like(frame, dtype=np.uint8)
                _draw_line_points(tmp, p1j, p2j)
                overlay = cv2.addWeighted(overlay, 1.0, tmp, float(a), 0.0)

            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0.0)

        # current frame draw
        p1 = xy1[i]
        p2 = xy2[i]
        if skip_if_nan and (not (_finite_xy(p1) and _finite_xy(p2))):
            writer.write(frame)
            continue

        x1, y1, x2, y2 = _draw_line_points(frame, p1, p2)

        # theta from interval mapping
        if draw_theta and theta_per_frame is not None:
            th = theta_per_frame[i]
        else:
            th = None

        if th is not None:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx, dy = (x2 - x1), (y2 - y1)
            norm = np.hypot(dx, dy)

            if norm < 1e-6:
                tx, ty = int(round(mx + text_offset_px)), int(round(my - text_offset_px))
            else:
                ux, uy = (-dy / norm), (dx / norm)
                tx = int(round(mx + ux * text_offset_px))
                ty = int(round(my + uy * text_offset_px))

            text = theta_fmt.format(float(th))
            tx = int(np.clip(tx, 0, w - 1))
            ty = int(np.clip(ty, 0, h - 1))

            cv2.putText(frame, text, (tx, ty), font, font_scale, (0, 0, 0),
                        font_thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255),
                        font_thickness, cv2.LINE_AA)

        writer.write(frame)

        if progress and (i % 500 == 0):
            print(f"[overlay] frame {i}/{n_frames_used}")

    cap.release()
    writer.release()

    return {
        "n_frames_in_video": n_frames_video,
        "n_frames_used": n_frames_used,
        "out_video_path": out_video_path,
        "trail_len": trail_len,
        "draw_theta": draw_theta,
        "theta_segments": 0 if theta_segments is None else len(theta_segments),
        "theta_inclusive_end": theta_inclusive_end,
        "theta_overlap": theta_overlap,
    }


import os
import cv2
import numpy as np
from typing import Optional

def overlay_metric_at_centers(
    in_video_path: str,
    out_video_path: str,
    centers_xy: np.ndarray,
    metric: np.ndarray,
    *,
    unit: str = "",                     # e.g. "cm/s", "px", "%", "s"
    value_fmt: str = "{:.2f}",          # formatting for the metric number
    label: Optional[str] = None,        # e.g. "Speed" -> "Speed: 1.23 cm/s"
    # drawing / text
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_dx: int = 8,                   # offset relative to center
    text_dy: int = -8,
    text_color_bgr=(255, 255, 255),     # white
    outline_color_bgr=(0, 0, 0),        # black outline
    outline_extra: int = 2,
    # marker at center
    draw_center_marker: bool = True,
    marker_radius: int = 4,
    marker_color_bgr=(0, 255, 255),     # yellow-ish (BGR)
    # NaN handling
    skip_if_nan: bool = True,           # if center or metric NaN -> write raw frame
    # video
    codec: str = "mp4v",
    progress: bool = True,
    # optional trail / fade for last N frames (center marker only)
    trail_len: int = 0,
    trail_alpha_max: float = 0.55,
    trail_alpha_min: float = 0.05,
):
    """
    Render a video where per frame a metric value is written near the mouse center.

    Parameters
    ----------
    centers_xy : ndarray, shape (n_frames, 2)
        Per-frame center pixel coordinates (x, y).
    metric : ndarray, shape (n_frames,)
        Per-frame metric values.
    unit : str
        Unit appended to the value (ASCII recommended for OpenCV fonts).
    value_fmt : str
        Numeric format string, e.g. "{:.2f}", "{:.1f}".
    label : str or None
        Optional label prefix. If set, text becomes "label: value unit".
    text_dx, text_dy : int
        Pixel offset from center to place the text (avoid overlapping marker).
    trail_len : int
        If >0, draw faded center markers from the last `trail_len` frames.

    Returns
    -------
    dict with metadata.
    """
    centers_xy = np.asarray(centers_xy)
    metric = np.asarray(metric)

    if centers_xy.ndim != 2 or centers_xy.shape[1] != 2:
        raise ValueError(f"centers_xy must be shape (n_frames, 2), got {centers_xy.shape}")
    if metric.ndim != 1:
        raise ValueError(f"metric must be shape (n_frames,), got {metric.shape}")
    if trail_len < 0:
        raise ValueError("trail_len must be >= 0")
    if not (0.0 <= trail_alpha_min <= 1.0 and 0.0 <= trail_alpha_max <= 1.0):
        raise ValueError("trail_alpha_min/max must be in [0, 1]")
    if trail_alpha_min > trail_alpha_max:
        raise ValueError("trail_alpha_min must be <= trail_alpha_max")

    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_used = min(n_frames_video, centers_xy.shape[0], metric.shape[0])
    if n_frames_used <= 0:
        cap.release()
        raise ValueError("No frames to process (check video and array lengths).")

    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_video_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    def _finite_xy(xy):
        return np.isfinite(xy[0]) and np.isfinite(xy[1])

    def _draw_center(img, x, y, radius, color):
        cv2.circle(img, (x, y), radius, color, -1)

    for i in range(n_frames_used):
        ok, frame = cap.read()
        if not ok:
            break

        # --- optional trail of center markers (past frames only) ---
        if trail_len > 0 and i > 0 and draw_center_marker:
            overlay = np.zeros_like(frame, dtype=np.uint8)
            start = max(0, i - trail_len)
            idxs = list(range(start, i))
            n = len(idxs)

            for k, j in enumerate(idxs):
                c = centers_xy[j]
                if skip_if_nan and (not _finite_xy(c)):
                    continue

                t = 0.0 if n == 1 else (k / (n - 1))  # oldest->newest
                a = trail_alpha_min + t * (trail_alpha_max - trail_alpha_min)

                tmp = np.zeros_like(frame, dtype=np.uint8)
                xj, yj = int(round(float(c[0]))), int(round(float(c[1])))
                _draw_center(tmp, xj, yj, marker_radius, marker_color_bgr)
                overlay = cv2.addWeighted(overlay, 1.0, tmp, float(a), 0.0)

            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0.0)

        # --- current center + metric text ---
        c = centers_xy[i]
        m = metric[i]

        if skip_if_nan and (not _finite_xy(c) or not np.isfinite(m)):
            writer.write(frame)
            continue

        x, y = int(round(float(c[0]))), int(round(float(c[1])))

        if draw_center_marker:
            _draw_center(frame, x, y, marker_radius, marker_color_bgr)

        # text content
        val = value_fmt.format(float(m))
        if unit:
            val = f"{val} {unit}"
        if label:
            text = f"{label}: {val}"
        else:
            text = val

        tx = int(np.clip(x + text_dx, 0, w - 1))
        ty = int(np.clip(y + text_dy, 0, h - 1))

        # outline then text (readability)
        cv2.putText(frame, text, (tx, ty), font, font_scale, outline_color_bgr,
                    font_thickness + outline_extra, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, font_scale, text_color_bgr,
                    font_thickness, cv2.LINE_AA)

        writer.write(frame)

        if progress and (i % 500 == 0):
            print(f"[overlay_metric] frame {i}/{n_frames_used}")

    cap.release()
    writer.release()

    return {
        "n_frames_in_video": n_frames_video,
        "n_frames_used": n_frames_used,
        "out_video_path": out_video_path,
        "unit": unit,
        "label": label,
        "trail_len": trail_len,
        "draw_center_marker": draw_center_marker,
    }