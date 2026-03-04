from __future__ import annotations
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

def overlay_metric_at_centers_old(
    in_video_path: str,
    out_video_path: str,
    centers_xy: np.ndarray,
    metric: np.ndarray,
    *,
    unit: str = "",                     # e.g. "cm/s", "px", "%", "s"
    value_fmt: str = "{:.2f}",          # formatting for the metric number
    label: Optional[str] = None,        # e.g. "Speed" -> "Speed: 1.23 cm/s"

    # NEW: per-frame mask to color text (1->green, 0->red). If None -> white.
    color_mask: Optional[np.ndarray] = None,

    # drawing / text
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_dx: int = 8,                   # offset relative to center
    text_dy: int = -8,
    text_color_bgr=(255, 255, 255),     # default white (used if color_mask is None)
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
    color_mask : ndarray or None, shape (n_frames,)
        Optional per-frame mask (0/1). If provided:
        - 1 -> metric text in green
        - 0 -> metric text in red
        If None, text is drawn in `text_color_bgr` (default white).
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

    # --- NEW: validate / normalize color_mask ---
    if color_mask is not None:
        color_mask = np.asarray(color_mask)
        if color_mask.ndim != 1:
            raise ValueError(f"color_mask must be shape (n_frames,), got {color_mask.shape}")
        # allow longer, we will clip with n_frames_used later; but require at least 1
        if color_mask.shape[0] <= 0:
            raise ValueError("color_mask is empty.")
        # We'll interpret "1" as True, "0" as False. Any nonzero -> True.
        # If you strictly want only {0,1}, uncomment below:
        # uniq = np.unique(color_mask[~np.isnan(color_mask)] if np.issubdtype(color_mask.dtype, np.floating) else np.unique(color_mask))
        # if not set(uniq.tolist()).issubset({0, 1}):
        #     raise ValueError(f"color_mask must contain only 0/1, got values: {uniq}")

    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_used = min(n_frames_video, centers_xy.shape[0], metric.shape[0])
    if color_mask is not None:
        n_frames_used = min(n_frames_used, color_mask.shape[0])

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

    GREEN_BGR = (0, 255, 0)
    RED_BGR   = (0, 0, 255)

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
        text = f"{label}: {val}" if label else val

        tx = int(np.clip(x + text_dx, 0, w - 1))
        ty = int(np.clip(y + text_dy, 0, h - 1))

        # --- NEW: choose text color per frame ---
        if color_mask is None:
            cur_text_color = text_color_bgr  # default behavior (white)
        else:
            # Any nonzero -> green, else red
            cur_text_color = GREEN_BGR if bool(color_mask[i]) else RED_BGR

        # outline then text (readability)
        cv2.putText(frame, text, (tx, ty), font, font_scale, outline_color_bgr,
                    font_thickness + outline_extra, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, font_scale, cur_text_color,
                    font_thickness, cv2.LINE_AA)

        writer.write(frame)

        if progress and (i % 500 == 0):
            print(f"[overlay_metric] frame {i}/{n_frames_used}")

    writer.release()
    cap.release()

    return {
        "in_video_path": in_video_path,
        "out_video_path": out_video_path,
        "fps": fps,
        "width": w,
        "height": h,
        "n_frames_video": n_frames_video,
        "n_frames_used": n_frames_used,
        "trail_len": trail_len,
        "used_color_mask": color_mask is not None,
    }




import cv2
import numpy as np
from typing import Optional, Tuple, Union


def overlay_rolling_plot_at_centers(
    in_video_path: str,
    out_video_path: str,
    centers_xy: np.ndarray,
    metric: np.ndarray,
    *,
    fps: float,
    window_s: float = 2.0,                      # rolling window length in seconds
    plot_size: Tuple[int, int] = (250, 180),     # (w, h) in pixels
    plot_dx: int = 12,                          # plot offset relative to center
    plot_dy: int = -12,
    line_color_bgr: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 5,
    alpha: float = 1.0,                         # 1.0 = draw directly (fully opaque line)
    symmetric_zero: bool = True,                # best for acceleration
    scale_mode: str = "window",                 # "window" or "global"
    global_abs_max: Optional[float] = None,     # used if scale_mode == "global"
    min_abs_max: float = 1e-6,                  # avoid division by zero
    draw_zero_line: bool = False,               # optional reference
    zero_line_thickness: int = 1,
    zero_line_alpha: float = 0.35,
    nan_policy: str = "break",                  # "break" or "skip"
    codec: str = "mp4v",
) -> None:
    """
    Overlays a rolling-window line plot of a per-frame metric near each individual's center coordinate.

    Parameters
    ----------
    in_video_path, out_video_path : str
        Input/output video paths.
    centers_xy : ndarray
        Mouse centers per frame.
        Supported shapes:
        - (n_frames, 2) for a single individual
        - (n_ind, n_frames, 2) for multiple individuals
    metric : ndarray
        Metric per frame.
        Supported shapes:
        - (n_frames,) for a single individual
        - (n_ind, n_frames) for multiple individuals
    fps : float
        Frames per second of the video.
    window_s : float
        Rolling window duration in seconds.
    plot_size : (int, int)
        Plot width/height in pixels.
    plot_dx, plot_dy : int
        Offset of the plot's top-left corner relative to the center (cx, cy).
    line_color_bgr : tuple
        Line color in BGR (OpenCV default). White = (255,255,255).
    line_thickness : int
        Thickness of plot line.
    alpha : float
        Opacity for line drawing. If <1.0, blends an overlay; if 1.0 draws directly.
    symmetric_zero : bool
        If True, y scaling is symmetric around 0 and the 0-line is at mid-height.
        Recommended for acceleration (positive/negative changes).
    scale_mode : str
        "window" scales by the window's max abs value; "global" uses global_abs_max.
    global_abs_max : float or None
        Used if scale_mode == "global". If None and scale_mode is "global",
        falls back to abs max of entire metric array (per individual).
    draw_zero_line : bool
        If True, draws a faint horizontal 0-line inside the plot box.
    nan_policy : str
        "break" splits the polyline at NaNs; "skip" connects across (usually not recommended).
    codec : str
        FourCC codec (e.g. "mp4v", "avc1", "XVID").

    Notes
    -----
    - Transparent background: we do NOT draw a filled rectangle; only the line (and optional 0-line).
    - For performance, the plot is drawn with OpenCV primitives (no matplotlib).
    """

    # ----------------------------
    # Normalize inputs to (n_ind, n_frames, ...)
    # ----------------------------
    centers_xy = np.asarray(centers_xy)
    metric = np.asarray(metric, dtype=float)

    if centers_xy.ndim == 2 and centers_xy.shape[1] == 2:
        centers_xy = centers_xy[None, ...]  # (1, n_frames, 2)
    if metric.ndim == 1:
        metric = metric[None, ...]          # (1, n_frames)

    if centers_xy.ndim != 3 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must have shape (n_frames,2) or (n_ind,n_frames,2).")
    if metric.ndim != 2:
        raise ValueError("metric must have shape (n_frames,) or (n_ind,n_frames).")

    n_ind, n_frames_c, _ = centers_xy.shape
    n_ind_m, n_frames_m = metric.shape
    if n_ind_m != n_ind:
        raise ValueError(f"metric n_ind ({n_ind_m}) != centers n_ind ({n_ind}).")
    if n_frames_m != n_frames_c:
        raise ValueError(f"metric n_frames ({n_frames_m}) != centers n_frames ({n_frames_c}).")

    window_len = max(2, int(round(window_s * fps)))

    # Precompute per-individual global scaling if needed
    if scale_mode not in ("window", "global"):
        raise ValueError("scale_mode must be 'window' or 'global'.")

    if scale_mode == "global":
        if global_abs_max is not None:
            global_abs = np.full((n_ind,), float(global_abs_max), dtype=float)
        else:
            global_abs = np.nanmax(np.abs(metric), axis=1)
            global_abs = np.where(np.isfinite(global_abs), global_abs, 1.0)

    # ----------------------------
    # Video I/O
    # ----------------------------
    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {in_video_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps and fps and abs(in_fps - fps) > 1e-3:
        # not fatal, but often indicates mismatch in caller assumptions
        pass

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_video_path, fourcc, float(fps), (W, H))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open VideoWriter: {out_video_path}")

    plot_w, plot_h = map(int, plot_size)

    def _blend_line_overlay(frame: np.ndarray, overlay: np.ndarray, a: float) -> np.ndarray:
        # Blend only where overlay has non-zero pixels (to keep background untouched).
        mask = np.any(overlay != 0, axis=2)
        if not np.any(mask):
            return frame
        out = frame.copy()
        out[mask] = (frame[mask].astype(np.float32) * (1.0 - a) + overlay[mask].astype(np.float32) * a).astype(np.uint8)
        return out

    def _draw_polyline_in_box(
        base_frame: np.ndarray,
        box_x: int,
        box_y: int,
        vals: np.ndarray,
        *,
        abs_max: float,
        color: Tuple[int, int, int],
        thickness: int,
        do_zero_line: bool,
        sym_zero: bool,
    ) -> np.ndarray:
        """
        Draws the rolling plot into base_frame in-place (or via overlay blending if alpha<1).
        """
        # Clip box to frame bounds
        x1 = max(0, box_x)
        y1 = max(0, box_y)
        x2 = min(W, box_x + plot_w)
        y2 = min(H, box_y + plot_h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return base_frame

        # Box-local dimensions (in case of clipping)
        bw = x2 - x1
        bh = y2 - y1

        # Optional overlay for alpha blending
        if alpha < 1.0:
            overlay = np.zeros_like(base_frame)
            canvas = overlay
        else:
            canvas = base_frame

        # Optional 0-line (faint)
        if do_zero_line:
            y_zero = y1 + (bh // 2) if sym_zero else (y2 - 1)
            if alpha < 1.0:
                z_col = tuple(int(c * zero_line_alpha) for c in color)
                cv2.line(canvas, (x1, y_zero), (x2 - 1, y_zero), z_col, zero_line_thickness, cv2.LINE_AA)
            else:
                # simulate faintness by drawing a thinner line; still visible on dark backgrounds
                cv2.line(canvas, (x1, y_zero), (x2 - 1, y_zero), color, 1, cv2.LINE_AA)

        # Prepare points
        n = len(vals)
        if n < 2:
            return base_frame

        # x spans the box width
        xs = np.linspace(0, bw - 1, n)

        # y mapping
        abs_max = max(float(abs_max), float(min_abs_max))
        if sym_zero:
            # 0 at mid-height, +abs_max at top, -abs_max at bottom
            ys = (bh - 1) / 2.0 - (vals / abs_max) * ((bh - 1) / 2.0)
        else:
            # Normalize to [0..1] within [-abs_max..abs_max] anyway
            ys = (bh - 1) - ((vals + abs_max) / (2.0 * abs_max)) * (bh - 1)

        # Build segments (handle NaNs)
        pts = []
        segments = []

        for x, y, v in zip(xs, ys, vals):
            if not np.isfinite(v) or not np.isfinite(y):
                if nan_policy == "break" and len(pts) >= 2:
                    segments.append(np.array(pts, dtype=np.int32))
                pts = []
                continue

            xi = int(round(x1 + x))
            yi = int(round(y1 + np.clip(y, 0, bh - 1)))
            pts.append([xi, yi])

        if len(pts) >= 2:
            segments.append(np.array(pts, dtype=np.int32))

        # Draw segments
        for seg in segments:
            cv2.polylines(canvas, [seg], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Blend back if needed
        if alpha < 1.0:
            return _blend_line_overlay(base_frame, overlay, alpha)
        return base_frame

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx >= n_frames_c:
            # video longer than provided arrays
            writer.write(frame)
            frame_idx += 1
            continue

        # Draw for each individual
        for ind in range(n_ind):
            cx, cy = centers_xy[ind, frame_idx]
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue

            # Rolling window slice
            start = max(0, frame_idx - window_len + 1)
            vals = metric[ind, start:frame_idx + 1]

            # Scaling
            if scale_mode == "window":
                abs_max = np.nanmax(np.abs(vals))
                if not np.isfinite(abs_max):
                    continue
            else:
                abs_max = global_abs[ind]

            # Plot box anchored near center
            box_x = int(round(cx + plot_dx))
            box_y = int(round(cy + plot_dy - plot_h))  # default above center

            frame = _draw_polyline_in_box(
                frame,
                box_x,
                box_y,
                vals,
                abs_max=abs_max,
                color=line_color_bgr,
                thickness=line_thickness,
                do_zero_line=draw_zero_line,
                sym_zero=symmetric_zero,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

import os
import cv2
import numpy as np
from typing import Optional, Sequence, Union, List, Tuple

def overlay_metric_at_centers(
    in_video_path: str,
    out_video_path: str,
    centers_xy: np.ndarray,
    metric: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    unit: Union[str, Sequence[str]] = "",
    value_fmt: Union[str, Sequence[str]] = "{:.2f}",
    label: Union[Optional[str], Sequence[Optional[str]]] = None,

    # NEW: per-metric masks: (n_frames,) OR (n_frames, n_metrics) OR list of (n_frames,)
    color_mask: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,

    # drawing / text
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_dx: int = 8,
    text_dy: int = -8,
    line_spacing: int = 2,              # extra pixels between stacked metric lines
    text_color_bgr=(255, 255, 255),
    outline_color_bgr=(0, 0, 0),
    outline_extra: int = 2,

    # marker at center
    draw_center_marker: bool = True,
    marker_radius: int = 4,
    marker_color_bgr=(0, 255, 255),

    # NaN handling
    skip_if_nan: bool = True,           # if center NaN -> write raw frame
    skip_metric_if_nan: bool = True,    # if a specific metric NaN -> skip that line only

    # video
    codec: str = "mp4v",
    progress: bool = True,

    # optional trail / fade for last N frames (center marker only)
    trail_len: int = 0,
    trail_alpha_max: float = 0.55,
    trail_alpha_min: float = 0.05,

    # rectangle overlay (as polygon)
    draw_rect: bool = False,
    rect_xy: Optional[np.ndarray] = None,      # shape (4,2) or (n_frames,4,2)
    rect_fill_alpha: float = 0.20,             # transparency of fill
    rect_fill_bgr: tuple = (0, 255, 0),        # fill color (green)
    rect_outline_bgr: tuple = (0, 255, 0),     # outline color (green)
    rect_outline_thickness: int = 2,

    # circle overlay
    draw_circle: bool = False,
    circle_xy: Optional[np.ndarray] = None,   # shape (n_frames,2) or (2,) for constant
    circle_radius: int = 30,                  # pixels
    circle_fill_alpha: float = 0.20,          # transparency
    circle_fill_bgr: tuple = (0, 255, 0),     # fill color
    circle_outline_bgr: tuple = (0, 255, 0),  # outline color
    circle_outline_thickness: int = 2,
    ):
    """
    Render a video where per frame one or multiple metric values are written near the mouse center.
    Multiple metrics are stacked vertically (under each other).

    Parameters
    ----------
    centers_xy : ndarray, shape (n_frames, 2)
        Per-frame center pixel coordinates (x, y).

    metric :
        - ndarray (n_frames,) OR
        - ndarray (n_frames, n_metrics) OR
        - list/tuple of 1D ndarrays (each shape (n_frames,))

    unit, value_fmt, label :
        Either a single value (applied to all metrics) or a sequence with length n_metrics.

    color_mask :
        None, or:
        - ndarray (n_frames,) applied to all metrics
        - ndarray (n_frames, n_metrics)
        - list/tuple of 1D masks (each shape (n_frames,))

    skip_if_nan :
        If True, frames with non-finite center are passed through unchanged.
    skip_metric_if_nan :
        If True, individual metric lines with NaN/inf are skipped (others can still render).

    Returns
    -------
    dict with metadata.
    """
    centers_xy = np.asarray(centers_xy)

    if centers_xy.ndim != 2 or centers_xy.shape[1] != 2:
        raise ValueError(f"centers_xy must be shape (n_frames, 2), got {centers_xy.shape}")
    if trail_len < 0:
        raise ValueError("trail_len must be >= 0")
    if not (0.0 <= trail_alpha_min <= 1.0 and 0.0 <= trail_alpha_max <= 1.0):
        raise ValueError("trail_alpha_min/max must be in [0, 1]")
    if trail_alpha_min > trail_alpha_max:
        raise ValueError("trail_alpha_min must be <= trail_alpha_max")

    # ----------------------------
    # Normalize metrics to (n_frames, n_metrics)
    # ----------------------------
    def _to_2d_metrics(m) -> np.ndarray:
        if isinstance(m, (list, tuple)):
            arrs = [np.asarray(a) for a in m]
            if len(arrs) == 0:
                raise ValueError("metric list is empty.")
            for a in arrs:
                if a.ndim != 1:
                    raise ValueError(f"Each metric in list must be 1D (n_frames,), got {a.shape}")
            # stack as columns
            return np.vstack([a.reshape(-1, 1) for a in arrs]).reshape(len(arrs), -1, 1).transpose(1, 0, 2).reshape(-1, len(arrs))
        else:
            a = np.asarray(m)
            if a.ndim == 1:
                return a.reshape(-1, 1)
            if a.ndim == 2:
                return a
            raise ValueError(f"metric must be 1D, 2D, or list of 1D arrays. Got shape {a.shape}")

    metrics_2d = _to_2d_metrics(metric)
    if metrics_2d.shape[0] <= 0:
        raise ValueError("metric is empty.")
    n_metrics = metrics_2d.shape[1]

    def _broadcast_param(p, name: str) -> List:
        # p can be scalar or sequence length n_metrics
        if isinstance(p, (list, tuple)):
            if len(p) != n_metrics:
                raise ValueError(f"{name} must have length n_metrics={n_metrics}, got {len(p)}")
            return list(p)
        else:
            return [p for _ in range(n_metrics)]

    units = _broadcast_param(unit, "unit")
    fmts  = _broadcast_param(value_fmt, "value_fmt")
    labels = _broadcast_param(label, "label")

    # ----------------------------
    # Normalize color masks to list of (n_frames,) or None per metric
    # ----------------------------
    masks_per_metric: Optional[List[Optional[np.ndarray]]] = None
    if color_mask is None:
        masks_per_metric = None
    else:
        if isinstance(color_mask, (list, tuple)):
            if len(color_mask) != n_metrics:
                raise ValueError(f"color_mask list must have length n_metrics={n_metrics}, got {len(color_mask)}")
            masks_per_metric = []
            for cm in color_mask:
                cm = np.asarray(cm)
                if cm.ndim != 1:
                    raise ValueError(f"Each color_mask must be 1D (n_frames,), got {cm.shape}")
                masks_per_metric.append(cm)
        else:
            cm = np.asarray(color_mask)
            if cm.ndim == 1:
                # one mask for all metrics
                masks_per_metric = [cm for _ in range(n_metrics)]
            elif cm.ndim == 2:
                if cm.shape[1] != n_metrics:
                    raise ValueError(f"2D color_mask must be shape (n_frames, n_metrics={n_metrics}), got {cm.shape}")
                masks_per_metric = [cm[:, j] for j in range(n_metrics)]
            else:
                raise ValueError(f"color_mask must be 1D, 2D, or list of 1D arrays. Got {cm.shape}")

        for j, cmj in enumerate(masks_per_metric):
            if cmj.shape[0] <= 0:
                raise ValueError(f"color_mask[{j}] is empty.")
            
    # ----------------------------
    # Normalize rectangle
    # ----------------------------            
    if rect_xy is not None:
        rect_xy = np.asarray(rect_xy, dtype=float)

        if rect_xy.ndim == 2:
            # (4,2) -> constant rectangle for all frames
            if rect_xy.shape != (4, 2):
                raise ValueError(f"rect_xy must be (4,2) or (n_frames,4,2), got {rect_xy.shape}")
        elif rect_xy.ndim == 3:
            # (n_frames,4,2) -> per-frame rectangle
            if rect_xy.shape[1:] != (4, 2):
                raise ValueError(f"rect_xy must be (4,2) or (n_frames,4,2), got {rect_xy.shape}")
        else:
            raise ValueError(f"rect_xy must be (4,2) or (n_frames,4,2), got {rect_xy.shape}")

        if not (0.0 <= rect_fill_alpha <= 1.0):
            raise ValueError("rect_fill_alpha must be in [0,1]")
        
    # ----------------------------
    # Normalize circle
    # ----------------------------
    if circle_xy is not None:
        circle_xy = np.asarray(circle_xy, dtype=float)

        if circle_xy.ndim == 1:
            # constant center (2,)
            if circle_xy.shape != (2,):
                raise ValueError(f"circle_xy must be (2,) or (n_frames,2), got {circle_xy.shape}")
        elif circle_xy.ndim == 2:
            # per-frame centers (n_frames,2)
            if circle_xy.shape[1] != 2:
                raise ValueError(f"circle_xy must be (2,) or (n_frames,2), got {circle_xy.shape}")
        else:
            raise ValueError(f"circle_xy must be (2,) or (n_frames,2), got {circle_xy.shape}")

        if circle_radius <= 0:
            raise ValueError("circle_radius must be > 0")
        if not (0.0 <= circle_fill_alpha <= 1.0):
            raise ValueError("circle_fill_alpha must be in [0,1]")

    # ----------------------------
    # Video IO
    # ----------------------------
    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_used = min(n_frames_video, centers_xy.shape[0], metrics_2d.shape[0])
    if masks_per_metric is not None:
        for cmj in masks_per_metric:
            n_frames_used = min(n_frames_used, cmj.shape[0])
    if rect_xy is not None and rect_xy.ndim == 3:
        n_frames_used = min(n_frames_used, rect_xy.shape[0])
    if circle_xy is not None and circle_xy.ndim == 2:
        n_frames_used = min(n_frames_used, circle_xy.shape[0])

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

    def _draw_transparent_polygon(frame, pts_xy, fill_bgr, alpha, outline_bgr, thickness):
        """
        pts_xy: array-like shape (4,2) (or any Nx2 polygon)
        """
        pts = np.asarray(pts_xy, dtype=float)

        # Skip if any point is non-finite
        if not np.isfinite(pts).all():
            return frame

        # OpenCV wants int32 in shape (N,1,2)
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        # Fill on an overlay, then alpha blend
        if alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_i], fill_bgr)
            frame = cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0)

        # Outline on top (not blended)
        if thickness > 0:
            cv2.polylines(frame, [pts_i], isClosed=True, color=outline_bgr, thickness=thickness, lineType=cv2.LINE_AA)

        return frame
    
    def _draw_transparent_circle(frame, center_xy, radius, fill_bgr, alpha, outline_bgr, thickness):
        """
        center_xy: array-like shape (2,)
        Draws filled circle on overlay and alpha blends. Outline is drawn on top.
        """
        c = np.asarray(center_xy, dtype=float)
        if c.shape != (2,) or (not np.isfinite(c).all()):
            return frame

        cx = int(round(float(c[0])))
        cy = int(round(float(c[1])))

        # Optional: clip center into frame to avoid OpenCV edge weirdness
        # (Circle may still extend beyond borders, which is fine.)
        cx = int(np.clip(cx, 0, frame.shape[1] - 1))
        cy = int(np.clip(cy, 0, frame.shape[0] - 1))

        if alpha > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), int(radius), fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0)

        if thickness > 0:
            cv2.circle(frame, (cx, cy), int(radius), outline_bgr, thickness=int(thickness), lineType=cv2.LINE_AA)

        return frame

    def _finite_xy(xy):
        return np.isfinite(xy[0]) and np.isfinite(xy[1])

    def _draw_center(img, x, y, radius, color):
        cv2.circle(img, (x, y), radius, color, -1)

    GREEN_BGR = (0, 255, 0)
    RED_BGR   = (0, 0, 255)

    # Estimate a robust line height (depends on font/scale/thickness)
    # We'll compute per line anyway, but this gives stable stacking.
    _, base = cv2.getTextSize("Ag", font, font_scale, font_thickness)
    line_h = cv2.getTextSize("Ag", font, font_scale, font_thickness)[0][1] + base + line_spacing

    for i in range(n_frames_used):
        ok, frame = cap.read()
        if not ok:
            break
        # --- rectangle overlay ---
        if draw_rect and rect_xy is not None:
            if rect_xy.ndim == 2:
                pts = rect_xy          # constant 4x2
            else:
                pts = rect_xy[i]       # per-frame 4x2

        # --- circle overlay ---
        if draw_circle and circle_xy is not None:
            if circle_xy.ndim == 1:
                cxy = circle_xy        # constant (2,)
            else:
                cxy = circle_xy[i]     # per-frame (2,)

            frame = _draw_transparent_circle(
                frame,
                center_xy=cxy,
                radius=circle_radius,
                fill_bgr=circle_fill_bgr,
                alpha=circle_fill_alpha,
                outline_bgr=circle_outline_bgr,
                thickness=circle_outline_thickness,
            )

            frame = _draw_transparent_polygon(
                frame,
                pts_xy=pts,
                fill_bgr=rect_fill_bgr,
                alpha=rect_fill_alpha,
                outline_bgr=rect_outline_bgr,
                thickness=rect_outline_thickness,
            )
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

        # --- current center + stacked metric text ---
        c = centers_xy[i]
        if skip_if_nan and (not _finite_xy(c)):
            writer.write(frame)
            continue

        x, y = int(round(float(c[0]))), int(round(float(c[1])))

        if draw_center_marker:
            _draw_center(frame, x, y, marker_radius, marker_color_bgr)

        tx0 = int(np.clip(x + text_dx, 0, w - 1))
        ty0 = int(np.clip(y + text_dy, 0, h - 1))

        # Draw each metric on its own line
        any_drawn = False
        for j in range(n_metrics):
            m = metrics_2d[i, j]

            if skip_metric_if_nan and (not np.isfinite(m)):
                continue

            # Format text
            val = fmts[j].format(float(m)) if np.isfinite(m) else "nan"
            if units[j]:
                val = f"{val} {units[j]}"
            text = f"{labels[j]}: {val}" if labels[j] else val

            # Line position (stacked)
            ty = ty0 + j * line_h
            # If we'd go off-screen at bottom, we clamp; still keeps relative order.
            ty = int(np.clip(ty, 0, h - 1))

            # Per-metric color selection
            if masks_per_metric is None:
                cur_text_color = text_color_bgr
            else:
                cur_text_color = GREEN_BGR if bool(masks_per_metric[j][i]) else RED_BGR

            # outline then text
            cv2.putText(frame, text, (tx0, ty), font, font_scale, outline_color_bgr,
                        font_thickness + outline_extra, cv2.LINE_AA)
            cv2.putText(frame, text, (tx0, ty), font, font_scale, cur_text_color,
                        font_thickness, cv2.LINE_AA)

            any_drawn = True

        # If all metrics were NaN and skip_metric_if_nan=True, we still write the frame (with marker/trail)
        writer.write(frame)

        if progress and (i % 500 == 0):
            print(f"[overlay_metric] frame {i}/{n_frames_used}")

    writer.release()
    cap.release()

    return {
        "in_video_path": in_video_path,
        "out_video_path": out_video_path,
        "fps": fps,
        "width": w,
        "height": h,
        "n_frames_video": n_frames_video,
        "n_frames_used": n_frames_used,
        "n_metrics": n_metrics,
        "trail_len": trail_len,
        "used_color_mask": masks_per_metric is not None,
    }