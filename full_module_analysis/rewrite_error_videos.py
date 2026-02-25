import os
import cv2
import numpy as np

def rewrite_error_video(in_video_path, out_video_path, codec="mp4v"):
    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_video_path}")
    
    for i in range(n_frames_video):
        ok, frame = cap.read()
        if not ok:
            break

        writer.write(frame)

    writer.release()
    cap.release()

rewrite_error_video(in_video_path=r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_33_47_48\error_videos\2025_11_11_14_28_49_mice_ommpgol_females_home_unfamiliar_top1_40439818.avi",
                    out_video_path=r"Z:\n2023_odor_related_behavior\2025_omm_mice\Clavel_paradigm\ommpgol\females_33_47_48\error_videos\2025_11_11_14_28_49_mice_ommpgol_females_home_unfamiliar_top1_40439818.mp4")