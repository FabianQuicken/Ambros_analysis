
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# Configuration
# --------------------------------------------------

folder = Path(r"Z:\n2023_odor_related_behavior\2026_sick_behavior\kamera-merge-test")

camera_names = ["stim_top1_40439817", "home_top2_40405188"]
output = folder / "combined_rotated.avi"

# --------------------------------------------------
# Read metadata
# --------------------------------------------------

def start_time(path):
    return datetime.strptime(
        "_".join(path.stem.split("_")[:6]),
        "%Y_%m_%d_%H_%M_%S"
    )


def load_videos(camera_name):
    files = sorted(
        (p for p in folder.glob("*.avi")
         if camera_name in p.stem and p != output),
        key=start_time
    )

    if not files:
        raise ValueError(f"No videos found: {camera_name}")

    return files


videos = [load_videos(name) for name in camera_names]

# Read reference video settings
cap = cv2.VideoCapture(str(videos[0][0]))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
cap.release()

if fps <= 0:
    raise ValueError("Invalid FPS")

# Global starting timestamp
global_start = min(start_time(v[0]) for v in videos)

# Build frame-indexed video timelines
timelines = []

for camera_videos in videos:
    timeline = []

    for path in camera_videos:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path.name}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if abs(video_fps - fps) > 0.01:
            raise ValueError(f"FPS mismatch: {path.name}")

        if (video_width, video_height) != (width, height):
            raise ValueError(f"Resolution mismatch: {path.name}")

        start_frame = round(
            (start_time(path) - global_start).total_seconds() * fps
        )

        timeline.append({
            "path": path,
            "start": start_frame,
            "end": start_frame + n_frames,
        })

    timelines.append(timeline)

# Last frame in either camera timeline
total_frames = max(
    video["end"]
    for timeline in timelines
    for video in timeline
)

# --------------------------------------------------
# Synchronized video reader
# --------------------------------------------------

class CameraReader:
    def __init__(self, timeline):
        self.timeline = timeline
        self.index = -1
        self.cap = None
        self.frame_index = -1
        self.last_frame = np.zeros(
            (height, width, 3), dtype=np.uint8
        )

    def get_frame(self, global_frame):
        # Switch to next recording when its timestamp is reached
        while (
            self.index + 1 < len(self.timeline)
            and self.timeline[self.index + 1]["start"] <= global_frame
        ):
            if self.cap is not None:
                self.cap.release()

            self.index += 1
            self.cap = cv2.VideoCapture(
                str(self.timeline[self.index]["path"])
            )
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open recording")

            self.frame_index = -1

        if self.index < 0:
            return self.last_frame

        video = self.timeline[self.index]
        local_frame = global_frame - video["start"]

        # Recording gap: retain last available frame
        if global_frame >= video["end"]:
            return self.last_frame

        # Seek only if necessary
        if local_frame != self.frame_index + 1:
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES, local_frame
            )

        ret, frame = self.cap.read()

        if ret:
            self.last_frame = frame
            self.frame_index = local_frame

        return self.last_frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------
# Write combined video
# --------------------------------------------------

readers = [CameraReader(timeline) for timeline in timelines]

writer = cv2.VideoWriter(
    str(output),
    fourcc,
    fps,
    (width * 2, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not initialize VideoWriter")

try:
    for i in range(total_frames):
        frame1 = readers[0].get_frame(i)
        frame2 = readers[1].get_frame(i)

        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

        combined = np.hstack((frame1, frame2))
        writer.write(combined)

        if i % 1000 == 0:
            print(f"Progress: {i / total_frames * 100:.1f}%")

finally:
    writer.release()

    for reader in readers:
        reader.close()

print(f"Saved: {output}")
