
import cv2
from pathlib import Path
from datetime import datetime

folder = Path(r"Z:\n2023_odor_related_behavior\2026_sick_behavior\kamera-merge-test")

cameras = ["stim_top1_40439817", "home_top2_40405188"]

def start_time(path):
    return datetime.strptime(
        "_".join(path.stem.split("_")[:6]),
        "%Y_%m_%d_%H_%M_%S"
    )

for camera in cameras:
    videos = sorted(
        [p for p in folder.glob("*.avi") if camera in p.stem],
        key=start_time
    )

    print(f"\n{'=' * 70}")
    print(f"CAMERA: {camera}")
    print(f"{'=' * 70}")

    for i, video in enumerate(videos):
        cap = cv2.VideoCapture(str(video))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration = frames / fps
        start = start_time(video)

        print(f"\n{video.name}")
        print(f"  Start:      {start}")
        print(f"  FPS:        {fps:.4f}")
        print(f"  Frames:     {frames}")
        print(f"  Duration:   {duration:.2f} s")

        if i + 1 < len(videos):
            next_start = start_time(videos[i + 1])

            interval = (next_start - start).total_seconds()
            difference = interval - duration

            print(f"  Next start: {next_start}")
            print(f"  Interval:   {interval:.2f} s")
            print(f"  Difference: {difference:+.2f} s")

            if difference > 1:
                print("  -> GAP between recordings")
            elif difference < -1:
                print("  -> OVERLAPPING recordings")
            else:
                print("  -> Recording lengths approximately match")
