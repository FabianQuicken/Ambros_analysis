import cv2
import numpy as np
from pathlib import Path


def plot_center_on_video(
    video_path,
    output_path,
    center_x,
    center_y,
    center_status=None,
    radius=8,
    show_frame_number=True
):
    video_path = Path(video_path)
    output_path = Path(output_path)

    center_x = np.asarray(center_x, dtype=float).reshape(-1)
    center_y = np.asarray(center_y, dtype=float).reshape(-1)

    if len(center_x) != len(center_y):
        raise ValueError(
            f"center_x and center_y have different lengths: "
            f"{len(center_x)} and {len(center_y)}"
        )

    # Optionales 0/1-Array vorbereiten
    if center_status is not None:
        center_status = np.asarray(center_status).reshape(-1)

        """
        if len(center_status) != len(center_x):
            raise ValueError(
                f"center_status contains {len(center_status)} values, "
                f"but the center coordinates contain {len(center_x)} frames."
            )
        """
        if not np.all(np.isin(center_status, [0, 1])):
            raise ValueError(
                "center_status may only contain the values 0 and 1."
            )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    center_x = center_x[:video_frames]
    center_y = center_y[:video_frames]
    center_status = center_status[:video_frames]

    if video_frames != len(center_x):
        print(
            f"Warning: Video contains {video_frames} frames, "
            f"but center data contains {len(center_x)} frames."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not create output video: {output_path}")

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index < len(center_x):
            x = center_x[frame_index]
            y = center_y[frame_index]

            if np.isfinite(x) and np.isfinite(y):
                x = int(round(x))
                y = int(round(y))

                if 0 <= x < width and 0 <= y < height:

                    if center_status is None:
                        color = (0, 0, 255)  # Rot
                        label = "center"
                    elif center_status[frame_index] == 1:
                        color = (0, 255, 0)  # Grün
                        label = "center: 1"
                    else:
                        color = (0, 0, 255)  # Rot
                        label = "center: 0"

                    cv2.circle(
                        frame,
                        center=(x, y),
                        radius=radius,
                        color=color,
                        thickness=-1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )

        if show_frame_number:
            cv2.putText(
                frame,
                f"Frame: {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print(f"Video saved to: {output_path}")