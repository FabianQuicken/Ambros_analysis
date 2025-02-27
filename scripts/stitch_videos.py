import cv2
import os
import numpy as np
import glob as glob

def stitch_videos(video_files, output_file):
    if not video_files:
        print("No video files provided.")
        return
    
    # Open the first video to get properties
    first_video = cv2.VideoCapture(video_files[0])
    if not first_video.isOpened():
        print("Error: Cannot open the first video file.")
        return
    
    frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(first_video.get(cv2.CAP_PROP_FPS))
    first_video.release()
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Warning: Cannot open {video_file}, skipping...")
            continue
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    
    out.release()
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/code_test/video_stitch/"
    video_list = glob.glob(os.path.join(path, '*.avi'))
    video_list.sort()
    
    output_video = path + "stitched_output.avi"
    stitch_videos(video_list, output_video)