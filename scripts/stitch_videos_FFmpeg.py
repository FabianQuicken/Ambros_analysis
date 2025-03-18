import os
import subprocess
import glob as glob

def concatenate_videos(video_files, output_file):
    if not video_files:
        print("No video files provided.")
        return
    
    # Create a temporary file list for FFmpeg
    list_file = "file_list.txt"
    with open(list_file, "w") as f:
        for video in video_files:
            f.write(f"file '{video}'\n")
    
    # FFmpeg command to concatenate videos
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_file
    ]
    
    # Run the FFmpeg command
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        print(f"Successfully created {output_file}")
    else:
        print(f"Error: {result.stderr}")
    
    # Clean up temporary file
    os.remove(list_file)

if __name__ == "__main__":

    path = "Z:/n2023_odor_related_behavior/2023_behavior_setup_seminatural_odor_presentation/analyse/code_test/video_stitch/"
    video_list = glob.glob(os.path.join(path, '*.avi'))
    video_list.sort()
    
    output_video = path + "stitched_output.avi"

    concatenate_videos(video_list, output_video)