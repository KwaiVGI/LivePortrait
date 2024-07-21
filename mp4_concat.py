from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def merge_mp4_files(input_folder, output_file):
    # Get all MP4 files in the input folder
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    if not mp4_files:
        print("No MP4 files found in the specified folder.")
        return

    # Sort files to ensure consistent order
    mp4_files.sort()

    # Create VideoFileClip objects for each file
    clips = [VideoFileClip(os.path.join(input_folder, mp4)) for mp4 in mp4_files]

    # Concatenate all clips
    final_clip = concatenate_videoclips(clips)

    # Write the result to a file
    final_clip.write_videofile(output_file)

    # Close all clips
    final_clip.close()
    for clip in clips:
        clip.close()

# Example usage
input_folder = "D:\LivePortrait/animations\concat"
output_file = "D:\LivePortrait/animations\s20--output_12-16.mp4"
merge_mp4_files(input_folder, output_file)
