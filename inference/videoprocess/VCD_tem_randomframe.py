import os
import cv2
import torch
from tqdm import tqdm
import random

def drop_frames_randomly(video_path, output_path, fps=1):
    """
    Randomly drop frames from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, original_fps // fps)  # Interval to sample frames

    selected_frames = []
    for i in range(0, total_frames, frame_interval):
        if random.random() > 0.5:  # Randomly decide whether to keep the frame
            selected_frames.append(i)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()
    return output_path

def process_videos_in_folder(input_folder, output_folder, noise_step, num_frames):
    """
    Process all videos in the input folder by dropping frames randomly.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to save processed videos.
        noise_step (int): Noise step for processing.
        num_frames (int): Number of frames to process.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.mp4")

        try:
            drop_frames_randomly(input_path, output_path, fps=1)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

# Example usage
input_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/videos'  
output_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/randomframe'  
noise_step = 500  
num_frames = 50  
process_videos_in_folder(input_folder, output_folder, noise_step, num_frames)
