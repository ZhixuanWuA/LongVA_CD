import os
import cv2
from tqdm import tqdm

def reverse_video(video_path, output_path):
    """
    Reverse the frames of the entire video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (frame_width, frame_height))

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    frames.reverse()

    for frame in frames:
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def process_videos_in_folder(input_folder, output_folder):
    """
    Process all videos in the input folder by reversing the order of frames.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.mp4")

        try:
            reverse_video(input_path, output_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

input_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/videos'  
output_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/reversed_videos'  
process_videos_in_folder(input_folder, output_folder)
