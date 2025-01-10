import os
import cv2
from tqdm import tqdm

def interval_sample(video_path, output_path, interval, min_duration=4, target_fps=3):
    """
    Sample frames from a video at a fixed interval, ensuring minimum duration and target FPS.

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate minimum frames required based on target FPS and minimum duration
    min_frames = min_duration * target_fps

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

    sampled_frames = []
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        sampled_frames.append(frame)

    # Ensure minimum duration by duplicating frames if necessary
    while len(sampled_frames) < min_frames:
        sampled_frames.extend(sampled_frames[:min_frames - len(sampled_frames)])

    for frame in sampled_frames[:min_frames]:
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def process_videos_in_folder(input_folder, output_folder, interval):
    """
    Process all videos in the input folder by sampling frames at a fixed interval.

    """
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.mp4")

        try:
            interval_sample(input_path, output_path, interval, min_duration=2, target_fps=3)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

# Example usage
input_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/videos'  
output_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/interval_sampled_4'  
interval = 30  
process_videos_in_folder(input_folder, output_folder, interval)
