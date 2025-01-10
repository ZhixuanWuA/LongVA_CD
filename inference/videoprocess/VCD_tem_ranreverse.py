import os
import cv2
import random
from tqdm import tqdm

def split_video_into_segments(video_path, segment_duration=4, target_fps=30):
    """
    Split the video into segments of fixed duration, keeping the frame order within each segment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the number of frames per segment
    segment_frames = segment_duration * fps
    
    # List to hold the segments
    segments = []
    
    # Split video into segments
    for i in range(0, total_frames, segment_frames):
        segment_frames_data = []
        for j in range(segment_frames):
            frame_idx = i + j
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames_data.append(frame)
        
        # If segment is not empty, append it
        if segment_frames_data:
            segments.append(segment_frames_data)
    
    cap.release()
    return segments, frame_width, frame_height, fps

def shuffle_video_segments(segments, output_path, target_fps=30, frame_width=640, frame_height=480):
    """
    Shuffle the segments and write them to a new video file, keeping the internal order of frames intact.
    """
    # Randomly shuffle the segments
    random.shuffle(segments)

    # Define video writer with 3fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

    # Write the shuffled segments to the output video
    for segment in segments:
        for frame in segment:
            out.write(frame)

    out.release()

def process_videos_in_folder(input_folder, output_folder, segment_duration=4, target_fps=30):
    """
    Process all videos in the input folder by splitting them into segments, shuffling the segments, 
    and saving the shuffled video to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.mp4")

        try:
            segments, frame_width, frame_height, fps = split_video_into_segments(input_path, segment_duration, target_fps)
            
            # If the video is shorter than one segment, treat the whole video as one segment
            if not segments:
                cap = cv2.VideoCapture(input_path)
                ret, frame = cap.read()
                frames = []
                while ret:
                    frames.append(frame)
                    ret, frame = cap.read()
                cap.release()
                segments.append(frames)
            
            shuffle_video_segments(segments, output_path, target_fps, frame_width, frame_height)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

# Example usage
input_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/videos'  
output_folder = '/home/zhangshaoxing/cv/datasets/MSVD_QA/shuffled_videos_4s'  
segment_duration = 4  
target_fps = 30  
process_videos_in_folder(input_folder, output_folder, segment_duration, target_fps)
