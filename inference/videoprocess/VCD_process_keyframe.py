import os
import cv2
import torch
from tqdm import tqdm

def add_diffusion_noise_to_video(video_path, output_path, noise_step, num_frames):
    num_steps = 1000  # Number of diffusion steps
    betas = torch.linspace(0, 0.01, num_steps)  # Adjusted beta range
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0) * 0.1  # Scale down noise
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    print(f"Processing {video_path}, Total frames: {total_frames}")
    
    if total_frames < num_frames:
        print(f"Error: Video {video_path} does not have enough frames to process.")
        cap.release()
        return

    # Calculate middle part
    start_frame = (total_frames - num_frames) // 2
    end_frame = start_frame + num_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0  # Counter for processed frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_tensor = torch.tensor(frame).float() / 255.0  # Normalize the frame
        
        if start_frame <= frame_count < end_frame:
            noisy_frame = q_x(frame_tensor.permute(2, 0, 1), noise_step).permute(1, 2, 0).clamp(0, 1) * 255
            out.write(noisy_frame.byte().numpy())
        else:
            out.write(frame)  # Write the original frame

        frame_count += 1  # Increment the counter

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def process_videos_in_folder(input_folder, output_folder, noise_step, num_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.avi', '.mp4', '.mov'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        add_diffusion_noise_to_video(video_path, output_path, noise_step, num_frames)

# Example usage
input_folder = '/home/wuzhixuan/dataset/MSVD_Zero_Shot_QA/videos'  # Replace with your input folder path
output_folder = '/home/wuzhixuan/dataset/MSVD_Zero_Shot_QA/VCD_keyframe'  # Replace with your output folder path
noise_step = 500  # Example noise step
num_frames = 50  # Number of frames to process
process_videos_in_folder(input_folder, output_folder, noise_step, num_frames)
