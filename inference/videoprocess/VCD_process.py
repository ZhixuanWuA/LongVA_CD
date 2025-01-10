import cv2
import os
import torch
from tqdm import tqdm

def add_diffusion_noise_to_video(video_path, output_path, noise_step):
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
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_tensor = torch.tensor(frame).float() / 255.0  # Normalize the frame
        frame_list.append(frame_tensor)

    frame_tensor = torch.stack(frame_list)  # Stack frames into a tensor
    key_frame_indices = get_key_frame_indices(frame_tensor, threshold=8000)

    for i in range(len(frame_tensor)):
        if i in key_frame_indices:
            # print(f'Processing key frame index: {i}')
            noisy_frame = q_x(frame_tensor[i].permute(2, 0, 1), noise_step).permute(1, 2, 0).clamp(0, 1) * 255
            out.write(noisy_frame.byte().numpy())
        else:
            out.write((frame_tensor[i] * 255).byte().numpy())  # Write unchanged frame

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def process_videos_in_folder(input_folder, output_folder, noise_step):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.avi', '.mp4', '.mov'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        add_diffusion_noise_to_video(input_path, output_path, noise_step)

def get_key_frame_indices(video_tensor, threshold=10):
    key_frame_indices = []
    video_tensor = video_tensor.float()  # Normalize values to [0, 1]
    total_frames = video_tensor.size(0)

    for i in range(1, total_frames):
        prev_frame = video_tensor[i-1]
        curr_frame = video_tensor[i]

        # Calculate absolute difference between frames
        diff = torch.abs(prev_frame - curr_frame).sum()
        print(f'Frame {i-1} to {i} difference: {diff.item()}')  # 输出差异值

        # Record current frame as a key frame if difference exceeds threshold
        if diff.item() > threshold:
            key_frame_indices.append(i)

    return key_frame_indices

# Example usage
input_folder = '/home/wuzhixuan/code/Video-LLaVA/videollava/serve/examples/test'  # Replace with your input folder path
output_folder = '/home/wuzhixuan/code/Video-LLaVA/videollava/serve/examples/output'  # Replace with your output folder path
noise_step = 500  # Example noise step
process_videos_in_folder(input_folder, output_folder, noise_step)
