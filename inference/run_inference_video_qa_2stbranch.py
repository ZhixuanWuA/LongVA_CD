import math
import os
import argparse
import json

import torch
import transformers
from transformers import set_seed
from tqdm import tqdm
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import shutil  # 导入 shutil 模块

from videoprocess.VCD_sample_2stbranch import evolve_vcd_sampling
evolve_vcd_sampling()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)  
    parser.add_argument('--cache_dir', help='', required=True)   
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)  
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)  
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)   
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True) 
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)  
    parser.add_argument("--num_chunks", type=int, default=1)  
    parser.add_argument("--chunk_idx", type=int, default=0)  
    parser.add_argument("--device", type=str, required=False, default='cuda:0')  
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)  
    parser.add_argument("--model_max_length", type=int, required=False, default=2048) 
    
    parser.add_argument("--max_frames_num", type=int, required=True, default=16) 

    parser.add_argument("--top_p", type=float, default=0.5) 
    parser.add_argument("--top_k", type=int, default=100)  

    parser.add_argument('--video_temcd_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--video_spacd_dir', help='Directory containing video files.', required=True)
    parser.add_argument("--noise_step", type=int, default=500)  
    parser.add_argument("--use_temcd", action='store_true', default=True) 
    parser.add_argument("--use_spacd", action='store_true', default=True)  
    parser.add_argument("--cd_alpha", type=float, default=1)  
    parser.add_argument("--cd_beta", type=float, default=0.5)  
    parser.add_argument("--seed", type=int, default=42)  

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, video_temcd, video_spacd, qs, args):

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n" + qs + " Please answer concisely based on the video content, and avoid providing any irrelevant or unrelated information.<|im_end|>\n<|im_start|>assistant\n"

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    vr = VideoReader(video, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = video_processor.preprocess(frames, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    
    vr_spa = VideoReader(video_spacd, ctx=cpu(0))
    total_frame_num_spa = len(vr_spa)
    uniform_sampled_frame_spa = np.linspace(0, total_frame_num_spa - 1, args.max_frames_num, dtype=int)
    frame_idx_spa = uniform_sampled_frame_spa.tolist()
    frames_spa = vr.get_batch(frame_idx_spa).asnumpy()
    
    vr_tem = VideoReader(video_temcd, ctx=cpu(0))
    total_frame_num_tem = len(vr_tem)
    uniform_sampled_frame_tem = np.linspace(0, total_frame_num_tem - 1, args.max_frames_num, dtype=int)
    frame_idx_tem = uniform_sampled_frame_tem.tolist()
    frames_tem = vr.get_batch(frame_idx_tem).asnumpy()

    if args.use_temcd:
            print(f'begining tem cd function...')
            video_tensor_cd = video_processor.preprocess(frames_tem, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    else:
            video_tensor_cd = None   
            
    if args.use_spacd:
            print(f'begining spa cd function...')
            video_tensor_spacd = video_processor.preprocess(frames_spa, return_tensors='pt')['pixel_values'][0].half().to(args.device)
            # print(f'tensor_cd:{video_tensor_cd}')
    else:
            
            video_tensor_spacd = None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            image_tem=(video_tensor_cd.unsqueeze(0).to(args.device) if video_tensor_cd is not None else None),
            images_mask=(video_tensor_spacd.unsqueeze(0).to(args.device) if video_tensor_spacd is not None else None),
            cd_alpha = args.cd_alpha,
            cd_beta = args.cd_beta,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=1024,
            use_cache=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    tokenizer, model, video_processor, _ = load_pretrained_model(args.model_path, args.model_base, "llava_qwen")
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            video_mask_folder = os.path.join(args.video_spacd_dir, str(id))
            mask_temp_path = os.path.join(video_mask_folder, f"{id}_output.mp4")
            video_temcd_dir = os.path.join(args.video_temcd_dir, f"{video_name}.mp4")
            
            if os.path.exists(temp_path):
                video_path = temp_path
                if os.path.exists(mask_temp_path):
                    video_spacd_dir = mask_temp_path
                    print(f'video_spacd_dir:{video_spacd_dir}')
                    if os.path.exists(video_temcd_dir):
                        print(f'video_temcd_dir:{video_temcd_dir}')
                    # try:
                    # Run inference on the video and add the output to the list
                        output = get_model_output(model, video_processor, tokenizer, video_path, video_temcd_dir, video_spacd_dir, question, args)
                        sample_set['pred'] = output
                        output_list.append(sample_set)
                        ans_file.write(json.dumps(sample_set) + "\n")
                        break

    ans_file.close()
    
    # 将当前脚本文件复制到 output_dir 目录
    current_file_path = os.path.abspath(__file__)  
    shutil.copy(current_file_path, args.output_dir)  


if __name__ == "__main__":
    args = parse_args()
    # set_seed(args.seed)
    run_inference(args)
