#!/bin/bash

pred_path="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/merge.jsonl"
output_dir="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/gpt/result"
output_json="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/results.json"
api_key=""
api_base=""
num_tasks=8



python3 /home/zhangshaoxing/cv/code/LongVA-main/inference/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
