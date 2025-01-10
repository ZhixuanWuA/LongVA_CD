#!/bin/bash

pred_path="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/merge.jsonl"
output_dir="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/gpt/result"
output_json="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_ori/tem1/LongVA/results.json"
# api_key="sk-r6DBaecYzPH9wr1sWQ5W5SFaDHdfxPRKSLTqtDEiZLcIvFEx"
# api_base="https://api.key77qiqi.cn/v1"
# api_key="sk-q9CcVdfnSCoeUlDE5285F6C0C1834aA08b52F68cB36190E0"
# api_base="https://api.key77qiqi.cn/v1"
api_key="sk-pzQO99KnMESlhTVt59454eD8B226499dB304996dE263F74e"
api_base="http://43.154.251.242:10050/v1"
num_tasks=8



python3 /home/zhangshaoxing/cv/code/LongVA-main/inference/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}