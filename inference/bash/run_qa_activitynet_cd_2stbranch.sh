CKPT_NAME="LongVA"
model_path="/model/LongVA"
cache_dir="./cache_dir"
video_dir="/datasets/ActivityNet_QA/videos"
gt_file_question="/datasets/ActivityNet_QA/test_q.json"
gt_file_answers="/datasets/ActivityNet_QA/test_a.json"
output_dir="/LongVA-main/output/ActivityNet_QA_VCD/tem1-2stbranch/${CKPT_NAME}"
max_frames_num=16
video_temcd_dir="datasets/ActivityNet_QA/shuffled_videos_4s"
video_spacd_dir="datasets/ActivityNet_QA/sam2_addmaskblack_withoutlabel_result"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
    #   --chat_model_path ${chat_model_path} \
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 code/LongVA-main/inference/run_inference_video_qa_activitynet_cd_2stbranch.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --max_frames_num $max_frames_num \
      --video_temcd_dir ${video_temcd_dir} \
      --video_spacd_dir ${video_spacd_dir} \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
