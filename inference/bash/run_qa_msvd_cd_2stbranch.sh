CKPT_NAME="LongVA"
model_path="/home/zhangshaoxing/cv/model/LongVA"
cache_dir="./cache_dir"
video_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/videos"
gt_file_question="/home/zhangshaoxing/cv/datasets/MSVD_QA/test_q.json"
gt_file_answers="/home/zhangshaoxing/cv/datasets/MSVD_QA/test_a.json"
output_dir="/home/zhangshaoxing/cv/code/LongVA-main/output/MSVD_QA_VCD/tem1-2stbranch/${CKPT_NAME}"
max_frames_num=16
video_temcd_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/shuffled_videos_4s"
video_spacd_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/sam2_addmaskblack_withoutlabel_result"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
    #   --chat_model_path ${chat_model_path} \
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /home/zhangshaoxing/cv/code/LongVA-main/inference/run_inference_video_qa_vcd.py \
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
