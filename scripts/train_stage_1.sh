export PROJECT=Qwen_2_5_VL_7B_Video_GRPO_stage_1
export OMP_NUM_THREADS=1

mkdir -p $PROJECT

export DEBUG_MODE="true"
export LOG_PATH="$PROJECT/debug_log.txt"
export VLLM_LOGGING_LEVEL=ERROR

torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/train.py \
    --deepspeed scripts/zero3.json \
    --output_dir $PROJECT \
    --model_name_or_path /path/to/model \
    --dataset_name data/train/stage_1.jsonl \
    --reward_funcs format segment_matching \
    --question_template frame_selection_thinking \
    --max_prompt_length 4096 \
    --min_pixels 12544 \
    --max_frames 448 \
    --total_pixels 2809856 \
    --learning_rate 1.5e-6 \
    --num_generations 8 \
    --max_completion_length 4096 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $PROJECT \
    --save_steps 100 \
    --save_total_limit 10 \
    --save_only_model true
