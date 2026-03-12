#!/bin/bash
# Spatial-TTT training with chunk size 2648 and Spatial-TTT-Data-97k

export TORCHCODEC_FFMPEG_LOG_LEVEL=QUIET

# ======================
# Distributed
# ======================
NPROC_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

# ======================
# Paths (set these for your environment)
# ======================
MODEL_PATH="PATH_TO_PRETRAINED_QWEN3VL"
OUTPUT_DIR="./checkpoints/spatial_ttt_2648"
CACHE_DIR="./cache"

# ======================
# Data (Spatial-TTT-Data-97k: https://huggingface.co/datasets/THU-SI/Spatial-TTT-Data-97k)
# ======================
DATASET="spatial_ttt_data_97k"
VIDEO_MAX_FRAMES=128
RESIZE_HEIGHT=352
RESIZE_WIDTH=480

# ======================
# Training
# ======================
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
NUM_EPOCHS=1
MAX_LENGTH=65536

# ======================
# TTT / LaCT (chunk 2648 setting)
# ======================
LACT_CHUNK_SIZE=2648
WINDOW_SIZE=2648

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    qwenvl/train/train_spatial_ttt.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --dataset_use $DATASET \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --model_max_length $MAX_LENGTH \
    --bf16 True \
    --gradient_checkpointing True \
    --optim adamw_torch \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --min_lr_rate 0.1 \
    --weight_decay 0.01 \
    --dataloader_num_workers 0 \
    --video_fps 30 \
    --video_max_frames $VIDEO_MAX_FRAMES \
    --resize_height $RESIZE_HEIGHT \
    --resize_width $RESIZE_WIDTH \
    --video_min_frames 16 \
    --video_min_pixels 0 \
    --video_max_pixels 0 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --data_flatten True \
    --data_packing True \
    --deepspeed "scripts/zero2.json" \
    --lora_enable False \
    --lact_enable True \
    --num_lact_heads 4 \
    --lact_chunk_size $LACT_CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    --window_decay False \
    --use_muon True \
    --use_momentum True \
    --use_conv_layer True \
    --w0_w2_low_rank 0 \
    --learnable_ttt_scale True \
    --lact_lr 1e-5 \
    --lact_layers "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26" \
    --use_fused_kernel False \
    --seed 42
