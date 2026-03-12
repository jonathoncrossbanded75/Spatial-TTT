#!/bin/bash

set -euo pipefail

# Optional: set HF_TOKEN, HF_ENDPOINT, HF_HOME if needed for dataset download
# export HF_TOKEN=""
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME=~/.cache/huggingface

ckpt_path=$1
output_name=$2
nprocs=${3:-8}
lact_chunk_size=${4:-2648}
window_size=${5:-2648}
resize_height=${6:-352}
resize_width=${7:-480}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../lmms-eval" || exit 1

MODEL="qwen3_vl_spatial_ttt"
PRETRAINED="${PRETRAINED:-Qwen/Qwen3-VL-2B-Instruct}"
frame_count=128

model_args=(
    "pretrained=${PRETRAINED}"
    "lact_checkpoint_path=${ckpt_path}"
    "num_lact_heads=4"
    "w0_w2_low_rank=0"
    "use_fused_kernel=False"
    "use_conv_layer=True"
    "lact_chunk_size=${lact_chunk_size}"
    "window_size=${window_size}"
    "resize_height=${resize_height}"
    "resize_width=${resize_width}"
    "lact_layers=0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26"
    "max_pixels=99999999999"
    "fps=16"
    "max_num_frames=${frame_count}"
    "attn_implementation=flash_attention_2"
    "interleave_visuals=false"
)
MODEL_ARGS="$(IFS=,; echo "${model_args[*]}")"

OUTPUT_DIR="${script_dir}/../results/vsibench/${output_name}-${frame_count}f"
mkdir -p "${OUTPUT_DIR}"
accelerate launch --main_process_port 29502 --num_processes=$nprocs -m lmms_eval \
    --model "${MODEL}" \
    --model_args="${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --verbosity DEBUG 2>&1 | tee "${OUTPUT_DIR}/log.txt"
