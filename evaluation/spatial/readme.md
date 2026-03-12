# VSI-Bench Evaluation

This directory contains evaluation code and scripts for **VSI-Bench** (Visual-based Spatial Intelligence Benchmark). The code is based on [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Script

- **`scripts/eval_spatial_ttt_2b.sh`** — Evaluate Spatial-TTT (Qwen3-VL-2B) on VSI-Bench.

  Usage:
  ```bash
  ./eval_spatial_ttt_2b.sh <ckpt_path> <output_name> [nprocs] [lact_chunk_size] [window_size] [resize_height] [resize_width]
  ```
  Evaluates on VSI-Bench with **128 frames** only. Example: `./eval_spatial_ttt_2b.sh /path/to/checkpoint my_model 8`

  Optional env: `PRETRAINED` to override base model (default: `Qwen/Qwen3-VL-2B-Instruct`).

## Results

Summarize results with `misc/summarize_vsibench_results.py`.
