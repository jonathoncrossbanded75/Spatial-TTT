# Spatial-TTT Training

This directory contains the training code for **Spatial-TTT** (chunk size 2648, Spatial-TTT-demo-data).

## Repository structure

- `spatial_ttt_train.sh` — Single training script (LaCT/TTT, chunk 2648).
- `qwenvl/train/` — `train_spatial_ttt.py`, trainer, arguments.
- `qwenvl/data/` — Dataset config (`__init__.py`: set Spatial-TTT-demo-data paths), data processor.
- `models/` — LaCT/TTT layers.
- `scripts/` — DeepSpeed config (e.g. `zero2.json`).

## Requirements

- Python 3.10+
- `torch>=2.6.0`, `torchvision`, `transformers>=4.57.0`
- `deepspeed`, `flash-attn`, `accelerate`, `peft`, `triton`, `torchcodec`

## Data

We provide **Spatial-TTT-demo-data** ([THU-SI/Spatial-TTT-demo-data](https://huggingface.co/datasets/THU-SI/Spatial-TTT-demo-data) on Hugging Face), a high-quality spatial dataset with ~100k samples. Configure it in `qwenvl/data/__init__.py`: set `annotation_path` and `data_path` for `SPATIAL_TTT_DEMO_DATA` after downloading.

## Run training

1. Set `MODEL_PATH` and `OUTPUT_DIR` in `spatial_ttt_train.sh`.
2. From this directory:

```bash
bash spatial_ttt_train.sh
```

For full instructions and options, see the root [README](../README.md).
