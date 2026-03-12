import torch
from transformers import AutoTokenizer

model_path = "/home/lff/bigdata1/huggingface/Qwen3-VL-2B-Instruct"
ids = torch.load("ids.pt", weights_only=False)[0].tolist()
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(tokenizer.decode(ids, skip_special_tokens=False))
