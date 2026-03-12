import json
import os


def read_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(jsonl_path, iterable):
    with open(jsonl_path, "w") as f:
        for obj in iterable:
            f.write(json.dumps(obj))
            f.write("\n")


def process_video_path(orig_video_path):
    # rootdir = "/home/lff/data1/cjw/projs/qwen3vl/qwen-vl-finetune/data_processed"
    # return f"{rootdir}/{orig_video_path.split('/')[-1].replace('.mp4', '.pt')}"
    rootdir = "/home/lff/data1/cjw/projs/qwen3vl/qwen-vl-finetune/data"
    return f"{rootdir}/{orig_video_path.split('/')[-1]}"


datadir = "/home/lff/bigdata1/cjw/projs/qwen3vl/qwen-vl-finetune/data"
video_path = set(obj for obj in os.listdir(datadir) if obj.endswith(".mp4"))

data = list(read_jsonl(f"{datadir}/sp-mllm-502k.jsonl"))
filtered_data = [item for item in data if item["video"].split("/")[-1] in video_path]

datadir = "/home/lff/bigdata1/cjw/projs/qwen3vl/qwen-vl-finetune/data"
video_path = set(obj for obj in os.listdir(datadir) if obj.endswith(".mp4"))

data = list(read_jsonl(f"{datadir}/sp-mllm-502k.jsonl"))
filtered_data = [
    {"video": process_video_path(item["video"]), "conversations": item["conversations"]}
    for item in data
    if item["video"].split("/")[-1] in video_path
]

write_jsonl(f"{datadir}/sp-mllm-502k_filtered.jsonl", filtered_data)
