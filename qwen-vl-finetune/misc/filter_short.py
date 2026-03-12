import json

input_file = "/mnt/public/wdk/proj/ECCV2026/VSI-Super/data/vsisuper-train-v2/vsc/vsc_train_regular_formatted_10mins.jsonl"
output_file = "/mnt/public/wdk/proj/ECCV2026/VSI-Super/data/vsisuper-train-v2/vsc/vsc_train_regular_formatted_10mins_filtered.jsonl"

suffix = " There maybe many rooms in the video, sum the counts from all rooms/scenes to produce the final total."

with (
    open(input_file, "r", encoding="utf-8") as fin,
    open(output_file, "w", encoding="utf-8") as fout,
):
    for line in fin:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        gpt_value = None
        for msg in data["conversations"]:
            if msg["from"] == "gpt":
                try:
                    gpt_value = int(msg["value"])
                except ValueError:
                    gpt_value = None
                break

        if gpt_value is not None and gpt_value < 10:
            continue

        for msg in data["conversations"]:
            if msg["from"] == "human":
                if not msg["value"].endswith(suffix):
                    msg["value"] += suffix

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Done. Saved to", output_file)
